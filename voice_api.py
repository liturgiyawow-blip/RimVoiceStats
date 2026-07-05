# ============ voice_api.py ============
# УПРОЩЁННЫЙ ПРОКСИ — диалоги пешек + TTS + Personality Web
# =====================================

import json
import queue
import re
from typing import Any

from flask import Flask, request, Response

from config import Config
from database import PawnDatabase
from cache import RAMWavCache
from tts_engine import PiperVoiceManager, PiperTTSEngine, TTSWorker
from llm_parser import LLMResponseParser

# === ПОДКЛЮЧЕНИЕ МОДУЛЯ ГЕНЕРАЦИИ ЛИЧНОСТЕЙ ===
try:
    from personality_web import register_personality_routes
    PERSONALITY_WEB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [PersonalityWeb] Модуль не найден: {e}")
    PERSONALITY_WEB_AVAILABLE = False
    def register_personality_routes(app, config):
        pass
# =============================================


def _json_response(data: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(data, ensure_ascii=False, indent=None),
        status=status,
        mimetype='application/json; charset=utf-8'
    )


def _build_prompt_context(messages: list, max_msgs: int = 10) -> str:
    recent = messages[-max_msgs:] if len(messages) > max_msgs else messages
    parts = [str(m.get("content", "")) for m in recent if m.get("content")]
    return " ".join(parts)


def create_app(config: Config,
               db: PawnDatabase,
               tts_queue: queue.Queue,
               parser: LLMResponseParser) -> Flask:
    app = Flask(__name__)

    _GLOBAL_SPEECH_VOL = max(0.0, min(2.0, config.get("speech_volume", 1.0)))
    _GLOBAL_THOUGHT_VOL = max(0.0, min(2.0, config.get("thought_volume", 0.4)))

    def _add_tts_task(text: str, context: str = "", is_thought: bool = False,
                      voice: str = None, profile: dict = None,
                      post_delay: float = None):
        """Положить задачу озвучки в очередь (одна фраза)."""
        clean = text.replace("*", "").replace("#", "").replace("_", "").strip()
        clean = clean.replace("[", "").replace("]", "")
        clean = clean.replace("{", "").replace("}", "")

        if not clean:
            return

        if profile is None and voice is not None:
            profile = {
                "voice": voice,
                "speaker_id": 0,
                "length_scale": 1.0,
                "noise_scale": 0.667,
                "noise_w": 0.8,
                "gender": "female",
                "volume": 1.0,
            }

        if profile is None:
            m = re.match(r"^([А-ЯЁа-яёA-Za-z0-9_\-]+)\s*:", clean)
            name = m.group(1).capitalize() if m else "Unknown_Pawn"
            profile = db.get_or_create_profile(name, clean, context)

        pawn_vol = max(0.0, min(2.0, profile.get("volume", 1.0)))

        if is_thought:
            vol = _GLOBAL_SPEECH_VOL * _GLOBAL_THOUGHT_VOL * pawn_vol
        else:
            vol = _GLOBAL_SPEECH_VOL * pawn_vol

        vol = max(0.0, min(1.5, vol))

        if post_delay is None:
            delay = config.get("thought_delay_seconds", 1.0) if is_thought else 0.0
        else:
            delay = post_delay

        task = {
            "text": clean,
            "voice": profile["voice"],
            "speaker_id": profile.get("speaker_id", 0),
            "length_scale": profile.get("length_scale", 1.0),
            "noise_scale": profile.get("noise_scale", 0.667),
            "noise_w": profile.get("noise_w", 0.8),
            "volume": vol,
            "post_delay": delay,
            "is_thought": is_thought,
        }

        try:
            tts_queue.put_nowait(task)
        except queue.Full:
            print(f"⚠️ Очередь TTS переполнена! Пропускаем: {clean[:40]}...")

    def _extract_speaker_name(prompt_context: str) -> str:
        """Ищет имя говорящего пешки в контексте системного промпта."""
        if not prompt_context:
            return "Unknown_Pawn"

        patterns = [
            (r"\[([А-ЯЁа-яёA-Za-z0-9_\-]+)'s\s+Status\]", 1),
            (r"\[Character:\s*([А-ЯЁа-яёA-Za-z0-9_\-]+)\]", 1),
            (r"Role-playing as\s+([А-ЯЁа-яёA-Za-z0-9_\-]+)", 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, prompt_context, re.IGNORECASE)
            if match:
                name = match.group(group).strip()
                if name.lower() not in {"reply", "thought", "rimmind", "current", 
                                         "colonist", "rimworld", "unknown_pawn", 
                                         "assistant", "system"}:
                    return name

        return "Unknown_Pawn"

    def _queue_name_then_speech(pawn_speech: str, prompt_context: str,
                                pawn_profile: dict = None):
        """
        Атомарная озвучка: имя тихо + реплика нормальной громкостью.
        Вся цепочка кладётся в очередь как ОДНА задача — 
        имена и реплики разных пешек никогда не перемешаются.
        """
        clean_text = pawn_speech.replace("*", "").replace("#", "").replace("_", "").strip()
        clean_text = re.sub(r'\[.*?\]', '', clean_text)
        clean_text = re.sub(r'\{.*?\}', '', clean_text)

        if not clean_text:
            return

        pawn_name = _extract_speaker_name(prompt_context)

        if pawn_profile is None:
            pawn_profile = db.get_or_create_profile(pawn_name, clean_text, prompt_context)

        segments = []

        # Шаг 1: Имя (тихо)
        if pawn_name and pawn_name != "Unknown_Pawn":
            name_vol_multiplier = config.get("name_volume_multiplier", 0.4)
            name_vol = _GLOBAL_SPEECH_VOL * name_vol_multiplier * max(0.0, min(2.0, pawn_profile.get("volume", 1.0)))
            name_vol = max(0.0, min(1.5, name_vol))

            segments.append({
                "text": f"{pawn_name}.",
                "volume": name_vol,
                "post_delay": config.get("name_post_delay", 0.8),
            })

        # Шаг 2: Реплика (убираем имя из начала текста, если LLM его продублировал)
        speech_body = re.sub(r'^([А-ЯЁа-яёA-Za-z0-9_\-]+)\s*:\s*', '', clean_text).strip()
        if speech_body:
            pawn_vol = max(0.0, min(2.0, pawn_profile.get("volume", 1.0)))
            speech_vol = _GLOBAL_SPEECH_VOL * pawn_vol
            speech_vol = max(0.0, min(1.5, speech_vol))

            segments.append({
                "text": speech_body,
                "volume": speech_vol,
                "post_delay": 0.0,
            })

        if not segments:
            return

        task = {
            "voice": pawn_profile.get("voice", config.get("eva_voice")),
            "speaker_id": pawn_profile.get("speaker_id", 0),
            "length_scale": pawn_profile.get("length_scale", 1.0),
            "noise_scale": pawn_profile.get("noise_scale", 0.667),
            "noise_w": pawn_profile.get("noise_w", 0.8),
            "segments": segments,
            "is_thought": False,
        }

        try:
            tts_queue.put_nowait(task)
            print(f"   🏷️ Озвучка: {pawn_name} → {speech_body[:50]}...")
        except queue.Full:
            print(f"⚠️ Очередь TTS переполнена! Пропускаем реплику {pawn_name}.")

    # ============ РОУТЫ ============

    @app.route('/v1/chat/completions', methods=['POST'])
    def proxy_chat():
        """Главный роут. Диалоги пешек → LM Studio → TTS."""
        if not request.is_json:
            return _json_response({"error": "Нужен JSON"}, 400)

        data = request.get_json()

        # --- Перехват для PersonalityWeb ---
        if hasattr(app, '_personality_interceptor') and app._personality_interceptor:
            app._personality_interceptor(data)
        # ----------------------------------

        messages = data.get('messages', [])
        context = _build_prompt_context(messages, config.get("max_context_messages", 10))

        try:
            import requests
            resp = requests.post(
                config.get("lm_studio_url"),
                json=data,
                timeout=(5, 60)
            )

            if resp.status_code != 200:
                return _json_response({
                    "error": "Ошибка LLM",
                    "status": resp.status_code,
                    "detail": resp.text[:500]
                }, resp.status_code)

            rj = resp.json()

            try:
                raw = rj['choices'][0]['message']['content'].strip()
            except (KeyError, IndexError, AttributeError):
                return _json_response(rj, 200)

            parsed = parser.parse(raw)
            speaker_name = _extract_speaker_name(context)
            speaker_profile = None
            if speaker_name != "Unknown_Pawn":
                speaker_profile = db.get_or_create_profile(speaker_name, "", context)

            # Мысль (тихо)
            thought_text = parsed.get("thought")
            if isinstance(thought_text, dict):
                thought_text = thought_text.get("description", "")

            if config.get("enable_thought_tts", False) and thought_text and str(thought_text).strip():
                print(f"\n🧠 [Рассуждение]: {thought_text}")
                _add_tts_task(
                    str(thought_text).strip(), 
                    context, 
                    is_thought=True, 
                    profile=speaker_profile,
                    post_delay=config.get("name_pre_delay", config.get("thought_delay_seconds", 1.0))
                )

            # Реплика (имя + речь, атомарно)
            if parsed.get("reply"):
                print(f"\n💬 [Диалог]: {parsed['reply']}")
                _queue_name_then_speech(parsed["reply"], context, pawn_profile=speaker_profile)

            return _json_response(rj, 200)

        except requests.exceptions.Timeout as e:
            print(f"❌ Таймаут LLM: {e}")
            return _json_response({"error": "Таймаут", "details": str(e)}, 504)
        except requests.exceptions.ConnectionError as e:
            print(f"❌ LLM недоступен: {e}")
            return _json_response({"error": "Нет соединения", "details": str(e)}, 503)
        except Exception as e:
            print(f"❌ Ошибка связи: {e}")
            return _json_response({"error": "Ошибка сети", "details": str(e)}, 503)

    @app.route('/rimmind/voice/speak', methods=['POST'])
    def speak_system():
        """Системная озвучка (для отладки или внешних скриптов)."""
        data = request.get_json(force=True, silent=True)
        if not data or 'text' not in data:
            return _json_response({"status": "error", "message": "Нет текста"}, 400)

        text = data['text'].replace("*", "").replace("#", "").replace("_", "").strip()
        if not text:
            return _json_response({"status": "error", "message": "Пустой текст"}, 400)

        voice = data.get('voice', config.get("eva_voice", "ru_RU-irina-medium"))
        _add_tts_task(text, voice=voice, is_thought=False)
        return _json_response({"status": "success", "queued": True})

    @app.route('/rimmind/voice/register_pawns', methods=['POST'])
    def register_pawns():
        """C# мод присылает список пешек (только для голосовых профилей)."""
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return _json_response({"status": "error"}, 400)

        pawns_list = data.get('pawns', [])
        added = 0
        for p in pawns_list:
            name = p.get('shortName') or p.get('name')
            if name:
                db.register_known_pawn(name)
                added += 1

        return _json_response({
            "status": "success",
            "registered": added
        })

    @app.route('/rimmind/voice/pawns', methods=['GET'])
    def get_pawns():
        """Список известных пешек (информационно)."""
        return _json_response({
            "pawns": db.get_known_pawns()
        })

    @app.route('/rimmind/voice/status', methods=['GET'])
    def voice_status():
        """Статус системы."""
        from tts_engine import PIPER_AVAILABLE
        return _json_response({
            "status": "online",
            "piper_available": PIPER_AVAILABLE,
            "gpu": config.get("use_gpu", False),
            "parser_stats": parser.get_stats(),
        })

    @app.route('/rimmind/voice/export', methods=['POST'])
    def export_db():
        db.export_to_json()
        return _json_response({"status": "success", "file": config.get("json_backup")})

    @app.route('/rimmind/voice/import', methods=['POST'])
    def import_db():
        db.import_from_json()
        return _json_response({"status": "success", "file": config.get("json_backup")})

    # === РЕГИСТРАЦИЯ РОУТОВ PERSONALITY WEB ===
    if PERSONALITY_WEB_AVAILABLE:
        register_personality_routes(app, config)
    # ==========================================

    return app