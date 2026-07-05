# personality_web/routes.py
import json
import os
from flask import Flask, request, Response, render_template
from typing import Dict, Any

from .pawn_cache import pawn_cache, PawnExtractor
from .llm_client import PersonalityLLMClient
from .personality_config import PersonalityConfig


def register_personality_routes(app: Flask, config):
    # Guard: не регистрировать повторно
    if app.config.get('_PERSONALITY_ROUTES_REGISTERED'):
        print("[PersonalityWeb] Роуты уже зарегистрированы, пропускаем.")
        return
    app.config['_PERSONALITY_ROUTES_REGISTERED'] = True

    lm_url = config.get("lm_studio_url", "http://127.0.0.1:1234/v1/chat/completions")
    llm_client = PersonalityLLMClient(lm_url)
    pcfg = PersonalityConfig()

    # --- Шаблоны: подключаем папку templates внутри personality_web ---
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if os.path.exists(templates_dir):
        if hasattr(app, 'jinja_loader') and hasattr(app.jinja_loader, 'searchpath'):
            if templates_dir not in app.jinja_loader.searchpath:
                app.jinja_loader.searchpath.insert(0, templates_dir)
        else:
            app.template_folder = templates_dir
        print(f"[PersonalityWeb] Шаблоны подключены: {templates_dir}")

    # --- Перехватчик: вытаскивает пешку из промпта и пишет в кэш ---
    def intercept_pawn_data(request_data: Dict):
        try:
            messages = request_data.get("messages", [])
            if len(messages) < 2:
                return
            user_content = ""
            for msg in messages:
                content = msg.get("content", "")
                if "'s Status]" in content or "Status]" in content:
                    user_content = content
                    break
            if not user_content:
                return
            extracted = PawnExtractor.extract_from_prompt(user_content)
            if extracted and extracted.get("name"):
                pawn_cache.upsert(extracted["name"], extracted)
                pawn_cache.save_if_dirty()
                print(f"[PersonalityWeb] Перехвачена пешка: {extracted['name']} (поля: {list(extracted.keys())})")
        except Exception as e:
            print(f"[PersonalityWeb] Ошибка перехвата: {e}")
            import traceback
            traceback.print_exc()

    app._personality_interceptor = intercept_pawn_data

    # ============ API ============

    @app.route("/rimmind/personality/pawns", methods=["GET"])
    def api_pawns_list():
        return _json_response({"pawns": pawn_cache.get_all()})

    @app.route("/rimmind/personality/pawns/<name>", methods=["GET"])
    def api_pawn_detail(name):
        pawn = pawn_cache.get(name)
        if not pawn:
            return _json_response({"error": "Пешка не найдена"}, 404)
        return _json_response(pawn)

    @app.route("/rimmind/personality/generate", methods=["POST"])
    def api_generate():
        data = request.get_json() or {}
        name = data.get("name")
        field = data.get("field", "description")
        if not name:
            return _json_response({"error": "Не указано имя пешки"}, 400)
        if field not in ("description", "workTendencies", "socialTendencies"):
            return _json_response({"error": "Неверное поле"}, 400)
        pawn = pawn_cache.get(name)
        if not pawn:
            return _json_response({"error": "Пешка не в кэше. Сначала открой её окно в игре."}, 404)
        result = llm_client.generate_field(pawn, field)
        pawn_cache.set_field(name, field, result)
        return _json_response({"field": field, "text": result})

    @app.route("/rimmind/personality/generate_all", methods=["POST"])
    def api_generate_all():
        data = request.get_json() or {}
        name = data.get("name")
        if not name:
            return _json_response({"error": "Не указано имя"}, 400)
        pawn = pawn_cache.get(name)
        if not pawn:
            return _json_response({"error": "Пешка не в кэше"}, 404)
        result = llm_client.generate_all(pawn)
        for field, value in result.items():
            pawn_cache.set_field(name, field, value)
        return _json_response(result)

    @app.route("/rimmind/personality/generate_pure", methods=["POST"])
    def api_generate_pure():
        data = request.get_json() or {}
        name = data.get("name")
        if not name:
            return _json_response({"error": "Не указано имя пешки"}, 400)
        pawn = pawn_cache.get(name)
        if not pawn:
            return _json_response({"error": "Пешка не в кэше. Открой её окно в игре."}, 404)
        result = llm_client.generate_pure_profile(pawn)
        pawn_cache.set_field(name, "pure_profile", result)
        return _json_response({"field": "pure_profile", "text": result})

    @app.route("/rimmind/personality/generate_story", methods=["POST"])
    def api_generate_story():
        data = request.get_json() or {}
        name = data.get("name")
        include_colony = data.get("include_colony", True)
        if not name:
            return _json_response({"error": "Не указано имя пешки"}, 400)
        pawn = pawn_cache.get(name)
        if not pawn:
            return _json_response({"error": "Пешка не в кэше. Открой её окно в игре."}, 404)
        colony_pawns = pawn_cache.get_all() if include_colony else None
        result = llm_client.generate_daily_story(pawn, colony_pawns)
        pawn_cache.set_field(name, "daily_story", result)
        return _json_response({"field": "daily_story", "text": result})

    @app.route("/rimmind/personality/clear", methods=["POST"])
    def api_clear():
        pawn_cache.clear()
        return _json_response({"status": "ok"})

    # ============ ВЕБ-ИНТЕРФЕЙС ============

    @app.route("/rimmind/personality", methods=["GET"])
    def web_interface():
        return render_template("main.html", refresh_interval=pcfg.REFRESH_INTERVAL_MS)

    @app.route("/rimmind/personality/pawn/<name>", methods=["GET"])
    def web_pawn_page(name):
        pawn = pawn_cache.get(name)
        if not pawn:
            return render_template("error.html", name=name), 404
        pawn_json = json.dumps(pawn, ensure_ascii=False)
        # Передаём голос для озвучки (если есть в кэше, иначе дефолт)
        voice = pawn.get("voice", "ru_RU-irina-medium")
        return render_template("pawn.html", pawn_json=pawn_json, voice=voice)

    print("[PersonalityWeb] Роуты зарегистрированы:")
    print("   GET  /rimmind/personality              -> главная (сетка пешек)")
    print("   GET  /rimmind/personality/pawn/<name>  -> страница пешки (новая вкладка)")
    print("   GET  /rimmind/personality/pawns        -> API список")
    print("   POST /rimmind/personality/generate     -> генерация поля")
    print("   POST /rimmind/personality/generate_all -> генерация всего")
    print("   POST /rimmind/personality/generate_pure-> чистый профиль")
    print("   POST /rimmind/personality/generate_story-> история дня")


def _json_response(data: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(data, ensure_ascii=False, indent=None),
        status=status,
        mimetype="application/json; charset=utf-8"
    )