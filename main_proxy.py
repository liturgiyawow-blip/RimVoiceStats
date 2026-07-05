# ============ main_proxy.py ============
# УПРОЩЁННЫЙ ПРОКСИ — только диалоги + TTS
# ======================================

import os
import sys
import gc
import queue
import atexit
import signal
import threading
import time

from config import Config
from database import PawnDatabase
from cache import RAMWavCache
from tts_engine import PiperVoiceManager, PiperTTSEngine, TTSWorker, PIPER_AVAILABLE
from llm_parser import LLMResponseParser
from voice_api import create_app

_shutdown_event = threading.Event()


def main():
    print("=" * 60)
    print("🤖 RimMind Proxy v7.1 (Финальная — только диалоги)")
    print("=" * 60)

    config = Config.load("config.json")
    db = PawnDatabase(config)
    cache = RAMWavCache(max_entries=config.get("max_ram_cache_entries", 300))

    voices_dir = config.get("piper_model_dir", "piper_models")
    use_gpu = config.get("use_gpu", False)
    voice_manager = PiperVoiceManager(voices_dir, use_gpu=use_gpu)
    engine = PiperTTSEngine(voice_manager, cache)
    parser = LLMResponseParser()

    tts_queue = queue.Queue(maxsize=config.get("max_queue_size", 50))
    worker = TTSWorker(tts_queue, engine, _shutdown_event)
    worker.start()

    # Предзагрузка голосов
    if PIPER_AVAILABLE:
        print("\n🎙️ Предзагрузка голосов...")
        voices_to_preload = [config.get("eva_voice", "ru_RU-irina-medium")]
        voices_to_preload.extend(config.get("male_voices", []))
        voices_to_preload.extend(config.get("female_voices", []))
        for v in voices_to_preload:
            if v:
                voice_manager.load(v)

    app = create_app(config, db, tts_queue, parser)
    port = config.get("proxy_port", 1235)

    print(f"\n🌐 Прокси: http://127.0.0.1:{port}")
    print("   /v1/chat/completions  → LLM (диалоги пешек)")
    print("   /rimmind/voice/speak  → Системная озвучка")
    print("   /rimmind/voice/status → Статус")
    print("=" * 60)

    def on_exit():
        print("\n🛑 Завершение...")
        _shutdown_event.set()
        db.save()
        cache.clear()
        try:
            tts_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        time.sleep(0.5)
        gc.collect()
        print("👋 До свидания.")

    atexit.register(on_exit)

    def signal_handler(signum, frame):
        print(f"\n🛑 Сигнал {signum}")
        on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run(host='127.0.0.1', port=port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except OSError as e:
        print(f"\n❌ Порт {port} занят!")
        input("Нажмите Enter...")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n💀 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter...")