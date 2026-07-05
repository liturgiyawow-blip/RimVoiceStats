# ============ config.py ============
# МОДУЛЬ НАСТРОЕК — "ПУЛЬТ УПРАВЛЕНИЯ" ПРОКСИ
# ===================================
# Все адреса, порты, пути к голосам и цифры хранятся ЗДЕСЬ.
# Если нужно что-то поменять — правь файл config.json рядом с программой,
# а этот модуль сам его прочитает при запуске.
#
# Почему это важно: раньше настройки были размазаны по коду и их было
# не найти. Теперь — одна точка правды.

import os
import json
from typing import Dict, Any


# === ЗАВОДСКИЕ НАСТРОЙКИ ===
# Это "запасной аэродром". Если config.json ещё не создан или в нём
# чего-то не хватает — программа возьмёт цифры отсюда.
DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Сеть и адреса ---
    "lm_studio_url": "http://localhost:1234/v1/chat/completions",
    "proxy_port": 1235,

    # --- Файлы и папки ---
    # pawn_voices.json — единственное хранилище профилей пешек.
    # Открыл блокнотом — поправил. Всё.
    "json_backup": "pawn_voices.json",
    "aliases_file": "voice_aliases.json",
    "tts_cache_dir": "tts_cache",
    "piper_model_dir": "piper_models",

    # --- Звук и голоса ---
    "speech_volume": 0.6,
    "thought_volume": 0.4,
    # Управление озвучкой мыслей и паузами
    "enable_thought_tts": False,
    "name_pre_delay": 0.5,
    "name_post_delay": 0.8,
    # Устаревший ключ (оставлен для обратной совместимости, используется как fallback)
    "thought_delay_seconds": 1.0,
    "eva_voice": "ru_RU-irina-medium",
    "male_voices": [
        "ru_RU-denis-medium",
        "ru_RU-ruslan-medium",
        "ru_RU-dmitri-medium"
    ],
    "female_voices": [
        "ru_RU-irina-medium"
    ],

    # --- Очереди и кэш ---
    "max_queue_size": 50,
    "use_gpu": False,
    # ВАЖНО: всё хранится только в оперативке (RAM). SSD не трогаем.
    "max_ram_cache_entries": 300,
}


class Config:
    """
    Этот класс — как переводчик между твоим config.json и программой.
    
    Один раз загружает файл при старте, и потом все модули спрашивают у него:
    "А какой порт?" или "А какой голос у Евы?"
    
    Если в config.json чего-то нет — он тихо подставит заводское значение,
    чтобы программа не упала.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """
        Фабричный метод — создаёт Config из файла на диске.
        
        Что происходит:
        1. Если config.json есть — читаем его.
        2. Если в нём чего-то нет — подставляем из DEFAULT_CONFIG.
        3. Если файла нет — создаём его с заводскими настройками.
        4. Возвращаем готовый объект Config.
        """
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)

                # Берём заводские настройки как основу...
                data = DEFAULT_CONFIG.copy()
                # ...и накладываем сверху те, что прочитали из файла.
                for key, val in loaded.items():
                    if (
                        key in data
                        and isinstance(data[key], dict)
                        and isinstance(val, dict)
                    ):
                        data[key].update(val)
                    else:
                        data[key] = val

                print(f"⚙️ [Config] Загружен '{path}'")

            except Exception as e:
                print(f"⚠️ [Config] Ошибка чтения '{path}': {e}")
                data = DEFAULT_CONFIG.copy()
        else:
            data = DEFAULT_CONFIG.copy()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"⚙️ [Config] Создан новый '{path}'")
            except Exception as e:
                print(f"⚠️ [Config] Не удалось сохранить '{path}': {e}")

        return cls(data)

    def get(self, key: str, default=None):
        """
        Безопасное получение одной настройки.
        
        Примеры:
            config.get("proxy_port")        → 1235
            config.get("use_gpu")           → False
            config.get("несуществующий")    → None
        """
        return self._data.get(key, default)

    @property
    def raw(self) -> Dict[str, Any]:
        """
        Отдаёт весь словарь настроек целиком.
        Нужно редко, но иногда удобно для отладки.
        """
        return self._data