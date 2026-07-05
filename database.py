# ============ database.py ============
# МОДУЛЬ БАЗЫ ДАННЫХ — "КАРТОТЕКА КОЛОНИИ" (JSON-версия)
# =====================================
# Хранит всё в pawn_voices.json. Без SQLite, без миграций, без хуйни.
# Открыл блокнотом — поправил. Всё.

import os
import json
import random
import re
import threading
from typing import Dict, Any, List, Optional

from config import Config

EXCLUDED_WORDS = {
    "reply", "thought", "rimmind", "current", "colonist", "rimworld",
    "unknown_pawn", "assistant", "размышляет", "думает", "реплика",
    "мысли", "система", "eva", "ева"
}


class PawnDatabase:
    def __init__(self, config: Config):
        self.config = config
        self.json_path = config.get("json_backup", "pawn_voices.json")
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        self._known: set = set()
        self._load()

    def _load(self):
        """Загрузить JSON с диска."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
                    self._known = set(loaded.keys())
                    print(f"💾 [База] Загружено {len(self._data)} пешек из '{self.json_path}'")
                else:
                    print(f"⚠️ [База] JSON битый, начинаем с чистого листа.")
            except Exception as e:
                print(f"⚠️ [База] Ошибка чтения: {e}")
        else:
            print(f"💾 [База] Файл не найден, создадим новый: '{self.json_path}'")

    def _save(self):
        """Сохранить JSON на диск. Вызывается автоматически при изменениях."""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"⚠️ [База] Ошибка сохранения: {e}")

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Найти пешку по имени."""
        if name.lower() in EXCLUDED_WORDS:
            return None
        canonical = self.get_canonical(name)
        name = canonical or name
        return self._data.get(name)

    def save_profile(self, name: str, profile: Dict[str, Any]):
        """Сохранить или обновить профиль."""
        if name.lower() in EXCLUDED_WORDS:
            return
        self._data[name] = {
            "voice": profile.get("voice", "ru_RU-denis-medium"),
            "speaker_id": int(profile.get("speaker_id", 0)),
            "length_scale": float(profile.get("length_scale", 1.0)),
            "noise_scale": float(profile.get("noise_scale", 0.667)),
            "noise_w": float(profile.get("noise_w", 0.8)),
            "gender": profile.get("gender", "male"),
            "volume": float(profile.get("volume", 1.0)),
            "preset": profile.get("preset", None),
        }
        self._known.add(name)
        self._save()
        print(f"👤 [База] Сохранён: '{name}' | {self._data[name]['voice']}")

    def delete_profile(self, name: str) -> bool:
        """Удалить пешку."""
        if name in self._data:
            del self._data[name]
            self._known.discard(name)
            self._save()
            return True
        return False

    def get_all_names(self) -> List[str]:
        """Все имена."""
        return sorted(list(self._data.keys()))

    def get_or_create_profile(self, name: str, pawn_speech: str, full_prompt_text: str) -> Dict[str, Any]:
        """Получить профиль или создать новый."""
        if name.lower() in EXCLUDED_WORDS:
            name = "Unknown_Pawn"

        canonical = self.get_canonical(name)
        if canonical:
            name = canonical

        existing = self.get_profile(name)
        if existing and name != "Unknown_Pawn":
            available = self._get_available_voices(existing.get("gender", "male"))
            if existing.get("voice") in available:
                return existing
            print(f"⚠️ Голос '{existing['voice']}' недоступен, переназначаем...")

        # Создаём нового
        gender = self._detect_gender(name, pawn_speech, full_prompt_text)
        available = self._get_available_voices(gender)

        if not available:
            all_voices = self._get_available_voices("male") + self._get_available_voices("female")
            available = all_voices if all_voices else None

        if not available:
            print("❌ Нет доступных голосов Piper!")
            return {
                "voice": None, "gender": gender, "speaker_id": 0,
                "length_scale": 1.0, "noise_scale": 0.667,
                "noise_w": 0.8, "volume": 1.0, "preset": None,
            }

        chosen_voice = random.choice(available)
        length_scale = round(random.uniform(0.85, 1.15), 2)
        noise_w = round(random.uniform(0.7, 0.9), 2)

        profile = {
            "voice": chosen_voice,
            "speaker_id": 0,
            "length_scale": length_scale,
            "noise_scale": 0.667,
            "noise_w": noise_w,
            "gender": gender,
            "volume": 1.0,
            "preset": None,
        }

        if name != "Unknown_Pawn":
            self.save_profile(name, profile)

        return profile

    def add_alias(self, alias: str, canonical_name: str):
        self._aliases[alias.lower().strip()] = canonical_name
        self._known.add(canonical_name)

    def get_canonical(self, alias: str) -> Optional[str]:
        return self._aliases.get(alias.lower().strip())

    def get_all_aliases(self) -> Dict[str, str]:
        return dict(self._aliases)

    def register_known_pawn(self, name: str):
        if name.lower() not in EXCLUDED_WORDS:
            self._known.add(name)

    def get_known_pawns(self) -> List[str]:
        return sorted(list(self._known | set(self._data.keys())))

    def export_to_json(self, path: Optional[str] = None):
        """Уже JSON, просто копируем если нужен другой путь."""
        target = path or self.json_path
        if target != self.json_path:
            try:
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=4)
                print(f"💾 [База] Экспорт в '{target}'")
            except Exception as e:
                print(f"⚠️ [База] Ошибка экспорта: {e}")

    def import_from_json(self, path: Optional[str] = None):
        """Загрузить из другого JSON."""
        source = path or self.json_path
        if not os.path.exists(source):
            print(f"⚠️ [База] Файл не найден: {source}")
            return
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data.update(data)
                self._known.update(data.keys())
                self._save()
                print(f"💾 [База] Импортировано {len(data)} пешек из '{source}'")
        except Exception as e:
            print(f"⚠️ [База] Ошибка импорта: {e}")

    def _get_available_voices(self, gender: str) -> List[str]:
        voices_dir = self.config.get("piper_model_dir", "piper_models")
        key = "female_voices" if gender == "female" else "male_voices"
        voices = self.config.get(key, [])
        return [v for v in voices if os.path.exists(os.path.join(voices_dir, v + ".onnx"))]

    @staticmethod
    def _detect_gender(name: str, pawn_speech: str, full_prompt_text: str) -> str:
        if name and name != "Unknown_Pawn":
            try:
                pattern = rf"\[{re.escape(name)}'s\s+Status\]([^\n]+)"
                match = re.search(pattern, full_prompt_text, re.IGNORECASE)
                if match:
                    status = match.group(1).lower()
                    if "мужчина" in status or " male " in status:
                        return "male"
                    if "женщина" in status or " female " in status:
                        return "female"
            except re.error:
                pass

        text = (full_prompt_text + " " + pawn_speech).lower()
        f_score = sum(1 for w in ["сказала", "пошла", "сделала", "заметила", "замучалась"] if w in text)
        m_score = sum(1 for w in ["сказал", "пошел", "сделал", "заметил", "решил"] if w in text)

        if f_score > m_score:
            return "female"
        if m_score > f_score:
            return "male"
        return random.choice(["male", "female"])

    def save(self):
        """Для совместимости с graceful shutdown."""
        self._save()