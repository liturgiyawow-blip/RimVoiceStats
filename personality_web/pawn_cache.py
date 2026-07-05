# personality_web/pawn_cache.py
import json
import os
import threading
import re
from typing import Dict, Optional, List, Any


class PawnCache:
    def __init__(self, cache_file: str = "personality_pawn_cache.json"):
        self._cache_file = cache_file
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._load()

    def upsert(self, name: str, fields: Dict[str, Any]):
        with self._lock:
            if name not in self._data:
                self._data[name] = {"name": name}
            for k, v in fields.items():
                if v is not None and v != "" and v != []:
                    self._data[name][k] = v
            self._data[name]["last_seen"] = self._now()
            self._dirty = True

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(name)

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._data.values())

    def set_field(self, name: str, field: str, value: str):
        with self._lock:
            if name not in self._data:
                self._data[name] = {"name": name}
            self._data[name][field] = value
            self._data[name]["last_updated"] = self._now()
            self._dirty = True
            self._save()

    def clear(self):
        with self._lock:
            self._data.clear()
            self._dirty = True
            self._save()

    def _load(self):
        if not os.path.exists(self._cache_file):
            return
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._data = loaded
                print(f"[PersonalityCache] Загружено {len(self._data)} пешек")
        except Exception as e:
            print(f"[PersonalityCache] Ошибка загрузки: {e}")

    def _save(self):
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception as e:
            print(f"[PersonalityCache] Ошибка сохранения: {e}")

    def save_if_dirty(self):
        with self._lock:
            if self._dirty:
                self._save()

    @staticmethod
    def _now():
        from datetime import datetime
        return datetime.now().isoformat()


class PawnExtractor:
    @classmethod
    def extract_from_prompt(cls, text: str) -> Optional[Dict[str, Any]]:
        if not text or "'s Status]" not in text:
            return None
        name = cls._extract_name(text)
        if not name:
            return None
        return {
            "name": name,
            "age": cls._extract_age(text),
            "gender": cls._extract_gender(text),
            "race": cls._extract_race(text),
            "genes": cls._extract_list(text, "Genes:"),
            "childhood": cls._extract_single(text, "Childhood:"),
            "adulthood": cls._extract_single(text, "Adulthood:"),
            "traits": cls._extract_list(text, "Traits:"),
            "skills": cls._extract_skills(text),
            "health": cls._extract_list(text, "Health Issues:"),
            "mood": cls._extract_mood(text),
            "ideology": cls._extract_single(text, "Ideology:"),
            "relations": cls._extract_list(text, "Relations:"),
            "personality_profile": cls._extract_personality_profile(text, name),
            "work_tendencies": cls._extract_work_tendencies(text, name),
            "social_tendencies": cls._extract_social_tendencies(text, name),
            "memories": cls._extract_memories(text, name),
        }

    @staticmethod
    def _extract_name(text: str) -> Optional[str]:
        m = re.search(r"\[([А-ЯЁа-яёA-Za-z0-9_\-]+)'s\s+Status\]", text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_age(text: str) -> Optional[int]:
        m = re.search(r"(\d+)yo\s+\w+", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_gender(text: str) -> Optional[str]:
        m = re.search(r"\d+yo\s+(мужчина|женщина|male|female)", text, re.I)
        return m.group(1).lower() if m else None

    @staticmethod
    def _extract_race(text: str) -> Optional[str]:
        m = re.search(r"\d+yo\s+\w+\s+([\w\s]+?)(?:\n|$)", text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_list(text: str, prefix: str) -> List[str]:
        pattern = rf"{re.escape(prefix)}\s*(.+?)(?:\n\w|\n\[|$)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        raw = m.group(1).strip()
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _extract_single(text: str, prefix: str) -> Optional[str]:
        pattern = rf"{re.escape(prefix)}\s*(.+?)(?:\n|$)"
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_skills(text: str) -> Dict[str, int]:
        skills = {}
        m = re.search(r"Skills:\s*(.+?)(?:\n\w|\n\[|$)", text, re.DOTALL)
        if not m:
            return skills
        raw = m.group(1)
        for match in re.finditer(r"([\w\s]+)\((\d+)\)", raw):
            name = match.group(1).strip()
            level = int(match.group(2))
            skills[name] = level
        return skills

    @staticmethod
    def _extract_mood(text: str) -> Optional[str]:
        m = re.search(r"Mood:\s*([\w\s]+)\s+(\d+)%", text)
        if m:
            return f"{m.group(1).strip()} {m.group(2)}%"
        return None

    @staticmethod
    def _extract_personality_profile(text: str, name: str) -> Optional[str]:
        pattern = rf"\[{re.escape(name)}'s\s+Personality\s+Profile\](.+?)(?=\n\[|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return None
        lines = [l.strip() for l in m.group(1).split("\n") if l.strip()]
        result = []
        for line in lines:
            if any(skip in line for skip in ["Work tendencies", "Social tendencies", "Recent psychological"]):
                break
            result.append(line)
        return " ".join(result) if result else None

    @staticmethod
    def _extract_work_tendencies(text: str, name: str) -> Optional[str]:
        pattern = rf"\[{re.escape(name)}'s\s+Personality\s+Profile\].*?Work\s+tendencies:\s*(.+?)(?:\n\w|\nSocial|$)"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_social_tendencies(text: str, name: str) -> Optional[str]:
        pattern = rf"\[{re.escape(name)}'s\s+Personality\s+Profile\].*?Social\s+tendencies:\s*(.+?)(?:\n\w|\nRecent|$)"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_memories(text: str, name: str) -> List[str]:
        pattern = rf"\[{re.escape(name)}\s+Recent\s+Memories\](.+?)(?=\n\[|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        lines = m.group(1).split("\n")
        return [l.strip().lstrip("-").strip() for l in lines if l.strip()]


pawn_cache = PawnCache()