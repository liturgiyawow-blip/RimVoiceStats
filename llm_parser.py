# ============ llm_parser.py ============
# МОДУЛЬ ПАРСИНГА ОТВЕТОВ LLM — "ПЕРЕВОДЧИК С МОДЕЛЬНОГО"
# ======================================
# LM Studio (и вообще любые LLM) — непредсказуемые собеседники.
# Они могут выдать:
#   1. Чистый JSON: {"reply": "Привет", "thought": "..."}
#   2. JSON в markdown: ```json\n{"reply": "..."}\n```
#   3. "Грязный" JSON: вперемешку с текстом, с лишними запятыми
#   4. Просто текст: без фигурных скобок вообще
#
# Этот модуль пробует всё по порядку, пока не поймёт, что сказала модель.
# Он НЕ знает про Flask, TTS или игру — ему дают строку, он отдаёт словарь.

import json
import re
from typing import Optional, Dict, Any


def _strip_markdown(text: str) -> str:
    """
    Убрать markdown-обёртку вокруг JSON.
    
    LLM любит оборачивать код в ```json ... ```.
    Эта функция вытаскивает содержимое между обратными апострофами.
    """
    # Ищем блок ```json ... ``` или просто ``` ... ```
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """
    Убрать висящие запятые перед закрывающими скобками.
    
    LLM часто генерирует:
        {"reply": "Привет", "thought": "...",}
    Последняя запятая перед } — битый JSON. Это исправляет.
    """
    # Запятая перед } или ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text


def _extract_field_dirty(text: str, field: str) -> Optional[str]:
    """
    ПОСЛЕДНИЙ РУБЕЖ ОБОРОНЫ.
    
    Если JSON совсем развалился — ищем поле через regex.
    Это не идеально, но лучше, чем ничего.
    Возвращает строку или None.
    """
    # Ищем "поле": "значение" (с экранированными кавычками внутри)
    pattern_str = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    m = re.search(pattern_str, text)
    if m:
        raw = m.group(1)
        # Раскрываем экранированные символы
        return raw.replace('\\\\', '\\').replace('\\n', '\n').replace('\\"', '"')

    # Ищем "поле": null или "поле": { ... }
    pattern_obj = rf'"{re.escape(field)}"\s*:\s*(null|None|\{{.*?\}}|\[.*?\])'
    m = re.search(pattern_obj, text, re.DOTALL | re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val.lower() in ("null", "none"):
            return None
        return val

    return None


class LLMResponseParser:
    """
    Главный переводчик. Получает сырую строку от LLM — отдаёт структурированный ответ.
    
    Результат — словарь:
        {
            "reply": str или None,      # то, что пешка говорит вслух
            "thought": str или None,    # внутренние мысли пешки
            "raw_type": str             # как мы распарсили: "clean", "markdown", "dirty", "text"
        }
    """

    def __init__(self):
        # Счётчики для статистики (можно посмотреть, насколько часто
        # LLM выдаёт грязный JSON)
        self.stats = {
            "clean": 0,
            "markdown": 0,
            "dirty": 0,
            "text": 0,
            "failed": 0
        }

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Главный метод. Пробует распарсить ответ LLM всеми доступными способами.
        """
        if not raw_text or not raw_text.strip():
            self.stats["failed"] += 1
            return {"reply": None, "thought": None, "raw_type": "empty"}

        raw = raw_text.strip()

        # === ЭТАП 1: Чистый JSON ===
        if raw.startswith('{') and ('"reply"' in raw or '"thought"' in raw):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return self._normalize(parsed, "clean")
            except json.JSONDecodeError:
                pass  # Не беда, идём дальше

        # === ЭТАП 2: JSON внутри markdown ===
        stripped = _strip_markdown(raw)
        if stripped.startswith('{') and ('"reply"' in stripped or '"thought"' in stripped):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return self._normalize(parsed, "markdown")
            except json.JSONDecodeError:
                pass

        # === ЭТАП 3: "Грязный" JSON — чистим и пробуем ===
        cleaned = _fix_trailing_commas(raw)
        if cleaned.startswith('{') and ('"reply"' in cleaned or '"thought"' in cleaned):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return self._normalize(parsed, "dirty")
            except json.JSONDecodeError:
                pass

        # === ЭТАП 4: Ищем JSON-блок внутри текста ===
        # Иногда LLM пишет: "Вот ответ: {"reply": "..."} Надеюсь, помог!"
        json_block = self._extract_json_block(raw)
        if json_block:
            try:
                parsed = json.loads(json_block)
                if isinstance(parsed, dict):
                    return self._normalize(parsed, "dirty")
            except json.JSONDecodeError:
                pass

        # === ЭТАП 5: Fallback на regex-поля ===
        reply = _extract_field_dirty(raw, "reply")
        thought = _extract_field_dirty(raw, "thought")
        if reply is not None or thought is not None:
            self.stats["dirty"] += 1
            return {
                "reply": reply,
                "thought": thought,
                "raw_type": "dirty_regex"
            }

        # === ЭТАП 6: Просто текст (не JSON) ===
        # LLM выдал обычную речь без фигурных скобок
        if not raw.startswith('{'):
            self.stats["text"] += 1
            return {
                "reply": raw,
                "thought": None,
                "raw_type": "text"
            }

        # === ЭТАП 7: Полный провал ===
        self.stats["failed"] += 1
        print(f"⚠️ [LLM Parser] Не удалось распарсить: {raw[:200]}...")
        return {"reply": None, "thought": None, "raw_type": "failed"}

    def _normalize(self, parsed: dict, raw_type: str) -> Dict[str, Any]:
        """
        Приводит распарсенный словарь к единому формату.
        Обрабатывает случай, когда thought — не строка, а словарь.
        """
        self.stats[raw_type] += 1

        # reply: берём как строку, если есть
        reply = parsed.get("reply")
        if reply is not None:
            reply = str(reply).strip()
            if reply.lower() in ("none", "null", ""):
                reply = None

        # thought: может быть строкой или словарём {"description": "..."}
        thought_raw = parsed.get("thought")
        thought = None
        if thought_raw is not None:
            if isinstance(thought_raw, dict):
                thought = str(thought_raw.get("description", "")).strip()
            else:
                thought = str(thought_raw).strip()
            if thought.lower() in ("none", "null", "", "{}", "[]"):
                thought = None

        return {
            "reply": reply,
            "thought": thought,
            "raw_type": raw_type
        }

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """
        Ищет первый JSON-объект внутри произвольного текста.
        Используем счётчик скобок, чтобы не сломаться на вложенности.
        """
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        for i, char in enumerate(text[start:], start=start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None

    def get_stats(self) -> Dict[str, int]:
        """Статистика парсинга для отладки."""
        return self.stats.copy()