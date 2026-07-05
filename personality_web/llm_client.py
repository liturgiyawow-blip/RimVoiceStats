# personality_web/llm_client.py
import requests
import json
import re
from typing import Dict, List, Optional

from .personality_config import PersonalityConfig


class PersonalityLLMClient:
    def __init__(self, lm_url: str = "http://127.0.0.1:1234/v1/chat/completions"):
        self.url = lm_url
        self.cfg = PersonalityConfig()
        self.timeout = self.cfg.LLM_TIMEOUT

    def generate_field(self, pawn_data: Dict, field_name: str) -> str:
        system = self._system_prompt(field_name)
        user = self._user_prompt(pawn_data, field_name)

        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": self.cfg.FIELD_MAX_TOKENS,
            "temperature": self.cfg.FIELD_TEMPERATURES.get(field_name, self.cfg.DEFAULT_TEMPERATURE),
            "stream": False
        }

        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._extract_field(raw, field_name)
        except Exception as e:
            return f"[Ошибка LLM: {e}]"

    def generate_all(self, pawn_data: Dict) -> Dict[str, str]:
        """Генерирует все три поля по очереди, используя полноценные промпты для каждого."""
        fields = ["description", "workTendencies", "socialTendencies"]
        result = {}
        for field in fields:
            result[field] = self.generate_field(pawn_data, field)
        return result

    def generate_pure_profile(self, pawn_data: Dict) -> str:
        system = self.cfg.PURE_PROFILE_PROMPT
        user = self._user_prompt(pawn_data, "all")
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": self.cfg.PURE_MAX_TOKENS,
            "temperature": self.cfg.PURE_PROFILE_TEMPERATURE,
            "stream": False
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._extract_field(raw, "pure_profile")
        except Exception as e:
            return f"[Ошибка LLM: {e}]"

    def generate_daily_story(self, pawn_data: Dict, colony_pawns: List[Dict] = None) -> str:
        system = self.cfg.STORY_PROMPT
        lines = [f"Ты — {pawn_data.get('name', 'колонист')}. Опиши свой день своими словами."]
        if pawn_data.get("gender"):
            lines.append(f"Пол: {pawn_data['gender']}")
        if pawn_data.get("race"):
            lines.append(f"Раса: {pawn_data['race']}")
        if pawn_data.get("traits"):
            lines.append(f"Черты: {', '.join(pawn_data['traits'])}")
        if pawn_data.get("mood"):
            lines.append(f"Настроение: {pawn_data['mood']}")

        if pawn_data.get("memories"):
            lines.append("\nСобытия сегодня:")
            recent_memories = pawn_data["memories"][-self.cfg.STORY_MEMORY_COUNT:]
            for mem in recent_memories:
                clean_mem = mem
                for old, new in self.cfg.STORY_MEMORY_REPLACEMENTS.items():
                    clean_mem = clean_mem.replace(old, new)
                lines.append(f"- {clean_mem}")

        if colony_pawns:
            lines.append("\nДругие колонисты (для контекста):")
            for p in (colony_pawns or [])[:self.cfg.STORY_COLONY_PAWNS_COUNT]:
                if p.get("name") != pawn_data.get("name"):
                    profile = p.get("personality_profile", "") or p.get("description", "")
                    lines.append(f"- {p.get('name', '?')}: {profile[:120]}...")

        user = "\n".join(lines)
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": self.cfg.STORY_MAX_TOKENS,
            "temperature": self.cfg.STORY_TEMPERATURE,
            "stream": False
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._extract_field(raw, "story")
        except Exception as e:
            return f"[Ошибка LLM: {e}]"

    def generate_colony_report(self, colony_pawns: List[Dict]) -> Dict[str, str]:
        """Составляет отчёт по колонии и рекомендации советника."""
        lines = ["Состояние колонистов:"]
        for p in colony_pawns:
            name = p.get("name", "?")
            parts = []

            if p.get("health"):
                serious = [h for h in p["health"] if any(k in h.lower() for k in [
                    "шрам", "рана", "трещина", "удалено", "уничтожено", "отгрызено",
                    "прострелено", "болит", "астма", "раздроблено", "отравление", "газ"
                ])]
                if serious:
                    parts.append(f"травмы: {', '.join(serious)}")

            if p.get("mood"):
                mood_lower = p["mood"].lower()
                if any(b in mood_lower for b in ["стресс", "паника", "ярость", "сломлен", "нервный", "депрессия"]):
                    parts.append(f"психика критическая: {p['mood']}")
                else:
                    parts.append(f"настроение: {p['mood']}")

            if p.get("memories"):
                recent = p["memories"][-3:]
                threats = [m for m in recent if any(t in m.lower() for t in [
                    "attacked", "атакован", "ранен", "убит", "пожар", "голод", "взрыв"
                ])]
                if threats:
                    parts.append(f"недавние события: {'; '.join(threats)}")

            status = "; ".join(parts) if parts else "в норме"
            lines.append(f"- {name}: {status}")

        user_text = "\n".join(lines)

        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": self.cfg.COLONY_REPORT_PROMPT},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": self.cfg.COLONY_REPORT_MAX_TOKENS,
            "temperature": self.cfg.COLONY_REPORT_TEMPERATURE,
            "stream": False
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            report = self._extract_field(raw, "report")
        except Exception as e:
            report = f"[Ошибка генерации отчёта: {e}]"

        rec_payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": self.cfg.COLONY_RECOMMENDATIONS_PROMPT},
                {"role": "user", "content": f"Данные колонии:\n{user_text}\n\nОтчёт:\n{report}\n\nДай рекомендации."}
            ],
            "max_tokens": self.cfg.COLONY_RECOMMENDATIONS_MAX_TOKENS,
            "temperature": self.cfg.COLONY_RECOMMENDATIONS_TEMPERATURE,
            "stream": False
        }
        try:
            resp = requests.post(self.url, json=rec_payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            recommendations = self._extract_field(raw, "recommendations")
        except Exception as e:
            recommendations = f"[Ошибка генерации рекомендаций: {e}]"

        return {"report": report, "recommendations": recommendations}

    # ============ ВСПОМОГАТЕЛЬНЫЕ ============

    def _system_prompt(self, field: str) -> str:
        base = self.cfg.SYSTEM_PROMPT_BASE
        field_prompt = self.cfg.FIELD_PROMPTS.get(field, self.cfg.FIELD_PROMPTS["all"])
        return base + "\n\n" + field_prompt

    def _user_prompt(self, pawn: Dict, field: str) -> str:
        lines = [f"Имя: {pawn.get('name', 'Unknown')}"]
        if pawn.get("gender"):
            lines.append(f"Пол: {pawn['gender']}")
        if pawn.get("race"):
            lines.append(f"Раса: {pawn['race']}")
        bg = []
        if pawn.get("childhood"): bg.append(f"детство: {pawn['childhood']}")
        if pawn.get("adulthood"): bg.append(f"взросление: {pawn['adulthood']}")
        if bg:
            lines.append(f"Прошлое: {', '.join(bg)}")
        if pawn.get("traits"):
            lines.append(f"Черты характера: {', '.join(pawn['traits'])}")
        if pawn.get("ideology"):
            lines.append(f"Идеология: {pawn['ideology']}")
        if pawn.get("genes"):
            lines.append(f"Генетические особенности: {', '.join(pawn['genes'])}")
        if pawn.get("skills"):
            top = sorted(pawn["skills"].items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"Чем занимается: {', '.join([k for k, v in top])}")
        if pawn.get("health"):
            lines.append(f"Жизнь оставила следы: {', '.join(pawn['health'])}")
        if pawn.get("relations"):
            lines.append(f"Связи: {', '.join(pawn['relations'])}")
        if pawn.get("memories"):
            recent = pawn["memories"][-5:]
            lines.append(f"Недавние события: {', '.join(recent)}")
        if pawn.get("personality_profile"):
            lines.append(f"Уже известно: {pawn['personality_profile']}")
        lines.append("\nСгенерируй психологический портрет. НЕ указывай возраст, цифры навыков, текущие статы. Опиши личность.")
        return "\n".join(lines)

    def _extract_field(self, raw: str, field: str) -> str:
        cleaned = self._clean_json(raw)
        try:
            data = json.loads(cleaned)
            result = data.get(field, "")
            if result and isinstance(result, str):
                return result
            return "[Поле не найдено в JSON]"
        except json.JSONDecodeError:
            pattern = rf'"{re.escape(field)}"\s*:\s*"\s*((?:[^"\\]|\\.)*)"'
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                raw_text = m.group(1)
                raw_text = raw_text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                return raw_text.strip()
            pattern2 = rf'"{re.escape(field)}"\s*:\s*"?([^"}}]+)"?'
            m = re.search(pattern2, raw, re.DOTALL)
            if m:
                return m.group(1).strip().replace('\\n', '\n')
            return raw[:500]

    def _extract_all(self, raw: str) -> Dict[str, str]:
        cleaned = self._clean_json(raw)
        try:
            data = json.loads(cleaned)
            return {
                "description": data.get("description", ""),
                "workTendencies": data.get("workTendencies", ""),
                "socialTendencies": data.get("socialTendencies", "")
            }
        except:
            return {"description": raw[:500], "workTendencies": "", "socialTendencies": ""}

    @staticmethod
    def _clean_json(raw: str) -> str:
        if "```" in raw:
            start = raw.find('{')
            end = raw.rfind('}')
            if start >= 0 and end > start:
                return raw[start:end+1]
        return raw