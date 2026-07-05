# ============ cache.py ============
# МОДУЛЬ RAM-КЭША АУДИО — "ОПЕРАТИВНЫЙ СКЛАД РЕЧИ"
# ==================================
# Когда Piper синтезирует фразу, результат (готовый WAV в виде байтов)
# сохраняется здесь. Если эта же фраза понадобится ещё раз —
# берём из памяти мгновенно, не нагружая нейросеть.
#
# Правила:
# - Только оперативная память (RAM). SSD не трогаем.
# - Если склад переполнен — выкидываем самую старую запись (LRU).
# - Все операции потокобезопасны (RLock).

import hashlib
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any


class RAMWavCache:
    """
    Оперативный склад синтезированной речи.
    
    Хранит пары: "хеш фразы" → "готовые аудио-байты".
    Когда Piper говорит "Привет" — мы запоминаем результат.
    Когда через минуту LLM снова говорит "Привет" — мгновенно отдаём
    из памяти, не тратя время на синтез.
    """

    def __init__(self, max_entries: int = 300):
        """
        max_entries — сколько фраз хранить одновременно.
        Каждая фраза ~ 50-200 КБ в памяти. 300 записей ≈ 15-60 МБ.
        При переполнении самые старые вытесняются автоматически.
        """
        self._max = max_entries
        # OrderedDict помнит порядок добавления. Когда добавляем новое —
        # оно в конце. Когда обращаемся к старому — переставляем в конец.
        # При переполнении выкидываем самое первое (самое старое).
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.RLock()
        # Счётчики для статистики и отладки
        self._hits = 0
        self._misses = 0

    def _make_key(self, voice_name: str, text: str,
                  length_scale: float, noise_scale: float,
                  noise_w: float, speaker_id: int) -> str:
        """
        Создаём уникальный "штрих-код" для фразы.
        
        Если ты поменяешь скорость речи (length_scale) с 1.0 на 1.2 —
        штрих-код изменится, и кэш не выдаст старую версию.
        Это важно: медленная и быстрая фраза — разные записи.
        """
        # Нормализуем текст: убираем лишние пробелы, приводим к нижнему регистру
        normalized = " ".join(text.lower().split())
        raw = (
            f"{voice_name}|{normalized}|"
            f"{length_scale:.4f}|{noise_scale:.4f}|"
            f"{noise_w:.4f}|{speaker_id}"
        )
        # MD5 — быстрый и короткий хеш. Для кэша криптостойкость не нужна.
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, voice_name: str, text: str,
            length_scale: float, noise_scale: float,
            noise_w: float, speaker_id: int = 0) -> Optional[bytes]:
        """
        Попытка найти готовое аудио в памяти.
        
        Возвращает:
            bytes — готовые WAV-данные, если нашли.
            None — если такой фразы ещё не синтезировали.
        """
        key = self._make_key(voice_name, text, length_scale, noise_scale, noise_w, speaker_id)

        with self._lock:
            if key in self._cache:
                # Переставляем в конец — эта запись "свежая", не вытеснять
                self._cache.move_to_end(key)
                self._hits += 1
                print(f"   ⚡ [Кэш] Попадание! '{text[:40]}...'")
                return self._cache[key]

            self._misses += 1
            return None

    def put(self, voice_name: str, text: str,
            length_scale: float, noise_scale: float,
            noise_w: float, speaker_id: int,
            wav_data: bytes) -> None:
        """
        Сохранить свежесинтезированное аудио в память.
        """
        if self._max <= 0:
            # Кэширование отключено (max_entries = 0)
            return

        key = self._make_key(voice_name, text, length_scale, noise_scale, noise_w, speaker_id)

        with self._lock:
            if key in self._cache:
                # Уже есть — обновляем и переставляем в конец
                self._cache.move_to_end(key)
                self._cache[key] = wav_data
                return

            # Если склад полон — выкидываем самую старую запись
            while len(self._cache) >= self._max:
                oldest_key, _ = self._cache.popitem(last=False)
                print(f"   🗑️ [Кэш] Вытеснено: {oldest_key[:8]}...")

            self._cache[key] = wav_data
            self._cache.move_to_end(key)

    def clear(self):
        """Полная очистка. Вызывается при graceful shutdown."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            print("🗑️ [Кэш] Оперативный склад очищен.")

    def stats(self) -> Dict[str, Any]:
        """
        Статистика для отладки: сколько записей, сколько попаданий,
        процент эффективности.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            # Примерный размер в МБ: 1 секунда 22050 Гц mono float32 ≈ 88 КБ
            # Но у нас уже готовые WAV (int16), примерно в 2 раза меньше.
            approx_mb = sum(len(b) for b in self._cache.values()) / (1024 * 1024)
            return {
                "entries": len(self._cache),
                "max": self._max,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
                "approx_mb": round(approx_mb, 2),
            }