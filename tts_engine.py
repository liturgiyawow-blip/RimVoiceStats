# ============ tts_engine.py ============
# МОДУЛЬ СИНТЕЗА И ВОСПРОИЗВЕДЕНИЯ РЕЧИ — "ГОЛОСОВОЙ ЦЕХ"
# ======================================
# Здесь живёт Piper TTS. Этот модуль умеет:
# 1. Загружать голосовые модели (.onnx) и держать их в памяти.
# 2. Превращать текст в звук — строго в оперативной памяти (RAM), без SSD.
# 3. Воспроизводить звук через колонки.
# 4. Работать в фоновом потоке, не мешая веб-серверу.
#
# Важно: этот модуль НИЧЕГО не знает про Flask, LLM или игру.
# Ему дают текст и имя голоса — он отдаёт звук.

import os
import io
import gc
import wave
import threading
import queue
import time
import warnings
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf

# ============ Piper TTS ============
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PiperVoice = None
    PIPER_AVAILABLE = False
    warnings.warn("⚠️ piper-tts не установлен! Установите: pip install piper-tts")

try:
    import onnxruntime as ort
    print(f"🔧 ONNX Runtime провайдеры: {ort.get_available_providers()}")
except ImportError:
    pass

from cache import RAMWavCache


# ============ 1. МЕНЕДЖЕР ГОЛОСОВ ============
class PiperVoiceManager:
    """
    Склад загруженных голосовых моделей.
    
    Piper-модели тяжёлые (сотни мегабайт). Загружать их каждый раз
    при новой фразе — смерть для скорости. Этот класс загружает модель
    один раз и держит её в памяти, пока программа работает.
    """

    def __init__(self, voices_dir: str, use_gpu: bool = False):
        self.voices_dir = voices_dir
        self.use_gpu = use_gpu
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_model_path(self, voice_name: str) -> str:
        """Полный путь к файлу модели."""
        return os.path.join(self.voices_dir, voice_name + ".onnx")

    def get_config_path(self, voice_name: str) -> str:
        """Полный путь к JSON-конфигу модели (нужен для Piper 1.2+)."""
        return os.path.join(self.voices_dir, voice_name + ".onnx.json")

    def load(self, voice_name: str) -> Optional[Any]:
        """
        Загрузить голос в память. Если уже загружен — вернуть из кэша.
        Возвращает None, если файла модели нет на диске.
        """
        if not PIPER_AVAILABLE or PiperVoice is None:
            return None

        with self._lock:
            if voice_name in self._cache:
                return self._cache[voice_name]

            onnx_path = self.get_model_path(voice_name)
            config_path = self.get_config_path(voice_name)

            if not os.path.exists(onnx_path):
                print(f"❌ Модель не найдена: {onnx_path}")
                print(f"   Скачай отсюда: https://huggingface.co/rhasspy/piper-voices/")
                return None

            try:
                voice = PiperVoice.load(onnx_path, config_path, use_cuda=self.use_gpu)
                self._cache[voice_name] = voice
                print(f"🎙️ [Piper] Загружен: {voice_name} (GPU={self.use_gpu})")
                return voice
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {voice_name}: {e}")
                return None

    def preload(self, voice_names: list):
        """Предзагрузить несколько голосов при старте программы."""
        for name in voice_names:
            if name:
                self.load(name)


# ============ 2. ДВИЖОК СИНТЕЗА ============
class PiperTTSEngine:
    """
    Сердце голосового цеха. Получает текст и настройки — отдаёт WAV-байты.
    
    Работает строго в памяти:
    1. Создаём пустой WAV-файл в памяти (io.BytesIO).
    2. Просим Piper синтезировать текст по чанкам.
    3. Складываем чанки в WAV.
    4. Возвращаем готовые байты.
    
    Никаких записей на SSD в процессе.
    """

    def __init__(self, voice_manager: PiperVoiceManager, cache: RAMWavCache):
        self.voice_manager = voice_manager
        self.cache = cache

    def synthesize(self, voice_name: str, text: str,
                   length_scale: float, noise_scale: float,
                   noise_w: float, speaker_id: int = 0) -> Optional[bytes]:
        """
        Главный метод. Возвращает готовые WAV-байты или None при ошибке.
        """
        # 1. Проверяем кэш (если эту фразу уже синтезировали)
        if self.cache._max > 0:
            cached = self.cache.get(voice_name, text, length_scale, noise_scale, noise_w, speaker_id)
            if cached is not None:
                return cached

        # 2. Загружаем голос (или берём из памяти)
        voice = self.voice_manager.load(voice_name)
        if voice is None:
            return None

        # 3. Синтезируем
        wav_data = self._synthesize_to_bytes(voice, text, length_scale, noise_scale, noise_w, speaker_id)

        # 4. Кладём в кэш для будущего использования
        if wav_data is not None and self.cache._max > 0:
            self.cache.put(voice_name, text, length_scale, noise_scale, noise_w, speaker_id, wav_data)

        return wav_data

    def _synthesize_to_bytes(self, voice, text: str,
                             length_scale: float, noise_scale: float,
                             noise_w: float, speaker_id: int) -> Optional[bytes]:
        """
        Синтез через wave + synthesize_wav (как в старом проекте).
        Надёжнее для Piper 1.4.2.
        """
        buf = io.BytesIO()
        syn_config = None
        
        # Создаём SynthesisConfig (как в старом проекте)
        try:
            from piper import SynthesisConfig
            try:
                syn_config = SynthesisConfig(
                    length_scale=float(length_scale),
                    noise_scale=float(noise_scale),
                    noise_w_scale=float(noise_w),
                )
            except Exception as e:
                print(f"   ⚠️ Не удалось создать SynthesisConfig: {e}")
        except ImportError:
            pass

        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            
            if syn_config:
                voice.synthesize_wav(text, wav_file, syn_config)
            else:
                voice.synthesize_wav(text, wav_file)

        return buf.getvalue()


# ============ 3. ВОСПРОИЗВЕДЕНИЕ ============
class AudioPlayer:
    """
    Проигрыватель звука. Получает готовые WAV-байты и отправляет в колонки.
    
    Этот класс изолирован от Piper. В будущем, если захочешь отправлять
    звук не в колонки, а по сети или в файл — просто заменишь этот класс.
    """

    @staticmethod
    def play(wav_data: bytes, volume: float = 1.0) -> None:
        """
        Воспроизвести WAV-байты.
        
        wav_data — готовые байты WAV-файла (заголовок + аудио).
        volume — от 0.0 (тишина) до 1.0 (максимум).
        """
        if not wav_data or len(wav_data) == 0:
            return

        try:
            buf = io.BytesIO(wav_data)
            data, samplerate = sf.read(buf, dtype='float32')

            if data.ndim > 1:
                data = data.mean(axis=1)

            data = data * max(0.0, min(1.0, volume))

            sd.play(data, samplerate)
            sd.wait()

        except Exception as e:
            print(f"⚠️ Ошибка воспроизведения: {e}")

    @staticmethod
    def generate_silence(sample_rate: int, duration_ms: int) -> np.ndarray:
        """
        Создать тишину заданной длины. Нужно для пауз между фразами.
        """
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        return np.zeros(num_samples, dtype=np.float32)


# ============ 4. ФОНОВЫЙ ВОРКЕР ============
class TTSWorker:
    """
    Фоновый рабочий, который круглосуточно ждёт задачи из очереди.
    
    Это как конвейер на фабрике: главный поток (Flask) бросает задачу
    в корзину (tts_queue), а этот воркер берёт задачи из корзины
    и отправляет в голосовой цех.
    
    Так веб-сервер не тормозит: он мгновенно кладёт задачу в очередь
    и отвечает игре "ОК", а озвучка идёт параллельно.
    """

    def __init__(self, tts_queue: queue.Queue,
                 engine: PiperTTSEngine,
                 shutdown_event: threading.Event):
        self.tts_queue = tts_queue
        self.engine = engine
        self.shutdown_event = shutdown_event
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Запустить фоновый поток."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="TTS-Worker"
        )
        self._thread.start()
        print("🎙️ [TTS Worker] Голосовой цех запущен.")

    def _run(self):
        """Бесконечный цикл обработки задач."""
        while not self.shutdown_event.is_set():
            try:
                task = self.tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                self.tts_queue.task_done()
                break

            try:
                self._process_task(task)
            except Exception as e:
                print(f"💀 [TTS Worker] Ошибка: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.tts_queue.task_done()

        print("[TTS Worker] Голосовой цех остановлен.")

    def _process_task(self, task: Dict[str, Any]):
        """
        Обработка одной задачи озвучки.
        
        Теперь поддерживает список сегментов (segments) для атомарного
        воспроизведения цепочки фраз (например, имя + реплика).
        Если segments нет — работает в обратно-совместимом режиме (одна фраза).
        """
        segments = task.get("segments")
        if not segments:
            # Обратная совместимость: одна фраза = один сегмент
            segments = [task]

        voice_name = task.get("voice")
        base_length = task.get("length_scale", 1.0)
        base_noise = task.get("noise_scale", 0.667)
        base_noise_w = task.get("noise_w", 0.8)
        base_speaker = task.get("speaker_id", 0)

        for seg in segments:
            text = seg.get("text", "")
            if not text:
                continue

            volume = max(0.0, min(1.5, seg.get("volume", task.get("volume", 1.0))))
            post_delay = seg.get("post_delay", task.get("post_delay", 0.0))
            length_scale = seg.get("length_scale", base_length)
            noise_scale = seg.get("noise_scale", base_noise)
            noise_w = seg.get("noise_w", base_noise_w)
            speaker_id = seg.get("speaker_id", base_speaker)

            label = "🧠 Мысль" if task.get("is_thought") else "💬 Реплика"
            print(f"🔊 [TTS] {label}: {text[:70]}{'...' if len(text) > 70 else ''}")
            print(f"   🎭 Голос: {voice_name} | length={length_scale:.2f}")

            wav_data = self.engine.synthesize(
                voice_name=voice_name,
                text=text,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                speaker_id=speaker_id
            )

            if wav_data is not None:
                AudioPlayer.play(wav_data, volume)
                if post_delay > 0:
                    time.sleep(post_delay)
                del wav_data
                gc.collect()
            else:
                print(f"   ❌ Синтез не удался для: {text[:50]}...")