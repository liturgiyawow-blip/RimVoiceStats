# debug_piper.py — файл для диагностики
from piper import PiperVoice

print("Загружаю голос...")
voice = PiperVoice.load(
    'piper_voices/ru_RU-irina-medium.onnx',
    'piper_voices/ru_RU-irina-medium.onnx.json',
    use_cuda=False
)

print("Синтезирую тестовую фразу...")
chunks = list(voice.synthesize('Привет, мир!'))

print(f"Количество чанков: {len(chunks)}")

if chunks:
    chunk = chunks[0]
    print(f"Тип чанка: {type(chunk)}")
    print(f"Поля чанка: {[x for x in dir(chunk) if not x.startswith('_')]}")
    
    for attr in ['audio_bytes', 'bytes', 'data', 'audio', 'audio_int16_bytes', 'wav_bytes', 'sample_rate', 'rate']:
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            if val is not None:
                try:
                    length = len(val)
                    print(f"{attr}: {type(val).__name__} | длина: {length}")
                except TypeError:
                    print(f"{attr}: {type(val).__name__} | значение: {val}")
            else:
                print(f"{attr}: None")
        else:
            print(f"{attr}: НЕТ ТАКОГО ПОЛЯ")
else:
    print("Чанков нет!")