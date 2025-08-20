from ctransformers import AutoModelForCausalLM
import os
import time
print("загрузка модели\n")
# Пути к модели
MODEL_DIR = "./models"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q2_K.gguf"  # Используем более легкую модель
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

# Загрузка модели с оптимизированными параметрами
model = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id=MODEL_PATH,
    model_type="mistral",
    gpu_layers=0,
    threads=8,               # Уменьшаем количество потоков
    context_length=1024,     # Уменьшаем длину контекста
    mmap=True,
    mlock=False              # Отключаем блокировку памяти
)
print("модель загружена\n")
# Упрощенный промпт

prompt = """
<s>[INST] <<SYS>>
Ты — русскоязычный, грамотно, коротко, ёмко, без транслита.
<</SYS>>
Объясни образование зиготы[/INST]
"""

# Измеряем время выполнения
start = time.time()

# Генерируем ответ
output = model(
    prompt,
    max_new_tokens=512,       # меньше токенов = быстрее
    temperature=0.1,         # низкая температура = меньше случайности, быстрее выбор
    top_p=0.95,              # высокое top_p не влияет на скорость напрямую, но с низким temperature работает стабильно
    top_k=2,                # ограничение количества рассматриваемых вариантов
    repetition_penalty=1.0,  # без штрафа за повторения = быстрее
    threads=int(os.cpu_count()),  # максимальное количество потоков CPU
    stream=False             # без потоковой выдачи = быстрее
)

duration = time.time() - start
word_count = len(output.split())

print(f"Скорость: {word_count / duration:.1f} слов/сек")
print(f"Время исполнения: {duration:.2f} сек")
print(output)