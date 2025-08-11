from ctransformers import AutoModelForCausalLM
import os
import time
print("загрузка модели\n")
# Пути к модели
MODEL_DIR = "./models"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Используем более легкую модель
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

# Загрузка модели с оптимизированными параметрами
model = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id=MODEL_PATH,
    model_type="mistral",
    gpu_layers=0,
    threads=8,               # Уменьшаем количество потоков
    context_length=2048,     # Уменьшаем длину контекста
    mmap=True,
    mlock=False              # Отключаем блокировку памяти
)
print("модель загружена\n")
# Упрощенный промпт

prompt = """
<s>[INST] <<SYS>>
Ты — русскоязычный, грамотно без транслита.
<</SYS>>
Объясни образование зиготы[/INST]
"""

# Измеряем время выполнения
start = time.time()

# Генерируем ответ
output = model(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

duration = time.time() - start
word_count = len(output.split())

print(f"Скорость: {word_count / duration:.1f} слов/сек")
print(f"Время исполнения: {duration:.2f} сек")
print(output)