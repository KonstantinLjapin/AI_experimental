from huggingface_hub import hf_hub_download

# Загружаем и сохраняем модель в локальную директорию
model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q2_K.gguf",
    local_dir="./models",  # Сохраняем в локальную директорию
    force_download=True,   # Принудительная загрузка (если нужно)
)

print(f"Модель сохранена в: {model_path}")