import os
import time
from openai import OpenAI

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
print(settings.api_key)
start = time.time()
api_key = settings.api_key
url = "https://foundation-models.api.cloud.ru/v1"

client = OpenAI(
    api_key=api_key,
    base_url=url
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    max_tokens=5000,
    temperature=0.5,
    presence_penalty=1.1,
    top_p=0.95,
    messages=[
        {
            "role": "user",
            "content": "<s>[INST] <<SYS>>Ты — русскоязычный, грамотно, коротко, ёмко, без транслита, не больше 15 слов"
                       "<</SYS>>Объясни образование зиготы[/INST]"
        }
    ]
)

print(response.choices[0].message.content)

duration = time.time() - start

print(f"Время исполнения: {duration:.2f} сек")

