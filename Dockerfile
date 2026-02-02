FROM python:3.10-slim-bookworm

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВЕСЬ проект (включая main.py)
COPY . .

# Проверяем, что файл main.py существует
RUN ls -la /app/*.py

# Создаем директорию для кеша huggingface
RUN mkdir -p /root/.cache/huggingface

# Запускаем приложение main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]