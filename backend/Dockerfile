# Используем официальный Python образ с CUDA поддержкой
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Устанавливаем Python 3.12 (доступен в Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Создаем символическую ссылку для python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PyTorch с CUDA поддержкой вручную
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124

# Копируем исходный код
COPY . .

# Создаем пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
