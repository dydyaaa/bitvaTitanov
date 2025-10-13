# CUDA 12.6 + cuDNN, Ubuntu 24.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Инструменты и Python 3.12 с venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# Изолируем окружение
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA 12.4
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124


COPY . .

# Непривилегированный пользователь
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
