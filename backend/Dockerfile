# GPU image. Not built on the dev laptop (no Docker there). Build from the project root: docker compose build api
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && pip3 install --no-cache-dir -r requirements.txt
COPY backend backend
COPY models models
COPY data/download_data.py data/make_scenarios.py data/
COPY cameras.json .
RUN python3 data/download_data.py && python3 data/make_scenarios.py
EXPOSE 8400
CMD ["python3", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8400"]
