FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 安装特定版本的TensorFlow (2.x版本，与SavedModel格式兼容)
RUN pip install --no-cache-dir tensorflow==2.12.0
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates templates/

# 创建必要的目录
RUN mkdir -p /uploads /models

# 修改app.py中的路径
RUN sed -i 's|UPLOAD_FOLDER = .*|UPLOAD_FOLDER = "/uploads"|g' app.py && \
    sed -i 's|MODEL_PATH = .*|MODEL_PATH = "/models/F_M1_90"|g' app.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]