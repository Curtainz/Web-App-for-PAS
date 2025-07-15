# 基础镜像：使用Python 3.9精简版，体积小
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# libgl1-mesa-glx: OpenCV和matplotlib可能需要的图形库
# libglib2.0-0: GTK图形界面库的依赖
RUN apt-get update -y || (rm -rf /var/lib/apt/lists/* && apt-get update -y) \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
# --no-cache-dir: 不缓存pip下载的包，减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模板
COPY flask-app.py .
COPY templates templates/

# 创建必要的目录
# uploads: 存放上传的图像
# Models: 存放机器学习模型
RUN mkdir -p uploads models

# 暴露应用端口
EXPOSE 5000

# 容器启动时执行的命令
# gunicorn: WSGI HTTP服务器，用于生产环境
# --bind 0.0.0.0:5000: 绑定到所有网络接口的5000端口
# app:app: 指定WSGI应用对象(app.py中的app变量)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]