import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，服务器环境中必须
import matplotlib.pyplot as plt

# 初始化Flask应用
app = Flask(__name__)
# 配置上传文件夹
app.config['UPLOAD_FOLDER'] = 'uploads'
# 限制上传文件大小为16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上传限制
# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 在应用启动时一次性加载模型，避免每次请求都加载
MODEL_PATH = "Models/F_M1_90"  # 根据实际情况更新路径
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理图像上传并进行预测"""
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': '请求中没有文件部分'}), 400
    
    file = request.files['file']
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 保存上传的文件
    filename = secure_filename(file.filename)  # 确保文件名安全
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 处理图像
    try:
        # 打开并预处理图像
        test_orig = Image.open(file_path).convert("L")  # 转换为灰度图
        test = test_orig.resize((1100, 1100), Image.LANCZOS)  # 调整大小
        test_tensor = tf.reshape(tf.constant(test, dtype=tf.float32), (1, 1100, 1100, 1))  # 转换为模型输入格式
        
        # 进行预测
        prediction = model.predict(test_tensor)
        result = np.argmax(prediction)  # 获取预测类别
        
        # 创建结果图像
        plt.figure(figsize=(12, 6))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(test_orig, cmap='gray')
        plt.title("原图像")
        
        # 显示处理后的图像
        plt.subplot(1, 2, 2)
        plt.imshow(tf.reshape(test_tensor, (1100, 1100)), cmap='gray')
        plt.title("模型输入图像")
        
        # 根据预测结果显示不同的标题
        if result == 0:
            result_text = "参考结果：正常"
        else:
            result_text = "参考结果：肺炎"
            
        plt.suptitle(result_text)
        plt.tight_layout()
        
        # 将图表保存为base64字符串，用于在网页中显示
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()  # 关闭图表，释放内存
        
        # 返回结果
        return jsonify({
            'result': int(result),  # 预测的类别
            'result_text': result_text,  # 显示给用户的文本
            'confidence': float(prediction[0][result]),  # 预测的置信度
            'prediction_values': prediction.tolist(),  # 所有预测值
            'image': img_str  # base64编码的结果图像
        })
    
    except Exception as e:
        # 捕获并返回处理过程中的任何错误
        return jsonify({'error': str(e)}), 500

# 应用程序入口点
if __name__ == '__main__':
    # 启动Flask服务器，监听所有网络接口
    app.run(host='0.0.0.0', port=5000, debug=False)