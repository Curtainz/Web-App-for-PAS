import traceback
import logging
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
import matplotlib.pyplot as plt

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.config['UPLOAD_FOLDER'] = '/uploads'  # 修改为容器内挂载点
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 限制上传文件大小为16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 更新模型路径以匹配挂载点
MODEL_PATH = "/models/F_M1_90"

# 兼容Keras 3的模型加载方式
try:
    # 首先尝试直接加载模型
    model = tf.keras.models.load_model(MODEL_PATH)
except ValueError as e:
    # 如果失败，尝试使用TFSMLayer加载
    print(f"标准加载失败，尝试使用TFSMLayer: {e}")
    try:
        model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
    except Exception as e2:
        # 如果仍然失败，尝试指定特定版本的TF兼容性
        print(f"TFSMLayer加载失败: {e2}")
        # 通过设置环境变量降级TF行为
        os.environ['TF_KERAS'] = '1'
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("收到预测请求")
    app.logger.info(f"请求表单: {request.form}")
    app.logger.info(f"请求文件: {request.files}")
    
    if 'file' not in request.files:
        app.logger.error("请求中没有文件部分")
        return jsonify({'error': '请求中没有文件部分'}), 400
    
    file = request.files['file']
    app.logger.info(f"文件名: {file.filename}")

    if file.filename == '':
        app.logger.error("未选择文件")
        return jsonify({'error': '未选择文件'}), 400
    
    try:
        filename = secure_filename(file.filename)
        
        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        app.logger.info(f"尝试保存文件到: {file_path}")
        
        # 保存文件
        file.save(file_path)
        app.logger.info(f"文件成功保存到: {file_path}")
        
        try:
            test_orig = Image.open(file_path).convert("L")
            test = test_orig.resize((1100, 1100), Image.LANCZOS)
            test_tensor = tf.reshape(tf.constant(test, dtype=tf.float32), (1, 1100, 1100, 1))
            
            prediction_values = model.predict(test_tensor)
            
            result = np.argmax(prediction_values)
            
            # 使用英文标题避免中文字体问题
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(test_orig, cmap='gray')
            plt.title("Original Image")

            plt.subplot(1, 2, 2)
            plt.imshow(tf.reshape(test_tensor, (1100, 1100)), cmap='gray')
            plt.title("Processed Image")

            if result == 0:
                result_text = "Result: Normal"
            else:
                result_text = "Result: Pneumonia"
    
            plt.suptitle(result_text)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)  # 降低分辨率
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return jsonify({
                'result': int(result),
                'result_text': result_text,
                'confidence': float(prediction_values[0][result]),
                'prediction_values': prediction_values.tolist(),
                'image': img_str
            })
        except Exception as e:
            app.logger.error(f"图像处理过程中出错: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            return jsonify({
                'error': f"图像处理错误: {str(e)}",
                'details': error_details
            }), 500
            
    except Exception as e:
        app.logger.error(f"上传处理过程中出错: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'error': f"文件上传错误: {str(e)}",
            'details': error_details
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)