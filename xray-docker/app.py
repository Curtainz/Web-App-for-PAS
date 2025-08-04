# 导入必要的库
import os
import sys
import numpy as np
import tensorflow as tf
import io
import base64
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

# 禁用oneDNN优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 尝试导入Resampling，如果失败则使用旧版本API
try:
    from PIL import Image, ImageResampling
    resample_method = ImageResampling.LANCZOS  # 高质量下采样
except ImportError:
    from PIL import Image
    # 忽略特定的Pillow弃用警告
    import warnings
    warnings.filterwarnings('ignore', message='LANCZOS is deprecated')
    resample_method = Image.LANCZOS

from typing import Optional, Tuple, List  # 类型注解
import matplotlib
matplotlib.use('Agg')  # 在无头环境中使用
import matplotlib.pyplot as plt

# 初始化Flask应用
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/saved_model')
CLASS_NAMES = ["正常", "肺炎"]

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 全局变量，用于存储加载的模型
global_model = None

# 将pb文件转化为keras或h5文件保证文件能够兼容tensorflow新版本
def model_upgrade(old_model_dir: str, new_keras_path: str) -> Optional[str]:
    try:
        if not os.path.isdir(old_model_dir):
            raise ValueError(f"输入路径必须是目录: {old_model_dir}")

        # 对于Keras 3，使用TFSMLayer加载SavedModel然后保存为.keras格式
        print(f"使用TFSMLayer加载SavedModel: {old_model_dir}")
        # 创建一个包含TFSMLayer的模型
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, None, 1)),  # 根据实际输入形状调整
            tf.keras.layers.TFSMLayer(old_model_dir, call_endpoint='serving_default')
        ])
        # 保存为.keras格式
        model.save(new_keras_path)
        print(f"[成功] 模型已保存为: {new_keras_path}")
        return new_keras_path
    except Exception as e:
        print(f"[失败] 转换错误: {str(e)}", file=sys.stderr)
        return None

# 模型加载函数
def load_model(model_path: str) -> Optional[tf.keras.Model]:
    try:
        model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"文件不存在：{model_path}")

        # 检查是否为目录 (可能是SavedModel)
        if os.path.isdir(model_path):  # 处理saved model 目录
            print(f"检测到SavedModel目录，使用TFSMLayer加载: {model_path}")
            # 使用TFSMLayer加载SavedModel
            # 首先尝试获取模型签名以确定输入形状
            try:
                import tensorflow.saved_model as sm
                loaded = sm.load(model_path)
                # 获取服务默认签名
                serving_default = loaded.signatures['serving_default']
                # 获取输入形状
                input_spec = list(serving_default.parameters.values())[0]
                input_shape = input_spec.shape
                print(f"检测到模型输入形状: {input_shape}")
            except Exception as e:
                print(f"获取模型输入形状失败: {str(e)}")
                # 使用默认输入形状
                input_shape = (None, 1100, 1100, 1)
                print(f"使用默认输入形状: {input_shape}")

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape[1:]),  # 排除批次维度
                tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
            ])
            print(f"模型加载成功，使用SavedModel格式")
            return model
        # 检查文件扩展名
        elif model_path.endswith('.pb'):
            raise ValueError("Keras 3 不支持 .pb 格式。请将模型转换为 .keras 或 .h5 格式。")
        elif model_path.endswith(('.keras', '.h5')):
            model = tf.keras.models.load_model(model_path)  # 直接加载
            print(f"模型加载成功，输入形状: {model.input_shape}")
            return model
        else:
            raise ValueError(f"不支持的文件格式: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}", file=sys.stderr)
        return None

# 图像预处理模块
def process_image(image_path: str, target_size: Optional[Tuple[int, int]] = None):
    try:
        img = Image.open(image_path).convert('L')
        orig_img = img.copy()
        if target_size is not None:
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError("target_size 应为 (width, height) 元组")
            # 确保target_size中的元素都是正整数
            if not all(isinstance(dim, int) and dim > 0 for dim in target_size):
                raise ValueError("target_size 中的元素必须是正整数")
            # 使用适当的重采样方法
            img = img.resize(target_size, resample_method)
        img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化
        # 正确处理维度扩展
        img_tensor = tf.expand_dims(img_array, axis=0)  # 添加批次维度
        img_tensor = tf.expand_dims(img_tensor, axis=-1)  # 添加通道维度
        return img_tensor, orig_img
    except Exception as e:
        print(f"图像处理失败: {str(e)}", file=sys.stderr)
        return None, None

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 预测函数
def predict(model: tf.keras.Model, image_path: str, class_names: Optional[List[str]] = None):
    if not (model and hasattr(model, "predict")):
        raise ValueError("无效模型对象")

    if class_names is None:
        class_names = ["正常", "肺炎"]

    # 获取输入形状
    try:
        # 尝试从模型中获取输入形状
        input_shape = model.input_shape
        print(f"模型输入形状: {input_shape}")
        target_size = input_shape[1:3]

        # 确保target_size是有效的整数元组
        if not all(isinstance(dim, int) and dim > 0 for dim in target_size):
            # 如果尺寸无效，尝试使用错误日志中提到的1100x1100
            target_size = (1100, 1100)
            print(f"模型输入形状无效，使用 fallback 尺寸: {target_size}")
        else:
            print(f"使用模型输入形状: {target_size}")
    except Exception as e:
        print(f"无法获取模型输入形状，使用默认值 (1100, 1100): {str(e)}")
        target_size = (1100, 1100)

    finish_img, orig_img = process_image(image_path, target_size)

    if finish_img is None or orig_img is None:
        print("图像处理失败", file=sys.stderr)
        return None

    try:
        predicts = model.predict(finish_img, verbose=0)
        # 处理TFSMLayer可能返回的嵌套结构
        if isinstance(predicts, dict):
            # 假设输出是{'output_name': array}
            predicts = list(predicts.values())[0]
        # 确保predicts是2D数组 [batch, classes]
        if len(predicts.shape) > 2:
            predicts = predicts.reshape(predicts.shape[0], -1)

        pred_class = np.argmax(predicts)
        confidence = np.max(predicts)

        if len(class_names) != predicts.shape[1]:
            print(f"警告: 标签数量({len(class_names)})与结果类别数量({predicts.shape[1]})不符")
            # 调整class_names以匹配结果
            class_names = [f"类别{i}" for i in range(predicts.shape[1])]

        # 创建可视化图表并返回
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title("原始图像")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(finish_img), cmap="gray")
        plt.title(f"调整尺寸 ({target_size[0]}x{target_size[1]})")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图表为内存中的图像
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        result = {
            "prediction": class_names[pred_class],
            "confidence": float(confidence),
            "probabilities": [float(p) for p in predicts[0]],
            "class_names": class_names,
            "visualization": plot_data
        }
        
        return result
    except Exception as e:
        print(f"预测错误: {str(e)}", file=sys.stderr)
        return None

# 在应用启动时尝试加载模型
print("尝试加载模型...")
try:
    global_model = load_model(MODEL_PATH)
    if global_model is None:
        print("警告：模型加载失败！应用将无法进行预测。", file=sys.stderr)
    else:
        print("模型加载成功！")
except Exception as e:
    print(f"模型加载出错: {str(e)}", file=sys.stderr)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 上传和预测路由
@app.route('/predict', methods=['POST'])
def upload_file():
    global global_model
    
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 检查模型是否已加载
        if global_model is None:
            try:
                global_model = load_model(MODEL_PATH)
                if global_model is None:
                    return jsonify({'error': '模型加载失败'}), 500
            except Exception as e:
                return jsonify({'error': f'模型加载错误: {str(e)}'}), 500
        
        # 进行预测
        try:
            result = predict(global_model, filepath, CLASS_NAMES)
            if result is None:
                return jsonify({'error': '预测失败'}), 500
                
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'预测过程错误: {str(e)}'}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

# 提供上传的文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 主函数
if __name__ == '__main__':
    # 确保模型目录存在
    if not os.path.exists(MODEL_PATH):
        print(f"警告：模型路径不存在: {MODEL_PATH}", file=sys.stderr)
        print("请确保模型已正确放置，或更新MODEL_PATH变量。", file=sys.stderr)
    
    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=True)