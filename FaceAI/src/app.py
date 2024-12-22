"""
# -*- coding=utf8 -*-
@Project ：Faceai
@File    ：app.py
@IDE     ：PyCharm 
@Author  ：爬虫工作室'
@Date    ：2024-12-22 10:19
"""
from recognition import *
from flask import Flask, request, jsonify, render_template
import os
from model import CNN3
from werkzeug.utils import secure_filename
# 初始化 Flask 应用
app = Flask(__name__)
# 配置文件上传目录
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
def predict(image_path):
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    return predict_expression(image_path, model)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')  # 渲染前端页面

# 接收图片并进行表情分析
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 保存上传的图片
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)
        image_bytes = file.read()

        # 调用 predict 函数进行分析
        try:
            emotion,_ = predict(image_bytes )
            return jsonify({'emotion': emotion})  # 返回表情分析结果
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)