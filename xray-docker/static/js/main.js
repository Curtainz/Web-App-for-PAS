document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressContainer = document.getElementById('progress-container');
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');
    const resultImage = document.getElementById('result-image');
    const probabilityBars = document.getElementById('probability-bars');

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 检查是否有文件
        if (!fileInput.files.length) {
            showError('请选择一个图像文件');
            return;
        }
        
        const file = fileInput.files[0];
        
        // 检查文件类型
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            showError('请选择有效的图像文件 (JPG, PNG, GIF)');
            return;
        }
        
        // 准备表单数据
        const formData = new FormData();
        formData.append('file', file);
        
        // 显示进度条
        progressContainer.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        analyzeBtn.disabled = true;
        
        // 发送请求
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || '服务器错误');
                });
            }
            return response.json();
        })
        .then(data => {
            // 处理结果
            displayResults(data);
        })
        .catch(error => {
            showError('分析过程中出错: ' + error.message);
        })
        .finally(() => {
            progressContainer.classList.add('d-none');
            analyzeBtn.disabled = false;
        });
    });
    
    function displayResults(data) {
        // 显示结果容器
        resultContainer.classList.remove('d-none');
        
        // 显示预测结果
        const prediction = data.prediction;
        const confidence = (data.confidence * 100).toFixed(2);
        
        // 设置结果文本
        resultContent.innerHTML = `
            <div class="alert ${prediction === '肺炎' ? 'alert-danger' : 'alert-success'}">
                <h4 class="alert-heading">${prediction}</h4>
                <p>置信度: ${confidence}%</p>
            </div>
        `;
        
        // 显示可视化图像
        if (data.visualization) {
            resultImage.src = 'data:image/png;base64,' + data.visualization;
            resultImage.classList.remove('d-none');
        } else {
            resultImage.classList.add('d-none');
        }
        
        // 显示概率条
        probabilityBars.innerHTML = '';
        if (data.probabilities && data.class_names) {
            data.probabilities.forEach((prob, index) => {
                const className = data.class_names[index];
                const percentage = (prob * 100).toFixed(2);
                
                const barColor = className === '肺炎' ? 'bg-danger' : 'bg-success';
                
                probabilityBars.innerHTML += `
                    <div class="mb-2">
                        <div class="d-flex justify-content-between">
                            <span>${className}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar ${barColor}" role="progressbar" 
                                 style="width: ${percentage}%" 
                                 aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        // 滚动到结果
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    function showError(message) {
        resultContainer.classList.remove('d-none');
        resultContent.innerHTML = `
            <div class="alert alert-danger">
                <h4 class="alert-heading">错误</h4>
                <p>${message}</p>
            </div>
        `;
        resultImage.classList.add('d-none');
        probabilityBars.innerHTML = '';
    }
});