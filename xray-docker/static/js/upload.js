// 创建static目录和js子目录，添加以下内容
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    
    if (form) {
        // 确保表单有正确的编码类型
        form.setAttribute('enctype', 'multipart/form-data');
        
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.querySelector('input[type="file"]');
            if (!fileInput || !fileInput.files.length) {
                alert('请选择一个文件');
                return;
            }
            
            const formData = new FormData(form);
            
            // 显示加载指示器
            const loadingElement = document.getElementById('loading');
            if (loadingElement) loadingElement.style.display = 'block';
            
            // 禁用提交按钮
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) submitButton.disabled = true;
            
            // 设置超时处理
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('请求超时，服务器处理时间过长')), 90000);
            });
            
            // 使用Promise.race同时处理fetch和超时
            Promise.race([
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                }),
                timeoutPromise
            ])
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(`服务器错误: ${err.error || response.statusText}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('收到服务器响应:', data);
                
                // 隐藏加载指示器
                if (loadingElement) loadingElement.style.display = 'none';
                
                // 重新启用提交按钮
                if (submitButton) submitButton.disabled = false;
                
                // 处理成功响应
                if (data.image) {
                    // 创建或更新结果容器
                    let resultContainer = document.getElementById('result-container');
                    if (!resultContainer) {
                        resultContainer = document.createElement('div');
                        resultContainer.id = 'result-container';
                        resultContainer.className = 'mt-4 p-3 border rounded';
                        form.parentNode.insertBefore(resultContainer, form.nextSibling);
                    }
                    
                    resultContainer.innerHTML = `
                        <h3 class="mb-3">${data.result_text || 'Analysis Result'}</h3>
                        <div class="text-center">
                            <img src="data:image/png;base64,${data.image}" class="img-fluid" alt="Analysis Result">
                        </div>
                        <p class="mt-3">Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                    `;
                    
                    resultContainer.style.display = 'block';
                    
                    // 滚动到结果区域
                    resultContainer.scrollIntoView({behavior: 'smooth'});
                } else {
                    alert('服务器响应中没有图像数据');
                }
            })
            .catch(error => {
                console.error('上传错误:', error);
                
                // 隐藏加载指示器
                if (loadingElement) loadingElement.style.display = 'none';
                
                // 重新启用提交按钮
                if (submitButton) submitButton.disabled = false;
                
                // 显示错误消息
                if (error.message.includes('超时')) {
                    alert('图像分析耗时过长，请尝试使用较小的图片或稍后再试');
                } else {
                    alert(`上传失败: ${error.message}`);
                }
            });
        });
    } else {
        console.error('未找到表单元素');
    }
});