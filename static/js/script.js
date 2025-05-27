document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('upload-form');
    const progressArea = document.getElementById('progress-area'); // You'll need to add this div in HTML
    const resultArea = document.getElementById('result-area');   // You'll need to add this div in HTML
    const submitButton = form.querySelector('button[type="submit"]');

    if (!form) {
        console.error('Upload form not found!');
        return;
    }
    if (!progressArea) {
        console.warn('Progress area element not found. Progress updates will not be displayed.');
    }
    if (!resultArea) {
        console.warn('Result area element not found. Results will not be displayed.');
    }

    form.addEventListener('submit', function (event) {
        event.preventDefault();
        if (progressArea) progressArea.innerHTML = ''; // Clear previous progress
        if (resultArea) resultArea.innerHTML = '';   // Clear previous results
        if (submitButton) submitButton.disabled = true;
        if (progressArea) progressArea.textContent = '上传中，请稍候...';

        const formData = new FormData(form);

        fetch('/', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                // Try to get error message from JSON response if server sent one
                return response.json().then(err => {
                    throw new Error(err.error || `服务器错误: ${response.status} ${response.statusText}`);
                }).catch(() => {
                    // Fallback if response is not JSON or err.error is not present
                    throw new Error(`服务器错误: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            if (data.job_id) {
                if (progressArea) progressArea.textContent = '处理已开始... 任务 ID: ' + data.job_id;
                pollStatus(data.job_id);
            } else {
                throw new Error('未能获取任务ID。');
            }
        })
        .catch(error => {
            console.error('提交错误:', error);
            if (progressArea) progressArea.textContent = '错误: ' + error.message;
            if (submitButton) submitButton.disabled = false;
        });
    });

    function pollStatus(jobId) {
        const intervalId = setInterval(() => {
            fetch(`/status/${jobId}`)
                .then(response => {
                    if (!response.ok) {
                        // Attempt to parse error from server if available
                        return response.json().then(err => {
                            throw new Error(err.status || `状态检查错误: ${response.status}`);
                        }).catch(() => {
                             throw new Error(`状态检查错误: ${response.status} ${response.statusText}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.status === 'Job not found'){
                        if (progressArea) progressArea.textContent = '任务未找到或已过期。';
                        clearInterval(intervalId);
                        if (submitButton) submitButton.disabled = false;
                        return;
                    }

                    let progressMessage = `状态: ${data.status}`;
                    if (data.progress !== undefined) {
                        progressMessage += ` (${data.progress}%)`;
                    }
                    if (progressArea) progressArea.textContent = progressMessage;

                    if (data.status === 'Done' || data.status.startsWith('Error:')) {
                        clearInterval(intervalId);
                        if (submitButton) submitButton.disabled = false;
                        if (data.status === 'Done' && data.result_video) {
                            if (resultArea) {
                                const downloadLink = document.createElement('a');
                                downloadLink.href = `/downloads/${data.result_video}`;
                                downloadLink.textContent = `下载处理完成的视频: ${data.result_video}`;
                                downloadLink.setAttribute('download', data.result_video);
                                resultArea.appendChild(downloadLink);
                            }
                        } else if (data.status.startsWith('Error:')) {
                            if (resultArea) resultArea.textContent = `处理失败: ${data.status}`;
                        }
                    }
                })
                .catch(error => {
                    console.error('轮询错误:', error);
                    if (progressArea) progressArea.textContent = '轮询状态时发生错误: ' + error.message;
                    clearInterval(intervalId);
                    if (submitButton) submitButton.disabled = false;
                });
        }, 2000); // Poll every 2 seconds
    }
});
