# 视频字幕翻译系统

这是一个使用Python Flask、Whisper和LLM API构建的视频字幕翻译系统。

## 功能

1.  上传视频文件。
2.  使用本地Whisper模型提取视频中的对话作为字幕。
    *   支持选择不同的Whisper模型 (tiny, base, small, medium, large)。
3.  调用大模型API（如OpenAI）将字幕翻译成指定语言。
4.  生成翻译后字幕和双语字幕。
5.  将可选的字幕（原始、翻译后、双语）及其样式嵌入（烧录）到视频中。
    *   支持自定义字体、大小、颜色、描边、阴影等样式。
6.  异步处理视频，并通过前端JavaScript轮询状态，显示实时进度和结果。
7.  提供多种视频输出质量选项 (1080p, 720p, 480p, 源质量)。

## 技术栈
*   **前端**: HTML, CSS, JavaScript (用于异步表单提交和状态轮询)
*   **后端**: Python (Flask)
*   **语音转文字**: Whisper (本地模型)

## 项目结构

```
windsurf-project/
├── app.py                  # Flask 应用主文件
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明
├── uploads/                # 存放用户上传的原始视频
├── processed_videos/       # 存放处理完成的带字幕视频
├── subtitles/              # 存放生成的SRT和ASS字幕文件
├── temp_audio/             # 存放提取的临时音频文件
├── templates/
│   └── index.html          # 前端HTML页面
└── static/
    ├── css/
    │   └── style.css       # CSS样式文件 (如果使用)
    └── js/
        └── script.js       # 前端JavaScript，处理异步任务和进度显示
```

## 安装与运行

1.  **克隆/下载项目**

2.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    您可能还需要单独安装 `ffmpeg` 和 `Whisper`。
    *   **FFmpeg:** 请根据您的操作系统从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并安装，确保它在系统的PATH中。
    *   **Whisper:** `openai-whisper` 会通过 pip 安装。确保您有 PyTorch 安装好，这是 Whisper 的一个依赖。可以参考 OpenAI Whisper [GitHub仓库](https://github.com/openai/whisper) 的指引进行更详细的安装（例如特定 CUDA 版本的 PyTorch）。
      ```bash
      # PyTorch (example for CPU, check https://pytorch.org/ for other versions)
      # pip install torch torchvision torchaudio
      pip install openai-whisper
      ```
    *   **Pysubs2:** 用于高级字幕处理和样式应用。
      ```bash
      pip install pysubs2
      ```

3.  **配置 (如果需要):**
    *   在 `app.py` 中或通过环境变量配置API密钥等。
    *   用户可在前端选择Whisper模型大小。

4.  **运行应用:**
    ```bash
    python app.py
    ```
    应用默认会在 `http://127.0.0.1:5000/` 启动。

## 后续开发计划

*   实现完整的视频处理流程：
    *   音频提取
    *   Whisper集成进行语音识别
    *   调用OpenAI API进行字幕翻译
    *   使用ffmpeg进行字幕烧录和视频转码
*   完善字幕样式自定义功能。
    *   提供更多样式选项 (位置, 边距, 背景框等)
*   添加实时进度显示。
*   错误处理和日志记录。
*   异步任务处理大文件。
    *   目前已实现基于ThreadPoolExecutor的后台处理和状态轮询。

## 已知问题/待办
*   完善异步任务处理，确保大文件处理时不会阻塞主线程。
*   实现更好的进度显示和状态更新机制。
    *   前端通过 `static/js/script.js` 实现异步提交和状态轮询。

## 未来可能的增强功能
*   支持更多的视频格式和编解码器。
*   实现字幕编辑功能。
*   支持更多的翻译语言和模型。
*   提供更好的用户界面和用户体验。
