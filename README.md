# 视频字幕翻译系统

这是一个使用Python Flask、Whisper和LLM API构建的视频字幕翻译系统。

## 功能

1.  上传视频文件。
2.  使用本地Whisper模型提取视频中的对话作为字幕。
    *   支持选择不同的Whisper模型 (tiny, base, small, medium, large)。
3.  调用大模型API（如OpenAI）将字幕翻译成指定语言。
    *   支持自定义OpenAI兼容的API基础URL
    *   自动回退机制，兼容不同模型版本
4.  生成翻译后字幕和双语字幕。
5.  将可选的字幕（原始、翻译后、双语）及其样式嵌入（烧录）到视频中。
    *   支持自定义字体、大小、颜色、描边、阴影等样式。
6.  异步处理视频，并通过前端JavaScript轮询状态，显示实时进度和结果。
7.  提供多种视频输出质量选项 (1080p, 720p, 480p, 源质量)。
8.  完整的错误处理和日志记录。

## 技术栈
*   **前端**: HTML, CSS, JavaScript (用于异步表单提交和状态轮询)
*   **后端**: Python (Flask)
*   **语音转文字**: Whisper (本地模型)

## 项目结构

```
burning-sub/
├── app.py                  # Flask 应用主文件
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明
├── uploads/                # 存放用户上传的原始视频
├── processed_videos/       # 存放处理完成的带字幕视频
├── subtitles/              # 存放生成的SRT和ASS字幕文件
├── temp_audio/             # 临时存放提取的音频文件（处理完成后自动清理）
├── static/
│   └── js/
│       └── script.js       # 前端JavaScript，处理异步任务和进度显示
└── templates/
    └── index.html          # 前端HTML页面
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

3.  **配置:**
    *   **OpenAI API 配置**
        - 在前端界面直接输入API密钥
        - 可选：指定自定义的OpenAI兼容API基础URL
    *   **Whisper 模型**
        - 用户可在前端选择不同大小的Whisper模型
        - 默认使用 base 模型，支持 tiny, small, medium, large
    *   **视频处理**
        - 支持多种输出质量 (1080p, 720p, 480p, 源质量)
        - 可自定义字幕样式（字体、大小、颜色等）

4.  **运行应用:**
    ```bash
    python app.py
    ```
    应用默认会在 `http://127.0.0.1:5000/` 启动。

## 已实现功能

*   完整的视频处理流程：
    *   音频提取（使用ffmpeg）
    *   Whisper语音识别集成
    *   OpenAI API翻译（支持自定义API端点）
    *   使用ffmpeg进行字幕烧录和视频转码
*   字幕样式自定义：
    *   字体、大小、颜色、描边、阴影等样式设置
    *   支持原始、翻译后或双语字幕
*   实时进度显示和状态更新
*   异步任务处理大文件
*   完整的错误处理和日志记录

## 已知问题/待办

*   大文件处理时可能需要优化内存使用
*   添加更多视频格式支持
*   实现字幕编辑功能
*   添加更多翻译语言支持

## 未来可能的增强功能

*   **性能优化**
    *   实现更高效的视频处理流程
    *   添加批处理功能
*   **功能增强**
    *   添加更多字幕样式选项（位置、边距、背景框等）
    *   支持更多视频格式和编解码器
    *   添加字幕编辑功能
    *   支持更多翻译语言和模型
*   **用户体验**
    *   提供更直观的用户界面
    *   添加更多交互式帮助和提示
    *   实现用户账户系统和历史记录功能
*   **部署与扩展**
    *   添加Docker支持
    *   实现水平扩展以支持更多并发用户
