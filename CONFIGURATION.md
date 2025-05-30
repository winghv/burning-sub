# 配置说明

## 环境变量配置

1. 复制 `.env.example` 文件并重命名为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，设置您的 OpenAI API 密钥和其他配置：
   ```
   # OpenAI API 配置
   OPENAI_API_KEY=your_openai_api_key_here
   
   # 可选：自定义 API 基础 URL（如果使用 Azure OpenAI 或其他兼容 API）
   # OPENAI_API_BASE=https://api.openai.com/v1/
   # OPENAI_MODEL=gpt-3.5-turbo
   
   # 翻译设置
   TRANSLATION_BATCH_SIZE=5
   TRANSLATION_MAX_RETRIES=3
   TRANSLATION_TIMEOUT=45
   TRANSLATION_TEMPERATURE=0.3
   TRANSLATION_MAX_TOKENS=2000
   
   # 文件路径
   UPLOAD_FOLDER=uploads
   PROCESSED_FOLDER=processed
   ```

## 翻译功能测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行翻译测试

1. 测试单条翻译：
   ```bash
   python test_translation.py --source en --target zh-CN --text "Hello, how are you?"
   ```

2. 测试文件翻译（每行一个句子）：
   ```bash
   python test_translation.py --source en --target zh-CN --file input.txt
   ```

3. 自定义批处理大小：
   ```bash
   python test_translation.py --source en --target zh-CN --file input.txt --batch-size 3
   ```

### 测试参数说明

- `--source`: 源语言代码（默认：en）
- `--target`: 目标语言代码（默认：zh-CN）
- `--batch-size`: 每批处理的句子数（默认：5）
- `--text`: 要翻译的文本（用引号括起来）
- `--file`: 包含要翻译的文本的文件路径（每行一个句子）

## 在应用中使用翻译功能

翻译功能已集成到主应用中，您可以通过以下方式使用：

1. 确保已正确配置 `.env` 文件
2. 在应用中上传视频并选择翻译选项
3. 系统会自动使用配置的 API 密钥和参数进行翻译

## 故障排除

1. **API 密钥无效**：检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确
2. **连接超时**：尝试增加 `TRANSLATION_TIMEOUT` 值
3. **批量大小问题**：如果遇到 API 限制，请减小 `TRANSLATION_BATCH_SIZE`
4. **查看日志**：应用日志会显示详细的错误信息，帮助诊断问题
