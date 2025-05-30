from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os
import subprocess
import time
import random
from werkzeug.utils import secure_filename
import whisper
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import re
import pysubs2
import uuid
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import config
from config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)

# 配置
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_FOLDER = config.UPLOAD_FOLDER
PROCESSED_FOLDER = config.PROCESSED_FOLDER
SUBTITLES_FOLDER = 'subtitles'
TEMP_AUDIO_FOLDER = 'temp_audio'
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'mov', 'avi', 'webm'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB 最大上传大小

# 应用配置
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-me'),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    PROCESSED_FOLDER=PROCESSED_FOLDER,
    SUBTITLES_FOLDER=SUBTITLES_FOLDER,
    TEMP_AUDIO_FOLDER=TEMP_AUDIO_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH
)

# 确保所有必要的目录都存在
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, SUBTITLES_FOLDER, TEMP_AUDIO_FOLDER]:
    folder_path = BASE_DIR / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'确保目录存在: {folder_path}')

# For managing background jobs and their statuses
executor = ThreadPoolExecutor(max_workers=2) # Adjust max_workers as needed
jobs = {} # Dictionary to store job status and results {job_id: {status: '...', result: '...'}}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# This function will contain the entire video processing logic
def process_video_pipeline(job_id, video_filepath, original_filename, form_data):
    """
    处理视频的完整流程：提取音频 -> 语音转文字 -> 翻译 -> 生成字幕 -> 嵌入视频
    """
    basename = os.path.splitext(original_filename)[0]
    output_filename = f"{basename}_subtitled.mp4"
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    # 初始化任务状态
    jobs[job_id] = {
        'status': 'Starting processing...',
        'progress': 0,
        'start_time': datetime.now().isoformat(),
        'result_video': None,
        'error': None
    }
    
    def update_status(status, progress=None):
        """更新任务状态"""
        if progress is not None:
            jobs[job_id]['progress'] = min(100, max(0, int(progress)))
        jobs[job_id]['status'] = status
        logger.info(f"Job {job_id}: {status} ({jobs[job_id]['progress']}%)")
    
    try:
        # 确保输出目录存在
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        os.makedirs(app.config['SUBTITLES_FOLDER'], exist_ok=True)
        os.makedirs(app.config['TEMP_AUDIO_FOLDER'], exist_ok=True)
        
        # 生成临时文件路径
        audio_filename = f"{basename}.aac"
        audio_filepath = os.path.join(app.config['TEMP_AUDIO_FOLDER'], audio_filename)
        
        # 确保字幕文件路径一致
        original_srt_filename = f"{basename}_original.srt"
        original_srt_filepath = os.path.abspath(os.path.join(app.config['SUBTITLES_FOLDER'], original_srt_filename))
        
        translated_srt_filename = f"{basename}_translated.srt"
        translated_srt_filepath = os.path.abspath(os.path.join(app.config['SUBTITLES_FOLDER'], translated_srt_filename))
        
        styled_ass_filepath = os.path.abspath(os.path.join(app.config['SUBTITLES_FOLDER'], f"{basename}_styled.ass"))
        
        # 记录文件路径
        logger.info(f"Original SRT path: {original_srt_filepath}")
        logger.info(f"Translated SRT path: {translated_srt_filepath}")
        logger.info(f"Styled ASS path: {styled_ass_filepath}")
        
        # 从表单获取参数
        target_language = form_data.get('target_language', '').strip()
        subtitle_type = form_data.get('subtitle_type', 'original_only')
        source_language = form_data.get('subtitle_source_lang', '').strip()
        video_quality = form_data.get('video_quality', '1080p')
        font_name = form_data.get('font_name', 'Arial')
        font_size = form_data.get('font_size', '24')
        primary_color = form_data.get('primary_color', '&H00FFFFFF')
        outline_color = form_data.get('outline_color', '&H00000000')
        outline_thickness = form_data.get('outline_thickness', '2')
        shadow_depth = form_data.get('shadow_depth', '1')
        whisper_model_size = form_data.get('whisper_model', 'base')
        
        # 从配置获取API密钥
        openai_api_key = config.OPENAI_API_KEY
        
        # 记录参数
        logger.info(f"Processing job {job_id} with params: {form_data}")

        # 1. 提取音频
        try:
            update_status('Extracting audio from video...', 10)
            
            # 构建ffmpeg命令
            ffmpeg_command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', video_filepath,  # 输入文件
                '-vn',  # 禁用视频流
                '-acodec', 'aac',  # 音频编码器
                '-b:a', '192k',  # 音频比特率
                '-ar', '16000',  # 采样率
                '-ac', '1',  # 单声道
                audio_filepath  # 输出文件
            ]
            
            logger.info(f"Running command: {' '.join(ffmpeg_command)}")
            result = subprocess.run(
                ffmpeg_command,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = f"Audio extraction failed: {result.stderr}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            update_status('Audio extracted successfully', 20)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Audio extraction failed: {e.stderr}"
            logger.error(error_msg)
            update_status(f'Error: {error_msg}', 0)
            jobs[job_id]['error'] = error_msg
            return
        except Exception as e:
            error_msg = f"Error during audio extraction: {str(e)}"
            logger.error(error_msg)
            update_status(f'Error: {error_msg}', 0)
            jobs[job_id]['error'] = error_msg
            return

        # 2. 语音转文字 (Whisper)
        try:
            update_status('Loading Whisper model...', 25)
            
            # 加载Whisper模型
            logger.info(f"Loading Whisper model: {whisper_model_size}")
            model = whisper.load_model(whisper_model_size)
            
            # 设置转录选项
            transcribe_options = {}
            if source_language:
                transcribe_options['language'] = source_language
                update_status(f'Transcribing in {source_language}...', 30)
            else:
                update_status('Detecting language and transcribing...', 30)
            
            # 执行转录
            logger.info("Starting transcription...")
            logger.info(f"Audio file exists: {os.path.exists(audio_filepath)}")
            logger.info(f"Audio file size: {os.path.getsize(audio_filepath) if os.path.exists(audio_filepath) else 0} bytes")
            
            try:
                whisper_result = model.transcribe(audio_filepath, **transcribe_options)
                logger.info("Transcription completed successfully")
                
                # 检查转录结果
                if not whisper_result or 'segments' not in whisper_result or not whisper_result['segments']:
                    raise ValueError("No transcription segments found in Whisper result")
                
                # 保存原始字幕文件 (SRT格式)
                logger.info(f"Saving transcription to {original_srt_filepath}")
                segments = whisper_result.get('segments', [])
                logger.info(f"Number of segments: {len(segments)}")
                
                # 确保字幕目录存在
                os.makedirs(os.path.dirname(original_srt_filepath), exist_ok=True)
                
                with open(original_srt_filepath, "w", encoding="utf-8") as srt_file:
                    for i, segment in enumerate(segments):
                        start_time = segment.get('start', 0)
                        end_time = segment.get('end', 0)
                        text = segment.get('text', '').strip()
                        
                        if not text:  # 跳过空文本段
                            logger.warning(f"Empty text in segment {i+1}, skipping")
                            continue
                            
                        srt_entry = f"{i+1}\n{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}\n{text}\n\n"
                        srt_file.write(srt_entry)
                
                # 验证SRT文件是否成功创建
                if not os.path.exists(original_srt_filepath):
                    raise IOError(f"Failed to create SRT file at {original_srt_filepath}")
                if os.path.getsize(original_srt_filepath) == 0:
                    raise ValueError("Created SRT file is empty")
                
                detected_language = whisper_result.get('language', 'unknown')
                update_status(f'Transcription completed in {detected_language}', 50)
                logger.info(f"Transcription completed. Detected language: {detected_language}")
                logger.info(f"SRT file created successfully at {original_srt_filepath}")
                
            except Exception as e:
                logger.error(f"Error during transcription or SRT creation: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            error_msg = f"Whisper transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            update_status(f'Error: {error_msg}', 0)
            jobs[job_id]['error'] = error_msg
            return

        # 3. 翻译字幕 (如果启用了翻译)
        if target_language and subtitle_type in ['translated_only', 'bilingual']:
            if not openai_api_key:
                logger.warning("OpenAI API key not found in configuration. Skipping translation.")
                translated_srt_filepath = None
            else:
                logger.info(f"Starting translation to {target_language}")
            try:
                update_status('Translating subtitles...', 60)
                
                # 解析原始SRT文件
                parsed_srt = parse_srt(original_srt_filepath)
                if not parsed_srt:
                    raise ValueError("Original SRT is empty or failed to parse")
                
                # 设置OpenAI API配置
                try:
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=config.OPENAI_API_BASE
                    )
                    logger.info(f"Using OpenAI API base URL: {config.OPENAI_API_BASE}")
                    logger.info(f"Using model: {config.OPENAI_MODEL}")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                    raise

                
                # 获取语言信息
                detected_lang_code = whisper_result.get('language', 'en')
                source_lang_name = get_language_name(detected_lang_code) or detected_lang_code
                target_lang_name = get_language_name(target_language) or target_language
                
                # 如果源语言和目标语言相同，跳过翻译
                if source_lang_name.lower() == target_lang_name.lower():
                    update_status('Source and target languages are the same, skipping translation', 70)
                    translated_srt_filepath = original_srt_filepath
                else:
                    # 批量翻译函数
                    def batch_translate(entries, batch_size=None, max_retries=None):
                        # 使用配置中的默认值
                        batch_size = batch_size or config.TRANSLATION_BATCH_SIZE
                        max_retries = max_retries or config.TRANSLATION_MAX_RETRIES
                        translated_entries = []
                        total_batches = (len(entries) + batch_size - 1) // batch_size
                        
                        for batch_idx in range(0, len(entries), batch_size):
                            batch = entries[batch_idx:batch_idx + batch_size]
                            batch_texts = [entry['text'] for entry in batch]
                            
                            # 构建批量提示
                            system_prompt = {
                                "role": "system",
                                "content": f"""You are a professional translator. 
                                Translate the following texts from {source_lang_name} to {target_lang_name}. 
                                For each input text, output ONLY the translated text in the target language. 
                                Keep the same order as the input. Separate translations with '---'. Do not add any additional text or numbering."""
                            }
                            
                            user_prompt = {
                                "role": "user",
                                "content": "\n---\n".join(batch_texts)
                            }
                            
                            # 重试逻辑
                            for attempt in range(max_retries):
                                try:
                                    # 指数退避
                                    if attempt > 0:
                                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                                        logger.warning(f"Attempt {attempt + 1} after {wait_time:.2f}s delay...")
                                        time.sleep(wait_time)
                                    
                                    # 发送批量请求
                                    response = client.chat.completions.create(
                                        model=config.OPENAI_MODEL,
                                        messages=[system_prompt, user_prompt],
                                        temperature=0.3,
                                        max_tokens=2000,
                                        timeout=45
                                    )
                                    
                                    # 处理响应
                                    if not response or not response.choices:
                                        raise ValueError("Invalid or empty response from OpenAI API")
                                    
                                    # 分割翻译结果
                                    translated_texts = response.choices[0].message.content.strip().split('---')
                                    translated_texts = [t.strip() for t in translated_texts if t.strip()]
                                    
                                    # 验证返回的翻译数量是否匹配
                                    if len(translated_texts) != len(batch):
                                        raise ValueError(f"Mismatched number of translations. Expected {len(batch)}, got {len(translated_texts)}")
                                    
                                    # 保存翻译结果
                                    for entry, translated_text in zip(batch, translated_texts):
                                        translated_entries.append({
                                            'index': entry['index'],
                                            'start_time_str': entry['start_time_str'],
                                            'end_time_str': entry['end_time_str'],
                                            'text': translated_text
                                        })
                                    
                                    # 更新进度
                                    current_batch = (batch_idx // batch_size) + 1
                                    progress = 60 + int(30 * (current_batch / total_batches))
                                    update_status(f'Translated batch {current_batch}/{total_batches}...', progress)
                                    logger.info(f"Successfully translated batch {current_batch}/{total_batches}")
                                    break  # 成功，退出重试循环
                                    
                                except openai.error.RateLimitError as e:
                                    if attempt == max_retries - 1:  # 最后一次重试也失败
                                        logger.error(f"Rate limit reached after {max_retries} attempts")
                                        raise
                                    continue
                                except Exception as e:
                                    if attempt == max_retries - 1:  # 最后一次重试也失败
                                        logger.error(f"Translation failed after {max_retries} attempts: {str(e)}")
                                        raise
                                    continue
                        
                        return translated_entries
                    
                    # 执行批量翻译
                    try:
                        update_status('Starting translation...', 60)
                        logger.info(f"Starting batch translation of {len(parsed_srt)} segments")
                        
                        # 分批处理
                        translated_entries = batch_translate(
                            parsed_srt, 
                            batch_size=config.TRANSLATION_BATCH_SIZE,
                            max_retries=config.TRANSLATION_MAX_RETRIES
                        )
                        
                        if len(translated_entries) != len(parsed_srt):
                            raise ValueError(f"Mismatch in number of translated entries. Expected {len(parsed_srt)}, got {len(translated_entries)}")
                        
                        # 确保目录存在并保存翻译后的SRT文件
                        os.makedirs(os.path.dirname(translated_srt_filepath), exist_ok=True)
                        with open(translated_srt_filepath, 'w', encoding='utf-8') as f:
                            for entry in translated_entries:
                                f.write(f"{entry['index']}\n")
                                f.write(f"{entry['start_time_str']} --> {entry['end_time_str']}\n")
                                f.write(f"{entry['text']}\n\n")
                        
                        update_status('Translation completed', 75)
                        logger.info(f"Translation completed. Saved to {translated_srt_filepath}")
                        
                    except Exception as e:
                        error_msg = f"Translation failed: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        update_status('Warning: Translation failed, using original subtitles', 70)
                        translated_srt_filepath = None
            
            except Exception as e:
                error_msg = f"Translation failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                update_status('Warning: Translation failed, using original subtitles', 70)
                translated_srt_filepath = None
                # 非致命错误，继续处理

        jobs[job_id]['status'] = 'Preparing final subtitles...'
        jobs[job_id]['progress'] = 70
        # 4. Determine/Merge Subtitles
        final_srt_for_embedding = None
        bilingual_srt_filepath = None
        
        # Debug logging
        logger.info(f"Determining final SRT for embedding. Subtitle type: {subtitle_type}")
        logger.info(f"Original SRT exists: {os.path.exists(original_srt_filepath) if original_srt_filepath else 'N/A'}")
        logger.info(f"Translated SRT exists: {os.path.exists(translated_srt_filepath) if translated_srt_filepath else 'N/A'}")
        
        if subtitle_type == "original_only":
            logger.info("Using original subtitles only")
            if os.path.exists(original_srt_filepath):
                final_srt_for_embedding = original_srt_filepath
                logger.info(f"Selected original SRT: {original_srt_filepath}")
            else:
                logger.warning(f"Original SRT not found at: {original_srt_filepath}")
                
        elif subtitle_type == "translated_only":
            logger.info("Using translated subtitles only")
            if translated_srt_filepath and os.path.exists(translated_srt_filepath):
                final_srt_for_embedding = translated_srt_filepath
                logger.info(f"Selected translated SRT: {translated_srt_filepath}")
            else:
                logger.warning(f"Translated SRT not found or not available")
                
        elif subtitle_type == "bilingual":
            logger.info("Attempting to create bilingual subtitles")
            if os.path.exists(original_srt_filepath) and translated_srt_filepath and os.path.exists(translated_srt_filepath):
                bilingual_srt_filename = f"{basename}_bilingual_{target_language}.srt"
                bilingual_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], bilingual_srt_filename)
                logger.info(f"Will create bilingual SRT at: {bilingual_srt_filepath}")
                try:
                    # Parse both SRT files
                    logger.info(f"Parsing original SRT: {original_srt_filepath}")
                    parsed_orig = parse_srt(original_srt_filepath)
                    logger.info(f"Parsed {len(parsed_orig)} entries from original SRT")
                    
                    logger.info(f"Parsing translated SRT: {translated_srt_filepath}")
                    parsed_trans = parse_srt(translated_srt_filepath)
                    logger.info(f"Parsed {len(parsed_trans)} entries from translated SRT")
                    
                    # Log any length mismatch
                    if len(parsed_orig) != len(parsed_trans): 
                        logger.warning(f"Bilingual SRTs differ in length. Original: {len(parsed_orig)}, Translated: {len(parsed_trans)}")
                    
                    # Create bilingual SRT
                    logger.info(f"Creating bilingual SRT at {bilingual_srt_filepath}")
                    with open(bilingual_srt_filepath, "w", encoding="utf-8") as bs_file:
                        num_entries = min(len(parsed_orig), len(parsed_trans))
                        logger.info(f"Merging {num_entries} entries into bilingual SRT")
                        
                        for i in range(num_entries):
                            try:
                                # Format: original text (newline) translated text
                                combined_text = f"{parsed_orig[i]['text']}\n{parsed_trans[i]['text']}"
                                bs_file.write(f"{i+1}\n{parsed_orig[i]['start_time_str']} --> {parsed_orig[i]['end_time_str']}\n{combined_text}\n\n")
                            except Exception as e:
                                logger.error(f"Error merging entry {i}: {e}", exc_info=True)
                                continue
                    
                    # Verify the file was created and has content
                    if os.path.exists(bilingual_srt_filepath) and os.path.getsize(bilingual_srt_filepath) > 0:
                        logger.info(f"Successfully created bilingual SRT with {num_entries} entries")
                        final_srt_for_embedding = bilingual_srt_filepath
                    else:
                        raise Exception("Failed to create bilingual SRT file or file is empty")
                        
                except Exception as e:
                    logger.error(f"Bilingual SRT creation failed: {str(e)}", exc_info=True)
                    logger.warning("Falling back to original subtitles due to bilingual creation failure")
        
        # Fallback logic if preferred type failed
        if not final_srt_for_embedding:
            logger.warning("Preferred subtitle type not available, falling back to alternatives")
            if os.path.exists(original_srt_filepath):
                final_srt_for_embedding = original_srt_filepath
                logger.info(f"Falling back to original SRT: {original_srt_filepath}")
            else:
                error_msg = f"No suitable SRT file available for embedding. Original SRT exists: {os.path.exists(original_srt_filepath) if original_srt_filepath else False}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        logger.info(f"Final SRT selected for embedding: {final_srt_for_embedding}")

        # Create styled ASS from the chosen SRT
        try:
            subs = pysubs2.load(final_srt_for_embedding, encoding="utf-8")
            if 'Default' not in subs.styles: subs.styles['Default'] = pysubs2.SSAStyle()
            style = subs.styles['Default']
            style.fontname = font_name
            style.fontsize = float(font_size)
            # Convert hex color to BGR format that pysubs2 expects (e.g., '&HBBGGRR&')
            if primary_color.startswith('#'):
                hex_color = primary_color.lstrip('#')
                bgr_color = f'&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}'
                style.primarycolor = bgr_color
            if outline_color.startswith('#'):
                hex_color = outline_color.lstrip('#')
                bgr_color = f'&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}'
                style.outlinecolor = bgr_color
            style.outline = float(outline_thickness)
            style.shadow = float(shadow_depth)
            style.borderstyle = 1
            for line in subs: line.style = 'Default'
            subs.save(styled_ass_filepath, encoding="utf-8")
        except Exception as e:
            raise Exception(f"Subtitle styling failed: {e}")

        jobs[job_id]['status'] = 'Embedding subtitles into video...'
        jobs[job_id]['progress'] = 85
        # 5. Embed Subtitles (ffmpeg)
        processed_video_filename = f"{basename}_subtitled.mp4"
        processed_video_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_video_filename)
        try:
            # 设置视频质量参数
            video_quality = form_data.get('video_quality', '1080p')
            crf_values = {
                '1080p': '20',
                '720p': '22',
                '480p': '25',
                'source': '18'  # 更高质量，文件更大
            }
            crf = crf_values.get(video_quality, '23')  # 默认值
            
            # 构建ffmpeg命令
            ffmpeg_cmd_embed = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', video_filepath,
                '-vf', f"ass='{os.path.abspath(styled_ass_filepath)}'",
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', crf,
                '-c:a', 'copy',  # 保留原始音频流
                '-movflags', '+faststart'  # 优化网络播放
            ]
            
            # 如果是源质量，使用更高质量的设置
            if video_quality == 'source':
                ffmpeg_cmd_embed.extend([
                    '-pix_fmt', 'yuv420p',
                    '-profile:v', 'high',
                    '-level', '4.2',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-y'
                ])
            # 添加输出文件路径
            ffmpeg_cmd_embed.append(processed_video_filepath)
            
            # 执行ffmpeg命令
            try:
                logger.info(f"Executing ffmpeg command: {' '.join(ffmpeg_cmd_embed)}")
                result = subprocess.run(
                    ffmpeg_cmd_embed,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.debug(f"FFmpeg output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"FFmpeg warnings: {result.stderr}")
                    
                # 更新任务状态
                jobs[job_id].update({
                    'status': 'Processing completed',
                    'progress': 100,
                    'result_video': processed_video_filename,
                    'download_url': f'/downloads/{processed_video_filename}'
                })
                
                logger.info(f"Video processing completed: {processed_video_filepath}")
                
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg error: {e.stderr}"
                logger.error(error_msg)
                jobs[job_id].update({
                    'status': f'Error: {error_msg}',
                    'progress': 0,
                    'error': error_msg
                })
                return
        except subprocess.CalledProcessError as e:
            raise Exception(f"Subtitle embedding failed: {e.stderr}")

        # 6. 清理临时文件
        try:
            update_status('Cleaning up temporary files...', 95)
            temp_files = [
                audio_filepath,
                original_srt_filepath,
                styled_ass_filepath
            ]
            
            if 'translated_srt_filepath' in locals() and translated_srt_filepath:
                temp_files.append(translated_srt_filepath)
            if 'bilingual_srt_filepath' in locals() and bilingual_srt_filepath:
                temp_files.append(bilingual_srt_filepath)
            
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file_path}: {e}")
        except Exception as e:
            # Non-fatal for translation, can proceed without it
            jobs[job_id]['status'] = f"Translation failed: {e}. Proceeding without translation."
            # We won't raise here, but translated_srt_filepath might not exist or be incomplete.
            # The logic for choosing final_srt_for_embedding needs to handle this.
            print(f"Translation error (non-fatal): {e}")
            # Ensure translated_srt_filepath is None if it failed to create properly
            if 'translated_srt_filepath' in locals() and translated_srt_filepath and (not os.path.exists(translated_srt_filepath) or os.path.getsize(translated_srt_filepath) == 0):
                translated_srt_filepath = None 
    
        jobs[job_id]['status'] = 'Preparing final subtitles...'
        jobs[job_id]['progress'] = 70
        # 4. Determine/Merge Subtitles
        final_srt_for_embedding = None
        bilingual_srt_filepath = None
        # (Logic from previous step to choose/create final_srt_for_embedding based on subtitle_type,
        #  original_srt_filepath, and translated_srt_filepath - this needs to be robust to translation failure)
        if subtitle_type == "original_only":
            if os.path.exists(original_srt_filepath): 
                final_srt_for_embedding = original_srt_filepath
        elif subtitle_type == "translated_only":
            if translated_srt_filepath and os.path.exists(translated_srt_filepath): 
                final_srt_for_embedding = translated_srt_filepath
        elif subtitle_type == "bilingual":
            if os.path.exists(original_srt_filepath) and translated_srt_filepath and os.path.exists(translated_srt_filepath):
                bilingual_srt_filename = f"{basename}_bilingual_{target_language}.srt"
                bilingual_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], bilingual_srt_filename)
                try:
                    parsed_orig = parse_srt(original_srt_filepath)
                    parsed_trans = parse_srt(translated_srt_filepath)
                    if len(parsed_orig) != len(parsed_trans): 
                        print("Warning: Bilingual SRTs differ in length.")
                    with open(bilingual_srt_filepath, "w", encoding="utf-8") as bs_file:
                        num_entries = min(len(parsed_orig), len(parsed_trans))
                        for i in range(num_entries):
                            # Get original and translated text, ensure they're not None
                            orig_text = parsed_orig[i]['text'].strip() if parsed_orig[i]['text'] else ''
                            trans_text = parsed_trans[i]['text'].strip() if i < len(parsed_trans) and parsed_trans[i]['text'] else ''
                            
                            # Only include non-empty lines
                            if orig_text and trans_text:
                                combined_text = f"{orig_text}\n{trans_text}"
                            elif orig_text:
                                combined_text = orig_text
                            elif trans_text:
                                combined_text = trans_text
                            else:
                                combined_text = ""
                                
                            if combined_text:  # Only write non-empty entries
                                bs_file.write(f"{i+1}\n{parsed_orig[i]['start_time_str']} --> {parsed_orig[i]['end_time_str']}\n{combined_text}\n\n")
                    final_srt_for_embedding = bilingual_srt_filepath
                except Exception as e:
                    print(f"Bilingual creation failed: {e}") # Non-fatal, might fallback
        
        # 记录字幕文件状态
        def log_file_status(filepath, desc):
            exists = os.path.exists(filepath) if filepath else False
            size = os.path.getsize(filepath) if exists and filepath else 0
            logger.info(f"{desc}: {exists} (size: {size} bytes) - {filepath}")
            return exists
            
        logger.info("\n=== Subtitle Files Status ===")
        orig_exists = log_file_status(original_srt_filepath, "Original SRT")
        trans_exists = log_file_status(translated_srt_filepath, "Translated SRT") if translated_srt_filepath else False
        bili_exists = log_file_status(bilingual_srt_filepath, "Bilingual SRT") if bilingual_srt_filepath else False
        logger.info("============================\n")
            
        if not final_srt_for_embedding: # Fallback if preferred type failed
            logger.warning("Preferred subtitle type not available, falling back to original SRT")
            if os.path.exists(original_srt_filepath): 
                final_srt_for_embedding = original_srt_filepath
                logger.info(f"Using original SRT: {original_srt_filepath}")
            else: 
                error_msg = f"No suitable SRT file available for embedding. Original SRT exists: {os.path.exists(original_srt_filepath)}"
                logger.error(error_msg)
                raise Exception(error_msg)

        logger.info(f"Using subtitle file for embedding: {final_srt_for_embedding}")
        
        # Create styled ASS from the chosen SRT
        try:
            subs = pysubs2.load(final_srt_for_embedding, encoding="utf-8")
            if 'Default' not in subs.styles: 
                subs.styles['Default'] = pysubs2.SSAStyle()
            style = pysubs2.SSAStyle()
            style.fontname = font_family
            style.fontsize = font_size
            # Convert hex color to BGR format that pysubs2 expects (e.g., '&HBBGGRR&')
            if primary_color.startswith('#'):
                hex_color = primary_color.lstrip('#')
                bgr_color = f'&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}&'
                style.primarycolor = bgr_color
            style.outlinecolor = pysubs2.Color.from_ass_string(outline_color)
            style.outline = float(outline_thickness)
            style.shadow = float(shadow_depth)
            style.borderstyle = 1
            for line in subs: 
                line.style = 'Default'
            subs.save(styled_ass_filepath, encoding="utf-8")
            
            jobs[job_id]['status'] = 'Embedding subtitles into video...'
            jobs[job_id]['progress'] = 85
            
            # 5. Embed Subtitles (ffmpeg)
            processed_video_filename = f"{basename}_subtitled.mp4"
            processed_video_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_video_filename)
            
            try:
                # 设置视频质量参数
                video_quality = form_data.get('video_quality', '1080p')
                crf_values = {
                    '1080p': '20',
                    '720p': '22',
                    '480p': '25',
                    'source': '18'  # 更高质量，文件更大
                }
                crf = crf_values.get(video_quality, '23')  # 默认值
                
                # 构建ffmpeg命令
                ffmpeg_cmd_embed = [
                    'ffmpeg',
                    '-y',  # 覆盖输出文件
                    '-i', video_filepath,
                    '-vf', f"ass='{os.path.abspath(styled_ass_filepath)}'",
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', crf,
                    '-c:a', 'copy',  # 保留原始音频流
                    '-movflags', '+faststart',  # 优化网络播放
                    '-hide_banner',
                    '-loglevel', 'error'
                ]
                
                # 如果是源质量，使用更高质量的设置
                if video_quality == 'source':
                    ffmpeg_cmd_embed.extend([
                        '-pix_fmt', 'yuv420p',
                        '-profile:v', 'high',
                        '-level', '4.2',
                        '-c:a', 'aac',
                        '-b:a', '192k'
                    ])
                
                # 添加输出文件路径
                ffmpeg_cmd_embed.append(processed_video_filepath)
                
                # 执行ffmpeg命令
                try:
                    logger.info(f"Executing ffmpeg command: {' '.join(ffmpeg_cmd_embed)}")
                    result = subprocess.run(
                        ffmpeg_cmd_embed,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.debug(f"FFmpeg output: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"FFmpeg warnings: {result.stderr}")
                        
                    # 更新任务状态
                    jobs[job_id].update({
                        'status': 'Processing completed',
                        'progress': 100,
                        'result_video': processed_video_filename,
                        'download_url': f'/downloads/{processed_video_filename}'
                    })
                    
                    logger.info(f"Video processing completed: {processed_video_filepath}")
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"FFmpeg error: {e.stderr}"
                    logger.error(error_msg)
                    jobs[job_id].update({
                        'status': f'Error: {error_msg}',
                        'progress': 0,
                        'error': error_msg
                    })
                    
                    # 6. 清理临时文件
                    try:
                        update_status('Cleaning up temporary files...', 95)
                        temp_files = [
                            audio_filepath,
                            original_srt_filepath,
                            styled_ass_filepath,
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp.mp4"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio.wav"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono.wav"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k.wav"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.srt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.ass"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.vtt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.json"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.txt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.srt.json"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.ass.json"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.vtt.json"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.txt.json"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.srt.txt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.ass.txt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.vtt.txt"),
                            os.path.join(app.config['UPLOAD_FOLDER'], f"{basename}_temp_audio_mono_16k_trimmed.wav.txt.txt"),
                        ]
                        
                        # 添加翻译后的字幕文件到临时文件列表
                        if 'translated_srt_filepath' in locals() and translated_srt_filepath:
                            temp_files.append(translated_srt_filepath)
                        
                        # 添加双语字幕文件到临时文件列表
                        if 'bilingual_srt_filepath' in locals() and bilingual_srt_filepath:
                            temp_files.append(bilingual_srt_filepath)
                        
                        # 删除临时文件
                        for file_path in temp_files:
                            try:
                                if file_path and os.path.exists(file_path):
                                    os.remove(file_path)
                                    logger.debug(f"Deleted temporary file: {file_path}")
                            except Exception as e:
                                logger.warning(f"Failed to delete temporary file {file_path}: {e}")
                        
                        logger.info("Temporary files cleaned up")
                        
                        # 更新任务状态
                        jobs[job_id].update({
                            'status': 'Processing completed',
                            'progress': 100,
                            'result_video': processed_video_filename,
                            'download_url': f'/downloads/{processed_video_filename}'
                        })
                        
                        logger.info(f"Job {job_id} completed successfully")
                        
                    except Exception as e:
                        error_msg = f"Cleanup failed: {e}"
                        logger.error(error_msg)
                        # 非致命错误，记录但继续
                        if 'error' not in jobs[job_id]:
                            jobs[job_id]['warning'] = error_msg
                    
                    # 最终状态更新
                    jobs[job_id].update({
                        'status': 'Done',
                        'progress': 100,
                        'completion_time': datetime.now().isoformat(),
                        'processing_time': (datetime.now() - datetime.fromisoformat(jobs[job_id]['start_time'])).total_seconds()
                    })
                    
                    print(f"Job {job_id} completed successfully. Result: {processed_video_filename}")
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"FFmpeg error: {e.stderr}"
                    logger.error(error_msg)
                    jobs[job_id].update({
                        'status': f'Error: {error_msg}',
                        'progress': 0,
                        'error': error_msg,
                        'completion_time': datetime.now().isoformat()
                    })
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in video processing pipeline: {error_msg}", exc_info=True)
                    jobs[job_id].update({
                        'status': f'Error: {error_msg}',
                        'progress': 0,
                        'error': error_msg,
                        'completion_time': datetime.now().isoformat()
                    })
            except Exception as e:
                error_msg = f"Error in video processing: {str(e)}"
                logger.error(error_msg, exc_info=True)
                jobs[job_id].update({
                    'status': f'Error: {error_msg}',
                    'progress': 0,
                    'error': error_msg,
                    'completion_time': datetime.now().isoformat()
                })
            finally:
                # Ensure we always clean up
                if 'temp_files' in locals():
                    for file_path in temp_files:
                        try:
                            if file_path and os.path.exists(file_path):
                                os.remove(file_path)
                                logger.debug(f"Cleaned up temporary file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up file {file_path}: {e}")
                
                # Update job completion time
                if job_id in jobs:
                    jobs[job_id]['completion_time'] = datetime.now().isoformat()
                    jobs[job_id]['processing_time'] = (
                        datetime.now() - 
                        datetime.fromisoformat(jobs[job_id]['start_time'])
                    ).total_seconds()

        except Exception as e:
            error_msg = f"Unexpected error in video processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if job_id in jobs:
                jobs[job_id].update({
                    'status': f'Error: {error_msg}',
                    'progress': 0,
                    'error': error_msg,
                    'completion_time': datetime.now().isoformat()
                })
        
        return jobs[job_id]
    except Exception as e:
        error_msg = f"Unexpected error in process_video_pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if job_id in jobs:
            jobs[job_id].update({
                'status': f'Error: {error_msg}',
                'progress': 0,
                'error': error_msg,
                'completion_time': datetime.now().isoformat()
            })
        return jobs[job_id]

# End of process_video_pipeline function

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                return jsonify({'error': 'No video file part'}), 400
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No selected video file'}), 400
            
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                video_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                file.save(video_filepath)
                job_id = str(uuid.uuid4())
                
                # 将任务添加到队列
                jobs[job_id] = {
                    'status': 'Queued',
                    'progress': 0,
                    'start_time': datetime.now().isoformat(),
                    'filename': original_filename,
                    'result_video': None
                }
                
                # 提交任务到线程池
                form_data = request.form.to_dict()  # Get form data
                executor.submit(process_video_pipeline, job_id, video_filepath, original_filename, form_data)
                
                return jsonify({
                    'job_id': job_id,
                    'status': 'Queued',
                    'message': '任务已加入处理队列'
                })
                
        except Exception as e:
            logger.error(f'处理上传时出错: {str(e)}', exc_info=True)
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500
    
    # GET 请求，渲染上传表单
    return render_template('index.html')

# End of process_video_pipeline function

@app.route('/status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id, {'status': 'Job not found'})
    response = jsonify(job)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/downloads/<filename>')
def download_file(filename):
    try:
        # Use absolute path for the directory
        processed_dir = os.path.join(BASE_DIR, app.config['PROCESSED_FOLDER'])
        return send_from_directory(
            processed_dir,
            filename,
            as_attachment=True,
            download_name=filename,
            as_attachment_filename=filename  # For compatibility with older Flask versions
        )
    except FileNotFoundError:
        return jsonify({'error': f'File not found: {filename}'}), 404

# Helper function for SRT timestamps
def format_timestamp_srt(seconds):
    assert seconds >= 0, "non-negative timestamp"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Helper function to parse SRT files
def parse_srt(srt_filepath):
    entries = []
    with open(srt_filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Regex to capture SRT blocks: index, timestamp, and text lines
    # This regex handles multi-line text for a single subtitle entry.
    pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n?)+)", re.MULTILINE)
    
    for match in pattern.finditer(content):
        index = match.group(1)
        start_time_str = match.group(2)
        end_time_str = match.group(3)
        text = match.group(4).strip()
        entries.append({
            'index': index,
            'start_time_str': start_time_str,
            'end_time_str': end_time_str,
            'text': text
        })
    return entries

# Helper to get full language names for prompts (can be expanded)
LANGUAGE_CODES = {
    "en": "English",
    "zh-CN": "Simplified Chinese",
    "zh": "Chinese", # Whisper might output 'zh'
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    # Add more as needed from your HTML select and Whisper outputs
}

def get_language_name(code):
    return LANGUAGE_CODES.get(code, code) # Return code itself if name not found

# 添加错误处理中间件
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 1GB.'}), 413

if __name__ == '__main__':
    # 检查ffmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("未找到ffmpeg。请确保已安装ffmpeg并添加到系统PATH中。")
        print("错误: 未找到ffmpeg。请确保已安装ffmpeg并添加到系统PATH中。")
        exit(1)
    
    # 检查Whisper模型是否可用
    try:
        import whisper
        whisper.load_model('base')
    except Exception as e:
        logger.error(f"Whisper模型加载失败: {str(e)}")
        print(f"警告: Whisper模型加载失败: {str(e)}")
        print("请确保已正确安装Whisper及其依赖项。")
    
    # 启动应用
    app.run(host='0.0.0.0', port=5001, debug=True)
