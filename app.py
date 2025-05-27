from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os
import subprocess
from werkzeug.utils import secure_filename
import whisper
import openai
import re
import pysubs2
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
SUBTITLES_FOLDER = 'subtitles'
TEMP_AUDIO_FOLDER = 'temp_audio'
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'mov', 'avi', 'webm'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB 最大上传大小

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-me')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SUBTITLES_FOLDER'] = SUBTITLES_FOLDER
app.config['TEMP_AUDIO_FOLDER'] = TEMP_AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

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
        original_srt_filename = f"{basename}_original.srt"
        original_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], original_srt_filename)
        translated_srt_filename = f"{basename}_translated.srt"
        translated_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], translated_srt_filename)
        styled_ass_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], f"{basename}_styled.ass")
        
        # 从表单获取参数
        target_language = form_data.get('target_language', '').strip()
        openai_api_key = form_data.get('openai_api_key', '').strip()
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
            whisper_result = model.transcribe(audio_filepath, **transcribe_options)
            
            # 保存原始字幕文件 (SRT格式)
            with open(original_srt_filepath, "w", encoding="utf-8") as srt_file:
                for i, segment in enumerate(whisper_result.get('segments', [])):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    if not text:  # 跳过空文本段
                        continue
                        
                    srt_file.write(f"{i+1}\n")
                    srt_file.write(f"{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}\n")
                    srt_file.write(f"{text}\n\n")
            
            detected_language = whisper_result.get('language', 'unknown')
            update_status(f'Transcription completed in {detected_language}', 50)
            logger.info(f"Transcription completed. Detected language: {detected_language}")
            
        except Exception as e:
            error_msg = f"Whisper transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            update_status(f'Error: {error_msg}', 0)
            jobs[job_id]['error'] = error_msg
            return

        # 3. 翻译字幕 (如果启用了翻译)
        if openai_api_key and target_language:
            try:
                update_status('Translating subtitles...', 60)
                
                # 解析原始SRT文件
                parsed_srt = parse_srt(original_srt_filepath)
                if not parsed_srt:
                    raise ValueError("Original SRT is empty or failed to parse")
                
                # 设置OpenAI API密钥
                openai.api_key = openai_api_key
                
                # 获取语言信息
                detected_lang_code = whisper_result.get('language', 'en')
                source_lang = source_language or detected_lang_code
                source_lang_name = get_language_name(source_lang) or source_lang
                target_lang_name = get_language_name(target_language) or target_language
                
                # 如果源语言和目标语言相同，跳过翻译
                if source_lang_name.lower() == target_lang_name.lower():
                    update_status('Source and target languages are the same, skipping translation', 70)
                    translated_srt_filepath = original_srt_filepath
                else:
                    # 逐段翻译
                    translated_entries = []
                    total_segments = len(parsed_srt)
                    
                    for i, entry in enumerate(parsed_srt):
                        try:
                            # 更新进度
                            progress = 60 + int(10 * (i / total_segments))
                            update_status(f'Translating segment {i+1}/{total_segments}...', progress)
                            
                            # 调用OpenAI API进行翻译
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": f"You are a professional translator. Translate the following text from {source_lang_name} to {target_lang_name}. Only output the translated text, nothing else."
                                    },
                                    {
                                        "role": "user",
                                        "content": entry['text']
                                    }
                                ],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            
                            translated_text = response.choices[0].message['content'].strip()
                            translated_entries.append({
                                'index': entry['index'],
                                'start_time_str': entry['start_time_str'],
                                'end_time_str': entry['end_time_str'],
                                'text': translated_text
                            })
                            
                        except Exception as e:
                            logger.error(f"Error translating segment {i+1}: {str(e)}")
                            # 如果翻译失败，使用原始文本
                            translated_entries.append(entry)
                    
                    # 保存翻译后的字幕文件
                    if translated_entries:
                        with open(translated_srt_filepath, 'w', encoding='utf-8') as f:
                            for entry in translated_entries:
                                f.write(f"{entry['index']}\n")
                                f.write(f"{entry['start_time_str']} --> {entry['end_time_str']}\n")
                                f.write(f"{entry['text']}\n\n")
                        update_status('Translation completed', 75)
                        logger.info(f"Translation completed. Saved to {translated_srt_filepath}")
                    else:
                        logger.warning("No translations were generated")
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
        # (Logic from previous step to choose/create final_srt_for_embedding based on subtitle_type,
        #  original_srt_filepath, and translated_srt_filepath - this needs to be robust to translation failure)
        if subtitle_type == "original_only":
            if os.path.exists(original_srt_filepath): final_srt_for_embedding = original_srt_filepath
        elif subtitle_type == "translated_only":
            if translated_srt_filepath and os.path.exists(translated_srt_filepath): final_srt_for_embedding = translated_srt_filepath
        elif subtitle_type == "bilingual":
            if os.path.exists(original_srt_filepath) and translated_srt_filepath and os.path.exists(translated_srt_filepath):
                bilingual_srt_filename = f"{basename}_bilingual_{target_language}.srt"
                bilingual_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], bilingual_srt_filename)
                try:
                    parsed_orig = parse_srt(original_srt_filepath)
                    parsed_trans = parse_srt(translated_srt_filepath)
                    if len(parsed_orig) != len(parsed_trans): print("Warning: Bilingual SRTs differ in length.")
                    with open(bilingual_srt_filepath, "w", encoding="utf-8") as bs_file:
                        num_entries = min(len(parsed_orig), len(parsed_trans))
                        for i in range(num_entries):
                            bs_file.write(f"{parsed_orig[i]['index']}\n{parsed_orig[i]['start_time_str']} --> {parsed_orig[i]['end_time_str']}\n{parsed_orig[i]['text']}\n{parsed_trans[i]['text']}\n\n")
                    final_srt_for_embedding = bilingual_srt_filepath
                except Exception as e:
                    print(f"Bilingual creation failed: {e}") # Non-fatal, might fallback
        
        if not final_srt_for_embedding: # Fallback if preferred type failed
            if os.path.exists(original_srt_filepath): final_srt_for_embedding = original_srt_filepath
            else: raise Exception("No suitable SRT file available for embedding.")

        # Create styled ASS from the chosen SRT
        try:
            subs = pysubs2.load(final_srt_for_embedding, encoding="utf-8")
            if 'Default' not in subs.styles: subs.styles['Default'] = pysubs2.SSAStyle()
            style = subs.styles['Default']
            style.fontname = font_name; style.fontsize = float(font_size)
            style.primarycolor = pysubs2.Color.from_ass_string(primary_color)
            style.outlinecolor = pysubs2.Color.from_ass_string(outline_color)
            style.outline = float(outline_thickness); style.shadow = float(shadow_depth)
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
                    '-level', '4.2'
                ])
                '-c:a', 'aac', '-b:a', '192k', '-y', processed_video_filepath
            ]
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
        if not os.path.exists(translated_srt_filepath) or os.path.getsize(translated_srt_filepath) == 0:
            translated_srt_filepath = None 

jobs[job_id]['status'] = 'Preparing final subtitles...'
jobs[job_id]['progress'] = 70
# 4. Determine/Merge Subtitles
final_srt_for_embedding = None
bilingual_srt_filepath = None
# (Logic from previous step to choose/create final_srt_for_embedding based on subtitle_type,
#  original_srt_filepath, and translated_srt_filepath - this needs to be robust to translation failure)
if subtitle_type == "original_only":
    if os.path.exists(original_srt_filepath): final_srt_for_embedding = original_srt_filepath
elif subtitle_type == "translated_only":
    if translated_srt_filepath and os.path.exists(translated_srt_filepath): final_srt_for_embedding = translated_srt_filepath
elif subtitle_type == "bilingual":
    if os.path.exists(original_srt_filepath) and translated_srt_filepath and os.path.exists(translated_srt_filepath):
        bilingual_srt_filename = f"{basename}_bilingual_{target_language}.srt"
        bilingual_srt_filepath = os.path.join(app.config['SUBTITLES_FOLDER'], bilingual_srt_filename)
        try:
            parsed_orig = parse_srt(original_srt_filepath)
            parsed_trans = parse_srt(translated_srt_filepath)
            if len(parsed_orig) != len(parsed_trans): print("Warning: Bilingual SRTs differ in length.")
            with open(bilingual_srt_filepath, "w", encoding="utf-8") as bs_file:
                num_entries = min(len(parsed_orig), len(parsed_trans))
                for i in range(num_entries):
                    bs_file.write(f"{parsed_orig[i]['index']}\n{parsed_orig[i]['start_time_str']} --> {parsed_orig[i]['end_time_str']}\n{parsed_orig[i]['text']}\n{parsed_trans[i]['text']}\n\n")
            final_srt_for_embedding = bilingual_srt_filepath
        except Exception as e:
            print(f"Bilingual creation failed: {e}") # Non-fatal, might fallback
        
if not final_srt_for_embedding: # Fallback if preferred type failed
    if os.path.exists(original_srt_filepath): final_srt_for_embedding = original_srt_filepath
    else: raise Exception("No suitable SRT file available for embedding.")

# Create styled ASS from the chosen SRT
try:
    subs = pysubs2.load(final_srt_for_embedding, encoding="utf-8")
    if 'Default' not in subs.styles: subs.styles['Default'] = pysubs2.SSAStyle()
    style = subs.styles['Default']
    style.fontname = font_name; style.fontsize = float(font_size)
    style.primarycolor = pysubs2.Color.from_ass_string(primary_color)
    style.outlinecolor = pysubs2.Color.from_ass_string(outline_color)
    style.outline = float(outline_thickness); style.shadow = float(shadow_depth)
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
            '-level', '4.2'
        ])
        '-c:a', 'aac', '-b:a', '192k', '-y', processed_video_filepath
    ]
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
            error_msg = f"Cleanup failed: {e}"
            logger.error(error_msg)
            # 非致命错误，记录但继续
            if 'error' not in jobs[job_id]:
                jobs[job_id]['warning'] = error_msg
    
    update_status('Processing completed successfully', 100)
    logger.info(f"Job {job_id} completed successfully")
    if os.path.exists(audio_filepath): os.remove(audio_filepath)
    if os.path.exists(styled_ass_filepath): os.remove(styled_ass_filepath)
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

except Exception as e:
    error_msg = str(e)
    logger.error(f"Error in video processing pipeline: {error_msg}", exc_info=True)
    jobs[job_id].update({
        'status': f'Error: {error_msg}',
        'progress': 0,
        'error': error_msg,
        'completion_time': datetime.now().isoformat()
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({'error': 'No video file part'}), 400
        file = request.files['video_file']
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
            executor.submit(process_video_pipeline, job_id, filepath, original_filename, form_data)
            
            return jsonify({
                'job_id': job_id,
                'status': 'Queued',
                'message': '任务已加入处理队列'
            })
            
        except Exception as e:
            logger.error(f'处理上传时出错: {str(e)}')
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500
    
    # GET 请求，渲染上传表单
    return render_template('index.html')

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
        return send_from_directory(
            app.config['PROCESSED_FOLDER'], 
            filename,
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

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
    app.run(host='0.0.0.0', port=5000, debug=True)
