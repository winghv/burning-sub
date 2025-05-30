"""
Configuration settings for the application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1/')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Translation Settings
    TRANSLATION_BATCH_SIZE = int(os.getenv('TRANSLATION_BATCH_SIZE', '5'))
    TRANSLATION_MAX_RETRIES = int(os.getenv('TRANSLATION_MAX_RETRIES', '3'))
    TRANSLATION_TIMEOUT = int(os.getenv('TRANSLATION_TIMEOUT', '45'))
    TRANSLATION_TEMPERATURE = float(os.getenv('TRANSLATION_TEMPERATURE', '0.3'))
    TRANSLATION_MAX_TOKENS = int(os.getenv('TRANSLATION_MAX_TOKENS', '2000'))
    
    # File Paths
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed')

# Create instance of config
config = Config()
