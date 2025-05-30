"""
Test script for translation functionality.
"""
import os
import sys
import json
import argparse
from config import config
from openai import OpenAI

def test_translation(input_text, source_lang, target_lang, batch_size=5):
    """
    Test the translation functionality with the given input text.
    
    Args:
        input_text (str or list): Text to translate (single string or list of strings)
        source_lang (str): Source language code (e.g., 'en', 'zh-CN')
        target_lang (str): Target language code
        batch_size (int): Number of segments to process in one batch
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE
    )
    
    # Prepare input as a list
    if isinstance(input_text, str):
        input_text = [input_text]
    
    print(f"Testing translation from {source_lang} to {target_lang}")
    print(f"Batch size: {batch_size}")
    print(f"Number of segments: {len(input_text)}")
    print("-" * 50)
    
    # Process in batches
    for i in range(0, len(input_text), batch_size):
        batch = input_text[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(input_text) + batch_size - 1)//batch_size}")
        
        # Prepare the prompt
        system_prompt = {
            "role": "system",
            "content": f"""You are a professional translator. 
            Translate the following texts from {source_lang} to {target_lang}. 
            For each input text, output ONLY the translated text in the target language. 
            Keep the same order as the input. Separate translations with '---'. 
            Do not add any additional text or numbering."""
        }
        
        user_prompt = {
            "role": "user",
            "content": "\n---\n".join(batch)
        }
        
        print(f"Sending request with {len(batch)} segments...")
        
        try:
            # Make the API call
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[system_prompt, user_prompt],
                temperature=config.TRANSLATION_TEMPERATURE,
                max_tokens=config.TRANSLATION_MAX_TOKENS,
                timeout=config.TRANSLATION_TIMEOUT
            )
            
            # Process the response
            if not response or not response.choices:
                print("Error: Invalid or empty response from API")
                continue
                
            translated_texts = response.choices[0].message.content.strip().split('---')
            translated_texts = [t.strip() for t in translated_texts if t.strip()]
            
            # Display results
            print("\nTranslation Results:")
            for orig, trans in zip(batch, translated_texts):
                print(f"\nOriginal ({source_lang}): {orig}")
                print(f"Translated ({target_lang}): {trans}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Test translation functionality')
    parser.add_argument('--source', default='en', help='Source language code (default: en)')
    parser.add_argument('--target', default='zh-CN', help='Target language code (default: zh-CN)')
    parser.add_argument('--batch-size', type=int, default=config.TRANSLATION_BATCH_SIZE, 
                       help=f'Number of segments per batch (default: {config.TRANSLATION_BATCH_SIZE})')
    parser.add_argument('--text', help='Text to translate (enclose in quotes)')
    parser.add_argument('--file', help='File containing text to translate (one per line)')
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        input_text = [args.text]
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Default test text
        input_text = [
            "Hello, how are you?",
            "This is a test of the translation system.",
            "Please translate this text to the target language."
        ]
    
    # Run the test
    test_translation(
        input_text=input_text,
        source_lang=args.source,
        target_lang=args.target,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
