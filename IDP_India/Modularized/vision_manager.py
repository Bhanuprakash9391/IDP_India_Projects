import streamlit as st
import base64
import time
import fitz  # PyMuPDF
import pickle
import re
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io

from openai import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import requests
from dotenv import load_dotenv

from html_manager import *
from chunk_manager import *
from llm_manager import *
from llm_manager import safe_llm_call
from excel_manager import *
from database_manager import *


VISION_CACHE_DIR = "vision_cache"


def ensure_directories():
    """Ensure necessary directories exist"""
    Path(VISION_CACHE_DIR).mkdir(exist_ok=True)

load_dotenv()

AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
}


def get_page_hash(page):
    """Create hash of page content for vision caching"""
    try:
        pix = page.get_pixmap(dpi=72)
        return hashlib.md5(pix.tobytes()).hexdigest()
    except:
        text = page.get_text()
        return hashlib.md5(text.encode()).hexdigest()

def load_vision_cache(page_hash: str) -> Optional[Dict]:
    """Load vision analysis from cache"""
    cache_file = os.path.join(VISION_CACHE_DIR, f"{page_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load vision cache: {e}")
    return None

def save_vision_cache(page_hash: str, result: Dict):
    """Save vision analysis to cache"""
    ensure_directories()
    cache_file = os.path.join(VISION_CACHE_DIR, f"{page_hash}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved vision cache: {cache_file}")
    except Exception as e:
        print(f"Failed to save vision cache: {e}")


def get_optimized_vision_analysis(client: AzureOpenAI, page: fitz.Page, page_num: int, file_name: str, use_cache: bool = True) -> dict:
    """Enhanced vision analysis with caching - with rate limit handling"""
    
    # Check cache first
    if use_cache:
        page_hash = get_page_hash(page)
        cached_result = load_vision_cache(page_hash)
        if cached_result:
            print(f"✓ Using cached vision result for {file_name} page {page_num}")
            cached_result['processing_method'] = 'cached'
            return cached_result
    
    print(f"🔍 Processing vision analysis for {file_name} page {page_num}...")
    
    try:
        # Increase DPI for better OCR results on scanned documents
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        vision_prompt = """
        Analyze this document page image and perform a comprehensive extraction:
        
        1. **TEXT EXTRACTION**: Extract ALL visible text from the image. This is critical.
           - Read every word, number, label, heading, and paragraph
           - Maintain the original structure and formatting as much as possible
           - If the document is scanned or photographed, perform OCR on all visible text
        
        2. **VISUAL ELEMENTS**: Describe any non-text elements:
           - Charts, graphs, diagrams (describe what they show)
           - Images or photographs (describe their content)
           - Logos or symbols
           - Tables or structured layouts
        
        3. **LAYOUT & STRUCTURE**: Describe how information is organized:
           - Document structure (sections, columns, etc.)
           - Formatting (bold, italic, underlined text if distinguishable)
           - Spatial relationships between elements
        
        BE THOROUGH AND COMPLETE. Extract every piece of text visible in the image.
        This may be a scanned document, so perform careful OCR on all text.
        """

        def vision_call():
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0
            )
            return response
        
        response = safe_llm_call(vision_call, initial_delay=3)
        
        basic_text = page.get_text().strip()
        enhanced_description = response.choices[0].message.content
        
        result = {
            "basic_text": basic_text,
            "enhanced_description": enhanced_description,
            "has_visuals": True,
            "processing_method": "vision"
        }
        
        # Save to cache
        if use_cache:
            page_hash = get_page_hash(page)
            save_vision_cache(page_hash, result)
            print(f"✓ Saved vision cache for {file_name} page {page_num}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in vision analysis for page {page_num}: {e}")
        return {
            "basic_text": page.get_text().strip(),
            "enhanced_description": "",
            "has_visuals": False,
            "processing_method": "error",
            "error": str(e)
        }

def should_use_vision_ai(page, file_name: str, page_num: int):
    """Decide whether to use vision AI - IMPROVED for scanned documents"""
    
    try:
        basic_text = page.get_text().strip()
        image_list = page.get_images()
        
        # Calculate text density (characters per page area)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
       
        # Check if there are substantial images (not just logos/icons)
        substantial_images = False
        if image_list:
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    from PIL import Image
                    import io
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    width, height = img_pil.size
                    
                    # Consider it substantial if larger than 200x200 pixels
                    if width > 200 and height > 200:
                        substantial_images = True
                        break
                except:
                    continue
        
        # IMPROVED: Multiple conditions for scanned documents
        is_scanned_page = (
            len(basic_text) < 100 or  # Very little text extracted
            (len(image_list) > 0 and len(basic_text) < 500)  # Has images and little text
        )
        
        # Check for poor quality text (lots of special characters, fragmented words)
        if len(basic_text) > 0:
            special_char_ratio = sum(1 for c in basic_text if not c.isalnum() and not c.isspace()) / len(basic_text)
            has_poor_quality_text = special_char_ratio > 0.3
        else:
            has_poor_quality_text = False
        
        conditions = {
            'has_substantial_images': substantial_images,
            'is_scanned_page': is_scanned_page,
            'has_poor_quality_text': has_poor_quality_text,
            
        }
        
        use_vision = any(conditions.values())
        
        return use_vision, {
            'conditions': conditions,
            'reasoning': [k for k, v in conditions.items() if v],
            'text_length': len(basic_text),
            
        }
        
    except Exception as e:
        print(f"Error in should_use_vision_ai: {e}")
        return False, {'error': str(e)}