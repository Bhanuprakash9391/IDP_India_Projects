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


from vision_manager import *
from excel_manager import *
from chunk_manager import *
from database_manager import *


AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
}



def classify_document_type_large(client: AzureOpenAI, text_content: str) -> Dict:
    """
    Classify document type - handles large documents by processing in chunks
    """
    # Extract heading (usually found in the first few lines)
    lines = text_content.strip().splitlines()
    heading_candidates = [line.strip() for line in lines[:10] if line.strip()]
    heading_text = " ".join(heading_candidates[:3])
    
    # For classification, use first 15000 chars (usually enough to determine type)
    classification_text = text_content[:15000]
    
    classification_prompt = f"""
    You are an expert document analyst.
    Your task is to identify the type of document based on both its heading and content.

    Heading (often reveals the type directly):
    "{heading_text}"

    Document Text:
    {classification_text}

    Analyze both the heading and overall content structure to determine:
    1. What type of document this is (e.g., Invoice, Report, Agreement, Certificate, Resume, Purchase Order, Contract, Medical Record, etc.)
    2. Your confidence level (0.0 to 1.0)
    3. Which words, phrases, or heading patterns led you to your conclusion.

    Provide your response strictly in JSON format:
    {{
        "document_type": "string",
        "confidence": float,
        "reasoning": "brief explanation referencing heading and content"
    }}
    """
    
    def classify_call():
        response = client.chat.completions.create(
            model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=500,
            temperature=0.0
        )
        return response
    
    try:
        response = safe_llm_call(classify_call)
        result_text = response.choices[0].message.content
        
        # Extract JSON safely
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            classification = json.loads(json_match.group())
            return classification
        else:
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "reasoning": "Failed to parse JSON classification"
            }
    
    except Exception as e:
        print(f"❌ Error in document classification: {e}")
        return {
            "document_type": "Unknown",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def extract_structured_data_large(client: AzureOpenAI, text_content: str, document_type: str) -> Dict:
    """
    Extract structured data from large documents by processing in chunks and merging
    """
    chunks = chunk_text_for_processing(text_content, max_chars=12000, overlap=500)
    
    print(f"📄 Extracting structured data from {len(chunks)} chunks...")
    
    all_extracted_data = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        
        extraction_prompt = f"""
        You are analyzing a {document_type} document. Your task is to extract meaningful structured data from this section.

        This is part {i+1} of {len(chunks)} from the document.

        Read the text carefully and identify key information that should be extracted as structured data. Extract ONLY information that is actually present and meaningful.

        Guidelines:
        - Extract factual information as key-value pairs
        - Use clear, descriptive keys
        - Group related information logically
        - For lists or multiple items, use arrays
        - For nested information, use nested objects
        - Do NOT make up or infer information that isn't explicitly stated
        - Focus on meaningful data points like dates, names, amounts, IDs, addresses, etc.
        
        Document Text Section:
        {chunk}
        
        Return a clean JSON object with the extracted data. Include only meaningful fields found in THIS section.
        
        Example structure (adapt based on actual content):
        {{
            "key_field": "value",
            "another_field": "value",
            "nested_info": {{
                "sub_field": "value"
            }},
            "list_items": ["item1", "item2"]
        }}
        """
        
        def extract_call():
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=4000,
                temperature=0.0
            )
            return response
        
        try:
            response = safe_llm_call(extract_call)
            result_text = response.choices[0].message.content
            
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                chunk_data = json.loads(json_match.group())
                all_extracted_data.append(chunk_data)
            else:
                print(f"  ⚠️ Failed to parse JSON from chunk {i+1}")
        
        except Exception as e:
            print(f"  ⚠️ Error extracting from chunk {i+1}: {e}")
            continue
    
    # Merge all extracted data intelligently
    if not all_extracted_data:
        return {"error": "No data could be extracted"}
    
    # If only one chunk, return it directly
    if len(all_extracted_data) == 1:
        return all_extracted_data[0]
    
    # Merge multiple chunks
    print("  Merging extracted data from all chunks...")
    merged_data = {}
    
    for chunk_data in all_extracted_data:
        for key, value in chunk_data.items():
            if key not in merged_data:
                merged_data[key] = value
            else:
                # Merge logic for duplicate keys
                if isinstance(value, list) and isinstance(merged_data[key], list):
                    # Merge lists, avoid duplicates
                    for item in value:
                        if item not in merged_data[key]:
                            merged_data[key].append(item)
                elif isinstance(value, dict) and isinstance(merged_data[key], dict):
                    # Merge dictionaries
                    merged_data[key].update(value)
                # For other types, keep the first occurrence
    
    return merged_data




def safe_llm_call(func, max_retries=5, initial_delay=2):
    """
    Wrapper for LLM calls with exponential backoff on rate limits
    Keeps retrying until successful, no matter how long it takes
    """
    attempt = 0
    delay = initial_delay
    
    while True:
        try:
            attempt += 1
            result = func()
            # Add a small delay between successful calls to avoid rate limits
            time.sleep(0.5)
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                print(f"⚠️ Rate limit hit on attempt {attempt}. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                
                # Exponential backoff: double the delay each time, max 60 seconds
                delay = min(delay * 2, 60)
                continue
            
            # Check if it's a timeout or connection error
            elif "timeout" in error_str or "connection" in error_str:
                print(f"⚠️ Connection error on attempt {attempt}. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay = min(delay * 1.5, 30)
                continue
            
            # For other errors, retry a few times then raise
            elif attempt < max_retries:
                print(f"⚠️ Error on attempt {attempt}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 1.5, 30)
                continue
            else:
                print(f"❌ Failed after {attempt} attempts: {e}")
                raise


def extract_table_with_llm_safe(client: AzureOpenAI, table_text: str) -> Dict:
    """Use LLM to structure table data - with rate limit handling"""
    
    table_prompt = f"""
    Convert this table text into structured JSON format.
    Identify the column headers and organize the data into rows.
    
    Table text:
    {table_text}
    
    Return JSON format:
    {{
        "headers": ["column1", "column2", ...],
        "rows": [
            {{"column1": "value", "column2": "value"}},
            ...
        ],
        "summary": "brief description of what this table contains"
    }}
    """
    
    def table_call():
        response = client.chat.completions.create(
            model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            messages=[{"role": "user", "content": table_prompt}],
            max_tokens=4096,
            temperature=0.0
        )
        return response
    
    try:
        response = safe_llm_call(table_call)
        result_text = response.choices[0].message.content
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Failed to parse table", "raw_text": table_text[:500]}
    
    except Exception as e:
        return {"error": str(e), "raw_text": table_text[:500]}
