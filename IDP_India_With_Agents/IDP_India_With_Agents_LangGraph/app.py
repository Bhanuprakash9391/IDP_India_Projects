import streamlit as st
import base64
import time
import fitz  # PyMuPDF
import re
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import logging
from PIL import Image
import io

from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from dotenv import load_dotenv

# Import Q&A Agent
from qna_agent_langgraph import integrate_qna_agent_langgraph_ui


st.set_page_config(
    page_title="Intelligent Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Updated CSS Styling
# CSS Styling
st.markdown("""
<style>
/* Import Segoe UI font family */
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

/* Global styles */
* {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* Main container */
.main {
    background-color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* CRITICAL: Ensure sidebar collapse/expand button is always visible and functional */
section[data-testid="stSidebar"] > div:first-child {
    position: relative;
}

/* Make sure the collapse button container is visible */
button[kind="header"] {
    display: block !important;
    visibility: visible !important;
}

/* Style the sidebar collapse/expand button */
section[data-testid="stSidebar"] button[kind="header"] {
    background-color: transparent;
    border: none;
    color: #6B2190;
}

section[data-testid="stSidebar"] button[kind="header"]:hover {
    background-color: rgba(107, 33, 144, 0.1);
}

/* Ensure collapsed control button is visible */
.css-1dp5vir {
    display: block !important;
    visibility: visible !important;
}

/* Header styling */
.header-container {
    background-color: #FFFFFF;
    padding: 1rem 0;
    margin-bottom: 2rem;
    text-align: center;
}

.main-title {
    color: #6B2190;
    font-size: 2rem;
    font-weight: 600;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 0;
}

/* Section titles */
.section-title {
    font-weight: 600;
    color: #2c3e50;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #6B2190;
    font-size: 1.1rem;
}

/* Classification box */
.classification-box {
    background: #F8F9FA;
    border-left: 4px solid #6B2190;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
    font-family: 'Segoe UI', sans-serif;
}


/* Buttons */
.stButton>button {
    background-color: #6B2190;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 2rem;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
    transition: background-color 0.3s;
}

.stButton>button:hover {
    background-color: #551A70;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: #FFFFFF;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 500;
    color: #2c3e50;
    background-color: #F8F9FA;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background-color: #6B2190;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F8F9FA;
    font-family: 'Segoe UI', sans-serif;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Segoe UI', sans-serif;
}

/* Input fields */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    font-family: 'Segoe UI', sans-serif;
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}

/* Select box */
.stSelectbox>div>div {
    font-family: 'Segoe UI', sans-serif;
}

/* File uploader */
.stFileUploader>div {
    font-family: 'Segoe UI', sans-serif;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
    color: #2c3e50;
    background-color: #F8F9FA;
    border-radius: 4px;
}

/* Chat messages */
.stChatMessage {
    font-family: 'Segoe UI', sans-serif;
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Chat input */
.stChatInput>div>div>textarea {
    font-family: 'Segoe UI', sans-serif;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.5rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Info, success, warning, error boxes */
.stAlert {
    font-family: 'Segoe UI', sans-serif;
    border-radius: 4px;
}

/* Spinner */
.stSpinner > div {
    font-family: 'Segoe UI', sans-serif;
}

/* Dataframe */
.dataframe {
    font-family: 'Segoe UI', sans-serif;
}

/* JSON display */
pre {
    font-family: 'Consolas', 'Monaco', monospace;
    background-color: #F8F9FA;
    border-radius: 4px;
    padding: 1rem;
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

load_dotenv()

AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
}

# Directory structure
VECTOR_DB_DIR = "vector_databases"
VISION_CACHE_DIR = "vision_cache"
STRUCTURED_DATA_DIR = "structured_data"

def ensure_directories():
    """Ensure necessary directories exist"""
    Path(VECTOR_DB_DIR).mkdir(exist_ok=True)
    Path(VISION_CACHE_DIR).mkdir(exist_ok=True)
    Path(STRUCTURED_DATA_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def validate_azure_config():
    """Validate Azure configuration"""
    required_fields = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
    ]
    
    for field in required_fields:
        if not AZURE_CONFIG.get(field):
            st.error(f"Missing Azure configuration: {field}")
            return False
    return True

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


def chunk_text_for_processing(text: str, max_chars: int = 12000, overlap: int = 500) -> List[str]:
    """
    Split large text into chunks for processing, with overlap to maintain context
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # If not the last chunk, try to break at a paragraph or sentence
        if end < len(text):
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start + max_chars // 2:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break > start + max_chars // 2:
                    end = sentence_break + 1
        
        chunks.append(text[start:end])
        start = end - overlap  # Overlap to maintain
    return chunks

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


def process_documents_with_intelligence(uploaded_files: list, client: AzureOpenAI):
    """
    Main processing function with intelligent extraction
    Handles documents of ANY SIZE with rate limit management
    """
    ensure_directories()
    log_filename = f"logs/processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')
    
    all_documents = {
        'text_documents': [],
        'image_documents': [],
        'per_file_data': {}
    }
    
    processing_stats = {
        "total_pages": 0,
        "vision_pages": 0,
        "text_pages": 0,
        "documents_classified": 0,
        "structured_extractions": 0,
        "cached_pages": 0,
        "total_characters": 0
    }
    
    for file_idx, file in enumerate(uploaded_files):
        try:
            file_name = file.name
            print(f"\n{'='*80}")
            print(f"📄 Processing file {file_idx + 1}/{len(uploaded_files)}: {file_name}")
            print(f"{'='*80}")
            logging.info(f"Processing: {file_name}")
            
            file_data = {
                'classification': {},
                'structured_data': {}
            }
            
            if file.type == "application/pdf":
                file_content = file.read()
                file.seek(0)
                doc = fitz.open(stream=file_content, filetype="pdf")
                
                total_pages = len(doc)
                print(f"📖 Document has {total_pages} pages")
                
                # STEP 1: First pass - collect all text/vision content
                print(f"\n🔄 STEP 1: Collecting content from all pages...")
                full_text_for_classification = ""
                page_contents = []
                
                for i, page in enumerate(doc):
                    page_num = i + 1
                    processing_stats["total_pages"] += 1
                    
                    print(f"\n  📄 Processing page {page_num}/{total_pages}...")
                    
                    # Decide if we need vision AI for this page
                    use_vision, analysis_info = should_use_vision_ai(page, file_name, page_num)
                    
                    page_text = ""
                    if use_vision:
                        reasons = ", ".join(analysis_info.get('reasoning', []))
                        print(f"    🔍 Using Vision AI: {reasons}")
                        logging.info(f"  Page {page_num}: Using vision AI - {reasons}")
                        
                        vision_result = get_optimized_vision_analysis(client, page, page_num, file_name)
                        
                        if vision_result['processing_method'] == 'cached':
                            processing_stats["cached_pages"] += 1
                            print(f"    ✓ Used cached result")
                        else:
                            processing_stats["vision_pages"] += 1
                            print(f"    ✓ Vision analysis complete")
                        
                        # Combine basic text and vision-enhanced description
                        page_text = vision_result['basic_text'] if vision_result['basic_text'] else ""
                        if vision_result['enhanced_description']:
                            page_text += "\n" + vision_result['enhanced_description']
                        
                        page_contents.append({
                            'page_num': page_num,
                            'text': page_text,
                            'vision_result': vision_result,
                            'use_vision': True
                        })
                    else:
                        processing_stats["text_pages"] += 1
                        page_text = page.get_text().strip()
                        print(f"    ✓ Text extraction complete ({len(page_text)} chars)")
                        
                        page_contents.append({
                            'page_num': page_num,
                            'text': page_text,
                            'vision_result': None,
                            'use_vision': False
                        })
                    
                    full_text_for_classification += page_text + "\n"
                    processing_stats["total_characters"] += len(page_text)
                
                total_chars = len(full_text_for_classification)
                print(f"\n✓ Content collection complete: {total_chars:,} characters")
                
                # STEP 2: Classify document
                print(f"\n🔄 STEP 2: Classifying document...")
                classification = classify_document_type_large(client, full_text_for_classification)
                file_data['classification'] = classification
                processing_stats["documents_classified"] += 1
                
                print(f"✓ Classification: {classification['document_type']} (confidence: {classification['confidence']:.1%})")
                logging.info(f"  Classified as: {classification['document_type']} (confidence: {classification['confidence']})")
                
                # STEP 3: Extract structured data
                print(f"\n🔄 STEP 3: Extracting structured data...")
                structured_data = extract_structured_data_large(
                    client, 
                    full_text_for_classification,
                    classification['document_type']
                )
                file_data['structured_data'] = structured_data
                processing_stats["structured_extractions"] += 1
                print(f"✓ Structured data extraction complete")
                
                # STEP 4: Process each page for document creation
                print(f"\n🔄 STEP 4: Creating document embeddings...")
                
                for page_idx, page_data in enumerate(page_contents):
                    page_num = page_data['page_num']
                    page_text = page_data['text']
                    page = doc[page_num - 1]
                    
                    if page_idx % 10 == 0:
                        print(f"  Processing pages {page_idx + 1}-{min(page_idx + 10, total_pages)}...")
                    
                    page_info = {
                        'page_num': page_num,
                        'processing_method': 'vision' if page_data['use_vision'] else 'text'
                    }
                    
                    # Add to appropriate document store
                    if page_data['use_vision']:
                        vision_result = page_data['vision_result']
                        if vision_result.get('has_visuals') or vision_result.get('enhanced_description'):
                            image_doc = Document(
                                page_content=vision_result.get('enhanced_description', '') if vision_result.get('enhanced_description') else page_text,
                                metadata={
                                    "source": file_name,
                                    "page": page_num,
                                    "type": "image",
                                    "document_type": classification['document_type']
                                }
                            )
                            all_documents['image_documents'].append(image_doc)
                    
                    # Always add text content to text documents
                    if page_text:
                        text_doc = Document(
                            page_content=page_text,
                            metadata={
                                "source": file_name,
                                "page": page_num,
                                "type": "text",
                                "document_type": classification['document_type']
                            }
                        )
                        all_documents['text_documents'].append(text_doc)
                    
                
                print(f"✓ Document processing complete for {file_name}")
                doc.close()
                
            all_documents['per_file_data'][file_name] = file_data
                
        except Exception as e:
            logging.error(f"Error processing {file.name}: {e}")
            print(f"❌ Error processing {file.name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            print(traceback.format_exc())
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Statistics:")
    print(f"  Total pages: {processing_stats['total_pages']}")
    print(f"  Vision processed: {processing_stats['vision_pages']}")
    print(f"  Cached pages: {processing_stats['cached_pages']}")
    print(f"  Text pages: {processing_stats['text_pages']}")
    print(f"  Documents classified: {processing_stats['documents_classified']}")
    print(f"  Total characters: {processing_stats['total_characters']:,}")
    print(f"{'='*80}\n")
    
    logging.info(f"Processing complete: {processing_stats}")
    return all_documents, processing_stats

def create_multiple_vectorstores(all_documents: Dict, db_name: str):
    """Create separate vectorstores for text, tables, and images"""
    ensure_directories()
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
        azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
        api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
    )
    
    vectorstores = {}
    
    # Create text vectorstore
    if all_documents['text_documents']:
        st.write(f"Creating text vectorstore ({len(all_documents['text_documents'])} documents)...")
        vectorstores['text'] = FAISS.from_documents(all_documents['text_documents'], embeddings)
        vectorstores['text'].save_local(os.path.join(VECTOR_DB_DIR, f"{db_name}_text"))
    
    
    # Create image vectorstore
    if all_documents['image_documents']:
        st.write(f"Creating image vectorstore ({len(all_documents['image_documents'])} documents)...")
        vectorstores['image'] = FAISS.from_documents(all_documents['image_documents'], embeddings)
        vectorstores['image'].save_local(os.path.join(VECTOR_DB_DIR, f"{db_name}_image"))
    
    # Save per-file data
    per_file_path = os.path.join(STRUCTURED_DATA_DIR, f"{db_name}_files.json")
    with open(per_file_path, 'w') as f:
        json.dump(all_documents['per_file_data'], f, indent=2)
    
    return vectorstores

def load_multiple_vectorstores(db_name: str, embeddings):
    """Load all related vectorstores and structured data"""
    vectorstores = {}
    
    # Load text vectorstore
    text_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_text")
    if os.path.exists(text_path):
        vectorstores['text'] = FAISS.load_local(text_path, embeddings, allow_dangerous_deserialization=True)
    
    
    # Load image vectorstore
    image_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_image")
    if os.path.exists(image_path):
        vectorstores['image'] = FAISS.load_local(image_path, embeddings, allow_dangerous_deserialization=True)
    
    # Load per-file data
    per_file_path = os.path.join(STRUCTURED_DATA_DIR, f"{db_name}_files.json")
    per_file_data = {}
    if os.path.exists(per_file_path):
        with open(per_file_path, 'r') as f:
            per_file_data = json.load(f)
    
    return vectorstores, per_file_data

def create_conversation_chain(vectorstores: Dict):
    """Create conversation chain that searches across all vectorstores"""
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
        azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
        api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
        temperature=0.0
    )
    
    # Use primary retriever
    primary_retriever = None
    if 'text' in vectorstores:
        primary_retriever = vectorstores['text'].as_retriever(search_kwargs={"k": 5})
    elif 'image' in vectorstores:
        primary_retriever = vectorstores['image'].as_retriever(search_kwargs={"k": 5})
    
    if primary_retriever:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=primary_retriever,
            memory=memory,
            return_source_documents=True
        )
    
    return None

def display_pdf_page(file_bytes, page_num: int):
    """Display a specific page from PDF"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, width="stretch")
        doc.close()
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")




def display_header():
    """Display professional header with logo"""
    logo_path = "./logo.png"  # Update this path
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    header_col1, header_col2, header_col3 = st.columns([1.5, 6, 1])
    
    with header_col1:
        try:
            st.image(logo_path, width=600, output_format="PNG")
        except Exception as e:
            st.markdown("""
            <div style="font-family: 'Segoe UI', sans-serif; font-size: 1.5rem; 
                        font-weight: bold; color: #6B2190; padding: 1rem;">
                AGENT
            </div>
            """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
            <h1 style="font-size: 2rem; margin: 0; 
                    font-family: 'Segoe UI', sans-serif; font-weight: 700; 
                    line-height: 1.2; text-align: center;">
                <span style="color: #C9A961;">Intelligent</span><span style="color: #000000;">Document</span><span style="color: #000000;">Processor</span>
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col3:
        st.empty()
    
    st.markdown("""
    <hr style='margin: 3rem 0 2rem 0; border: none; border-top: 2px solid #E8E8E8;'>
    """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    if not validate_azure_config():
        st.error("Please configure Azure OpenAI settings before using the application.")
        st.stop()
    
    ensure_directories()
    display_header()
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = {}
    if "per_file_data" not in st.session_state:
        st.session_state.per_file_data = {}
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    if "current_db_name" not in st.session_state:
        st.session_state.current_db_name = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = {}
    if "task_execution_stats" not in st.session_state:
        st.session_state.task_execution_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "recent_tasks": []
        }
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-title">Document Upload</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files or images",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        db_name = st.text_input(
            "Database Name:",
            value=f"intelligent_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if st.button("Process Documents"):
            if uploaded_files and db_name:
                with st.spinner("Analyzing documents with AI..."):
                    try:
                        uploaded_files_data = {file.name: file.read() for file in uploaded_files}
                        for file in uploaded_files:
                            file.seek(0)
                        st.session_state.uploaded_file_bytes = uploaded_files_data
                        
                        client = AzureOpenAI(
                            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                        )
                        
                        all_documents, stats = process_documents_with_intelligence(uploaded_files, client)
                        
                        st.session_state.processing_stats = stats
                        st.session_state.per_file_data = all_documents['per_file_data']
                        
                        vectorstores = create_multiple_vectorstores(all_documents, db_name)
                        st.session_state.vectorstores = vectorstores
                        
                        conversation = create_conversation_chain(vectorstores)
                        if conversation:
                            st.session_state.conversation = conversation
                            st.session_state.current_db_name = db_name
                            
                            st.success("Documents processed successfully!")
                            st.info(f"Created {len(vectorstores)} specialized databases")
                        else:
                            st.error("Failed to create conversation chain")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Please upload files and provide a database name")
        
        st.markdown("---")
        
        # Load existing database
        st.markdown('<div class="section-title">Load Existing Database</div>', unsafe_allow_html=True)
        
        if os.path.exists(VECTOR_DB_DIR):
            db_files = [f.replace("_text", "").replace("_table", "").replace("_image", "") 
                       for f in os.listdir(VECTOR_DB_DIR) if "_text" in f or "_table" in f or "_image" in f]
            db_files = list(set(db_files))
            
            if db_files:
                selected_db = st.selectbox("Select database:", ["None"] + db_files)
                
                if st.button("Load Database") and selected_db != "None":
                    with st.spinner("Loading database..."):
                        try:
                            embeddings = AzureOpenAIEmbeddings(
                                azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
                                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                            )
                            
                            vectorstores, per_file_data = load_multiple_vectorstores(selected_db, embeddings)
                            
                            if vectorstores:
                                st.session_state.vectorstores = vectorstores
                                st.session_state.per_file_data = per_file_data
                                st.session_state.current_db_name = selected_db
                                
                                conversation = create_conversation_chain(vectorstores)
                                if conversation:
                                    st.session_state.conversation = conversation
                                    st.success(f"Loaded: {selected_db}")
                                    st.info(f"Loaded {len(vectorstores)} databases")
                                else:
                                    st.error("Failed to create conversation chain")
                            else:
                                st.error("No vectorstores found")
                                
                        except Exception as e:
                            st.error(f"Error loading database: {e}")
            else:
                st.info("No saved databases found")
        
        st.markdown("---")
        
        # Action Buttons Section
        st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)
        
        # Clear Chat History button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if 'response_sources' in st.session_state:
                st.session_state.response_sources = []
            if 'qna_conversation' in st.session_state:
                st.session_state.qna_conversation = []
            st.success("Chat history cleared")
            st.rerun()
        
        # View Chat History expander
        if st.session_state.chat_history:
            with st.expander("View Chat History", expanded=False):
                for i, message in enumerate(st.session_state.chat_history):
                    role = "user" if message.type == 'human' else "assistant"
                    with st.chat_message(role):
                        st.markdown(message.content)
                        
                        if role == "assistant" and i // 2 < len(st.session_state.get('response_sources', [])):
                            sources = st.session_state.response_sources[i // 2]
                            if sources:
                                with st.expander(f"Sources ({len(sources)} documents)"):
                                    for idx, doc in enumerate(sources):
                                        st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')} "
                                                  f"(Page {doc.metadata.get('page', '?')}) - "
                                                  f"Type: {doc.metadata.get('type', 'unknown')}")
                                        st.text(doc.page_content[:200] + "...")
                                        st.markdown("---")
        
        
            # Database Actions
            if st.session_state.per_file_data:
                st.markdown("---")
                st.markdown('<div class="section-title">Database Actions</div>', unsafe_allow_html=True)
                
                # Stack buttons vertically instead of side by side
                if st.button("View Database Tables"):
                    try:
                        from database_agent import DatabaseAgent
                        db_agent = DatabaseAgent()
                        tables = db_agent.get_table_info()
                        
                        if tables:
                            st.success(f"Found {len(tables)} tables in database")
                            for table in tables:
                                with st.expander(f"Table: {table['table_name']}"):
                                    st.write(f"Description: {table['description']}")
                                    st.write(f"Created: {table['created_at']}")
                                    
                                    # Show sample data
                                    sample_data = db_agent.get_table_data(table['table_name'], limit=5)
                                    if sample_data:
                                        st.write("Sample Data:")
                                        st.json(sample_data)
                        else:
                            st.info("No document tables found in database")
                    except Exception as e:
                        st.error(f"Error viewing database tables: {e}")
                
                if st.button("Clear Database"):
                    try:
                        from database_agent import DatabaseAgent
                        db_agent = DatabaseAgent()
                        result = db_agent.clear_database()
                        st.success(result)
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
        
        st.markdown("---")
        
        # Task Execution Statistics
        st.markdown('<div class="section-title">Task Execution Statistics</div>', unsafe_allow_html=True)
        task_stats = st.session_state.task_execution_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tasks", task_stats.get('total_tasks', 0))
            st.metric("Successful", task_stats.get('successful_tasks', 0))
        with col2:
            st.metric("Failed", task_stats.get('failed_tasks', 0))
            success_rate = (task_stats.get('successful_tasks', 0) / task_stats.get('total_tasks', 1)) * 100 if task_stats.get('total_tasks', 0) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Recent Tasks
        if task_stats.get('recent_tasks'):
            with st.expander("Recent Tasks", expanded=False):
                for task in task_stats['recent_tasks'][-5:]:  # Show last 5 tasks
                    status_icon = "✅" if task.get('status') == 'success' else "❌"
                    st.write(f"{status_icon} **{task.get('query', 'Unknown')}**")
                    st.write(f"   Status: {task.get('status', 'unknown')}")
                    if task.get('result'):
                        st.write(f"   Result: {task.get('result', '')[:100]}...")
                    st.markdown("---")
        
        st.markdown("---")
        
        # Display statistics
        if st.session_state.processing_stats:
            st.markdown('<div class="section-title">Processing Statistics</div>', unsafe_allow_html=True)
            stats = st.session_state.processing_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Pages", stats.get('total_pages', 0))
                st.metric("Vision Pages", stats.get('vision_pages', 0))
                st.metric("Cached Pages", stats.get('cached_pages', 0))
            
            with col2:
                st.metric("Text Pages", stats.get('text_pages', 0))
                st.metric("Documents", stats.get('documents_classified', 0))
                st.metric("Structured Data", stats.get('structured_extractions', 0))
    
    # Main content area - Create tabs
    if st.session_state.per_file_data:
        # Create tab list: Smart Q&A Agent + one tab per document
        tab_names = ["Smart Q&A Agent"] + list(st.session_state.per_file_data.keys())
        tabs = st.tabs(tab_names)
        
        # Tab 0: Smart Q&A Agent (handles both questions and tasks)
        with tabs[0]:
            integrate_qna_agent_langgraph_ui()
        
        # Tabs 1+: One tab per document
        for tab_idx, (file_name, file_data) in enumerate(st.session_state.per_file_data.items(), start=1):
            with tabs[tab_idx]:
                st.markdown(f"### {file_name}")
                
                # Create sub-layout: PDF viewer on left, analysis on right
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### Document Viewer")
                    
                    if file_name in st.session_state.uploaded_file_bytes:
                        file_bytes = st.session_state.uploaded_file_bytes[file_name]
                        
                        # Page navigation
                        try:
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            total_pages = len(doc)
                            doc.close()
                            
                            page_num = st.number_input(
                                f"Page (1-{total_pages})",
                                min_value=1,
                                max_value=total_pages,
                                value=1,
                                key=f"page_nav_{file_name}"
                            )
                            
                            display_pdf_page(file_bytes, page_num)
                            
                        except Exception as e:
                            st.error(f"Error loading PDF: {e}")
                    else:
                        st.warning("PDF not available for viewing")
                
                with col2:
                    st.markdown("#### Document Analysis")
                    
                    # Document Classification
                    classification = file_data.get('classification', {})
                    if classification:
                        st.markdown('<div class="classification-box">', unsafe_allow_html=True)
                        st.markdown(f"**Document Type:** {classification.get('document_type', 'Unknown')}")
                        st.markdown(f"**Confidence:** {classification.get('confidence', 0):.1%}")
                        st.markdown(f"**Reasoning:** {classification.get('reasoning', 'N/A')}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Structured Data
                    structured_data = file_data.get('structured_data', {})
                    if structured_data and 'error' not in structured_data:
                        with st.expander("Structured Data Extracted", expanded=True):
                            st.json(structured_data)
                    elif structured_data and 'error' in structured_data:
                        with st.expander("Structured Data Extraction"):
                            st.warning(f"Extraction encountered an issue: {structured_data.get('error', 'Unknown error')}")
                            if 'raw_response' in structured_data:
                                st.text(structured_data['raw_response'])
                    else:
                        st.info("No structured data available")
                    
        
    
    elif st.session_state.conversation:
        # Q&A interface only (when database is loaded but no file data)
        st.markdown("### Question & Answer Interface")
        
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if message.type == 'human' else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
                
                if role == "assistant" and i // 2 < len(st.session_state.get('response_sources', [])):
                    sources = st.session_state.response_sources[i // 2]
                    if sources:
                        with st.expander(f"Sources ({len(sources)} documents)"):
                            for idx, doc in enumerate(sources):
                                st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')} "
                                          f"(Page {doc.metadata.get('page', '?')}) - "
                                          f"Type: {doc.metadata.get('type', 'unknown')}")
                                st.text(doc.page_content[:200] + "...")
                                st.markdown("---")
        
        if user_question := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation.invoke({'question': user_question})
                        answer = response.get('answer', '')
                        source_docs = response.get('source_documents', [])
                        
                        st.markdown(answer)
                        
                        if source_docs:
                            with st.expander(f"Sources ({len(source_docs)} documents)"):
                                for idx, doc in enumerate(source_docs):
                                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')} "
                                              f"(Page {doc.metadata.get('page', '?')}) - "
                                              f"Type: {doc.metadata.get('type', 'unknown')}")
                                    st.text(doc.page_content[:200] + "...")
                                    st.markdown("---")
                        
                        st.session_state.chat_history = response.get('chat_history', [])
                        
                        if 'response_sources' not in st.session_state:
                            st.session_state.response_sources = []
                        st.session_state.response_sources.append(source_docs)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 8px; text-align: center; 
                    border: 1px solid #E0E0E0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h2 style="color: #6B2190; font-family: 'Segoe UI', sans-serif; font-size: 1.5rem; 
                    font-weight: 600; margin-bottom: 1rem;">
                Welcome to Intelligent Document Processing
            </h2>
            <ul style="text-align: left; max-width: 550px; margin: 0 auto; 
                    font-family: 'Segoe UI', sans-serif; font-size: 0.9rem; 
                    color: #2c3e50; line-height: 1.8;">
                <li><strong style="color: #6B2190;">Automatic Classification</strong> - AI identifies document types</li>
                <li><strong style="color: #6B2190;">Structured Data Extraction</strong> - Intelligent field extraction</li>
                <li><strong style="color: #6B2190;">Multi-Database Architecture</strong> - Separate stores for text, images</li>
                <li><strong style="color: #6B2190;">Smart Q&A</strong> - Query across all document types</li>
                <li><strong style="color: #6B2190;">Visual Caching</strong> - Efficient processing with result caching</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    

if __name__ == "__main__":
    main()
