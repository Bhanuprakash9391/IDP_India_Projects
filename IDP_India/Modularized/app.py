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
from vision_manager import *
from llm_manager import *
from excel_manager import *

from database_manager import *  
from email_manager import *

st.set_page_config(
    page_title="Intelligent Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
EXPORTED_DATA_DIR = "exported_data"

def ensure_directories():
    """Ensure necessary directories exist"""
    Path(VECTOR_DB_DIR).mkdir(exist_ok=True)
    Path(VISION_CACHE_DIR).mkdir(exist_ok=True)
    Path(STRUCTURED_DATA_DIR).mkdir(exist_ok=True)
    Path(EXPORTED_DATA_DIR).mkdir(exist_ok=True)
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
                'structured_data': {},
                'pages_info': []
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
                    
                    file_data['pages_info'].append(page_info)
                
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
    
    # Load table vectorstore
    table_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_table")
    if os.path.exists(table_path):
        vectorstores['table'] = FAISS.load_local(table_path, embeddings, allow_dangerous_deserialization=True)
    
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
    elif 'table' in vectorstores:
        primary_retriever = vectorstores['table'].as_retriever(search_kwargs={"k": 5})
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


# def send_documents_via_email(
#     recipient_email: str,
#     per_file_data: Dict,
#     excel_files: List[str] = None
# ) -> Dict:
#     """Send processed documents via email with Excel attachments"""
    
#     email_mgr = EmailManager()
    
#     if not email_mgr.validate_config():
#         return {
#             "success": False,
#             "error": "Email configuration is missing. Please set SENDER_EMAIL and SENDER_PASSWORD in .env file"
#         }
    
#     try:
#         # Create document summary
#         documents_info = []
#         for file_name, file_data in per_file_data.items():
#             classification = file_data.get('classification', {})
#             documents_info.append({
#                 'file_name': file_name,
#                 'document_type': classification.get('document_type', 'Unknown'),
#                 'confidence': classification.get('confidence', 0),
#                 'total_pages': len(file_data.get('pages_info', []))
#             })
        
#         # Create email body
#         subject = f"Document Processing Complete - {len(documents_info)} documents processed"
#         body = email_mgr.create_document_summary_email(documents_info)
        
#         # Send email
#         success = email_mgr.send_email_with_attachments(
#             recipient_email=recipient_email,
#             subject=subject,
#             body=body,
#             attachments=excel_files or []
#         )
        
#         if success:
#             return {
#                 "success": True,
#                 "message": f"Email sent successfully to {recipient_email}",
#                 "documents_sent": len(documents_info),
#                 "attachments": len(excel_files) if excel_files else 0
#             }
#         else:
#             return {
#                 "success": False,
#                 "error": "Failed to send email. Check console logs for details."
#             }
    
#     except Exception as e:
#         return {"success": False, "error": str(e)}




def display_save_and_email_section():
    """Display save to database and email functionality in sidebar"""
    
    if st.session_state.per_file_data:
        st.markdown("---")
        st.markdown('<div class="section-title">Save & Share</div>', unsafe_allow_html=True)
        
        # Two buttons side by side
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save to DB", use_container_width=True):
                with st.spinner("Saving to database..."):
                    try:
                        result = result = save_to_database(st.session_state.per_file_data, st.session_state.excel_exports)
                        
                        if result['success']:
                            st.success(f"Saved {result['total_rows_inserted']} rows to database!")
                            
                            # Show statistics
                            st.info(f"Tables processed: {result['tables_processed']}, Skipped duplicates: {result['total_rows_skipped']}")
                            
                        else:
                            st.error(f"Save failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("Send Email", use_container_width=True):
                st.session_state.show_email_form = True
        
        # Email form
        if st.session_state.get('show_email_form', False):
            st.markdown("#### Email Details")
            
            recipient_email = st.text_input(
                "Recipient Email:",
                placeholder="example@email.com",
                key="email_recipient"
            )
            
            # Get available Excel files
            excel_files = []
            if st.session_state.excel_exports:
                excel_files = [
                    info['filepath'] 
                    for info in st.session_state.excel_exports.values()
                    if os.path.exists(info['filepath'])
                ]
            
            include_attachments = st.checkbox(
                f"Include Excel files ({len(excel_files)} files)",
                value=True
            )
            
            col_send, col_cancel = st.columns(2)
            
            with col_send:
                if st.button("Send", use_container_width=True):
                    if recipient_email:
                        with st.spinner(f"Sending email to {recipient_email}..."):
                            try:
                                attachments = excel_files if include_attachments else None
                                
                                result = send_documents_via_email(
                                    recipient_email=recipient_email,
                                    per_file_data=st.session_state.per_file_data,
                                    excel_files=attachments
                                )
                                
                                if result['success']:
                                    st.success(f"✅ {result['message']}")
                                    st.session_state.show_email_form = False
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"❌ Error sending email: {e}")
                    else:
                        st.warning("⚠️ Please enter recipient email")
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_email_form = False
                    st.rerun()
        
        # Database statistics
        st.markdown("---")
        if st.button("View DB Statistics", use_container_width=True):
                with st.spinner("Loading statistics..."):
                    try:
                        
                        db = DatabaseManager()
                        if db.connect():
                            stats = db.get_all_statistics()
                            db.close()
                            
                            st.markdown("#### Database Statistics")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Tables", stats.get('total_tables', 0))
                            
                            tables_info = stats.get('tables', {})
                            if tables_info:
                                st.markdown("**Tables in Database:**")
                                for table_name, info in tables_info.items():
                                    st.write(f"- **{table_name}**: {info['row_count']} rows")
                        else:
                            st.error("Database connection failed")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")

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
    if "excel_exports" not in st.session_state:
        st.session_state.excel_exports = {}
    if "show_email_form" not in st.session_state:
        st.session_state.show_email_form = False
    
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
                            
                            # AUTO-EXPORT TO EXCEL
                            with st.spinner("Exporting structured data to Excel..."):
                                try:
                                    saved_files = save_structured_data_to_excel(
                                        st.session_state.per_file_data,
                                        output_dir=EXPORTED_DATA_DIR
                                    )
                                    st.session_state.excel_exports = saved_files
                                    
                                    st.success("Documents processed and exported successfully!")
                                    st.info(f"Created {len(saved_files)} Excel file(s) by document type")
                                    
                                except Exception as e:
                                    st.warning(f"Processing succeeded but Excel export failed: {e}")
                            
                        else:
                            st.error("Failed to create conversation chain")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Please upload files and provide a database name")
        
        st.markdown("---")
        
        # Export Section
        if st.session_state.per_file_data:
            st.markdown('<div class="section-title">Export Options</div>', unsafe_allow_html=True)
            
            if st.button("Export to Excel"):
                with st.spinner("Generating Excel files..."):
                    try:
                        saved_files = save_structured_data_to_excel(
                            st.session_state.per_file_data,
                            output_dir=EXPORTED_DATA_DIR
                        )
                        st.session_state.excel_exports = saved_files
                        
                        st.success(f"Exported {len(saved_files)} file(s)")
                        
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            # Download buttons for exported files
            if st.session_state.excel_exports:
                st.markdown("**Download Exports:**")
                for doc_type, info in st.session_state.excel_exports.items():
                    filepath = info['filepath']
                    if os.path.exists(filepath):
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                label=f"{doc_type} ({info['count']} docs)",
                                data=f.read(),
                                file_name=info['filename'],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_{doc_type}"
                            )
        
        st.markdown("---")
        
        # Save & Email Section
        if st.session_state.per_file_data:
            st.markdown('<div class="section-title">Save & Share</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save to DB", use_container_width=True):
                    with st.spinner("Saving to database..."):
                        try:
                            result = save_to_database(st.session_state.per_file_data, st.session_state.excel_exports)
                            
                            if result['success']:
                                st.success(f"Saved {result['total_rows_inserted']} rows to database!")
                                st.info(f"Tables processed: {result['tables_processed']}, Skipped duplicates: {result['total_rows_skipped']}")
                            else:
                                st.error(f"Save failed: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f" Error: {e}")
            
            with col2:
                if st.button("Send Email", use_container_width=True):
                    st.session_state.show_email_form = True
            
            # Email form
            if st.session_state.get('show_email_form', False):
                st.markdown("#### Email Details")
                
                recipient_email = st.text_input(
                    "Recipient Email:",
                    placeholder="example@email.com",
                    key="email_recipient"
                )
                
                excel_files = []
                if st.session_state.excel_exports:
                    excel_files = [
                        info['filepath'] 
                        for info in st.session_state.excel_exports.values()
                        if os.path.exists(info['filepath'])
                    ]
                
                include_attachments = st.checkbox(
                    f"Include Excel files ({len(excel_files)} files)",
                    value=True
                )
                
                col_send, col_cancel = st.columns(2)
                
                with col_send:
                    if st.button("Send", use_container_width=True):
                        if recipient_email:
                            with st.spinner(f"Sending email to {recipient_email}..."):
                                try:
                                    attachments = excel_files if include_attachments else None
                                    
                                    result = send_documents_via_email(
                                        recipient_email=recipient_email,
                                        per_file_data=st.session_state.per_file_data,
                                        excel_files=attachments
                                    )
                                    
                                    if result['success']:
                                        st.success(f"{result['message']}")
                                        if result.get('attachments', 0) > 0:
                                            st.info(f" Sent with {result['attachments']} Excel file(s)")
                                        st.session_state.show_email_form = False
                                        st.rerun()
                                    else:
                                        error_msg = result.get('error', 'Unknown error')
                                        st.error(f"{error_msg}")
                                        
                                        # Show helpful hints based on error
                                        if "configuration" in error_msg.lower():
                                            st.warning("Please set SENDER_EMAIL and SENDER_PASSWORD in your .env file")
                                        elif "authentication" in error_msg.lower() or "login" in error_msg.lower():
                                            st.warning("Check your email credentials. For Gmail, you may need an App Password")
                                        elif "connection" in error_msg.lower():
                                            st.warning(" Check your SMTP server settings and internet connection")
                                
                                except Exception as e:
                                    st.error(f" Error sending email: {e}")
                        else:
                            st.warning(" Please enter recipient email")
                
                with col_cancel:
                    if st.button(" Cancel", use_container_width=True):
                        st.session_state.show_email_form = False
                        st.rerun()
            
            # Database statistics
            st.markdown("---")
            if st.button("View DB Statistics", use_container_width=True):
                with st.spinner("Loading statistics..."):
                    try:
                        
                        db = DatabaseManager()
                        if db.connect():
                            stats = db.get_all_statistics()
                            db.close()
                            
                            st.markdown("#### Database Statistics")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Tables", stats.get('total_tables', 0))
                            
                            tables_info = stats.get('tables', {})
                            if tables_info:
                                st.markdown("**Tables in Database:**")
                                for table_name, info in tables_info.items():
                                    st.write(f"- **{table_name}**: {info['row_count']} rows")
                        else:
                            st.error("Database connection failed")
                    
                    except Exception as e:
                        st.error(f" Error: {e}")
        
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
                
    
    # Main content area - Create tabs
    if st.session_state.per_file_data:
        tab_names = ["Q&A Interface"] + list(st.session_state.per_file_data.keys()) + [" Export Summary"]
        tabs = st.tabs(tab_names)
        
        # Tab 0: Q&A Interface
        with tabs[0]:
            st.markdown("### Question & Answer Interface")
            
            if not st.session_state.conversation:
                st.info("Process documents first to enable Q&A")
            else:
                for i, message in enumerate(st.session_state.chat_history):
                    role = "user" if message.type == 'human' else "assistant"
                    with st.chat_message(role):
                        st.markdown(message.content)
                
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
        
        # Tabs 1 to N-1: One tab per document
        for tab_idx, (file_name, file_data) in enumerate(st.session_state.per_file_data.items(), start=1):
            with tabs[tab_idx]:
                st.markdown(f"### {file_name}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### Document Viewer")
                    
                    if file_name in st.session_state.uploaded_file_bytes:
                        file_bytes = st.session_state.uploaded_file_bytes[file_name]
                        
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
                    
                    classification = file_data.get('classification', {})
                    if classification:
                        st.markdown('<div class="classification-box">', unsafe_allow_html=True)
                        st.markdown(f"**Document Type:** {classification.get('document_type', 'Unknown')}")
                        st.markdown(f"**Confidence:** {classification.get('confidence', 0):.1%}")
                        st.markdown(f"**Reasoning:** {classification.get('reasoning', 'N/A')}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    structured_data = file_data.get('structured_data', {})
                    if structured_data and 'error' not in structured_data:
                        with st.expander("Structured Data Extracted", expanded=True):
                            st.json(structured_data)
                    
                   
        
        # Last Tab: Export Summary
        with tabs[-1]:
            st.markdown("###  Export Summary")
            
            if st.session_state.excel_exports:
                st.success("Excel files have been generated!")
                
                summary_data = []
                for doc_type, info in st.session_state.excel_exports.items():
                    summary_data.append({
                        'Document Type': doc_type,
                        'Document Count': info['count'],
                        'File Name': info['filename']
                    })
                
                if summary_data:
                    import pandas as pd
                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)
                
                st.markdown("---")
                st.markdown("####  Download Files")
                
                cols = st.columns(2)
                col_idx = 0
                
                for doc_type, info in st.session_state.excel_exports.items():
                    with cols[col_idx % 2]:
                        filepath = info['filepath']
                        if os.path.exists(filepath):
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label=f" {doc_type} ({info['count']})",
                                    data=f.read(),
                                    file_name=info['filename'],
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"summary_download_{doc_type}",
                                    use_container_width=True
                                )
                    col_idx += 1
            else:
                st.info("No exports available yet. Click 'Export to Excel' in the sidebar to generate files.")
    
    elif st.session_state.conversation:
        st.markdown("### Question & Answer Interface")
        
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if message.type == 'human' else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
        
        if user_question := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation.invoke({'question': user_question})
                        answer = response.get('answer', '')
                        st.markdown(answer)
                        st.session_state.chat_history = response.get('chat_history', [])
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.rerun()
    
    else:
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
                <li><strong style="color: #6B2190;">Excel Export by Type</strong> - Group similar documents automatically</li>
                <li><strong style="color: #6B2190;">Save to Database</strong> - PostgreSQL storage with deduplication</li>
                <li><strong style="color: #6B2190;">Email Sharing</strong> - Send processed documents with attachments</li>
                <li><strong style="color: #6B2190;">Smart Q&A</strong> - Query across all document types</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()