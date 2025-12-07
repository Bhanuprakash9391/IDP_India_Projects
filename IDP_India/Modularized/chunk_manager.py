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
        start = end - overlap  # Overlap to maintain context
    
    return chunks