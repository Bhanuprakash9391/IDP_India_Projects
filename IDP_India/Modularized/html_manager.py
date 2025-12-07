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

/* Structured data box */
.structured-data {
    background: #F8F9FA;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
    border: 1px solid #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
}

/* Table box */
.table-box {
    background: #FFF9E6;
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


