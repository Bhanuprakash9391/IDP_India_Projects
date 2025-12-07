import streamlit as st
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

# Import existing agents
from unified_agent import UnifiedAgent
from logging_config import logger

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
}

class QnAAgent:
    """
    Q&A Agent that can handle both document questions and task requests
    Automatically determines if user wants information or wants to perform actions
    """
    
    def __init__(self):
        self.unified_agent = UnifiedAgent()
    
    def classify_query_intent(self, query: str) -> Dict:
        """Classify whether the query is for information or task execution"""
        logger.info(f"Classifying query intent: {query}")
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            prompt = f"""
            You are an expert at classifying user queries. Your task is to determine if the user wants:
            
            1. **INFORMATION** - Asking questions about document content, seeking facts, explanations, or summaries
            2. **TASK** - Requesting actions like creating files, sending emails, saving data, or performing operations
            
            User Query: "{query}"
            
            Examples:
            
            **INFORMATION QUERIES:**
            - "What is the total amount due?" → INFORMATION
            - "Show me all invoice numbers" → INFORMATION  
            - "What are the payment terms?" → INFORMATION
            - "Summarize the contract" → INFORMATION
            - "Find all documents related to client ABC" → INFORMATION
            
            **TASK QUERIES:**
            - "Create excel with invoice numbers" → TASK
            - "Send email to john@example.com with payment details" → TASK
            - "Save all invoice data to database" → TASK
            - "Export purchase orders to spreadsheet" → TASK
            - "Create a report with all document summaries" → TASK
            
            **MIXED QUERIES:**
            - "What are the invoice numbers and create an excel file with them" → TASK (contains action request)
            - "Show me payment due dates and email them to client@example.com" → TASK (contains action request)
            
            Rules:
            - If the query contains ANY action words (create, send, save, export, email, etc.), classify as TASK
            - If the query is purely informational (what, show, find, summarize, etc.), classify as INFORMATION
            - When in doubt, prefer TASK classification if there's any hint of action
            
            Return ONLY a JSON object with:
            {{
                "intent": "INFORMATION" or "TASK",
                "confidence": float between 0.0 and 1.0,
                "reasoning": "brief explanation of classification"
            }}
            """
            
            logger.info("Sending LLM request for intent classification")
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response received: {result_text}")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                intent_result = json.loads(json_match.group())
                logger.info(f"Intent classification: {intent_result['intent']} (confidence: {intent_result['confidence']})")
                return intent_result
            else:
                logger.warning("Failed to parse JSON from LLM response, using fallback classification")
                return self._fallback_intent_classification(query)
                
        except Exception as e:
            logger.error(f"LLM intent classification error: {e}")
            return self._fallback_intent_classification(query)
    
    def _fallback_intent_classification(self, query: str) -> Dict:
        """Fallback intent classification when LLM fails"""
        query_lower = query.lower()
        
        # Task action keywords
        task_keywords = [
            'create', 'send', 'save', 'export', 'email', 'download', 
            'generate', 'make', 'build', 'attach', 'upload', 'store',
            'export to', 'create a', 'send an', 'save to', 'email to'
        ]
        
        # Check for task keywords
        is_task = any(keyword in query_lower for keyword in task_keywords)
        
        if is_task:
            return {
                "intent": "TASK",
                "confidence": 0.8,
                "reasoning": "Contains action keywords"
            }
        else:
            return {
                "intent": "INFORMATION",
                "confidence": 0.7,
                "reasoning": "Appears to be informational query"
            }
    
    def extract_document_type_filter(self, query: str) -> Optional[str]:
        """Extract document type filter from query (e.g., 'from invoice', 'in statement')"""
        query_lower = query.lower()
        
        # Document type patterns
        document_patterns = {
            'invoice': ['invoice', 'invoices'],
            'statement': ['statement', 'statements', 'account statement'],
            'contract': ['contract', 'agreement', 'agreements'],
            'receipt': ['receipt', 'receipts'],
            'purchase order': ['purchase order', 'po', 'purchase orders'],
            'report': ['report', 'reports'],
            'certificate': ['certificate', 'certificates'],
            'resume': ['resume', 'cv', 'curriculum vitae']
        }
        
        # Look for patterns like "from invoice", "in statement", "for contract"
        for doc_type, patterns in document_patterns.items():
            for pattern in patterns:
                # Check for patterns like "from invoice", "in statement", "for contract"
                if f"from {pattern}" in query_lower or f"in {pattern}" in query_lower or f"for {pattern}" in query_lower:
                    logger.info(f"Detected document type filter: {doc_type}")
                    return doc_type
        
        return None
    
    def filter_source_documents_by_type(self, source_docs: List, doc_type_filter: str, per_file_data: Dict) -> List:
        """Filter source documents by document type"""
        if not doc_type_filter or not source_docs:
            return source_docs
        
        filtered_docs = []
        for doc in source_docs:
            source_file = doc.metadata.get('source', '')
            if source_file in per_file_data:
                file_classification = per_file_data[source_file].get('classification', {})
                file_doc_type = file_classification.get('document_type', '').lower()
                
                # Check if document type matches filter
                if doc_type_filter.lower() in file_doc_type:
                    filtered_docs.append(doc)
        
        logger.info(f"Filtered {len(source_docs)} documents to {len(filtered_docs)} {doc_type_filter} documents")
        return filtered_docs
    
    def process_query(self, query: str, conversation_chain=None, per_file_data: Dict = None) -> Dict:
        """Process user query - either answer question or execute task"""
        logger.info(f"Processing query: {query}")
        
        # Extract document type filter if specified
        doc_type_filter = self.extract_document_type_filter(query)
        
        # Classify intent
        intent_result = self.classify_query_intent(query)
        
        if intent_result["intent"] == "TASK":
            logger.info("Query classified as TASK - delegating to unified agent")
            # Use unified agent for task execution
            if per_file_data:
                result = self.unified_agent.process_unified_request(query, per_file_data)
                return {
                    "type": "TASK",
                    "result": result,
                    "intent_classification": intent_result,
                    "document_type_filter": doc_type_filter
                }
            else:
                return {
                    "type": "TASK",
                    "result": {
                        "status": "failed",
                        "message": "No processed document data available. Please process documents first.",
                        "details": "Task execution requires processed document data."
                    },
                    "intent_classification": intent_result,
                    "document_type_filter": doc_type_filter
                }
        
        else:
            logger.info("Query classified as INFORMATION - using Q&A conversation")
            # Use existing Q&A conversation chain
            if conversation_chain:
                try:
                    response = conversation_chain.invoke({'question': query})
                    answer = response.get('answer', '')
                    source_docs = response.get('source_documents', [])
                    
                    # Apply document type filter if specified
                    if doc_type_filter and per_file_data:
                        source_docs = self.filter_source_documents_by_type(source_docs, doc_type_filter, per_file_data)
                        
                        # If no documents match the filter, update the answer
                        if not source_docs:
                            answer = f"I couldn't find any information matching your query in {doc_type_filter} documents. The original answer was:\n\n{answer}"
                    
                    return {
                        "type": "INFORMATION",
                        "answer": answer,
                        "source_documents": source_docs,
                        "chat_history": response.get('chat_history', []),
                        "intent_classification": intent_result,
                        "document_type_filter": doc_type_filter
                    }
                except Exception as e:
                    logger.error(f"Error in Q&A conversation: {e}")
                    return {
                        "type": "INFORMATION",
                        "answer": f"Error processing your question: {str(e)}",
                        "source_documents": [],
                        "chat_history": [],
                        "intent_classification": intent_result,
                        "document_type_filter": doc_type_filter
                    }
            else:
                return {
                    "type": "INFORMATION",
                    "answer": "Q&A system is not available. Please process documents first to enable question answering.",
                    "source_documents": [],
                    "chat_history": [],
                    "intent_classification": intent_result,
                    "document_type_filter": doc_type_filter
                }


def integrate_qna_agent_ui():
    """Integrate Q&A Agent UI into the main application with continuous dialogue"""
    
    # Check if we have processed data
    if not st.session_state.get('per_file_data'):
        st.warning("Please process documents first to use the Q&A Agent.")
        return
    
    st.markdown("### Smart Q&A Agent")
    st.markdown("Ask questions about your documents or request actions - I'll understand both!")
    
    # Add custom CSS for chat input styling
    st.markdown("""
    <style>
    /* Style the chat input box with app theme color */
    .stChatInput {
        border: 2px solid #6B2190 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        background-color: white !important;
    }
    
    /* Style the textarea inside the chat input */
    .stChatInput textarea {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Focus state for chat input */
    .stChatInput:focus-within {
        border-color: #551A70 !important;
        box-shadow: 0 0 0 2px rgba(107, 33, 144, 0.2) !important;
    }
    
    /* Ensure the chat input container has proper styling */
    div[data-testid="stChatInputContainer"] {
        border: 2px solid #6B2190 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        background-color: white !important;
    }
    
    /* Alternative selector for chat input */
    [data-testid="stChatInput"] {
        border: 2px solid #6B2190 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize conversation history if not exists
    if "qna_conversation" not in st.session_state:
        st.session_state.qna_conversation = []
    
    # Display conversation history
    for message in st.session_state.qna_conversation:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
            
            # Show additional info for assistant messages
            if role == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Show intent classification info
                if "intent_info" in metadata:
                    intent_info = metadata["intent_info"]
                    intent_type = intent_info.get("intent", "UNKNOWN")
                    confidence = intent_info.get("confidence", 0)
                    reasoning = intent_info.get("reasoning", "")
                    
                    st.caption(f"**Detected as {intent_type}** (confidence: {confidence:.0%}) - {reasoning}")
                
                # Show document type filter info
                if "doc_type_filter" in metadata and metadata["doc_type_filter"]:
                    st.caption(f"**Filtering results from {metadata['doc_type_filter']} documents only**")
    
    # Single query input at the bottom for continuous conversation
    user_query = st.chat_input("Ask a question or request an action (e.g., 'What are the invoice numbers?')...")
    
    if user_query:
        # Add user message to conversation
        st.session_state.qna_conversation.append({"role": "user", "content": user_query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                try:
                    qna_agent = QnAAgent()
                    result = qna_agent.process_query(
                        user_query, 
                        st.session_state.conversation,
                        st.session_state.per_file_data
                    )
                    
                    # Prepare response content and metadata
                    response_content = ""
                    metadata = {
                        "intent_info": result.get("intent_classification", {}),
                        "doc_type_filter": result.get("document_type_filter")
                    }
                    
                    # Display based on result type
                    if result["type"] == "INFORMATION":
                        # Show Q&A answer
                        response_content = result["answer"]
                        st.markdown(response_content)
                        
                        # Update main chat history
                        st.session_state.chat_history = result.get("chat_history", [])
                        
                        # Track task execution statistics for information queries
                        if 'task_execution_stats' in st.session_state:
                            st.session_state.task_execution_stats['total_tasks'] += 1
                            st.session_state.task_execution_stats['successful_tasks'] += 1
                            st.session_state.task_execution_stats['recent_tasks'].append({
                                'query': user_query,
                                'status': 'success',
                                'result': 'Information query answered successfully',
                                'timestamp': datetime.now().isoformat()
                            })
                            # Keep only last 10 tasks
                            if len(st.session_state.task_execution_stats['recent_tasks']) > 10:
                                st.session_state.task_execution_stats['recent_tasks'] = st.session_state.task_execution_stats['recent_tasks'][-10:]
                    
                    else:  # TASK result
                        # Show task execution results
                        task_result = result["result"]
                        
                        if task_result["status"] == "success":
                            st.success(task_result["summary"])
                        elif task_result["status"] == "partial_success":
                            st.warning(task_result["summary"])
                        else:
                            st.error(task_result["summary"])
                        
                        # Show detailed breakdown
                        st.markdown("#### Task Execution Details")
                        st.markdown(task_result["details"])
                        
                        # Build response content for conversation history
                        response_content = f"**Task Execution Result:**\n\n{task_result['summary']}\n\n{task_result['details']}"
                        
                        
                        # Track task execution statistics for task queries
                        if 'task_execution_stats' in st.session_state:
                            st.session_state.task_execution_stats['total_tasks'] += 1
                            if task_result["status"] == "success":
                                st.session_state.task_execution_stats['successful_tasks'] += 1
                            elif task_result["status"] == "failed":
                                st.session_state.task_execution_stats['failed_tasks'] += 1
                            
                            st.session_state.task_execution_stats['recent_tasks'].append({
                                'query': user_query,
                                'status': task_result["status"],
                                'result': task_result["summary"],
                                'timestamp': datetime.now().isoformat()
                            })
                            # Keep only last 10 tasks
                            if len(st.session_state.task_execution_stats['recent_tasks']) > 10:
                                st.session_state.task_execution_stats['recent_tasks'] = st.session_state.task_execution_stats['recent_tasks'][-10:]
                    
                    # Show intent classification info
                    intent_info = result.get("intent_classification", {})
                    if intent_info:
                        intent_type = intent_info.get("intent", "UNKNOWN")
                        confidence = intent_info.get("confidence", 0)
                        reasoning = intent_info.get("reasoning", "")
                        
                        st.caption(f"**Detected as {intent_type}** (confidence: {confidence:.0%}) - {reasoning}")
                    
                    # Show document type filter info
                    doc_type_filter = result.get("document_type_filter")
                    if doc_type_filter:
                        st.caption(f"**Filtering results from {doc_type_filter} documents only**")
                    
                    # Add assistant response to conversation history
                    st.session_state.qna_conversation.append({
                        "role": "assistant", 
                        "content": response_content,
                        "metadata": metadata
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.qna_conversation.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "metadata": {}
                    })
        
        # Rerun to update the conversation display
        st.rerun()
