import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import re
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st
from excel_agent import ExcelAgent

# Setup logging
def setup_logging():
    """Setup comprehensive logging for Email Agent"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/email_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
}

# Email configuration
EMAIL_CONFIG = {
    "SMTP_SERVER": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "SMTP_PORT": int(os.getenv("SMTP_PORT", "587")),
    "EMAIL_USERNAME": os.getenv("SENDER_EMAIL"),
    "EMAIL_PASSWORD": os.getenv("SENDER_PASSWORD"),
    "EMAIL_FROM": os.getenv("SENDER_EMAIL")
}

def integrate_email_agent_ui():
    """Integrate Email Agent UI into the main application"""
    
    # Check if we have processed data
    if not st.session_state.get('per_file_data'):
        st.warning("Please process documents first to use the Email Agent.")
        return
    
    st.markdown("### 📧 Email Agent")
    st.markdown("Send emails with document information extracted from your processed data")
    
    # Email query input
    email_query = st.text_area(
        "Enter your email request:",
        placeholder="Examples:\n- Send an email to john@example.com with the due date for payment\n- Email sarah@company.com with invoice numbers and amounts\n- Send reminder to client@domain.com about overdue payments",
        height=100,
        key="email_query"
    )
    
    # Email configuration status
    email_configured = all([
        EMAIL_CONFIG["EMAIL_USERNAME"],
        EMAIL_CONFIG["EMAIL_PASSWORD"], 
        EMAIL_CONFIG["EMAIL_FROM"]
    ])
    
    if not email_configured:
        st.warning("⚠️ Email configuration is incomplete. Please set EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_FROM in your .env file.")
    
    # Send email button
    if st.button("Send Email", use_container_width=True, type="primary", disabled=not email_configured):
        if email_query:
            with st.spinner("Processing email request..."):
                email_agent = EmailAgent()
                result = email_agent.process_email_request(email_query, st.session_state.per_file_data)
                
                if "successfully" in result.lower():
                    st.success(result)
                else:
                    st.error(result)
        else:
            st.warning("Please enter an email request.")

class EmailAgent:
    """
    Email Agent for sending emails based on document data and user queries
    Can extract relevant information and send formatted emails
    """
    
    def __init__(self):
        self.email_config = EMAIL_CONFIG
    
    def extract_email_info_from_query(self, query: str, per_file_data: Dict) -> Dict:
        """Extract email information from user query using LLM"""
        logger.info(f"Starting email info extraction for query: {query}")
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            # Prepare document summary for context
            document_summary = self._prepare_document_summary(per_file_data)
            logger.info(f"Document summary prepared with {len(per_file_data)} documents")
            
            prompt = f"""
            You are an expert at parsing user queries for email sending.
            
            User Query: "{query}"
            
            Available document data summary:
            {document_summary}
            
            Your task is to extract the following information from the query:
            1. Recipient email address
            2. Email subject
            3. Email body content (what information to include from documents)
            4. Document filters (which documents to include)
            5. Whether to include an Excel file attachment (detect if user mentions "excel file", "spreadsheet", "attached file", etc.)
            
            Examples:
            - "Send an email to john@example.com with the due date for payment" → 
              {{"recipient": "john@example.com", "subject": "Payment Due Date Information", "body": "Include payment due dates from relevant documents", "filters": {{"document_types": ["invoice", "statement"]}}, "include_excel": false}}
            
            - "Email sarah@company.com with invoice numbers and amounts" → 
              {{"recipient": "sarah@company.com", "subject": "Invoice Information", "body": "Include invoice numbers and amounts", "filters": {{"document_types": ["invoice"]}}, "include_excel": false}}
            
            - "Send reminder to client@domain.com about overdue payments" → 
              {{"recipient": "client@domain.com", "subject": "Overdue Payment Reminder", "body": "Include information about overdue payments", "filters": {{"document_types": ["invoice", "statement"]}}, "include_excel": false}}
            
            - "Send an email to bhanu@example.com with excel file have invoice number and po number and date" → 
              {{"recipient": "bhanu@example.com", "subject": "Document Data Export", "body": "I've attached an Excel file with the requested information including invoice numbers, PO numbers, and dates", "filters": {{"document_types": ["invoice", "purchase_order"]}}, "include_excel": true, "excel_fields": ["invoice_number", "po_number", "date"]}}
            
            Return ONLY a JSON object with these keys: recipient, subject, body, filters, include_excel, excel_fields (if include_excel is true)
            """
            
            logger.info("Sending LLM request for email info extraction")
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response received: {result_text[:200]}...")
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                email_info = json.loads(json_match.group())
                logger.info(f"Successfully extracted email info: recipient={email_info.get('recipient')}, include_excel={email_info.get('include_excel', False)}")
                return email_info
            else:
                logger.warning("Failed to parse JSON from LLM response, using fallback extraction")
                # Fallback to basic extraction
                return self._fallback_email_extraction(query)
                
        except Exception as e:
            logger.error(f"LLM email extraction error: {e}")
            return self._fallback_email_extraction(query)
    
    def _fallback_email_extraction(self, query: str) -> Dict:
        """Fallback email information extraction when LLM fails"""
        # Basic email extraction using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        
        return {
            "recipient": emails[0] if emails else "",
            "subject": "Document Information",
            "body": "Here is the requested information from your documents.",
            "filters": {}
        }
    
    def _prepare_document_summary(self, per_file_data: Dict) -> str:
        """Prepare a summary of available document data for LLM context"""
        summary = []
        
        for file_name, file_data in per_file_data.items():
            doc_type = file_data.get('classification', {}).get('document_type', 'Unknown')
            structured_data = file_data.get('structured_data', {})
            
            # Extract key fields for summary
            key_fields = self._extract_key_fields(structured_data)
            
            summary.append(f"- {file_name} ({doc_type}): {key_fields}")
        
        return "\n".join(summary) if summary else "No document data available."
    
    def _extract_key_fields(self, structured_data: Dict) -> str:
        """Extract key fields from structured data for summary"""
        key_fields = []
        
        # Common important fields to include in summary
        important_fields = ['number', 'date', 'amount', 'total', 'due_date', 'customer', 'vendor']
        
        def search_fields(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}{key}"
                    if any(field in full_key.lower() for field in important_fields):
                        if isinstance(value, (str, int, float)) and str(value).strip():
                            key_fields.append(f"{full_key}: {value}")
                    if isinstance(value, dict):
                        search_fields(value, f"{full_key}_")
        
        search_fields(structured_data)
        return ", ".join(key_fields[:5])  # Limit to first 5 fields
    
    def prepare_email_content(self, email_info: Dict, per_file_data: Dict) -> Dict:
        """Prepare email content by extracting relevant data from documents"""
        logger.info(f"Preparing email content for recipient: {email_info.get('recipient')}")
        
        # Filter documents based on email info
        filtered_data = self._filter_documents_for_email(per_file_data, email_info.get('filters', {}))
        logger.info(f"Filtered {len(filtered_data)} documents from {len(per_file_data)} total")
        
        # Extract relevant data for email body
        email_data = self._extract_email_data(filtered_data, email_info['body'])
        logger.info(f"Extracted email data from {len(email_data)} documents")
        
        # Format email content
        subject = email_info['subject']
        body = self._format_email_body(email_info['body'], email_data, filtered_data)
        logger.info(f"Formatted email body with {len(body)} characters")
        
        # Create Excel file if requested
        excel_file_path = None
        if email_info.get('include_excel', False):
            logger.info(f"Creating Excel attachment with fields: {email_info.get('excel_fields', [])}")
            excel_file_path = self._create_excel_attachment(filtered_data, email_info.get('excel_fields', []))
            if excel_file_path:
                logger.info(f"Excel file created successfully: {excel_file_path}")
            else:
                logger.warning("Excel file creation failed")
        
        logger.info(f"Email content prepared successfully for {email_info['recipient']}")
        return {
            "recipient": email_info['recipient'],
            "subject": subject,
            "body": body,
            "documents_used": len(filtered_data),
            "excel_file_path": excel_file_path
        }
    
    def _create_excel_attachment(self, filtered_data: Dict, excel_fields: List[str]) -> str:
        """Create Excel file using Excel Agent for email attachment"""
        try:
            excel_agent = ExcelAgent()
            
            # Create a temporary file for the Excel attachment
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                excel_file_path = temp_file.name
            
            # Use Excel Agent to create the Excel file
            if excel_fields:
                # Create query for specific fields
                field_query = f"Create Excel with {', '.join(excel_fields)}"
                result = excel_agent.create_excel_from_query(field_query, filtered_data, excel_file_path)
            else:
                # Create comprehensive Excel with all data
                result = excel_agent.create_comprehensive_excel(filtered_data, excel_file_path)
            
            # Check if Excel creation was successful
            if "successfully" in result.lower():
                return excel_file_path
            else:
                print(f"Excel creation failed: {result}")
                # Clean up the temporary file
                if os.path.exists(excel_file_path):
                    os.remove(excel_file_path)
                return None
            
        except Exception as e:
            print(f"Error creating Excel attachment: {e}")
            # Clean up the temporary file on error
            if 'excel_file_path' in locals() and os.path.exists(excel_file_path):
                os.remove(excel_file_path)
            return None
    
    def _filter_documents_for_email(self, per_file_data: Dict, filters: Dict) -> Dict:
        """Filter documents based on email filters"""
        if not filters:
            return per_file_data
        
        filtered_data = {}
        
        for file_name, file_data in per_file_data.items():
            include_doc = True
            
            # Filter by document type
            if 'document_types' in filters:
                doc_type = file_data.get('classification', {}).get('document_type', '').lower()
                requested_types = [t.lower() for t in filters['document_types']]
                
                # Check if document type matches any requested type
                type_match = any(req_type in doc_type or doc_type in req_type 
                               for req_type in requested_types)
                
                if not type_match:
                    include_doc = False
            
            if include_doc:
                filtered_data[file_name] = file_data
        
        return filtered_data
    
    def _extract_email_data(self, filtered_data: Dict, body_request: str) -> List[Dict]:
        """Extract relevant data from filtered documents for email content"""
        email_data = []
        
        for file_name, file_data in filtered_data.items():
            doc_data = {
                'filename': file_name,
                'document_type': file_data.get('classification', {}).get('document_type', 'Unknown'),
                'extracted_data': {}
            }
            
            # Use LLM to extract relevant fields based on body request
            structured_data = file_data.get('structured_data', {})
            relevant_fields = self._extract_relevant_fields(structured_data, body_request)
            
            doc_data['extracted_data'] = relevant_fields
            email_data.append(doc_data)
        
        return email_data
    
    def _extract_relevant_fields(self, structured_data: Dict, body_request: str) -> Dict:
        """Extract relevant fields from structured data based on email body request"""
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            # Get all available fields
            available_fields = self._get_all_field_names(structured_data)
            
            prompt = f"""
            User wants to include this information in email: "{body_request}"
            
            Available fields in document data:
            {', '.join(sorted(available_fields))}
            
            Task: Identify which fields from the available data are relevant to include in the email.
            Return ONLY a JSON object with field names as keys and their values.
            Only include fields that are directly relevant to the user's request.
            """
            
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                requested_fields = json.loads(json_match.group())
                
                # Now get actual values for these fields
                relevant_data = {}
                for field in requested_fields.keys():
                    value = self._find_field_value(structured_data, field)
                    if value and value != "Not found":
                        relevant_data[field] = value
                
                return relevant_data
            else:
                return {}
                
        except Exception as e:
            print(f"LLM field extraction error: {e}")
            return {}
    
    def _get_all_field_names(self, data: Dict, prefix: str = "") -> set:
        """Recursively get all field names from structured data"""
        field_names = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}{key}" if prefix else key
                field_names.add(full_key)
                
                if isinstance(value, dict):
                    field_names.update(self._get_all_field_names(value, f"{full_key}_"))
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    for item in value:
                        field_names.update(self._get_all_field_names(item, f"{full_key}_"))
        
        return field_names
    
    def _find_field_value(self, data: Dict, field_name: str) -> str:
        """Find field value in structured data"""
        if isinstance(data, dict):
            # Check for exact match
            if field_name in data:
                value = data[field_name]
                return self._format_field_value(value)
            
            # Check nested structures
            for key, value in data.items():
                if isinstance(value, dict):
                    result = self._find_field_value(value, field_name)
                    if result:
                        return result
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result = self._find_field_value(item, field_name)
                            if result:
                                return result
        
        return "Not found"
    
    def _format_field_value(self, value) -> str:
        """Format field value for display"""
        if isinstance(value, (str, int, float)):
            return str(value)
        elif isinstance(value, list):
            return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)
    
    def _format_email_body(self, body_request: str, email_data: List[Dict], filtered_data: Dict) -> str:
        """Format the email body using LLM to create beautiful, human-readable content"""
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            # Prepare clean, human-readable data for LLM
            document_summary = self._prepare_human_readable_summary(email_data)
            
            prompt = f"""
            You are an expert at writing professional, well-formatted emails. 
            
            USER'S REQUEST: "{body_request}"
            
            INFORMATION EXTRACTED FROM DOCUMENTS:
            {document_summary}
            
            TASK: Create a beautifully formatted email body that:
            1. Directly addresses the user's request in a professional tone
            2. Presents the extracted information in a clear, organized manner
            3. Uses proper email formatting with paragraphs, bullet points, and clear structure
            4. Highlights the most important information first
            5. Is easy to read and understand
            6. Includes a professional closing
            7. DO NOT include any JSON formatting, brackets, or technical data structures
            8. Make it sound natural and human-written
            
            IMPORTANT: The email should flow naturally and read like it was written by a helpful assistant.
            Do not mention that you're an AI or that the data came from JSON.
            
            Format the email as if you are sending it on behalf of the Document Processing System.
            Make it warm, professional, and helpful.
            
            Return ONLY the email body content (no subject line, no headers).
            """
            
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            formatted_body = response.choices[0].message.content.strip()
            
            # Add a small footer
            formatted_body += f"\n\n---\nThis email was automatically generated by the Document Processing System based on {len(filtered_data)} processed documents."
            
            return formatted_body
            
        except Exception as e:
            print(f"LLM email formatting error: {e}")
            # Fallback to basic formatting
            return self._fallback_email_formatting(body_request, email_data, filtered_data)
    
    def _prepare_human_readable_summary(self, email_data: List[Dict]) -> str:
        """Prepare a clean, human-readable summary of extracted data"""
        summary_lines = []
        
        for doc in email_data:
            if doc['extracted_data']:
                doc_summary = f"Document: {doc['filename']} (Type: {doc['document_type']})\n"
                
                for field, value in doc['extracted_data'].items():
                    # Convert field names to readable format
                    readable_field = ' '.join(word.capitalize() for word in field.split('_'))
                    doc_summary += f"  - {readable_field}: {value}\n"
                
                summary_lines.append(doc_summary)
        
        if not summary_lines:
            return "No specific data was extracted from the documents."
        
        return "\n".join(summary_lines)
    
    def _fallback_email_formatting(self, body_request: str, email_data: List[Dict], filtered_data: Dict) -> str:
        """Fallback email formatting when LLM fails"""
        email_body = f"""Dear Recipient,

I hope this email finds you well.

{body_request}

Based on your request, here is the relevant information extracted from the processed documents:

"""
        
        for doc in email_data:
            if doc['extracted_data']:
                email_body += f"\n📄 **{doc['filename']}** ({doc['document_type']}):\n"
                for field, value in doc['extracted_data'].items():
                    # Format field names to be more readable
                    readable_field = ' '.join(word.capitalize() for word in field.split('_'))
                    email_body += f"   • **{readable_field}**: {value}\n"
        
        if not any(doc['extracted_data'] for doc in email_data):
            email_body += "\nNo specific data matching your request was found in the available documents.\n"
        
        email_body += f"""
Total documents analyzed: {len(filtered_data)}

Please let me know if you need any additional information.

Best regards,
Document Processing System
"""
        
        return email_body
    
    def send_email(self, email_content: Dict) -> str:
        """Send email using SMTP with optional Excel attachment"""
        try:
            # Check if email configuration is available
            if not all([self.email_config["EMAIL_USERNAME"], 
                       self.email_config["EMAIL_PASSWORD"],
                       self.email_config["EMAIL_FROM"]]):
                return "Email configuration is incomplete. Please check SMTP settings."
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config["EMAIL_FROM"]
            msg['To'] = email_content['recipient']
            msg['Subject'] = email_content['subject']
            
            # Add body to email
            msg.attach(MIMEText(email_content['body'], 'plain'))
            
            # Add Excel attachment if available
            if email_content.get('excel_file_path'):
                try:
                    with open(email_content['excel_file_path'], 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename=document_data_export.xlsx'
                    )
                    msg.attach(part)
                except Exception as e:
                    print(f"Error attaching Excel file: {e}")
            
            # Send email with better error handling
            server = smtplib.SMTP(self.email_config["SMTP_SERVER"], self.email_config["SMTP_PORT"])
            server.ehlo()  # Identify ourselves to the SMTP server
            server.starttls()  # Secure the connection
            server.ehlo()  # Re-identify ourselves over TLS connection
            server.login(self.email_config["EMAIL_USERNAME"], self.email_config["EMAIL_PASSWORD"])
            text = msg.as_string()
            server.sendmail(self.email_config["EMAIL_FROM"], email_content['recipient'], text)
            server.quit()
            
            # Clean up temporary Excel file
            if email_content.get('excel_file_path') and os.path.exists(email_content['excel_file_path']):
                try:
                    os.remove(email_content['excel_file_path'])
                except:
                    pass
            
            attachment_status = " with Excel attachment" if email_content.get('excel_file_path') else ""
            return f"Email sent successfully to {email_content['recipient']} using {email_content.get('documents_used', 0)} documents{attachment_status}."
            
        except Exception as e:
            # Clean up temporary Excel file on error
            if email_content.get('excel_file_path') and os.path.exists(email_content['excel_file_path']):
                try:
                    os.remove(email_content['excel_file_path'])
                except:
                    pass
            return f"Error sending email: {str(e)}"
    
    def process_email_request(self, query: str, per_file_data: Dict) -> str:
        """Process complete email request from user query"""
        # Extract email information from query
        email_info = self.extract_email_info_from_query(query, per_file_data)
        
        if not email_info.get('recipient'):
            return "No valid email address found in your query. Please specify a recipient email."
        
        # Prepare email content
        email_content = self.prepare_email_content(email_info, per_file_data)
        
        # Send email
        result = self.send_email(email_content)
        
        return result
