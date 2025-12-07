import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2 import sql
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from pathlib import Path
import logging
import pandas as pd

from app import *
from database_manager import *


# Email Configuration (you'll need to set these in .env)
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "sender_email": os.getenv("SENDER_EMAIL", ""),
    "sender_password": os.getenv("SENDER_PASSWORD", "")
}

def send_documents_via_email(
    recipient_email: str,
    per_file_data: Dict,
    excel_files: List[str] = None
) -> Dict:
    """Send processed documents via email with Excel attachments"""
    
    email_mgr = EmailManager()
    
    if not email_mgr.validate_config():
        return {
            "success": False,
            "error": "Email configuration is missing. Please set SENDER_EMAIL and SENDER_PASSWORD in .env file"
        }
    
    try:
        # Create document summary
        documents_info = []
        for file_name, file_data in per_file_data.items():
            classification = file_data.get('classification', {})
            documents_info.append({
                'file_name': file_name,
                'document_type': classification.get('document_type', 'Unknown'),
                'confidence': classification.get('confidence', 0),
                'total_pages': len(file_data.get('pages_info', []))
            })
        
        # Create email body
        subject = f"Document Processing Complete - {len(documents_info)} documents processed"
        body = email_mgr.create_document_summary_email(documents_info)
        
        # Send email
        success = email_mgr.send_email_with_attachments(
            recipient_email=recipient_email,
            subject=subject,
            body=body,
            attachments=excel_files or []
        )
        
        if success:
            return {
                "success": True,
                "message": f"Email sent successfully to {recipient_email}",
                "documents_sent": len(documents_info),
                "attachments": len(excel_files) if excel_files else 0
            }
        else:
            return {
                "success": False,
                "error": "Failed to send email. Check console logs for details."
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}



class EmailManager:
    """Manages email sending functionality"""
    
    def __init__(self):
        self.smtp_server = EMAIL_CONFIG['smtp_server']
        self.smtp_port = EMAIL_CONFIG['smtp_port']
        self.sender_email = EMAIL_CONFIG['sender_email']
        self.sender_password = EMAIL_CONFIG['sender_password']
    
    def validate_config(self) -> bool:
        """Validate email configuration"""
        if not self.sender_email or not self.sender_password:
            print("Email configuration missing. Set SENDER_EMAIL and SENDER_PASSWORD in .env")
            return False
        return True
    
    def send_email_with_attachments(
        self, 
        recipient_email: str, 
        subject: str, 
        body: str, 
        attachments: List[str] = None
    ) -> bool:
        """Send email with optional attachments"""
        
        if not self.validate_config():
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={Path(file_path).name}'
                            )
                            msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_document_summary_email(self, documents_info: List[Dict]) -> str:
        """Create HTML email body with document summary"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .header { 
                    background: linear-gradient(135deg, #6B2190 0%, #C9A961 100%); 
                    color: white; 
                    padding: 30px; 
                    text-align: center; 
                    border-radius: 10px 10px 0 0;
                }
                .content { padding: 30px; background: #f9f9f9; }
                .summary-box {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-top: 20px;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }
                th { 
                    background-color: #6B2190; 
                    color: white;
                    font-weight: 600;
                }
                tr:nth-child(even) { background-color: #f2f2f2; }
                tr:hover { background-color: #e8f4f8; }
                .footer { 
                    margin-top: 30px; 
                    padding: 20px;
                    font-size: 12px; 
                    color: #666;
                    text-align: center;
                    background: white;
                    border-radius: 0 0 10px 10px;
                }
                .stats {
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }
                .stat-item {
                    text-align: center;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .stat-number {
                    font-size: 32px;
                    font-weight: bold;
                    color: #6B2190;
                }
                .stat-label {
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Processing Complete</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px;">Your intelligent document analysis is ready</p>
            </div>
            <div class="content">
                <div class="summary-box">
                    <h2 style="color: #6B2190; margin-top: 0;"> Processing Summary</h2>
                    <p>Your documents have been successfully processed, analyzed, and stored. All extracted data has been saved to the database.</p>
                </div>
        """
        
        # Add statistics
        total_docs = len(documents_info)
        doc_types = {}
        total_pages = 0
        
        for doc in documents_info:
            doc_type = doc.get('document_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_pages += doc.get('total_pages', 0)
        
        html += f"""
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number">{total_docs}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{len(doc_types)}</div>
                        <div class="stat-label">Document Types</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{total_pages}</div>
                        <div class="stat-label">Total Pages</div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>File Name</th>
                        <th>Document Type</th>
                        <th>Confidence</th>
                        <th>Pages</th>
                    </tr>
        """
        
        for doc in documents_info:
            html += f"""
                    <tr>
                        <td>{doc.get('file_name', 'Unknown')}</td>
                        <td><strong>{doc.get('document_type', 'Unknown')}</strong></td>
                        <td>{doc.get('confidence', 0):.1%}</td>
                        <td>{doc.get('total_pages', 0)}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <div class="summary-box" style="margin-top: 30px;">
                    <h3 style="color: #6B2190; margin-top: 0;">📎 Attached Files</h3>
                    <p>The following Excel files are attached to this email:</p>
                    <ul>
                        <li>Structured data organized by document type</li>
                        <li>Extracted tables from all documents</li>
                        <li>Complete summary report</li>
                    </ul>
                </div>
            </div>
            <div class="footer">
                <p><strong>💾 Data Storage:</strong> All data has been securely saved to the PostgreSQL database.</p>
                <p><em>This is an automated email from the Intelligent Document Processor.</em></p>
                <p style="margin-top: 15px; color: #999;">Powered by Azure OpenAI • Vision AI • Advanced NLP</p>
            </div>
        </body>
        </html>
        """
        
        return html


