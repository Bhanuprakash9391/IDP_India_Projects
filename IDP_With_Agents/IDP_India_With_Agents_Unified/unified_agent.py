import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st

# Import existing agents
from excel_agent import ExcelAgent
from email_agent import EmailAgent
from database_agent import DatabaseAgent
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

class UnifiedAgent:
    """
    Unified Agent that orchestrates multiple agents (Excel, Email, Database)
    Can handle complex queries and delegate tasks to appropriate agents
    """
    
    def __init__(self):
        self.excel_agent = ExcelAgent()
        self.email_agent = EmailAgent()
        self.database_agent = DatabaseAgent()
    
    def decompose_query(self, query: str) -> Dict:
        """Decompose user query into sub-tasks using LLM"""
        logger.info(f"Decomposing query: {query}")
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            prompt = f"""
            You are an expert task decomposition agent. Your job is to analyze user queries and break them down into specific sub-tasks that can be handled by specialized agents.
            
            Available Agents:
            1. Excel Agent - Creates Excel files with document data
            2. Email Agent - Sends emails with document information
            3. Database Agent - Saves document data to database tables
            
            User Query: "{query}"
            
            Analyze this query and identify which agents need to be involved and what specific tasks they should perform.
            
            Examples:
            - "Create excel with invoice number and po number and also save the same to database and send email to abc@gmail.com with the same information" → 
              {{
                "tasks": [
                  {{
                    "agent": "excel",
                    "action": "create_excel",
                    "parameters": {{
                      "fields": ["invoice_number", "po_number"],
                      "document_type": "invoice"
                    }},
                    "description": "Create Excel file with invoice numbers and PO numbers"
                  }},
                  {{
                    "agent": "database", 
                    "action": "save_data",
                    "parameters": {{
                      "table_name": "invoice_po_data",
                      "fields": ["invoice_number", "po_number"],
                      "document_type": "invoice"
                    }},
                    "description": "Save invoice numbers and PO numbers to database"
                  }},
                  {{
                    "agent": "email",
                    "action": "send_email",
                    "parameters": {{
                      "recipient": "abc@gmail.com",
                      "subject": "Invoice and PO Data",
                      "body": "Attached is the Excel file with invoice numbers and PO numbers",
                      "include_excel": true,
                      "excel_fields": ["invoice_number", "po_number"]
                    }},
                    "description": "Send email with Excel attachment"
                  }}
                ]
              }}
            
            - "Save all invoice data to database and create excel with key fields" → 
              {{
                "tasks": [
                  {{
                    "agent": "database",
                    "action": "save_data", 
                    "parameters": {{
                      "table_name": "invoices",
                      "data_scope": "full",
                      "document_type": "invoice"
                    }},
                    "description": "Save all invoice data to database"
                  }},
                  {{
                    "agent": "excel",
                    "action": "create_excel",
                    "parameters": {{
                      "fields": [],
                      "document_type": "invoice"
                    }},
                    "description": "Create Excel with all invoice data"
                  }}
                ]
              }}
            
            - "Send email to client@example.com with payment due dates" → 
              {{
                "tasks": [
                  {{
                    "agent": "email",
                    "action": "send_email",
                    "parameters": {{
                      "recipient": "client@example.com",
                      "subject": "Payment Due Dates",
                      "body": "Include payment due dates from relevant documents",
                      "include_excel": false
                    }},
                    "description": "Send email with payment due dates"
                  }}
                ]
              }}
            
            Important Rules:
            1. Identify ALL agents mentioned in the query
            2. Extract specific parameters for each agent (fields, email addresses, table names, etc.)
            3. Infer document types from context when mentioned
            4. Ensure tasks are logically ordered
            5. Include all information the user explicitly requested
            
            Return ONLY a JSON object with the "tasks" array containing task objects.
            """
            
            logger.info("Sending LLM request for task decomposition")
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response received: {result_text[:200]}...")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                decomposition = json.loads(json_match.group())
                logger.info(f"Successfully decomposed query into {len(decomposition.get('tasks', []))} tasks")
                return decomposition
            else:
                logger.warning("Failed to parse JSON from LLM response, using fallback decomposition")
                return self._fallback_decomposition(query)
                
        except Exception as e:
            logger.error(f"LLM decomposition error: {e}")
            return self._fallback_decomposition(query)
    
    def _fallback_decomposition(self, query: str) -> Dict:
        """Fallback task decomposition when LLM fails"""
        query_lower = query.lower()
        tasks = []
        
        # Check for Excel tasks
        if any(keyword in query_lower for keyword in ['excel', 'spreadsheet', 'export']):
            tasks.append({
                "agent": "excel",
                "action": "create_excel",
                "parameters": {"fields": []},
                "description": "Create Excel file with document data"
            })
        
        # Check for Database tasks
        if any(keyword in query_lower for keyword in ['database', 'save', 'store', 'table']):
            tasks.append({
                "agent": "database",
                "action": "save_data",
                "parameters": {"data_scope": "full"},
                "description": "Save document data to database"
            })
        
        # Check for Email tasks
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        if emails:
            tasks.append({
                "agent": "email",
                "action": "send_email",
                "parameters": {"recipient": emails[0], "include_excel": False},
                "description": f"Send email to {emails[0]}"
            })
        
        return {"tasks": tasks}
    
    def execute_tasks(self, tasks: List[Dict], per_file_data: Dict) -> Dict:
        """Execute all decomposed tasks and return results"""
        logger.info(f"Executing {len(tasks)} tasks")
        results = {
            "successful_tasks": [],
            "failed_tasks": [],
            "overall_status": "success",
            "details": {}
        }
        
        # Track created Excel files for email attachments
        excel_files = {}
        
        for i, task in enumerate(tasks):
            task_id = f"task_{i+1}"
            logger.info(f"Executing {task_id}: {task['agent']} - {task['action']}")
            
            try:
                if task["agent"] == "excel":
                    result = self._execute_excel_task(task, per_file_data, excel_files)
                elif task["agent"] == "database":
                    result = self._execute_database_task(task, per_file_data)
                elif task["agent"] == "email":
                    result = self._execute_email_task(task, per_file_data, excel_files)
                else:
                    result = {"status": "failed", "message": f"Unknown agent: {task['agent']}"}
                
                if result.get("status") == "success":
                    results["successful_tasks"].append(task_id)
                    results["details"][task_id] = result
                    logger.info(f"✓ {task_id} completed successfully")
                else:
                    results["failed_tasks"].append(task_id)
                    results["details"][task_id] = result
                    logger.error(f"✗ {task_id} failed: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"Error executing {task_id}: {str(e)}"
                results["failed_tasks"].append(task_id)
                results["details"][task_id] = {"status": "failed", "message": error_msg}
                logger.error(f"✗ {task_id} failed with exception: {e}")
        
        # Update overall status
        if results["failed_tasks"]:
            results["overall_status"] = "partial_success" if results["successful_tasks"] else "failed"
        else:
            results["overall_status"] = "success"
        
        logger.info(f"Task execution completed: {len(results['successful_tasks'])} successful, {len(results['failed_tasks'])} failed")
        return results
    
    def _execute_excel_task(self, task: Dict, per_file_data: Dict, excel_files: Dict) -> Dict:
        """Execute Excel agent task"""
        try:
            parameters = task.get("parameters", {})
            fields = parameters.get("fields", [])
            document_type = parameters.get("document_type")
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if document_type:
                filename = f"exports/{document_type}_data_{timestamp}.xlsx"
            else:
                filename = f"exports/document_export_{timestamp}.xlsx"
            
            # Apply document type filter if specified, but be flexible
            working_data = per_file_data
            if document_type:
                working_data = self.excel_agent.filter_documents_by_type(per_file_data, document_type)
                # If no documents match the specified type, use all available data
                if not working_data:
                    logger.warning(f"No {document_type} documents found, using all available documents")
                    working_data = per_file_data
            
            # Create Excel file
            if fields:
                # Create query for specific fields
                field_query = f"Create Excel with {', '.join(fields)}"
                result = self.excel_agent.create_excel_from_query(field_query, working_data, filename)
            else:
                # Create comprehensive Excel
                result = self.excel_agent.create_comprehensive_excel(working_data, filename, document_type)
            
            if "successfully" in result.lower():
                # Store file path for potential email attachments
                excel_files[document_type or "all"] = filename
                return {
                    "status": "success", 
                    "message": result,
                    "file_path": filename,
                    "document_type": document_type,
                    "fields": fields
                }
            else:
                return {"status": "failed", "message": result}
                
        except Exception as e:
            return {"status": "failed", "message": f"Excel task failed: {str(e)}"}
    
    def _execute_database_task(self, task: Dict, per_file_data: Dict) -> Dict:
        """Execute Database agent task"""
        try:
            parameters = task.get("parameters", {})
            table_name = parameters.get("table_name", "documents")
            data_scope = parameters.get("data_scope", "full")
            fields = parameters.get("fields", [])
            document_type = parameters.get("document_type")
            
            # Build database query
            if fields:
                query = f"Save {', '.join(fields)} to database table {table_name}"
            elif data_scope == "full":
                query = f"Save all {document_type or ''} data to database table {table_name}"
            else:
                query = f"Save {document_type or 'document'} data to database"
            
            if document_type:
                query += f" for {document_type} documents"
            
            # Execute database operation
            result = self.database_agent.process_database_request(query, per_file_data)
            
            if "successfully" in result.lower():
                return {
                    "status": "success",
                    "message": result,
                    "table_name": table_name,
                    "document_type": document_type,
                    "fields": fields
                }
            else:
                return {"status": "failed", "message": result}
                
        except Exception as e:
            return {"status": "failed", "message": f"Database task failed: {str(e)}"}
    
    def _execute_email_task(self, task: Dict, per_file_data: Dict, excel_files: Dict) -> Dict:
        """Execute Email agent task"""
        try:
            parameters = task.get("parameters", {})
            recipient = parameters.get("recipient")
            subject = parameters.get("subject", "Document Information")
            body = parameters.get("body", "Here is the requested information from your documents.")
            include_excel = parameters.get("include_excel", False)
            excel_fields = parameters.get("excel_fields", [])
            document_type = parameters.get("document_type")
            
            if not recipient:
                return {"status": "failed", "message": "No recipient email address specified"}
            
            # Build email query
            email_query = f"Send email to {recipient} with {body}"
            if include_excel:
                email_query += f" and include excel file with {', '.join(excel_fields) if excel_fields else 'all data'}"
            
            if document_type:
                email_query += f" for {document_type} documents"
            
            # Execute email operation
            result = self.email_agent.process_email_request(email_query, per_file_data)
            
            if "successfully" in result.lower():
                return {
                    "status": "success",
                    "message": result,
                    "recipient": recipient,
                    "subject": subject,
                    "document_type": document_type
                }
            else:
                return {"status": "failed", "message": result}
                
        except Exception as e:
            return {"status": "failed", "message": f"Email task failed: {str(e)}"}
    
    def process_unified_request(self, query: str, per_file_data: Dict) -> Dict:
        """Process unified request - main entry point"""
        logger.info(f"Processing unified request: {query}")
        
        # Step 1: Decompose query into tasks
        decomposition = self.decompose_query(query)
        tasks = decomposition.get("tasks", [])
        
        if not tasks:
            return {
                "status": "failed",
                "message": "Could not identify any tasks from your query. Please be more specific.",
                "details": {}
            }
        
        # Step 2: Execute all tasks
        execution_results = self.execute_tasks(tasks, per_file_data)
        
        # Step 3: Format final response
        return self._format_final_response(query, tasks, execution_results)
    
    def _format_final_response(self, original_query: str, tasks: List[Dict], execution_results: Dict) -> Dict:
        """Format the final response for the user"""
        successful_count = len(execution_results["successful_tasks"])
        failed_count = len(execution_results["failed_tasks"])
        total_tasks = len(tasks)
        
        # Build summary message
        if execution_results["overall_status"] == "success":
            summary = f"✅ All {total_tasks} tasks completed successfully!"
        elif execution_results["overall_status"] == "partial_success":
            summary = f"⚠️ {successful_count} out of {total_tasks} tasks completed successfully. {failed_count} tasks failed."
        else:
            summary = f"❌ All {total_tasks} tasks failed."
        
        # Build detailed breakdown
        details = f"**Query:** {original_query}\n\n"
        details += f"**Summary:** {summary}\n\n"
        details += "**Task Breakdown:**\n"
        
        for i, task in enumerate(tasks):
            task_id = f"task_{i+1}"
            task_result = execution_results["details"].get(task_id, {})
            
            status_icon = "✅" if task_id in execution_results["successful_tasks"] else "❌"
            details += f"\n{status_icon} **{task['description']}**\n"
            details += f"   - Agent: {task['agent']}\n"
            details += f"   - Status: {task_result.get('status', 'unknown')}\n"
            details += f"   - Result: {task_result.get('message', 'No result')}\n"
        
        return {
            "status": execution_results["overall_status"],
            "summary": summary,
            "details": details,
            "execution_details": execution_results
        }
