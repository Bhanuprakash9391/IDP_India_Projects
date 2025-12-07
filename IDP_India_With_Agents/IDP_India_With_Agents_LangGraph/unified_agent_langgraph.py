import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st

# Import LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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

class AgentState(TypedDict):
    """State definition for LangGraph workflow"""
    query: str
    per_file_data: Dict
    tasks: List[Dict]
    current_task: Optional[Dict]
    execution_results: Dict
    excel_files: Dict
    status: str
    message: str
    step: str

class UnifiedAgentLangGraph:
    """
    Unified Agent that orchestrates multiple agents using LangGraph
    Provides better workflow management and state tracking
    """
    
    def __init__(self):
        self.excel_agent = ExcelAgent()
        self.email_agent = EmailAgent()
        self.database_agent = DatabaseAgent()
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("decompose_query", self._decompose_query_node)
        workflow.add_node("execute_excel_task", self._execute_excel_task_node)
        workflow.add_node("execute_database_task", self._execute_database_task_node)
        workflow.add_node("execute_email_task", self._execute_email_task_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Define edges
        workflow.set_entry_point("decompose_query")
        
        # After decomposition, route to appropriate task nodes
        workflow.add_conditional_edges(
            "decompose_query",
            self._route_to_tasks,
            {
                "excel": "execute_excel_task",
                "database": "execute_database_task", 
                "email": "execute_email_task",
                "format": "format_response",
                "end": END
            }
        )
        
        # After each task, check if more tasks exist
        workflow.add_conditional_edges(
            "execute_excel_task",
            self._route_after_task,
            {
                "next_task": "execute_excel_task",
                "database": "execute_database_task",
                "email": "execute_email_task",
                "format": "format_response",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "execute_database_task",
            self._route_after_task,
            {
                "next_task": "execute_database_task",
                "excel": "execute_excel_task",
                "email": "execute_email_task",
                "format": "format_response",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "execute_email_task",
            self._route_after_task,
            {
                "next_task": "execute_email_task",
                "excel": "execute_excel_task",
                "database": "execute_database_task",
                "format": "format_response",
                "end": END
            }
        )
        
        # After formatting response, end
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _decompose_query_node(self, state: AgentState) -> AgentState:
        """Decompose user query into sub-tasks using LLM"""
        logger.info(f"LangGraph: Decomposing query: {state['query']}")
        
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
            
            User Query: "{state['query']}"
            
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
            
            logger.info("LangGraph: Sending LLM request for task decomposition")
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LangGraph: LLM response received: {result_text[:200]}...")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                decomposition = json.loads(json_match.group())
                tasks = decomposition.get('tasks', [])
                logger.info(f"LangGraph: Successfully decomposed query into {len(tasks)} tasks")
                
                return {
                    **state,
                    "tasks": tasks,
                    "execution_results": {
                        "successful_tasks": [],
                        "failed_tasks": [],
                        "overall_status": "success",
                        "details": {}
                    },
                    "excel_files": {},
                    "step": "decomposed"
                }
            else:
                logger.warning("LangGraph: Failed to parse JSON from LLM response")
                return self._fallback_decomposition(state)
                
        except Exception as e:
            logger.error(f"LangGraph: LLM decomposition error: {e}")
            return self._fallback_decomposition(state)
    
    def _fallback_decomposition(self, state: AgentState) -> AgentState:
        """Fallback task decomposition when LLM fails"""
        query_lower = state['query'].lower()
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
        emails = re.findall(email_pattern, state['query'])
        if emails:
            tasks.append({
                "agent": "email",
                "action": "send_email",
                "parameters": {"recipient": emails[0], "include_excel": False},
                "description": f"Send email to {emails[0]}"
            })
        
        return {
            **state,
            "tasks": tasks,
            "execution_results": {
                "successful_tasks": [],
                "failed_tasks": [],
                "overall_status": "success",
                "details": {}
            },
            "excel_files": {},
            "step": "decomposed"
        }
    
    def _route_to_tasks(self, state: AgentState) -> str:
        """Route to appropriate task after decomposition"""
        if not state.get('tasks'):
            return "format"
        
        # Get the first task
        current_task = state['tasks'][0]
        agent = current_task.get('agent')
        
        if agent == "excel":
            return "excel"
        elif agent == "database":
            return "database"
        elif agent == "email":
            return "email"
        else:
            return "format"
    
    def _route_after_task(self, state: AgentState) -> str:
        """Route after completing a task"""
        # Remove the completed task
        remaining_tasks = state.get('tasks', [])[1:]
        
        if not remaining_tasks:
            return "format"
        
        # Get the next task
        next_task = remaining_tasks[0]
        agent = next_task.get('agent')
        
        if agent == "excel":
            return "excel"
        elif agent == "database":
            return "database"
        elif agent == "email":
            return "email"
        else:
            return "format"
    
    def _execute_excel_task_node(self, state: AgentState) -> AgentState:
        """Execute Excel agent task"""
        try:
            tasks = state.get('tasks', [])
            if not tasks:
                return {**state, "step": "excel_complete"}
            
            task = tasks[0]
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
            working_data = state['per_file_data']
            if document_type:
                working_data = self.excel_agent.filter_documents_by_type(state['per_file_data'], document_type)
                # If no documents match the specified type, use all available data
                if not working_data:
                    logger.warning(f"LangGraph: No {document_type} documents found, using all available documents")
                    working_data = state['per_file_data']
            
            # Create Excel file
            if fields:
                # Create query for specific fields
                field_query = f"Create Excel with {', '.join(fields)}"
                result = self.excel_agent.create_excel_from_query(field_query, working_data, filename)
            else:
                # Create comprehensive Excel
                result = self.excel_agent.create_comprehensive_excel(working_data, filename, document_type)
            
            execution_results = state['execution_results'].copy()
            excel_files = state['excel_files'].copy()
            
            if "successfully" in result.lower():
                # Store file path for potential email attachments
                excel_files[document_type or "all"] = filename
                
                execution_results["successful_tasks"].append(f"excel_task_{len(execution_results['successful_tasks'])}")
                execution_results["details"][f"excel_task_{len(execution_results['successful_tasks'])}"] = {
                    "status": "success", 
                    "message": result,
                    "file_path": filename,
                    "document_type": document_type,
                    "fields": fields
                }
                logger.info(f"LangGraph: ✓ Excel task completed successfully")
            else:
                execution_results["failed_tasks"].append(f"excel_task_{len(execution_results['failed_tasks'])}")
                execution_results["details"][f"excel_task_{len(execution_results['failed_tasks'])}"] = {
                    "status": "failed", 
                    "message": result
                }
                logger.error(f"LangGraph: ✗ Excel task failed: {result}")
            
            return {
                **state,
                "execution_results": execution_results,
                "excel_files": excel_files,
                "step": "excel_complete"
            }
                
        except Exception as e:
            error_msg = f"Excel task failed: {str(e)}"
            execution_results = state['execution_results'].copy()
            execution_results["failed_tasks"].append(f"excel_task_{len(execution_results['failed_tasks'])}")
            execution_results["details"][f"excel_task_{len(execution_results['failed_tasks'])}"] = {
                "status": "failed", 
                "message": error_msg
            }
            
            logger.error(f"LangGraph: ✗ Excel task failed with exception: {e}")
            return {
                **state,
                "execution_results": execution_results,
                "step": "excel_complete"
            }
    
    def _execute_database_task_node(self, state: AgentState) -> AgentState:
        """Execute Database agent task"""
        try:
            tasks = state.get('tasks', [])
            if not tasks:
                return {**state, "step": "database_complete"}
            
            task = tasks[0]
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
            result = self.database_agent.process_database_request(query, state['per_file_data'])
            
            execution_results = state['execution_results'].copy()
            
            if "successfully" in result.lower():
                execution_results["successful_tasks"].append(f"database_task_{len(execution_results['successful_tasks'])}")
                execution_results["details"][f"database_task_{len(execution_results['successful_tasks'])}"] = {
                    "status": "success",
                    "message": result,
                    "table_name": table_name,
                    "document_type": document_type,
                    "fields": fields
                }
                logger.info(f"LangGraph: ✓ Database task completed successfully")
            else:
                execution_results["failed_tasks"].append(f"database_task_{len(execution_results['failed_tasks'])}")
                execution_results["details"][f"database_task_{len(execution_results['failed_tasks'])}"] = {
                    "status": "failed",
                    "message": result
                }
                logger.error(f"LangGraph: ✗ Database task failed: {result}")
            
            return {
                **state,
                "execution_results": execution_results,
                "step": "database_complete"
            }
                
        except Exception as e:
            error_msg = f"Database task failed: {str(e)}"
            execution_results = state['execution_results'].copy()
            execution_results["failed_tasks"].append(f"database_task_{len(execution_results['failed_tasks'])}")
            execution_results["details"][f"database_task_{len(execution_results['failed_tasks'])}"] = {
                "status": "failed",
                "message": error_msg
            }
            
            logger.error(f"LangGraph: ✗ Database task failed with exception: {e}")
            return {
                **state,
                "execution_results": execution_results,
                "step": "database_complete"
            }
    
    def _execute_email_task_node(self, state: AgentState) -> AgentState:
        """Execute Email agent task"""
        try:
            tasks = state.get('tasks', [])
            if not tasks:
                return {**state, "step": "email_complete"}
            
            task = tasks[0]
            parameters = task.get("parameters", {})
            recipient = parameters.get("recipient")
            subject = parameters.get("subject", "Document Information")
            body = parameters.get("body", "Here is the requested information from your documents.")
            include_excel = parameters.get("include_excel", False)
            excel_fields = parameters.get("excel_fields", [])
            document_type = parameters.get("document_type")
            
            if not recipient:
                error_msg = "No recipient email address specified"
                execution_results = state['execution_results'].copy()
                execution_results["failed_tasks"].append(f"email_task_{len(execution_results['failed_tasks'])}")
                execution_results["details"][f"email_task_{len(execution_results['failed_tasks'])}"] = {
                    "status": "failed",
                    "message": error_msg
                }
                return {
                    **state,
                    "execution_results": execution_results,
                    "step": "email_complete"
                }
            
            # Build email query
            email_query = f"Send email to {recipient} with {body}"
            if include_excel:
                email_query += f" and include excel file with {', '.join(excel_fields) if excel_fields else 'all data'}"
            
            if document_type:
                email_query += f" for {document_type} documents"
            
            # Execute email operation
            result = self.email_agent.process_email_request(email_query, state['per_file_data'])
            
            execution_results = state['execution_results'].copy()
            
            if "successfully" in result.lower():
                execution_results["successful_tasks"].append(f"email_task_{len(execution_results['successful_tasks'])}")
                execution_results["details"][f"email_task_{len(execution_results['successful_tasks'])}"] = {
                    "status": "success",
                    "message": result,
                    "recipient": recipient,
                    "subject": subject,
                    "document_type": document_type
                }
                logger.info(f"LangGraph: ✓ Email task completed successfully")
            else:
                execution_results["failed_tasks"].append(f"email_task_{len(execution_results['failed_tasks'])}")
                execution_results["details"][f"email_task_{len(execution_results['failed_tasks'])}"] = {
                    "status": "failed",
                    "message": result
                }
                logger.error(f"LangGraph: ✗ Email task failed: {result}")
            
            return {
                **state,
                "execution_results": execution_results,
                "step": "email_complete"
            }
                
        except Exception as e:
            error_msg = f"Email task failed: {str(e)}"
            execution_results = state['execution_results'].copy()
            execution_results["failed_tasks"].append(f"email_task_{len(execution_results['failed_tasks'])}")
            execution_results["details"][f"email_task_{len(execution_results['failed_tasks'])}"] = {
                "status": "failed",
                "message": error_msg
            }
            
            logger.error(f"LangGraph: ✗ Email task failed with exception: {e}")
            return {
                **state,
                "execution_results": execution_results,
                "step": "email_complete"
            }
    
    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the final response for the user"""
        execution_results = state['execution_results']
        successful_count = len(execution_results["successful_tasks"])
        failed_count = len(execution_results["failed_tasks"])
        total_tasks = successful_count + failed_count
        
        # Build summary message
        if successful_count == total_tasks and total_tasks > 0:
            summary = f"✅ All {total_tasks} tasks completed successfully!"
            overall_status = "success"
        elif successful_count > 0:
            summary = f"⚠️ {successful_count} out of {total_tasks} tasks completed successfully. {failed_count} tasks failed."
            overall_status = "partial_success"
        else:
            summary = f"❌ All {total_tasks} tasks failed."
            overall_status = "failed"
        
        # Build detailed breakdown
        details = f"**Query:** {state['query']}\n\n"
        details += f"**Summary:** {summary}\n\n"
        details += "**Task Breakdown:**\n"
        
        for task_id, task_result in execution_results["details"].items():
            status_icon = "✅" if task_result.get('status') == 'success' else "❌"
            task_description = f"Task {task_id}"
            if 'description' in state.get('tasks', [{}])[0]:
                task_description = state['tasks'][0]['description']
            
            details += f"\n{status_icon} **{task_description}**\n"
            details += f"   - Status: {task_result.get('status', 'unknown')}\n"
            details += f"   - Result: {task_result.get('message', 'No result')}\n"
        
        return {
            **state,
            "status": overall_status,
            "message": summary,
            "details": details,
            "step": "complete"
        }
    
    def process_unified_request(self, query: str, per_file_data: Dict) -> Dict:
        """Process unified request using LangGraph workflow"""
        logger.info(f"LangGraph: Processing unified request: {query}")
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            per_file_data=per_file_data,
            tasks=[],
            current_task=None,
            execution_results={
                "successful_tasks": [],
                "failed_tasks": [],
                "overall_status": "success",
                "details": {}
            },
            excel_files={},
            status="pending",
            message="",
            step="start"
        )
        
        # Execute the graph
        try:
            config = {"configurable": {"thread_id": f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "status": final_state.get("status", "unknown"),
                "summary": final_state.get("message", ""),
                "details": final_state.get("details", ""),
                "execution_details": final_state.get("execution_results", {})
            }
            
        except Exception as e:
            logger.error(f"LangGraph: Error executing workflow: {e}")
            return {
                "status": "failed",
                "summary": f"Error processing request: {str(e)}",
                "details": "The LangGraph workflow encountered an error.",
                "execution_details": {}
            }
