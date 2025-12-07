import psycopg2
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
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

# Database configuration
DB_CONFIG = {
    "DB_HOST": os.getenv("DB_HOST"),
    "DB_PORT": os.getenv("DB_PORT"),
    "DB_NAME": os.getenv("DB_NAME"),
    "DB_USER": os.getenv("DB_USER"),
    "DB_PASSWORD": os.getenv("DB_PASSWORD"),
}

class DatabaseAgent:
    """
    Database Agent for saving extracted document data to PostgreSQL database
    Can handle natural language queries to save specific data to tables
    """
    
    def __init__(self):
        self.db_config = DB_CONFIG
        self._ensure_database_tables()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_config["DB_HOST"],
                port=self.db_config["DB_PORT"],
                database=self.db_config["DB_NAME"],
                user=self.db_config["DB_USER"],
                password=self.db_config["DB_PASSWORD"]
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _ensure_database_tables(self):
        """Ensure required database tables exist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create document_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(500) NOT NULL,
                    document_type VARCHAR(200),
                    confidence FLOAT,
                    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document_data table for storing structured data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_data (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES document_metadata(id),
                    field_name VARCHAR(300) NOT NULL,
                    field_value TEXT,
                    field_type VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document_tables for custom table storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_tables (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(200) NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Database tables ensured successfully")
            
        except Exception as e:
            logger.error(f"Error ensuring database tables: {e}")
    
    def extract_database_info_from_query(self, query: str, per_file_data: Dict) -> Dict:
        """Extract database saving information from user query using LLM"""
        logger.info(f"Starting database info extraction for query: {query}")
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            # Prepare document summary for context
            document_summary = self._prepare_document_summary(per_file_data)
            
            prompt = f"""
            You are an expert at parsing user queries for database operations.
            
            User Query: "{query}"
            
            Available data summary:
            {document_summary}
            
            Your task is to extract the following information from the query:
            1. Target table name (if specified, otherwise generate appropriate name)
            2. Data filters (which data to include based on type or category)
            3. Data scope (what specific data to save - full data, specific fields, etc.)
            4. Operation type (save, update, create table, etc.)
            
            Important: If the query mentions specific document types (like invoice, purchase order, contract, etc.), 
            include appropriate filters. If the query mentions specific fields that are typically associated with 
            certain document types, infer the document type from context.
            
            Examples:
            - "Save the full extracted information in table" → 
              {{"table_name": "extracted_data", "filters": {{}}, "data_scope": "full", "operation": "save"}}
            
            - "Save all invoice data to database" → 
              {{"table_name": "invoices", "filters": {{"type": "invoice"}}, "data_scope": "full", "operation": "save"}}
            
            - "Save only invoice numbers and amounts to database" → 
              {{"table_name": "invoice_summary", "filters": {{"type": "invoice"}}, "data_scope": "specific", "fields": ["invoice_number", "amount"], "operation": "save"}}
            
            - "Save purchase order numbers and dates" → 
              {{"table_name": "purchase_orders", "filters": {{"type": "purchase order"}}, "data_scope": "specific", "fields": ["po_number", "date"], "operation": "save"}}
            
            - "Save only numbers and amounts to database" → 
              {{"table_name": "summary_data", "filters": {{}}, "data_scope": "specific", "fields": ["number", "amount"], "operation": "save"}}
            
            - "Create table with key details" → 
              {{"table_name": "key_details", "filters": {{}}, "data_scope": "key_fields", "operation": "create_table"}}
            
            - "Save all metadata to database" → 
              {{"table_name": "metadata", "filters": {{}}, "data_scope": "metadata_only", "operation": "save"}}
            
            Return ONLY a JSON object with these keys: table_name, filters, data_scope, operation, fields (if data_scope is specific)
            """
            
            logger.info("Sending LLM request for database info extraction")
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response received: {result_text[:200]}...")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                db_info = json.loads(json_match.group())
                logger.info(f"Successfully extracted database info: table_name={db_info.get('table_name')}, operation={db_info.get('operation')}")
                
                # Post-process to add document type filters based on field context
                db_info = self._add_document_type_filters(db_info, query)
                
                return db_info
            else:
                logger.warning("Failed to parse JSON from LLM response, using fallback extraction")
                return self._fallback_database_extraction(query)
                
        except Exception as e:
            logger.error(f"LLM database extraction error: {e}")
            return self._fallback_database_extraction(query)
    
    def _add_document_type_filters(self, db_info: Dict, query: str) -> Dict:
        """Add document type filters based on field context in the query"""
        if not db_info.get('filters'):
            db_info['filters'] = {}
        
        # If no document type filter is present, try to infer from fields
        if not db_info['filters'].get('type') and not db_info['filters'].get('document_types'):
            fields = db_info.get('fields', [])
            query_lower = query.lower()
            
            # Infer document type from field names - prioritize invoice over other types
            if any('invoice' in field.lower() for field in fields) or 'invoice' in query_lower:
                db_info['filters']['type'] = 'invoice'
                logger.info(f"Inferred document type 'invoice' from fields: {fields}")
            elif any('po' in field.lower() for field in fields) or 'purchase order' in query_lower:
                db_info['filters']['type'] = 'purchase order'
                logger.info(f"Inferred document type 'purchase order' from fields: {fields}")
            elif any('contract' in field.lower() for field in fields) or 'contract' in query_lower:
                db_info['filters']['type'] = 'contract'
                logger.info(f"Inferred document type 'contract' from fields: {fields}")
            elif any('statement' in field.lower() for field in fields) or 'statement' in query_lower:
                db_info['filters']['type'] = 'statement'
                logger.info(f"Inferred document type 'statement' from fields: {fields}")
        
        # Ensure filter value is a string, not a list
        if db_info['filters'].get('type') and isinstance(db_info['filters']['type'], list):
            # If multiple types are detected, use the first one (prioritize invoice)
            if 'invoice' in db_info['filters']['type']:
                db_info['filters']['type'] = 'invoice'
            else:
                db_info['filters']['type'] = db_info['filters']['type'][0]
            logger.info(f"Converted filter type from list to string: {db_info['filters']['type']}")
        
        return db_info
    
    def _fallback_database_extraction(self, query: str) -> Dict:
        """Fallback database information extraction when LLM fails"""
        query_lower = query.lower()
        
        # Basic extraction logic
        table_name = "documents"
        filters = {}
        data_scope = "full"
        operation = "save"
        
        # Look for document types in query
        for doc_type in ['invoice', 'purchase order', 'contract', 'report', 'statement']:
            if doc_type in query_lower:
                table_name = f"{doc_type.replace(' ', '_')}s"
                filters = {"document_types": [doc_type]}
                break
        
        # Look for specific fields
        if 'number' in query_lower and 'amount' in query_lower:
            data_scope = "specific"
            fields = ["number", "amount"]
        elif 'metadata' in query_lower:
            data_scope = "metadata_only"
        
        return {
            "table_name": table_name,
            "filters": filters,
            "data_scope": data_scope,
            "operation": operation,
            "fields": fields if data_scope == "specific" else []
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
        important_fields = ['number', 'date', 'amount', 'total', 'due_date', 'customer', 'vendor', 'name', 'address']
        
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
    
    def filter_documents_by_type(self, per_file_data: Dict, document_type: str) -> Dict:
        """Filter documents by document type with flexible matching"""
        filtered_data = {}
        for file_name, file_data in per_file_data.items():
            classification = file_data.get('classification', {})
            actual_doc_type = classification.get('document_type', '').lower()
            requested_doc_type = document_type.lower()
            
            # Flexible matching: check if requested type is contained in actual type or vice versa
            if (requested_doc_type in actual_doc_type or 
                actual_doc_type in requested_doc_type or
                requested_doc_type == actual_doc_type):
                filtered_data[file_name] = file_data
        return filtered_data
    
    def save_data_to_database(self, db_info: Dict, per_file_data: Dict) -> str:
        """Save document data to database based on extracted information"""
        logger.info(f"Saving data to database with info: {db_info}")
        
        try:
            # Filter documents based on database info
            filtered_data = self._filter_documents_for_database(per_file_data, db_info.get('filters', {}))
            logger.info(f"Filtered {len(filtered_data)} documents for database saving")
            
            if not filtered_data:
                return "No documents found matching your criteria."
            
            # Determine saving strategy based on data_scope
            data_scope = db_info.get('data_scope', 'full')
            table_name = db_info.get('table_name', 'documents')
            
            if data_scope == 'metadata_only':
                result = self._save_metadata_only(filtered_data)
            elif data_scope == 'specific':
                fields = db_info.get('fields', [])
                result = self._save_specific_fields(filtered_data, table_name, fields)
            else:  # full or key_fields
                result = self._save_full_data(filtered_data, table_name, data_scope)
            
            logger.info(f"Database save operation completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            return f"Error saving data to database: {str(e)}"
    
    def _filter_documents_for_database(self, per_file_data: Dict, filters: Dict) -> Dict:
        """Filter documents based on database filters with flexible matching"""
        if not filters:
            return per_file_data
        
        filtered_data = {}
        
        for file_name, file_data in per_file_data.items():
            include_doc = True
            
            # Handle different filter formats from LLM
            # Case 1: filters with 'document_types' key
            if 'document_types' in filters:
                doc_type = file_data.get('classification', {}).get('document_type', '').lower()
                requested_types = [t.lower() for t in filters['document_types']]
                
                # Check if document type matches any requested type
                type_match = any(req_type in doc_type or doc_type in req_type 
                               for req_type in requested_types)
                
                if not type_match:
                    include_doc = False
            
            # Case 2: filters with 'type' key (from LLM response)
            elif 'type' in filters:
                doc_type = file_data.get('classification', {}).get('document_type', '').lower()
                requested_type = filters['type'].lower()
                
                # Check if document type matches the requested type
                type_match = (requested_type in doc_type or 
                            doc_type in requested_type or
                            requested_type == doc_type)
                
                if not type_match:
                    include_doc = False
            
            # Case 3: filters with other keys - apply generic matching
            elif filters:
                doc_type = file_data.get('classification', {}).get('document_type', '').lower()
                # Check if any filter value matches the document type
                type_match = any(filter_value.lower() in doc_type or 
                               doc_type in filter_value.lower()
                               for filter_value in filters.values() 
                               if isinstance(filter_value, str))
                
                if not type_match:
                    include_doc = False
            
            if include_doc:
                filtered_data[file_name] = file_data
        
        # If no documents match the specified filters, use all available documents
        if not filtered_data and filters:
            logger.warning(f"No documents found matching filters {filters}, using all available documents")
            filtered_data = per_file_data
        
        return filtered_data
    
    def _save_metadata_only(self, filtered_data: Dict) -> str:
        """Save only document metadata to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            saved_count = 0
            for file_name, file_data in filtered_data.items():
                classification = file_data.get('classification', {})
                
                # Insert into document_metadata
                cursor.execute("""
                    INSERT INTO document_metadata (filename, document_type, confidence)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (
                    file_name,
                    classification.get('document_type', 'Unknown'),
                    classification.get('confidence', 0)
                ))
                
                document_id = cursor.fetchone()[0]
                saved_count += 1
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return f"Successfully saved metadata for {saved_count} documents to database."
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return f"Error saving metadata: {str(e)}"
    
    def _save_specific_fields(self, filtered_data: Dict, table_name: str, fields: List[str]) -> str:
        """Save specific fields to a custom table"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create custom table if it doesn't exist
            self._create_custom_table(cursor, table_name, fields)
            
            saved_count = 0
            for file_name, file_data in filtered_data.items():
                classification = file_data.get('classification', {})
                structured_data = file_data.get('structured_data', {})
                
                # Extract specific fields
                field_values = self._extract_specific_fields(structured_data, fields)
                
                # Prepare insert statement
                placeholders = ', '.join(['%s'] * (len(fields) + 2))  # +2 for filename and document_type
                field_names = ['filename', 'document_type'] + fields
                field_names_str = ', '.join(field_names)
                
                values = [file_name, classification.get('document_type', 'Unknown')] + field_values
                
                cursor.execute(f"""
                    INSERT INTO {table_name} ({field_names_str})
                    VALUES ({placeholders})
                """, values)
                
                saved_count += 1
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return f"Successfully saved {saved_count} documents with specific fields to table '{table_name}'."
            
        except Exception as e:
            logger.error(f"Error saving specific fields: {e}")
            return f"Error saving specific fields: {str(e)}"
    
    def _save_full_data(self, filtered_data: Dict, table_name: str, data_scope: str) -> str:
        """Save full document data to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First, collect all unique fields from all documents
            all_unique_fields = {}
            for file_name, file_data in filtered_data.items():
                structured_data = file_data.get('structured_data', {})
                document_fields = self._extract_all_fields(structured_data)
                # Merge fields properly - don't overwrite, just collect unique field names
                for field_name in document_fields.keys():
                    all_unique_fields[field_name] = None  # We only care about field names, not values
            
            logger.info(f"Collected {len(all_unique_fields)} unique fields from {len(filtered_data)} documents")
            
            # Create comprehensive table for full data
            if data_scope == 'key_fields':
                # Create table with common important fields
                self._create_key_fields_table(cursor, table_name)
            else:
                # Create dynamic table with all unique fields
                self._create_dynamic_table(cursor, table_name, all_unique_fields)
            
            saved_count = 0
            for file_name, file_data in filtered_data.items():
                classification = file_data.get('classification', {})
                structured_data = file_data.get('structured_data', {})
                
                if data_scope == 'key_fields':
                    self._save_key_fields_data(cursor, table_name, file_name, classification, structured_data)
                else:
                    self._save_flexible_data(cursor, table_name, file_name, classification, structured_data, all_unique_fields)
                
                saved_count += 1
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return f"Successfully saved {saved_count} documents with full data to table '{table_name}'."
            
        except Exception as e:
            logger.error(f"Error saving full data: {e}")
            return f"Error saving full data: {str(e)}"
    
    def _create_custom_table(self, cursor, table_name: str, fields: List[str]):
        """Create custom table for specific fields"""
        # Basic columns
        columns = [
            "id SERIAL PRIMARY KEY",
            "filename VARCHAR(500) NOT NULL",
            "document_type VARCHAR(200)",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        # Add custom fields as TEXT columns
        for field in fields:
            columns.append(f"{field.replace(' ', '_').lower()} TEXT")
        
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
        """
        
        cursor.execute(create_table_sql)
        
        # Register table in document_tables
        cursor.execute("""
            INSERT INTO document_tables (table_name, description)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO NOTHING
        """, (table_name, f"Table for storing {', '.join(fields)} fields"))
    
    def _create_key_fields_table(self, cursor, table_name: str):
        """Create table with common key fields"""
        columns = [
            "id SERIAL PRIMARY KEY",
            "filename VARCHAR(500) NOT NULL",
            "document_type VARCHAR(200)",
            "document_number TEXT",
            "document_date TEXT",
            "amount TEXT",
            "total TEXT",
            "due_date TEXT",
            "customer_name TEXT",
            "vendor_name TEXT",
            "address TEXT",
            "additional_data JSONB",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
        """
        
        cursor.execute(create_table_sql)
        
        # Register table in document_tables
        cursor.execute("""
            INSERT INTO document_tables (table_name, description)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO NOTHING
        """, (table_name, "Table for storing key document fields"))
    
    def _create_flexible_table(self, cursor, table_name: str):
        """Create flexible table for full document data"""
        columns = [
            "id SERIAL PRIMARY KEY",
            "filename VARCHAR(500) NOT NULL",
            "document_type VARCHAR(200)",
            "confidence FLOAT",
            "structured_data JSONB",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
        """
        
        cursor.execute(create_table_sql)
        
        # Register table in document_tables
        cursor.execute("""
            INSERT INTO document_tables (table_name, description)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO NOTHING
        """, (table_name, "Table for storing full document data"))
    
    def _extract_specific_fields(self, structured_data: Dict, fields: List[str]) -> List[str]:
        """Extract specific field values from structured data"""
        field_values = []
        
        for field in fields:
            value = self._find_field_value(structured_data, field)
            field_values.append(value if value != "Not found" else None)
        
        return field_values
    
    def _save_key_fields_data(self, cursor, table_name: str, filename: str, classification: Dict, structured_data: Dict):
        """Save key fields data to table"""
        # Extract common key fields
        document_number = self._find_field_value(structured_data, 'number')
        document_date = self._find_field_value(structured_data, 'date')
        amount = self._find_field_value(structured_data, 'amount')
        total = self._find_field_value(structured_data, 'total')
        due_date = self._find_field_value(structured_data, 'due_date')
        customer_name = self._find_field_value(structured_data, 'customer')
        vendor_name = self._find_field_value(structured_data, 'vendor')
        address = self._find_field_value(structured_data, 'address')
        
        # Store additional data as JSON
        additional_data = {}
        for key, value in structured_data.items():
            if key not in ['number', 'date', 'amount', 'total', 'due_date', 'customer', 'vendor', 'address']:
                additional_data[key] = value
        
        cursor.execute(f"""
            INSERT INTO {table_name} (
                filename, document_type, document_number, document_date, 
                amount, total, due_date, customer_name, vendor_name, address, additional_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            filename,
            classification.get('document_type', 'Unknown'),
            document_number if document_number != "Not found" else None,
            document_date if document_date != "Not found" else None,
            amount if amount != "Not found" else None,
            total if total != "Not found" else None,
            due_date if due_date != "Not found" else None,
            customer_name if customer_name != "Not found" else None,
            vendor_name if vendor_name != "Not found" else None,
            address if address != "Not found" else None,
            json.dumps(additional_data) if additional_data else None
        ))
    
    def _save_flexible_data(self, cursor, table_name: str, filename: str, classification: Dict, structured_data: Dict, all_unique_fields: Dict = None):
        """Save flexible full data to table with separate columns for each field"""
        # Extract all fields from structured data
        document_fields = self._extract_all_fields(structured_data)
        
        # Use all_unique_fields if provided (for multi-document consistency), otherwise use document_fields
        if all_unique_fields is None:
            all_unique_fields = document_fields
        
        # Prepare insert statement with all unique fields
        field_names = ['filename', 'document_type', 'confidence'] + list(all_unique_fields.keys())
        placeholders = ', '.join(['%s'] * len(field_names))
        field_names_str = ', '.join(field_names)
        
        # Prepare values
        values = [
            filename,
            classification.get('document_type', 'Unknown'),
            classification.get('confidence', 0)
        ]
        
        # Add values for each field - use document value if exists, otherwise None
        for field_name in all_unique_fields.keys():
            if field_name in document_fields:
                values.append(document_fields[field_name])
            else:
                values.append(None)  # Use NULL for missing fields
        
        cursor.execute(f"""
            INSERT INTO {table_name} ({field_names_str})
            VALUES ({placeholders})
        """, values)
    
    def _extract_all_fields(self, data: Dict, prefix: str = "") -> Dict:
        """Extract all fields from structured data as flat dictionary"""
        fields = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}{key}" if prefix else key
                # Sanitize field name for PostgreSQL - replace invalid characters with underscores
                full_key = self._sanitize_field_name(full_key)
                
                if isinstance(value, dict):
                    # Recursively extract nested fields
                    nested_fields = self._extract_all_fields(value, f"{full_key}_")
                    fields.update(nested_fields)
                elif isinstance(value, list):
                    # Handle lists by converting to string
                    if all(isinstance(item, (str, int, float)) for item in value):
                        fields[full_key] = ", ".join(str(item) for item in value)
                    else:
                        # For complex lists, store as JSON string
                        fields[full_key] = json.dumps(value)
                elif isinstance(value, (str, int, float)):
                    # Store simple values directly
                    fields[full_key] = str(value)
                else:
                    # Convert other types to string
                    fields[full_key] = str(value)
        
        return fields
    
    def _sanitize_field_name(self, field_name: str) -> str:
        """Sanitize field name for PostgreSQL column names"""
        # Replace spaces, hyphens, and other invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', field_name.lower())
        
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading and trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it starts with a letter or underscore
        if not sanitized or sanitized[0].isdigit():
            sanitized = 'field_' + sanitized
        
        # Ensure reasonable length (PostgreSQL limit is 63 chars, but we'll use 50 for safety)
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized
    
    def _create_dynamic_table(self, cursor, table_name: str, fields: Dict):
        """Create dynamic table with columns for each field"""
        # Basic columns
        columns = [
            "id SERIAL PRIMARY KEY",
            "filename VARCHAR(500) NOT NULL",
            "document_type VARCHAR(200)",
            "confidence FLOAT",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        # Add dynamic columns for each field
        for field_name in fields.keys():
            # Use TEXT for all dynamic fields to handle various data types
            columns.append(f"{field_name} TEXT")
        
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
        """
        
        cursor.execute(create_table_sql)
        
        # Register table in document_tables
        cursor.execute("""
            INSERT INTO document_tables (table_name, description)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO NOTHING
        """, (table_name, f"Table with {len(fields)} dynamic columns for extracted data"))
    
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
                    if result != "Not found":
                        return result
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result = self._find_field_value(item, field_name)
                            if result != "Not found":
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
    
    def process_database_request(self, query: str, per_file_data: Dict) -> str:
        """Process complete database request from user query"""
        logger.info(f"Processing database request: {query}")
        
        # Extract database information from query
        db_info = self.extract_database_info_from_query(query, per_file_data)
        
        if not db_info.get('table_name'):
            return "Could not determine table name from your query. Please specify what data you want to save."
        
        # Save data to database
        result = self.save_data_to_database(db_info, per_file_data)
        
        return result
    
    def get_table_info(self) -> List[Dict]:
        """Get information about all document tables in database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT table_name, description, created_at 
                FROM document_tables 
                ORDER BY created_at DESC
            """)
            
            tables = []
            for row in cursor.fetchall():
                tables.append({
                    'table_name': row[0],
                    'description': row[1],
                    'created_at': row[2]
                })
            
            cursor.close()
            conn.close()
            
            return tables
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return []
    
    def get_table_data(self, table_name: str, limit: int = 10) -> List[Dict]:
        """Get sample data from a specific table"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT %s", (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            data = []
            
            for row in cursor.fetchall():
                row_data = {}
                for i, column in enumerate(columns):
                    row_data[column] = row[i]
                data.append(row_data)
            
            cursor.close()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting table data: {e}")
            return []
    
    def clear_database(self) -> str:
        """Clear all document tables and data from the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get all document tables
            cursor.execute("""
                SELECT table_name FROM document_tables
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            # Drop all document tables
            for table_name in tables:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped table: {table_name}")
                except Exception as e:
                    logger.error(f"Error dropping table {table_name}: {e}")
            
            # Clear document_tables registry
            cursor.execute("DELETE FROM document_tables")
            
            # Clear document_data table
            cursor.execute("DELETE FROM document_data")
            
            # Clear document_metadata table
            cursor.execute("DELETE FROM document_metadata")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Database cleared successfully")
            return f"Successfully cleared database. Removed {len(tables)} tables and all document data."
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return f"Error clearing database: {str(e)}"
