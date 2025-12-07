import pandas as pd
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import re
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

class ExcelAgent:
    """
    Excel Agent for creating Excel files from extracted document data
    Can handle various document types and custom queries
    """
    
    def __init__(self, structured_data_dir: str = "structured_data"):
        self.structured_data_dir = structured_data_dir
    
    def load_structured_data(self, db_name: str) -> Dict:
        """Load structured data from the processed files"""
        per_file_path = os.path.join(self.structured_data_dir, f"{db_name}_files.json")
        if os.path.exists(per_file_path):
            with open(per_file_path, 'r') as f:
                return json.load(f)
        return {}
    
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
    
    def extract_fields_from_query(self, query: str) -> List[str]:
        """Extract field names from user query using LLM for intelligent parsing"""
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            prompt = f"""
            You are an expert at parsing user queries for document field extraction.
            
            User Query: "{query}"
            
            Your task is to identify the specific fields the user wants to extract from documents.
            
            Examples:
            - "Create Excel with filename, document number and reference number" → ["filename", "document number", "reference number"]
            - "Extract document date and amount" → ["document date", "amount"]
            - "Create Excel with only filename and document type" → ["filename", "document type"]
            - "Create Excel with all data" → []
            
            Important rules:
            1. Treat compound field names as single fields (e.g., "document number" not ["document", "number"])
            2. Always include "filename" if mentioned
            3. Return an empty list if the user wants "all data" or doesn't specify specific fields
            4. Return only the field names the user explicitly requested
            5. Return the fields in the order they appear in the query
            
            Return ONLY a JSON array of field names, or an empty array if no specific fields are requested.
            """
            
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', result_text)
            if json_match:
                fields = json.loads(json_match.group())
                return fields
            else:
                # Fallback to basic extraction if LLM fails
                return self._fallback_field_extraction(query)
                
        except Exception as e:
            print(f"LLM field extraction error: {e}")
            # Fallback to basic extraction
            return self._fallback_field_extraction(query)
    
    def _fallback_field_extraction(self, query: str) -> List[str]:
        """Fallback field extraction when LLM fails"""
        query_lower = query.lower()
        
        # Look for field lists after common patterns
        patterns = [
            r'create.*?with\s+(.*?)$',
            r'extract\s+(.*?)$',
            r'only\s+(.*?)$',
            r'include\s+(.*?)$'
        ]
        
        fields = []
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Split by common separators and clean up
                field_list = re.split(r'[,\s]+and\s+|\s*,\s*|\s+|\?', match)
                fields.extend([field.strip() for field in field_list if field.strip() and field not in ['', 'a', 'an', 'the']])
        
        return fields
    
    def flatten_structured_data(self, structured_data: Dict) -> Dict:
        """Flatten nested structured data for Excel export"""
        flattened = {}
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    flattened[f"{prefix}{key}"] = ", ".join(str(item) for item in value)
                else:
                    flattened[f"{prefix}{key}"] = str(value)
        
        flatten_dict(structured_data)
        return flattened
    
    def create_excel_from_query(self, query: str, per_file_data: Dict, output_path: str) -> str:
        """Create Excel file based on user query - completely generic approach"""
        try:
            # Extract fields from query
            fields = self.extract_fields_from_query(query)
            
            # Check if user is asking for specific document types
            query_lower = query.lower()
            requested_doc_type = None
            
            # Look for document type mentions in the query
            for doc_type in ['invoice', 'report', 'statement', 'agreement', 'certificate', 'resume', 'purchase order', 'contract']:
                if doc_type in query_lower:
                    requested_doc_type = doc_type
                    break
            
            # Filter documents by requested type if specified
            if requested_doc_type:
                filtered_data = self.filter_documents_by_type(per_file_data, requested_doc_type)
                if not filtered_data:
                    return f"No {requested_doc_type} documents found in the processed data."
                working_data = filtered_data
            else:
                working_data = per_file_data
            
            # Prepare data for Excel
            excel_data = []
            
            for file_name, file_data in working_data.items():
                row_data = {}
                
                # Always include filename
                row_data['filename'] = file_name
                
                # Include document type from classification metadata
                classification = file_data.get('classification', {})
                row_data['document_type'] = classification.get('document_type', 'Unknown')
                row_data['confidence'] = classification.get('confidence', 0)
                
                # Extract specific fields from structured data
                structured_data = file_data.get('structured_data', {})
                flattened_data = self.flatten_structured_data(structured_data)
                
                # If no specific fields requested, include all structured data
                if not fields:
                    row_data.update(flattened_data)
                else:
                    # Add requested fields
                    for field in fields:
                        field_lower = field.lower()
                        
                        # Handle special cases
                        if field_lower in ['filename', 'file_name']:
                            row_data[field] = file_name
                        elif field_lower in ['document_type', 'type']:
                            row_data[field] = classification.get('document_type', 'Unknown')
                        elif field_lower in ['confidence']:
                            row_data[field] = classification.get('confidence', 0)
                        else:
                            # Look for matching field in flattened data
                            found = False
                            for data_key, data_value in flattened_data.items():
                                if field_lower in data_key.lower() or data_key.lower() in field_lower:
                                    row_data[field] = data_value
                                    found = True
                                    break
                            
                            # If not found, try to extract from structured data using intelligent matching
                            if not found:
                                row_data[field] = self._extract_specific_field(structured_data, field)
                
                excel_data.append(row_data)
            
            if not excel_data:
                return "No data found matching your query criteria."
            
            # Create DataFrame and save to Excel
            df = pd.DataFrame(excel_data)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            doc_type_info = f" ({requested_doc_type} documents only)" if requested_doc_type else ""
            return f"Excel file created successfully at: {output_path}\n" \
                   f"Documents processed: {len(excel_data)}{doc_type_info}\n" \
                   f"Fields included: {', '.join(fields)}"
            
        except Exception as e:
            return f"Error creating Excel file: {str(e)}"
    
    def _extract_specific_field(self, structured_data: Dict, field: str) -> str:
        """Extract specific field from structured data using intelligent matching with LLM"""
        field_lower = field.lower()
        
        # Try LLM-powered field matching first for intelligent mapping
        llm_result = self._llm_field_mapping(field, structured_data)
        if llm_result and llm_result != "Not found":
            return llm_result
        
        # Try direct field matching as fallback
        return self._find_field_value(structured_data, field_lower, [field_lower])
    
    def _llm_field_mapping(self, user_field: str, structured_data: Dict) -> str:
        """Use LLM to intelligently map user field names to actual field names in structured data"""
        try:
            # Get all available field names from structured data
            available_fields = self._get_all_field_names(structured_data)
            
            if not available_fields:
                return "Not found"
            
            client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
            )
            
            prompt = f"""
            You are an expert at mapping user field requests to actual field names in document data.
            
            User is asking for: "{user_field}"
            
            Available field names in the structured data:
            {', '.join(sorted(available_fields))}
            
            Task: Find the best matching field from the available fields for what the user is asking for.
            Consider synonyms, abbreviations, and common variations.
            
            Return ONLY the exact field name from the available fields that best matches, or "Not found" if no good match exists.
            """
            
            response = client.chat.completions.create(
                model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate that the result is actually in our available fields
            if result != "Not found" and result in available_fields:
                # Now find the actual value for this field
                return self._find_field_value(structured_data, result, [result])
            else:
                return "Not found"
                
        except Exception as e:
            print(f"LLM field mapping error: {e}")
            return "Not found"
    
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
    
    def _find_field_value(self, data: Dict, primary_field: str, aliases: List[str]) -> str:
        """Find field value in structured data using primary field and aliases"""
        if isinstance(data, dict):
            # Check for exact match first
            for alias in aliases:
                if alias in data:
                    value = data[alias]
                    if isinstance(value, (str, int, float)):
                        return str(value)
                    elif isinstance(value, list):
                        return ", ".join(str(item) for item in value)
                    elif isinstance(value, dict):
                        return str(value)
            
            # Check nested structures
            for key, value in data.items():
                if isinstance(value, dict):
                    result = self._find_field_value(value, primary_field, aliases)
                    if result:
                        return result
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result = self._find_field_value(item, primary_field, aliases)
                            if result:
                                return result
        
        return "Not found"
    
    def create_comprehensive_excel(self, per_file_data: Dict, output_path: str, document_type: str = None) -> str:
        """Create comprehensive Excel files - separate files for different document types"""
        try:
            # Filter by document type if specified
            if document_type:
                filtered_data = self.filter_documents_by_type(per_file_data, document_type)
                if not filtered_data:
                    return f"No {document_type} documents found."
                
                # Prepare comprehensive data for single document type
                all_data = []
                
                for file_name, file_data in filtered_data.items():
                    row_data = {
                        'filename': file_name,
                        'document_type': file_data.get('classification', {}).get('document_type', 'Unknown'),
                        'confidence': file_data.get('classification', {}).get('confidence', 0)
                    }
                    
                    # Add all structured data (flattened)
                    structured_data = file_data.get('structured_data', {})
                    flattened_data = self.flatten_structured_data(structured_data)
                    row_data.update(flattened_data)
                    
                    all_data.append(row_data)
                
                if not all_data:
                    return "No data available for Excel export."
                
                # Create DataFrame and save to Excel
                df = pd.DataFrame(all_data)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save to Excel
                df.to_excel(output_path, index=False, engine='openpyxl')
                
                return f"Comprehensive Excel file created successfully at: {output_path}\n" \
                       f"Documents included: {len(all_data)}\n" \
                       f"Total fields: {len(df.columns)}"
            
            else:
                # Create separate Excel files for each document type
                document_types = {}
                
                for file_name, file_data in per_file_data.items():
                    doc_type = file_data.get('classification', {}).get('document_type', 'Unknown')
                    if doc_type not in document_types:
                        document_types[doc_type] = []
                    document_types[doc_type].append((file_name, file_data))
                
                created_files = []
                
                for doc_type, files_data in document_types.items():
                    # Create filename for this document type
                    base_name = os.path.splitext(output_path)[0]
                    extension = os.path.splitext(output_path)[1]
                    type_output_path = f"{base_name}_{doc_type.replace(' ', '_').lower()}{extension}"
                    
                    # Prepare data for this document type
                    all_data = []
                    
                    for file_name, file_data in files_data:
                        row_data = {
                            'filename': file_name,
                            'document_type': file_data.get('classification', {}).get('document_type', 'Unknown'),
                            'confidence': file_data.get('classification', {}).get('confidence', 0)
                        }
                        
                        # Add all structured data (flattened)
                        structured_data = file_data.get('structured_data', {})
                        flattened_data = self.flatten_structured_data(structured_data)
                        row_data.update(flattened_data)
                        
                        all_data.append(row_data)
                    
                    if all_data:
                        # Create DataFrame and save to Excel
                        df = pd.DataFrame(all_data)
                        
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(type_output_path), exist_ok=True)
                        
                        # Save to Excel
                        df.to_excel(type_output_path, index=False, engine='openpyxl')
                        created_files.append((doc_type, type_output_path, len(all_data), len(df.columns)))
                
                if not created_files:
                    return "No data available for Excel export."
                
                result = "Comprehensive Excel files created successfully:\n"
                for doc_type, file_path, doc_count, field_count in created_files:
                    result += f"- {doc_type}: {doc_count} documents, {field_count} fields → {file_path}\n"
                
                return result
            
        except Exception as e:
            return f"Error creating comprehensive Excel file: {str(e)}"
