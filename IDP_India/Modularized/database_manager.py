import psycopg2
from psycopg2 import sql
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import json
import re
from dotenv import load_dotenv
import os


load_dotenv()

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

class DatabaseManager:
    """Manages PostgreSQL database operations - Direct table storage from Excel"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            print("Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def sanitize_table_name(self, doc_type: str) -> str:
        """Convert document type to valid PostgreSQL table name"""
        # Remove special characters, replace spaces/slashes with underscores
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', doc_type.lower())
        # Remove consecutive underscores
        table_name = re.sub(r'_+', '_', table_name)
        # Remove leading/trailing underscores
        table_name = table_name.strip('_')
        # Ensure it doesn't start with a number
        if table_name and table_name[0].isdigit():
            table_name = 'doc_' + table_name
        # Default if empty
        if not table_name:
            table_name = 'unknown_document'
        return table_name
    
    def sanitize_column_name(self, col_name: str) -> str:
        """Convert column name to valid PostgreSQL column name"""
        # Replace spaces and special chars with underscores
        col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_name.lower())
        col_name = re.sub(r'_+', '_', col_name).strip('_')
        # Ensure it doesn't start with a number
        if col_name and col_name[0].isdigit():
            col_name = 'col_' + col_name
        if not col_name:
            col_name = 'unnamed_column'
        # Avoid PostgreSQL reserved words
        reserved = ['user', 'table', 'order', 'group', 'index', 'select', 'where']
        if col_name in reserved:
            col_name = col_name + '_value'
        return col_name
    
    def generate_row_hash(self, row_data: Dict) -> str:
        """Generate unique hash for a row to detect duplicates"""
        # Create hash from all column values
        content = json.dumps(row_data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> bool:
        """Create table with proper schema based on DataFrame"""
        try:
            # Sanitize column names
            df.columns = [self.sanitize_column_name(col) for col in df.columns]
            
            # Build CREATE TABLE statement
            columns_def = []
            columns_def.append("id SERIAL PRIMARY KEY")
            columns_def.append("row_hash VARCHAR(64) UNIQUE NOT NULL")  # For duplicate detection
            columns_def.append("inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            
            # Add columns based on DataFrame dtypes
            for col in df.columns:
                dtype = df[col].dtype
                
                if pd.api.types.is_integer_dtype(dtype):
                    pg_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    pg_type = "DOUBLE PRECISION"
                elif pd.api.types.is_bool_dtype(dtype):
                    pg_type = "BOOLEAN"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    pg_type = "TIMESTAMP"
                else:
                    pg_type = "TEXT"
                
                columns_def.append(f"{col} {pg_type}")
            
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns_def)}
                )
            """
            
            self.cursor.execute(create_sql)
            
            # Create index on row_hash for faster duplicate checking
            index_sql = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_row_hash ON {table_name}(row_hash)"
            self.cursor.execute(index_sql)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating table {table_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def row_exists(self, table_name: str, row_hash: str) -> bool:
        """Check if row with given hash already exists in table"""
        try:
            query = sql.SQL("SELECT 1 FROM {} WHERE row_hash = %s LIMIT 1").format(
                sql.Identifier(table_name)
            )
            self.cursor.execute(query, (row_hash,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking row existence: {e}")
            return False
    
    def insert_dataframe_to_table(self, table_name: str, df: pd.DataFrame) -> Dict:
        """Insert DataFrame rows into table, skipping duplicates"""
        try:
            # Sanitize column names to match table
            df.columns = [self.sanitize_column_name(col) for col in df.columns]
            
            inserted_count = 0
            skipped_count = 0
            
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                
                # Generate hash for duplicate detection
                row_hash = self.generate_row_hash(row_dict)
                
                # Check if row already exists
                if self.row_exists(table_name, row_hash):
                    skipped_count += 1
                    continue
                
                # Prepare insert statement
                columns = list(row_dict.keys())
                values = [row_dict[col] for col in columns]
                
                # Add row_hash
                columns.insert(0, 'row_hash')
                values.insert(0, row_hash)
                
                # Build INSERT query
                insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(map(sql.Identifier, columns)),
                    sql.SQL(', ').join(sql.Placeholder() * len(values))
                )
                
                # Handle None values and special types
                clean_values = []
                for val in values:
                    if pd.isna(val):
                        clean_values.append(None)
                    elif isinstance(val, (list, dict)):
                        clean_values.append(json.dumps(val))
                    else:
                        clean_values.append(val)
                
                self.cursor.execute(insert_sql, clean_values)
                inserted_count += 1
            
            self.conn.commit()
            
            return {
                'inserted': inserted_count,
                'skipped': skipped_count,
                'total': len(df)
            }
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting data into {table_name}: {e}")
            import traceback
            traceback.print_exc()
            return {'inserted': 0, 'skipped': 0, 'total': len(df), 'error': str(e)}
    
    def save_excel_to_database(self, per_file_data: Dict, excel_files: Dict[str, Dict]) -> Dict:
        """
        Save Excel data directly to database tables
        
        Args:
            per_file_data: Original extracted data (for reference)
            excel_files: Dict with document types and their Excel file info
                        Format: {doc_type: {'filepath': path, 'count': n, 'filename': name}}
        
        Returns:
            Dictionary with results per document type
        """
        results = {}
        
        for doc_type, file_info in excel_files.items():
            try:
                # Read the Excel file
                excel_path = file_info['filepath']
                df = pd.read_excel(excel_path)
                
                if df.empty:
                    print(f"⚠️ Skipping {doc_type}: Empty DataFrame")
                    continue
                
                # Create sanitized table name
                table_name = self.sanitize_table_name(doc_type)
                
                print(f"\n📊 Processing {doc_type} → Table: {table_name}")
                
                # Create table if doesn't exist
                if not self.create_table_from_dataframe(table_name, df):
                    results[doc_type] = {
                        'success': False,
                        'error': 'Failed to create table'
                    }
                    continue
                
                # Insert data
                insert_result = self.insert_dataframe_to_table(table_name, df)
                
                results[doc_type] = {
                    'success': True,
                    'table_name': table_name,
                    'inserted': insert_result['inserted'],
                    'skipped': insert_result['skipped'],
                    'total': insert_result['total']
                }
                
                print(f"{doc_type}: Inserted {insert_result['inserted']}, "
                      f"Skipped {insert_result['skipped']} duplicates")
                
            except Exception as e:
                print(f"Error processing {doc_type}: {e}")
                results[doc_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a table"""
        try:
            # Get row count
            count_sql = sql.SQL("SELECT COUNT(*) FROM {}").format(
                sql.Identifier(table_name)
            )
            self.cursor.execute(count_sql)
            row_count = self.cursor.fetchone()[0]
            
            # Get column info
            self.cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = [{'name': row[0], 'type': row[1]} for row in self.cursor.fetchall()]
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': columns
            }
            
        except Exception as e:
            print(f"Error getting table info: {e}")
            return None
    
    def list_all_tables(self) -> List[str]:
        """List all document tables in the database"""
        try:
            self.cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                AND tablename NOT LIKE 'pg_%'
                AND tablename NOT LIKE 'sql_%'
                ORDER BY tablename
            """)
            
            tables = [row[0] for row in self.cursor.fetchall()]
            return tables
            
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []
    
    def get_all_statistics(self) -> Dict:
        """Get statistics for all tables"""
        try:
            tables = self.list_all_tables()
            stats = {
                'total_tables': len(tables),
                'tables': {}
            }
            
            for table in tables:
                info = self.get_table_info(table)
                if info:
                    stats['tables'][table] = info
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}


def save_to_database(per_file_data: Dict, excel_files: Dict[str, Dict]) -> Dict:
    """
    Save Excel data directly to PostgreSQL database as tables
    
    Args:
        per_file_data: Original extracted data
        excel_files: Dict from save_structured_data_to_excel() function
                    Format: {doc_type: {'filepath': path, 'count': n, 'filename': name}}
    
    Returns:
        Dictionary with results
    """
    db = DatabaseManager()
    
    if not db.connect():
        return {"success": False, "error": "Database connection failed"}
    
    try:
        # Save all Excel files to database
        results = db.save_excel_to_database(per_file_data, excel_files)
        
        # Get overall statistics
        stats = db.get_all_statistics()
        
        # Count successes
        successful = sum(1 for r in results.values() if r.get('success'))
        total_inserted = sum(r.get('inserted', 0) for r in results.values() if r.get('success'))
        total_skipped = sum(r.get('skipped', 0) for r in results.values() if r.get('success'))
        
        return {
            "success": True,
            "tables_processed": len(results),
            "tables_successful": successful,
            "total_rows_inserted": total_inserted,
            "total_rows_skipped": total_skipped,
            "details": results,
            "statistics": stats
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()