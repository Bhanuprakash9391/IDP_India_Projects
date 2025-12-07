# Intelligent Document Processor with Excel Agent

A powerful document processing system that automatically classifies documents, extracts structured data, and provides intelligent Excel export capabilities through an integrated Excel Agent.

## Features

### Core IDP Functionality
- **Automatic Document Classification**: AI-powered document type identification (Invoice, Report, Agreement, etc.)
- **Structured Data Extraction**: Intelligent field extraction from documents
- **Multi-Database Architecture**: Separate vector stores for text and images
- **Smart Q&A**: Query across all processed documents
- **Visual Caching**: Efficient processing with result caching

### Excel Agent Features
- **Query-Based Excel Creation**: Create Excel files based on natural language queries
- **Document Type Filtering**: Filter by specific document types (invoices, reports, etc.)
- **Intelligent Field Extraction**: Automatically extract and map fields from structured data
- **Comprehensive Export**: Export all extracted data or specific fields only
- **Flexible Output**: Customizable Excel file names and field selection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_chat_deployment
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your_embeddings_deployment
```

## Usage

### 1. Process Documents
1. Upload PDF files or images through the sidebar
2. Provide a database name
3. Click "Process Documents" to analyze and extract data

### 2. Using the Excel Agent
1. Navigate to the "Excel Agent" tab
2. Enter your query in natural language:
   - "Create Excel with all invoice data"
   - "Create Excel with only filename and invoice number"
   - "Extract all purchase order information to Excel"
   - "Create Excel for reports with filename and document type"

3. (Optional) Filter by document type
4. (Optional) Specify output filename
5. Click "Create Excel from Query" or "Create Comprehensive Excel"

### 3. Download Results
- Download the generated Excel file directly from the interface
- Files are saved in the `exports/` directory

## Excel Agent Query Examples

### Simple Queries
- "Create Excel with all invoice data"
- "Extract purchase order information"
- "Create Excel with filename and document type"

### Specific Field Queries
- "Create Excel with only filename and invoice number"
- "Extract invoice date and amount to Excel"
- "Create Excel with vendor name and total amount"

### Document Type Specific
- "Create Excel for all invoices"
- "Extract report data to Excel"
- "Create Excel with agreement information"

## Supported Document Types

The Excel Agent can handle various document types including:
- Invoices
- Reports
- Agreements
- Certificates
- Resumes
- Purchase Orders
- Contracts
- Medical Records
- And more...

## File Structure

```
IDP_India_With_Agents/
├── app.py                    # Main application
├── excel_agent.py           # Excel Agent implementation
├── requirements.txt         # Dependencies
├── .env                    # Environment variables
├── vector_databases/       # Vector stores
├── structured_data/        # Extracted structured data
├── vision_cache/           # Vision analysis cache
├── logs/                   # Processing logs
└── exports/                # Generated Excel files
```

## Excel Agent Capabilities

### Intelligent Field Mapping
The Excel Agent automatically maps common field names:
- `invoice_number` → invoice_no, invoice_number, invoice_id, number
- `invoice_date` → date, invoice_date, document_date
- `amount` → total, amount, total_amount, invoice_amount
- `vendor` → vendor, supplier, company, from
- `customer` → customer, client, to, bill_to

### Data Flattening
- Automatically flattens nested JSON structures
- Handles arrays and complex data types
- Maintains data integrity and relationships

### Error Handling
- Graceful handling of missing fields
- Clear error messages for troubleshooting
- Fallback mechanisms for data extraction

## Running the Application

```bash
streamlit run app.py
```

## API Integration

The Excel Agent can be used programmatically:

```python
from excel_agent import ExcelAgent

# Initialize agent
agent = ExcelAgent()

# Create Excel from query
result = agent.create_excel_from_query(
    query="Create Excel with all invoice data",
    per_file_data=processed_data,
    output_path="exports/invoice_data.xlsx"
)

# Create comprehensive Excel
result = agent.create_comprehensive_excel(
    per_file_data=processed_data,
    output_path="exports/all_data.xlsx",
    document_type="Invoice"  # Optional filter
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
