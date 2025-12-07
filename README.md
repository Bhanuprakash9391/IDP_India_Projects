# IDP India Projects

A collection of Intelligent Document Processing (IDP) projects developed for India-specific use cases. This repository contains multiple implementations and architectures for document processing, classification, extraction, and analysis.

## Projects Overview

### 1. IDP_India/Modularized
**Modular Architecture IDP System**
- **Description**: A comprehensive modular IDP system with separate managers for different functionalities
- **Features**:
  - Document classification using Azure OpenAI
  - Structured data extraction from large documents
  - Vision AI integration for scanned documents
  - Vector database storage (FAISS) with separate stores for text and images
  - Excel export functionality
  - Database integration with PostgreSQL
  - Email sharing capabilities
- **Key Files**:
  - `app.py`: Main Streamlit application
  - `chunk_manager.py`: Text chunking for large documents
  - `database_manager.py`: Database operations
  - `email_manager.py`: Email functionality
  - `excel_manager.py`: Excel export operations
  - `html_manager.py`: HTML generation
  - `llm_manager.py`: LLM interactions
  - `vision_manager.py`: Vision AI processing

### 2. IDP_India_With_Agents
**Agent-Based IDP System**
- **Description**: IDP system with specialized agents for different tasks
- **Features**:
  - Excel Agent: Handles spreadsheet operations and data export
  - Email Agent: Manages email communications and attachments
  - Multi-database architecture with separate vector stores
  - Vision caching for efficient processing
  - Rate limit handling with exponential backoff
- **Key Files**:
  - `app.py`: Main application with agent integration
  - `excel_agent.py`: Excel operations agent
  - `email_agent.py`: Email operations agent
  - Subprojects:
    - `IDP_India_With_Agents_Backup/`: Backup version
    - `IDP_India_With_Agents_LangGraph/`: LangGraph implementation

### 3. IDP_With_Agents/IDP_India_With_Agents_Unified
**Unified Agent Architecture**
- **Description**: Unified agent system with integrated Q&A capabilities
- **Features**:
  - Unified agent architecture
  - Database agent for storage operations
  - Q&A agent for document querying
  - Comprehensive logging configuration
  - Structured data management
- **Key Files**:
  - `unified_agent.py`: Main unified agent
  - `qna_agent.py`: Question-answering agent
  - `database_agent.py`: Database operations agent
  - `logging_config.py`: Logging configuration

### 4. SSA_India_IDP
**SSA India Specific IDP**
- **Description**: IDP system tailored for SSA India requirements
- **Features**:
  - Multiple app versions (v1 and current)
  - Vector database with separate stores for images, tables, and text
  - Vision cache for efficient image processing
  - Document processing pipeline
- **Key Files**:
  - `app.py`: Main application
  - `app_v1.py`: Version 1 application
  - `requirements.txt`: Dependencies

## Common Features Across Projects

### Document Processing
- PDF text extraction using PyMuPDF (fitz)
- Vision AI integration for scanned documents
- Intelligent document classification
- Structured data extraction with chunking for large documents

### AI/ML Capabilities
- Azure OpenAI integration for LLM operations
- Embedding generation for semantic search
- Conversational retrieval chains
- Vision analysis with caching

### Data Management
- Vector databases (FAISS) for semantic search
- Structured data storage in JSON format
- Excel export functionality
- Database integration (PostgreSQL)

### User Interface
- Streamlit-based web applications
- Professional UI with custom CSS
- Document viewer with page navigation
- Interactive Q&A interface
- Multi-tab layouts for different functionalities

## Technology Stack

### Core Technologies
- **Python 3.9+**
- **Streamlit**: Web application framework
- **Azure OpenAI**: LLM and embeddings
- **LangChain**: RAG pipeline and chains
- **FAISS**: Vector similarity search
- **PyMuPDF (fitz)**: PDF processing
- **PostgreSQL**: Relational database

### Supporting Libraries
- **Pillow**: Image processing
- **OpenCV**: Computer vision
- **pandas**: Data manipulation
- **SQLAlchemy**: Database ORM
- **python-dotenv**: Environment management

## Setup and Installation

### Prerequisites
1. Python 3.9 or higher
2. Azure OpenAI account with API keys
3. PostgreSQL database (optional)
4. Git

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/Bhanuprakash9391/IDP_India_Projects.git
cd IDP_India_Projects

# Set up environment variables
cp IDP_India/Modularized/.env.example .env
# Edit .env with your Azure OpenAI credentials

# Install dependencies for a specific project
cd IDP_India/Modularized
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Environment Variables
Required environment variables (set in `.env` file):
```
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_chat_deployment
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your_embeddings_deployment
```

## Usage

### Processing Documents
1. Upload PDF documents through the web interface
2. The system automatically classifies document types
3. Structured data is extracted and stored
4. Vector databases are created for semantic search
5. Export results to Excel or database

### Querying Documents
1. Use the Q&A interface to ask questions about processed documents
2. The system retrieves relevant information from vector stores
3. Responses include source references

### Export Options
- Excel export by document type
- Database storage with deduplication
- Email sharing with attachments

## Project Structure
```
IDP_India_Projects/
├── IDP_India/Modularized/          # Modular architecture
├── IDP_India_With_Agents/          # Agent-based system
├── IDP_With_Agents/                # Unified agents
└── SSA_India_IDP/                  # SSA-specific implementation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Proprietary - For internal use only.

## Contact

For questions or support, contact the development team.

## Repository Information
- **GitHub**: https://github.com/Bhanuprakash9391/IDP_India_Projects
- **Created**: December 2025
- **Last Updated**: December 2025
- **Status**: Active development
