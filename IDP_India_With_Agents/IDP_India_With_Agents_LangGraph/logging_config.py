import logging
import os
from datetime import datetime

def setup_logging():
    """Setup centralized logging for the entire application"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Single log file for the entire application
    log_filename = f"{log_dir}/application_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Global logger instance
logger = setup_logging()
