import logging
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the application"""
    
    # Create logs directory if logging to file
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Basic configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a' if log_file else None
    )
    
    # Add console handler if logging to file
    if log_file:
        console = logging.StreamHandler()
        console.setLevel(log_level)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)   # No provided name implies the root logger. Also the subsequent loggers use the root logger's settings (unless overridden) 
    
    return logging.getLogger('')

def get_logger(name):
    """
    Get a logger with the specified name

    Example usage:
        from document_vectorizer.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return logging.getLogger(name)