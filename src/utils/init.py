import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InitUtil")

def init_application():
    """
    Initialize the application by creating necessary directories and checking configuration.
    
    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required API keys
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
            
        # Create necessary directories
        directories = [
            "data",
            "data/raw",
            "data/processed",
            "data/analyzed",
            "reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during application initialization: {e}")
        return False

def create_example_env_file():
    """
    Create an example .env file if one doesn't exist.
    This helps users understand what environment variables are needed.
    """
    env_path = Path(".env")
    example_env_path = Path(".env.example")
    
    # Don't overwrite existing .env file
    if env_path.exists():
        logger.info(".env file already exists, skipping creation of example")
        return
    
    # Create example .env file
    example_content = """# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here

# Configuration
LOG_LEVEL=INFO
"""
    
    with open(example_env_path, "w") as f:
        f.write(example_content)
    
    logger.info(f"Created {example_env_path} file. Please rename to .env and update with your API keys.")

if __name__ == "__main__":
    # When run directly, initialize the application
    success = init_application()
    create_example_env_file()
    
    if success:
        print("Application initialization completed successfully.")
    else:
        print("Application initialization failed. See logs for details.") 