import os
import time
import logging
from typing import Optional, Callable, Any, Dict, List, Union
from functools import wraps
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMUtils")

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 60  # seconds
MAX_RETRY_DELAY = 60  # seconds

def retry_with_backoff(func: Callable) -> Callable:
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        func: The function to wrap with retry logic
        
    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        retry_count = 0
        current_delay = INITIAL_RETRY_DELAY
        
        while retry_count < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                if retry_count == MAX_RETRIES:
                    logger.error(f"Max retries ({MAX_RETRIES}) exceeded for function {func.__name__}: {str(e)}")
                    raise
                
                # Use exponential backoff
                current_delay = min(current_delay * 2, MAX_RETRY_DELAY)
                logger.warning(f"Attempt {retry_count}/{MAX_RETRIES} failed for {func.__name__}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
    
    return wrapper

# Load environment variables
load_dotenv()

def get_gemini_llm(model_name: str = "gemini-2.0-flash", temperature: float = 0.5) -> ChatGoogleGenerativeAI:
    """
    Initialize a Gemini LLM with the specified configuration.
    
    Args:
        model_name: The Gemini model to use
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        
    Returns:
        An initialized Gemini LLM
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )

class RetryableChain:
    """A wrapper class that adds retry functionality to a LangChain chain."""
    
    def __init__(self, chain: RunnableSequence):
        self.chain = chain
    
    @retry_with_backoff
    def invoke(self, input_data: Union[Dict, List, str]) -> Any:
        """
        Invoke the chain with retry logic.
        
        Args:
            input_data: The input data for the chain
            
        Returns:
            The chain's output
        """
        return self.chain.invoke(input_data)

def create_structured_llm_chain(system_prompt: str, human_template: str, output_parser=None):
    """
    Create a LangChain chain with a structured prompt and optional output parser.
    
    Args:
        system_prompt: The system prompt for the LLM
        human_template: The template for user input
        output_parser: Optional parser for structured output (default: StrOutputParser)
        
    Returns:
        A runnable LangChain chain with retry functionality
    """
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    # Initialize the LLM
    llm = get_gemini_llm()
    
    # Set default output parser if none provided
    if output_parser is None:
        output_parser = StrOutputParser()
    
    # Create the chain
    chain = prompt | llm | output_parser
    
    # Wrap the chain with retry functionality
    return RetryableChain(chain) 