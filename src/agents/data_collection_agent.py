import os
import json
import uuid
from datetime import datetime
import requests
from typing import List, Dict, Any, Optional
import logging
import os
import dotenv
from firecrawl import FirecrawlApp
import time

dotenv.load_dotenv()

from src.models.food_recall import RawRecallData, RecallSource

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCollectionAgent")

class DataCollectionAgent:
    """
    Agent responsible for collecting food recall data from FDA and USDA sources.
    """
    
    # Source URLs
    FDA_RECALLS_URL = "https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
    USDA_RECALLS_URL = "https://www.fsis.usda.gov/recalls"
    
    # API parameters
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    
    # Local storage paths
    DATA_DIR = "data/raw"
    
    # Retry configuration
    MAX_RETRIES = 5
    RETRY_DELAY = 60  # seconds
    
    def __init__(self):
        """Initialize the Data Collection Agent."""
        # Ensure data directory exists
        os.makedirs(self.DATA_DIR, exist_ok=True)
        # Initialize Firecrawl
        self.firecrawl = FirecrawlApp(api_key=self.FIRECRAWL_API_KEY)
        logger.info("Data Collection Agent initialized")
    
    def _scrape_with_retry(self, url: str, params: Dict[str, Any], retry_count: int = 0) -> Optional[Dict]:
        """
        Scrape URL with retry mechanism for rate limiting.
        
        Args:
            url: The URL to scrape
            params: Parameters for the scrape request
            retry_count: Current retry attempt number
            
        Returns:
            Optional[Dict]: The scrape response or None if all retries failed
        """
        try:
            response = self.firecrawl.scrape_url(url=url, params=params)
            return response
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error (429)
            if "Status code 429" in error_msg and "Rate limit exceeded" in error_msg:
                if retry_count < self.MAX_RETRIES:
                    # Extract the wait time from the error message if available
                    try:
                        import re
                        wait_match = re.search(r"retry after (\d+)s", error_msg)
                        if wait_match:
                            wait_time = int(wait_match.group(1))
                        else:
                            wait_time = self.RETRY_DELAY * (retry_count + 1)  # Exponential backoff
                    except:
                        wait_time = self.RETRY_DELAY * (retry_count + 1)  # Fallback to exponential backoff
                    
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count + 1}/{self.MAX_RETRIES}")
                    time.sleep(wait_time)
                    return self._scrape_with_retry(url, params, retry_count + 1)
                else:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) reached for {url}")
                    return None
            else:
                logger.error(f"Error scraping {url}: {e}")
                return None
    
    def run(self) -> List[str]:
        """
        Execute the data collection process for all sources.
        
        Returns:
            List of file paths containing the collected data
        """
        logger.info("Starting data collection process")
        file_paths = []
        
        # Collect FDA recalls
        fda_files = self.collect_fda_recalls()
        file_paths.extend(fda_files)
        
        # Collect USDA recalls
        usda_files = self.collect_usda_recalls()
        file_paths.extend(usda_files)
        
        logger.info(f"Data collection complete. Collected {len(file_paths)} recall announcements")
        return file_paths
    
    def _is_valid_fda_recall_link(self, link: str) -> bool:
        """
        Check if a link is a valid FDA recall detail page.
        
        Args:
            link: The URL to check
            
        Returns:
            bool: True if the link is a valid recall detail page
        """
        # Skip navigation links, search forms, and data exports
        skip_patterns = [
            '#main-content',
            '#search-form',
            '#section-nav',
            '#footer-heading',
            'datatables-data',
            'about-fda',
            'govdelivery',
            'archive',
            'additional-information-about-recalls'
        ]
        
        return (
            link.startswith('https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/') and
            not any(pattern in link for pattern in skip_patterns)
        )
    
    def collect_fda_recalls(self) -> List[str]:
        """
        Collect food recall data from FDA website using Firecrawl.
        
        Returns:
            List of file paths containing FDA recall data
        """
        logger.info("Collecting FDA recalls using Firecrawl")
        file_paths = []
        
        try:
            # Get the main recalls page content
            response = self._scrape_with_retry(
                url=self.FDA_RECALLS_URL,
                params={'formats': ['markdown', 'links']}
            )
            
            if not response or 'links' not in response:
                logger.error("Failed to get FDA recalls page content")
                return file_paths
            
            # Extract valid recall links
            recall_links = [
                link for link in response['links']
                if self._is_valid_fda_recall_link(link)
            ]
            
            logger.info(f"Found {len(recall_links)} valid FDA recall links")
            
            # Process each recall link
            for link in recall_links:
                try:
                    # Get the detail page content
                    detail_response = self._scrape_with_retry(
                        url=link,
                        params={'formats': ['markdown']}
                    )
                    
                    if not detail_response or 'markdown' not in detail_response:
                        logger.error(f"Failed to get detail content for {link}")
                        continue
                    
                    # Store the raw recall data
                    raw_data = RawRecallData(
                        source=RecallSource.FDA,
                        url=link,
                        html_content=detail_response['markdown']
                    )
                    
                    # Save to file
                    file_path = self._save_raw_data(raw_data)
                    file_paths.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing FDA recall {link}: {e}")
            
            logger.info(f"Collected {len(file_paths)} FDA recall announcements")
            
        except Exception as e:
            logger.error(f"Error collecting FDA recalls: {e}")
        
        return file_paths
    
    def collect_usda_recalls(self) -> List[str]:
        """
        Collect food recall data from USDA website using Firecrawl.
        
        Returns:
            List of file paths containing USDA recall data
        """
        logger.info("Collecting USDA recalls using Firecrawl")
        file_paths = []
        
        try:
            # Get the main recalls page content
            response = self._scrape_with_retry(
                url=self.USDA_RECALLS_URL,
                params={'formats': ['markdown', 'links']}
            )
            
            if not response or 'links' not in response:
                logger.error("Failed to get USDA recalls page content")
                return file_paths
            
            # Extract recall alert links
            recall_links = [
                link for link in response['links']
                if '/recalls-alerts/' in link
            ]
            
            logger.info(f"Found {len(recall_links)} recall alert links")
            
            # Process each recall link
            for link in recall_links:
                try:
                    # Get the detail page content
                    detail_response = self._scrape_with_retry(
                        url=link,
                        params={'formats': ['markdown']}
                    )
                    
                    if not detail_response or 'markdown' not in detail_response:
                        logger.error(f"Failed to get detail content for {link}")
                        continue
                    
                    # Store the raw recall data
                    raw_data = RawRecallData(
                        source=RecallSource.USDA,
                        url=link,
                        html_content=detail_response['markdown']
                    )
                    
                    # Save to file
                    file_path = self._save_raw_data(raw_data)
                    file_paths.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing USDA recall {link}: {e}")
            
            logger.info(f"Collected {len(file_paths)} USDA recall announcements")
            
        except Exception as e:
            logger.error(f"Error collecting USDA recalls: {e}")
        
        return file_paths
    
    def _save_raw_data(self, raw_data: RawRecallData) -> str:
        """
        Save raw recall data to a file.
        
        Args:
            raw_data: The raw recall data to save
            
        Returns:
            The path to the saved file
        """
        # Create a unique ID for the file
        unique_id = str(uuid.uuid4())
        
        # Use current timestamp for file naming
        date_part = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create the file path
        file_name = f"{raw_data.source.value.lower()}_{date_part}_{unique_id}.json"
        file_path = os.path.join(self.DATA_DIR, file_name)
        
        # Save the data as JSON
        with open(file_path, 'w') as f:
            # Convert to dict and handle datetime serialization
            raw_data_dict = raw_data.model_dump()
            raw_data_dict['collected_at'] = raw_data_dict['collected_at'].isoformat()
            
            json.dump(raw_data_dict, f, indent=2)
        
        return file_path


if __name__ == "__main__":
    # Run the agent standalone
    agent = DataCollectionAgent()
    collected_files = agent.run()
    print(f"Collected {len(collected_files)} recall announcements")
    for file in collected_files:
        print(f"  - {file}") 