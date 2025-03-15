import os
import json
import uuid
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any
from pathlib import Path
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.models.food_recall import FoodRecall, RawRecallData, RecallSource, HealthRisk, DistributionScope
from src.utils.llm import create_structured_llm_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InformationExtractionAgent")

# Define the output schema for the LLM
class RecallExtraction(BaseModel):
    """Schema for the structured information extracted from a recall announcement."""
    title: str = Field(description="The title of the recall announcement")
    product_name: str = Field(description="The name of the recalled product")
    brand_name: str = Field(description="The brand name of the recalled product")
    recalling_firm: str = Field(description="The name of the company recalling the product")
    recall_date: str = Field(description="The date when the recall was reported/published in YYYY-MM-DD format (not the original recall start date)")
    timestamp: str = Field(description="The timestamp when the recall was announced in YYYY-MM-DD HH:MM:SS format")
    reason: str = Field(description="The reason for the recall")
    health_risk: str = Field(description="The health risk level (low, medium, high, unknown)")
    distribution_scope: str = Field(description="The geographic scope (local, regional, national, international, unknown)")
    distribution_states: List[str] = Field(description="List of US states where the product was distributed")
    lot_codes: List[str] = Field(description="List of lot codes or identifiers for the recalled products")

class InformationExtractionAgent:
    """
    Agent responsible for extracting structured information from raw food recall data.
    """
    
    # Local storage paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    
    def __init__(self):
        """Initialize the Information Extraction Agent."""
        # Ensure data directories exist
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        
        # Initialize the extraction chains
        self.extraction_chain = self._create_extraction_chain()
        self.fda_extraction_chain = self._create_fda_extraction_chain()
        self.usda_extraction_chain = self._create_usda_extraction_chain()
        
        logger.info("Information Extraction Agent initialized")
    
    def run(self, raw_data_files: List[str] = None) -> List[str]:
        """
        Execute the information extraction process for all raw data files.
        
        Args:
            raw_data_files: Optional list of raw data files to process.
                            If None, all files in the raw data directory will be processed.
        
        Returns:
            List of file paths containing the processed data
        """
        logger.info("Starting information extraction process")
        
        # If no files provided, get all JSON files in the raw data directory
        if raw_data_files is None:
            raw_data_files = [
                os.path.join(self.RAW_DATA_DIR, f)
                for f in os.listdir(self.RAW_DATA_DIR)
                if f.endswith('.json')
            ]
        
        # Process each file
        processed_files = []
        for file_path in raw_data_files:
            try:
                # Extract information from the raw data
                processed_file = self._process_file(file_path)
                if processed_file:
                    processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Information extraction complete. Processed {len(processed_files)} recall announcements")
        return processed_files
    
    def _create_extraction_chain(self):
        """
        Create the LangChain chain for extracting information from general recall announcements.
        
        Returns:
            A LangChain chain for extraction
        """
        # Create the system prompt
        system_prompt = """
        You are a data extraction specialist. Your task is to extract specific information from food recall announcements and format it according to the provided schema.

        Your output must be a JSON object with the following fields:
        - title: The title of the recall announcement
        - product_name: The name of the recalled product
        - brand_name: The brand name of the recalled product
        - recalling_firm: The name of the company recalling the product
        - recall_date: The date when the recall was reported/published in YYYY-MM-DD format (not the original recall start date)
        - timestamp: The timestamp when the recall was announced in YYYY-MM-DD HH:MM:SS format
        - reason: The reason for the recall
        - health_risk: One of: "high", "medium", "low", "unknown"
        - distribution_scope: One of: "local", "regional", "national", "international", "unknown"
        - distribution_states: List of US states where the product was distributed
        - lot_codes: List of lot codes or identifiers for the recalled products

        Rules:
        1. Extract only factual information present in the announcement
        2. Use empty strings for missing text fields
        3. Use empty lists for missing list fields
        4. Do not make assumptions or inferences
        5. Format dates as YYYY-MM-DD
        6. Format timestamps as YYYY-MM-DD HH:MM:SS
        7. If timestamp is not available, use the recall date with time set to 00:00:00
        8. For recall_date, use the date when the recall was reported/published, not when the recall originally started
        """
        
        # Create the human template
        human_template = """
        Please extract information from the following food recall announcement:
        
        URL: {url}
        
        Markdown Content:
        {html_content}
        
        Extract and return the information in JSON format.
        """
        
        # Create the output parser
        parser = JsonOutputParser(pydantic_object=RecallExtraction)
        
        # Create and return the chain
        return create_structured_llm_chain(
            system_prompt=system_prompt,
            human_template=human_template,
            output_parser=parser
        )
    
    def _create_fda_extraction_chain(self):
        """
        Create the LangChain chain for extracting information from FDA recall announcements.
        
        Returns:
            A LangChain chain for extraction
        """
        # Create the system prompt
        system_prompt = """
        You are a data extraction specialist. Your task is to extract specific information from FDA food recall announcements and format it according to the provided schema.

        Your output must be a JSON object with the following fields:
        - title: The title of the recall announcement (extract from the first line after navigation)
        - product_name: The specific name of the recalled product (include dosage/form if medication)
        - brand_name: The brand name of the recalled product (look for "Brand Name:" field)
        - recalling_firm: The name of the company recalling the product (look for "Company Name:" field)
        - recall_date: The date when the recall was reported/published in YYYY-MM-DD format (extract from "FDA Publish Date:" field)
        - timestamp: The timestamp when the recall was announced in YYYY-MM-DD HH:MM:SS format
        - reason: The reason for the recall (extract from "Reason for Announcement:" or "Recall Reason Description" field)
        - health_risk: One of: "high", "medium", "low", "unknown"
        - distribution_scope: One of: "local", "regional", "national", "international", "unknown"
        - distribution_states: List of US states where the product was distributed
        - lot_codes: List of lot codes or identifiers for the recalled products (extract from "Lot Number:" or similar fields)

        FDA-Specific Rules:
        1. For title:
           - Extract from the first line after navigation
           - Do not include company name if it's already in the title
           - Keep the exact wording from the announcement

        2. For product_name:
           - Include the full product name with dosage/form if it's a medication
           - For food products, include the specific product name
           - Remove any HTML formatting

        3. For brand_name:
           - Look specifically for "Brand Name:" field
           - If multiple brands are listed, combine them with commas
           - Use empty string if no brand name is found

        4. For recalling_firm:
           - Look for "Company Name:" field
           - Include both company name and any subsidiaries mentioned
           - Remove any HTML formatting

        5. For recall_date:
           - Look for "FDA Publish Date:" field
           - Convert date to YYYY-MM-DD format
           - If not found, use the date from the URL or announcement header
           - IMPORTANT: Do not use future dates. If a future date is found, use the current date
           - Example: If FDA Publish Date shows "March 14, 2025", use current date instead

        6. For timestamp:
           - Look for "FDA Publish Date:" field for both date and time
           - If time is not specified, use "00:00:00"
           - Format as YYYY-MM-DD HH:MM:SS
           - IMPORTANT: Do not use future dates/times. If a future date is found:
             * Use the current date
             * Use the current time if available, otherwise use "00:00:00"
           - Example: If FDA Publish Date shows "March 14, 2025 15:30", use current date with "15:30:00"

        7. For reason:
           - Look for "Reason for Announcement:" or "Recall Reason Description" field
           - Include the complete reason statement
           - Remove any HTML formatting

        8. For health_risk:
           - Analyze the risk statement and potential health impacts
           - Consider factors like:
             * Potential for serious injury or death
             * Number of people affected
             * Severity of potential health effects
           - Default to "high" if there's any mention of serious health risks

        9. For distribution_scope:
           - Look for terms like "nationwide", "regional", "statewide"
           - Consider the number of states mentioned
           - Default to "national" if nationwide distribution is mentioned

        10. For distribution_states:
            - Extract specific state names if listed
            - If nationwide, include all US states
            - Remove any HTML formatting

        11. For lot_codes:
            - Look for "Lot Number:", "Batch/Lot No:", or similar fields
            - Include all lot numbers mentioned
            - Format consistently (remove any prefixes like "Lot:" or "Batch:")

        Date and Time Validation Rules:
        1. Never use future dates or times
        2. If a future date is found:
           - Use the current date instead
           - Keep the time if it's valid (not in the future)
           - Use "00:00:00" if time is not specified or is in the future
        3. Validate all dates are in the past or present
        4. Format all dates as YYYY-MM-DD
        5. Format all times as HH:MM:SS
        6. Use 24-hour format for times

        Data Cleaning Rules:
        1. Remove all HTML tags and formatting
        2. Remove redundant information
        3. Standardize state names
        4. Clean up any extra whitespace or special characters
        5. Ensure dates are in YYYY-MM-DD format
        6. Remove any duplicate information

        General Rules:
        1. Extract only factual information present in the announcement
        2. Use empty strings for missing text fields
        3. Use empty lists for missing list fields
        4. Do not make assumptions or inferences
        5. Format dates as YYYY-MM-DD
        6. Format timestamps as YYYY-MM-DD HH:MM:SS
        7. If timestamp is not available, use the recall date with time set to 00:00:00
        8. For recall_date, use the date when the recall was reported/published, not when the recall originally started
        """
        
        # Create the human template
        human_template = """
        Please extract information from the following FDA food recall announcement:
        
        URL: {url}
        
        Markdown Content:
        {html_content}
        
        Extract and return the information in JSON format.
        """
        
        # Create the output parser
        parser = JsonOutputParser(pydantic_object=RecallExtraction)
        
        # Create and return the chain
        return create_structured_llm_chain(
            system_prompt=system_prompt,
            human_template=human_template,
            output_parser=parser
        )
    
    def _create_usda_extraction_chain(self):
        """
        Create the LangChain chain for extracting information from USDA recall announcements.
        
        Returns:
            A LangChain chain for extraction
        """
        # Create the system prompt
        system_prompt = """
        You are a data extraction specialist. Your task is to extract specific information from USDA food recall announcements and format it according to the provided schema.

        Your output must be a JSON object with the following fields:
        - title: The title of the recall announcement (usually starts with the company name)
        - product_name: The name of the recalled product (extract only the product name without dates or package sizes)
        - brand_name: The brand name of the recalled product (if available)
        - recalling_firm: The name of the company recalling the product (look for "doing business as" or "dba")
        - recall_date: The date when the recall was reported/published in YYYY-MM-DD format (extract from the first line in format "Day, MM/DD/YYYY - Current")
        - timestamp: The timestamp when the recall was announced in YYYY-MM-DD HH:MM:SS format
        - reason: The reason for the recall (look for "recalls" followed by reason)
        - health_risk: One of: "high", "medium", "low", "unknown" (look for "[High - Class I]", "[Medium - Class II]", etc.)
        - distribution_scope: One of: "local", "regional", "national", "international", "unknown" (determine based on number of states)
        - distribution_states: List of US states where the product was distributed (look for comma-separated list of states)
        - lot_codes: List of lot codes or identifiers for the recalled products (extract "BEST BY" dates, establishment numbers, or lot numbers)

        USDA-Specific Rules:
        1. For recall_date: 
           - Look for the date in the first line of the content in format "Day, MM/DD/YYYY - Current"
           - Extract only the date part (MM/DD/YYYY) and convert to YYYY-MM-DD format
           - Example: "Tue, 02/25/2025 - Current" → "2025-02-25"
        2. For health_risk: 
           - "High - Class I" → "high"
           - "Medium - Class II" → "medium"
           - "Low - Class III" → "low"
           - If no classification found, determine based on reason and impact
        3. For distribution_scope:
           - 1-3 states → "local"
           - 4-10 states → "regional"
           - 11+ states → "national"
           - If states not listed but mentioned as "nationwide" → "national"
        4. For recalling_firm: Include both the main company name and any "dba" names
        5. For product_name: 
           - Extract only the product name without dates or package sizes
           - Remove HTML tags and formatting
           - If multiple products, use the most general name that covers all variants
        6. For reason: Look for the specific reason after "recalls" in the announcement
        7. For lot_codes: 
           - Extract "BEST BY" dates if available
           - Extract establishment numbers if available
           - Extract lot numbers if available
           - Format dates as YYYY-MM-DD
        8. For distribution_states:
           - Extract actual state names from comma-separated list
           - If "nationwide" is mentioned, list all US states
           - Remove any HTML formatting
           - Use standard state abbreviations or full names consistently

        Data Cleaning Rules:
        1. Remove all HTML tags and formatting
        2. Remove redundant information (dates, package sizes) from product names
        3. Standardize state names (either all abbreviations or all full names)
        4. Clean up any extra whitespace or special characters
        5. Ensure dates are in YYYY-MM-DD format
        6. Remove any duplicate information

        General Rules:
        1. Extract only factual information present in the announcement
        2. Use empty strings for missing text fields
        3. Use empty lists for missing list fields
        4. Do not make assumptions or inferences
        5. Format dates as YYYY-MM-DD
        6. Format timestamps as YYYY-MM-DD HH:MM:SS
        7. If timestamp is not available, use the recall date with time set to 00:00:00
        8. For recall_date, use the date when the recall was reported/published, not when the recall originally started
        """
        
        # Create the human template
        human_template = """
        Please extract information from the following USDA food recall announcement:
        
        URL: {url}
        
        Markdown Content:
        {html_content}
        
        Extract and return the information in JSON format.
        """
        
        # Create the output parser
        parser = JsonOutputParser(pydantic_object=RecallExtraction)
        
        # Create and return the chain
        return create_structured_llm_chain(
            system_prompt=system_prompt,
            human_template=human_template,
            output_parser=parser
        )
    
    def _validate_and_correct_dates(self, extracted_data: dict) -> dict:
        """
        Validate and correct dates in the extracted data to ensure no future dates are used.
        
        Args:
            extracted_data: Dictionary containing the extracted data
            
        Returns:
            Dictionary with validated and corrected dates
        """
        # Helper function to validate date format
        def validate_date(date_str: str) -> str:
            if not date_str:
                return datetime.now().strftime("%Y-%m-%d")
            try:
                # Just validate the format, don't compare dates
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                return datetime.now().strftime("%Y-%m-%d")
        
        # Helper function to validate timestamp format
        def validate_timestamp(timestamp_str: str) -> str:
            if not timestamp_str:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Just validate the format, don't compare timestamps
                datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                return timestamp_str
            except ValueError:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Validate and correct dates
        if "recall_date" in extracted_data:
            extracted_data["recall_date"] = validate_date(extracted_data["recall_date"])
        
        if "timestamp" in extracted_data:
            extracted_data["timestamp"] = validate_timestamp(extracted_data["timestamp"])
        
        return extracted_data

    def _extract_fda_date(self, html_content: str) -> str:
        """
        Extract FDA publish date using rule-based pattern matching.
        
        Args:
            html_content: The HTML content of the FDA recall announcement
            
        Returns:
            The extracted date in YYYY-MM-DD format, or current date if not found
        """
        # Pattern to match FDA Publish Date in various formats
        patterns = [
            r"FDA Publish Date:\s*([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
            r"FDA Publish Date:\s*(\d{1,2})/(\d{1,2})/(\d{4})",
            r"FDA Publish Date:\s*(\d{4})-(\d{2})-(\d{2})"
        ]
        
        # Month name to number mapping
        month_map = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        
        for pattern in patterns:
            match = re.search(pattern, html_content)
            if match:
                try:
                    if len(match.groups()) == 3:
                        if pattern == patterns[0]:  # Month name format
                            month, day, year = match.groups()
                            month = month_map.get(month)
                            if month:
                                return f"{year}-{month}-{int(day):02d}"
                        else:  # Numeric format
                            month, day, year = match.groups()
                            return f"{year}-{int(month):02d}-{int(day):02d}"
                except (ValueError, TypeError):
                    continue
        
        # If no date found, return current date
        return datetime.now().strftime("%Y-%m-%d")

    def _extract_usda_date(self, html_content: str) -> str:
        """
        Extract USDA recall date using rule-based pattern matching.
        
        Args:
            html_content: The HTML content of the USDA recall announcement
            
        Returns:
            The extracted date in YYYY-MM-DD format, or current date if not found
        """
        # Pattern to match USDA date in format "Day, MM/DD/YYYY - Current"
        pattern = r"([A-Za-z]+),\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*-\s*Current"
        
        match = re.search(pattern, html_content)
        if match:
            try:
                # Extract date components (day name is ignored)
                _, month, day, year = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
            except (ValueError, TypeError):
                pass
        
        # If no date found, return current date
        return datetime.now().strftime("%Y-%m-%d")

    def _process_file(self, file_path: str) -> str:
        """
        Process a single raw data file to extract structured information.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            Path to the processed data file, or None if processing failed
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Load the raw data
            with open(file_path, 'r') as f:
                raw_data_dict = json.load(f)
            
            # Create RawRecallData object
            raw_data = RawRecallData(
                source=RecallSource(raw_data_dict['source']),
                url=raw_data_dict['url'],
                html_content=raw_data_dict['html_content'],
                collected_at=datetime.fromisoformat(raw_data_dict['collected_at'])
            )
            
            # Extract recall date using rule-based method based on source
            if raw_data.source == RecallSource.FDA:
                recall_date = self._extract_fda_date(raw_data.html_content)
                logger.info(f"Extracted FDA publish date: {recall_date}")
            elif raw_data.source == RecallSource.USDA:
                recall_date = self._extract_usda_date(raw_data.html_content)
                logger.info(f"Extracted USDA recall date: {recall_date}")
            
            # Select the appropriate extraction chain based on the source
            if raw_data.source == RecallSource.FDA:
                extraction_chain = self.fda_extraction_chain
                logger.info(f"Using FDA extraction chain for {file_path}")
            elif raw_data.source == RecallSource.USDA:
                extraction_chain = self.usda_extraction_chain
                logger.info(f"Using USDA extraction chain for {file_path}")
            else:
                extraction_chain = self.extraction_chain
                logger.info(f"Using general extraction chain for {file_path}")
            
            # Extract information using the appropriate LLM chain
            extraction_input = {
                "html_content": raw_data.html_content,
                "url": raw_data.url
            }
            
            extracted_data = extraction_chain.invoke(extraction_input)
            logger.info(f"Extracted data: {extracted_data}")
            
            # Use rule-based date if available
            if raw_data.source in [RecallSource.FDA, RecallSource.USDA]:
                extracted_data["recall_date"] = recall_date
            
            # Validate and correct dates
            extracted_data = self._validate_and_correct_dates(extracted_data)
            logger.info(f"Validated data: {extracted_data}")
            
            # Create the FoodRecall object
            recall_id = Path(file_path).stem  # Use the filename without extension as ID
            
            food_recall = FoodRecall(
                id=recall_id,
                source=raw_data.source,
                url=raw_data.url,
                title=extracted_data.get("title", ""),
                product_name=extracted_data.get("product_name", ""),
                brand_name=extracted_data.get("brand_name"),
                recalling_firm=extracted_data.get("recalling_firm"),
                recall_date=datetime.fromisoformat(extracted_data.get("recall_date")) if extracted_data.get("recall_date") else None,
                reason=extracted_data.get("reason", ""),
                health_risk=HealthRisk(extracted_data.get("health_risk", "unknown").lower()),
                distribution_scope=DistributionScope(extracted_data.get("distribution_scope", "unknown").lower()),
                distribution_states=extracted_data.get("distribution_states", []),
                lot_codes=extracted_data.get("lot_codes", []),
                analyzed_at=datetime.now()
            )
            
            # Save the processed data
            processed_file_path = os.path.join(self.PROCESSED_DATA_DIR, f"{recall_id}.json")
            
            with open(processed_file_path, 'w') as f:
                # Convert to dict and handle datetime serialization
                food_recall_dict = food_recall.model_dump()
                
                # Convert datetime objects to ISO format strings
                if food_recall_dict.get('recall_date'):
                    food_recall_dict['recall_date'] = food_recall_dict['recall_date'].isoformat()
                if food_recall_dict.get('analyzed_at'):
                    food_recall_dict['analyzed_at'] = food_recall_dict['analyzed_at'].isoformat()
                
                json.dump(food_recall_dict, f, indent=2)
            
            logger.info(f"Successfully processed recall announcement: {food_recall.title}")
            return processed_file_path
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Run the agent standalone
    agent = InformationExtractionAgent()
    processed_files = agent.run()
    print(f"Processed {len(processed_files)} recall announcements")
    for file in processed_files:
        print(f"  - {file}") 