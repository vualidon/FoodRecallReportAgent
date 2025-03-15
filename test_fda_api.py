#!/usr/bin/env python3
"""
Test script for FDA API integration in the Food Recall Report Agent.
This script tests the data collection from the FDA API directly.
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FDA_API_Test")

# FDA API configuration
FDA_API_URL = "https://api.fda.gov/food/enforcement.json"
FDA_API_KEY = "RKtcYjoy3nogxlsb4dhBCKMnHV94Q6dmS0bkedwD"
FDA_API_LIMIT = 5  # Limit to a few results for testing

def test_fda_api():
    """Test the FDA API connection and data retrieval."""
    logger.info("Testing FDA API connection...")
    
    try:
        # Set up API parameters
        params = {
            "api_key": FDA_API_KEY,
            "limit": FDA_API_LIMIT,
            "sort": "report_date:desc"
        }
        
        # Call the FDA API
        response = requests.get(FDA_API_URL, params=params)
        response.raise_for_status()
        
        # Parse the API response
        recall_data = response.json()
        
        # Check for expected structure
        if "meta" not in recall_data or "results" not in recall_data:
            logger.error("Unexpected FDA API response structure")
            return False
        
        # Display summary info
        meta = recall_data.get("meta", {})
        results = recall_data.get("results", [])
        
        logger.info(f"FDA API request successful. Retrieved {len(results)} records.")
        logger.info(f"Total available records: {meta.get('results', {}).get('total', 'unknown')}")
        logger.info(f"Last updated: {meta.get('last_updated', 'unknown')}")
        
        # Display sample of results
        if results:
            # Check if all results have report_date
            has_report_dates = all("report_date" in result for result in results)
            logger.info(f"All results have report_date: {has_report_dates}")
            
            # Display all report dates
            logger.info("\nReport dates from results:")
            for i, result in enumerate(results):
                report_date = result.get("report_date", "N/A")
                logger.info(f"Result {i+1}: report_date = {report_date}")
            
            logger.info("\nSample recall information:")
            sample = results[0]
            logger.info(f"Recall Number: {sample.get('recall_number', 'N/A')}")
            logger.info(f"Product: {sample.get('product_description', 'N/A')}")
            logger.info(f"Reason: {sample.get('reason_for_recall', 'N/A')}")
            logger.info(f"Classification: {sample.get('classification', 'N/A')}")
            logger.info(f"Recalling Firm: {sample.get('recalling_firm', 'N/A')}")
            logger.info(f"Report Date: {sample.get('report_date', 'N/A')}")
            logger.info(f"Recall Initiation Date: {sample.get('recall_initiation_date', 'N/A')}")
            
            # Create a filename example
            report_date = sample.get('report_date', '')
            example_filename = f"fda_{report_date}_{sample.get('recall_number', 'unknown')}.json"
            logger.info(f"\nExample filename using report_date: {example_filename}")
            
            # Save sample to file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sample_file = f"fda_api_sample_{timestamp}.json"
            
            with open(sample_file, "w") as f:
                json.dump(sample, f, indent=2)
            
            logger.info(f"\nSaved sample data to {sample_file}")
            
        return True
        
    except requests.RequestException as e:
        logger.error(f"Error connecting to FDA API: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during FDA API test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting FDA API test")
    success = test_fda_api()
    
    if success:
        logger.info("FDA API test completed successfully")
    else:
        logger.error("FDA API test failed") 