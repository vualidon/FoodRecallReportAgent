import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from src.utils.llm import create_structured_llm_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReportingAgent")

class ReportingAgent:
    """
    Agent responsible for generating weekly food recall reports.
    """
    
    # Local storage paths
    ANALYZED_DATA_DIR = "data/analyzed"
    REPORTS_DIR = "reports"
    
    def __init__(self):
        """Initialize the Reporting Agent."""
        # Ensure directories exist
        os.makedirs(self.ANALYZED_DATA_DIR, exist_ok=True)
        os.makedirs(self.REPORTS_DIR, exist_ok=True)
        
        # Initialize the report generation chain
        self.report_chain = self._create_report_chain()
        
        logger.info("Reporting Agent initialized")
    
    def run(self, analyzed_files: List[str] = None, days: int = 7) -> str:
        """
        Generate a weekly report of food recalls.
        
        Args:
            analyzed_files: Optional list of analyzed data files to include in the report.
                           If None, all files in the analyzed data directory will be used.
            days: Number of days to include in the report (default: 7 for weekly report)
        
        Returns:
            Path to the generated report file
        """
        logger.info(f"Generating report for the last {days} days")
        
        # Calculate the start date for the report period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # If no files provided, get all JSON files in the analyzed data directory
        if analyzed_files is None:
            analyzed_files = [
                os.path.join(self.ANALYZED_DATA_DIR, f)
                for f in os.listdir(self.ANALYZED_DATA_DIR)
                if f.endswith('.json')
            ]
        
        # Load the analyzed recall data
        recalls = []
        for file_path in analyzed_files:
            try:
                with open(file_path, 'r') as f:
                    recall_data = json.load(f)
                
                # Parse the recall date
                if recall_data.get('recall_date'):
                    recall_date = datetime.fromisoformat(recall_data['recall_date'])
                    
                    # Only include recalls within the report period
                    if start_date <= recall_date <= end_date:
                        recalls.append(recall_data)
                else:
                    # If no date available, include it anyway (could filter by analyzed_at instead)
                    recalls.append(recall_data)
                    
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
        
        # If no recalls found, return early
        if not recalls:
            logger.warning("No recalls found for the report period")
            report_path = os.path.join(self.REPORTS_DIR, f"food_recall_report_{start_date.strftime('%Y%m%d')}_empty.md")
            with open(report_path, 'w') as f:
                f.write(f"# Food Recall Report: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n")
                f.write("No food recalls were reported during this period.\n")
            return report_path
        
        # Rank the recalls by impact score (higher score = higher rank)
        recalls.sort(key=lambda x: float(x.get('impact_score', 0)), reverse=True)
        
        # Prepare the report data
        report_data = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "recall_count": len(recalls),
            "recalls": recalls
        }
        
        # Generate the report using the LLM chain
        report_content = self.report_chain.invoke(report_data)
        
        # Save the report
        report_file = f"food_recall_report_{start_date.strftime('%Y%m%d')}.md"
        report_path = os.path.join(self.REPORTS_DIR, report_file)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated successfully: {report_path}")
        return report_path
    
    def _create_report_chain(self):
        """
        Create the LangChain chain for generating reports.
        
        Returns:
            A LangChain chain for report generation
        """
        # Create the system prompt
        system_prompt = """
        You are a professional food safety analyst specializing in recall reporting. 
        Your task is to generate a comprehensive weekly report on food recalls.
        
        The report should include:
        
        1. An executive summary highlighting key trends and the most significant recalls
        2. A ranked list of food recalls, organized by economic impact and health risk
        3. Details for each recall including:
           - Product information (name, brand, firm)
           - Reason for recall
           - Health risk assessment
           - Economic impact analysis
           - Distribution information
        
        Format the report in Markdown, making it well-structured, professional, and easy to read.
        Use tables where appropriate for better readability.
        
        Your audience includes food industry executives, regulatory officials, and business analysts,
        so focus on clear communication of business-critical information.
        """
        
        # Create the human template
        human_template = """
        Please generate a food recall report for the period {start_date} to {end_date}.
        
        There were {recall_count} recalls during this period.
        
        Here are the details of the recalls (already ranked by economic impact):
        
        {recalls}
        """
        
        # Create and return the chain
        return create_structured_llm_chain(
            system_prompt=system_prompt,
            human_template=human_template,
            output_parser=StrOutputParser()
        )


if __name__ == "__main__":
    # Run the agent standalone
    agent = ReportingAgent()
    report_path = agent.run()
    print(f"Report generated: {report_path}")
    
    # Print the report content
    with open(report_path, 'r') as f:
        print("\n" + f.read()) 