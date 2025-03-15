import os
import argparse
import logging
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.information_extraction_agent import InformationExtractionAgent
from src.agents.economic_impact_agent import EconomicImpactAgent
from src.agents.reporting_agent import ReportingAgent
from src.utils.llm import get_gemini_llm

# Define log directory and file path
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "food_recall_orchestrator.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

# Log startup information
logger.info("Logging initialized. Log file: %s", LOG_FILE)

class FoodRecallOrchestrator:
    """
    Orchestrator that coordinates the execution of the food recall analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the Food Recall Orchestrator."""
        # Initialize the agents
        self.data_collection_agent = DataCollectionAgent()
        self.information_extraction_agent = InformationExtractionAgent()
        self.economic_impact_agent = EconomicImpactAgent()
        self.reporting_agent = ReportingAgent()
        
        logger.info("Food Recall Orchestrator initialized")
    
    def run_pipeline(self, days: int = 7) -> str:
        """
        Execute the complete food recall analysis pipeline.
        
        Args:
            days: Number of days to include in the report (default: 7 for weekly report)
            
        Returns:
            Path to the generated report file
        """
        logger.info(f"Starting food recall analysis pipeline for the past {days} days")
        start_time = datetime.now()
        
        try:
            # Step 1: Collect food recall data
            logger.info("Step 1: Collecting food recall data")
            raw_data_files = self.data_collection_agent.run()
            logger.info(f"Collected {len(raw_data_files)} raw data files")
            
            # Step 2: Extract information from raw data
            logger.info("Step 2: Extracting information from raw data")
            processed_files = self.information_extraction_agent.run(raw_data_files)
            logger.info(f"Processed {len(processed_files)} files")
            
            # Step 3: Analyze economic impact
            logger.info("Step 3: Analyzing economic impact")
            analyzed_files = self.economic_impact_agent.run(processed_files)
            logger.info(f"Analyzed {len(analyzed_files)} files")
            
            # Step 4: Generate the report
            logger.info("Step 4: Generating report")
            report_path = self.reporting_agent.run(analyzed_files, days=days)
            logger.info(f"Report generated: {report_path}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error executing pipeline: {e}")
            raise
    
    def run_step(self, step: str, input_files: list = None, days: int = 7) -> list:
        """
        Execute a specific step of the pipeline.
        
        Args:
            step: The step to execute ('collect', 'extract', 'analyze', 'report')
            input_files: Optional list of input files for the step
            days: Number of days to include in the report (for 'report' step only)
            
        Returns:
            List of output files from the step, or report path for 'report' step
        """
        logger.info(f"Executing step: {step}")
        
        if step == 'collect':
            return self.data_collection_agent.run()
        elif step == 'extract':
            return self.information_extraction_agent.run(input_files)
        elif step == 'analyze':
            return self.economic_impact_agent.run(input_files)
        elif step == 'report':
            return self.reporting_agent.run(input_files, days=days)
        else:
            raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Food Recall Analysis System')
    parser.add_argument('--step', choices=['collect', 'extract', 'analyze', 'report', 'all'],
                        default='all', help='Step to execute (default: all)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to include in the report (default: 7)')
    parser.add_argument('--input', nargs='*',
                        help='Input files for the specified step')
    args = parser.parse_args()
    
    # Create the orchestrator
    orchestrator = FoodRecallOrchestrator()
    
    # Execute the requested step or full pipeline
    if args.step == 'all':
        report_path = orchestrator.run_pipeline(days=args.days)
        print(f"\nPipeline completed successfully. Report generated: {report_path}")
        
        # Print the report content
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    else:
        result = orchestrator.run_step(args.step, args.input, args.days)
        if args.step == 'report':
            print(f"\nReport generated: {result}")
            
            # Print the report content
            with open(result, 'r') as f:
                print("\n" + f.read())
        else:
            print(f"\nStep '{args.step}' completed successfully.")
            print(f"Output files: {result}") 