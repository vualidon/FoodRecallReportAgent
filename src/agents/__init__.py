"""
Agent module for the Food Recall Report System.

Contains specialized agents for different stages of the food recall analysis pipeline:
- DataCollectionAgent: Collects data from FDA and USDA websites
- InformationExtractionAgent: Extracts structured information from raw recall data
- EconomicImpactAgent: Analyzes economic impact of food recalls
- ReportingAgent: Generates weekly reports on food recalls
"""

from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.information_extraction_agent import InformationExtractionAgent
from src.agents.economic_impact_agent import EconomicImpactAgent
from src.agents.reporting_agent import ReportingAgent

__all__ = [
    'DataCollectionAgent',
    'InformationExtractionAgent',
    'EconomicImpactAgent',
    'ReportingAgent'
] 