import os
import json
import logging
import time
import re
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import TavilySearchResults

from src.models.food_recall import FoodRecall, EconomicImpact
from src.utils.llm import create_structured_llm_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EconomicImpactAgent")

# Define the output schema for the LLM
class ImpactAnalysis(BaseModel):
    """Schema for the economic impact analysis of a food recall."""
    impactCategory: str = Field(description="The impact category (low, medium, high)")
    impactScore: float = Field(
        description="A numerical score from 0.0 to 10.0 representing the economic impact",
        ge=0.0,
        le=10.0,
    )
    reasoning: str = Field(description="Explanation of the economic impact assessment")
    affectedIndustry: str = Field(description="The specific food industry sector affected")
    estimatedCostRange: str = Field(description="Estimated financial impact range (e.g., '$10K-$100K')")
    marketContext: str = Field(description="Market context and industry trends from external sources")

    class Config:
        """Pydantic model configuration."""
        populate_by_name = True  # Allow both alias and original field names
        allow_population_by_field_name = True  # Allow both alias and original field names

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 60  # seconds
MAX_RETRY_DELAY = 300  # seconds

class EconomicImpactAgent:
    """
    Agent responsible for analyzing the economic impact of food recalls.
    """
    
    # Local storage paths
    PROCESSED_DATA_DIR = "data/processed"
    ANALYZED_DATA_DIR = "data/analyzed"
    
    # Product category weights - higher values indicate higher economic impact
    PRODUCT_CATEGORIES = {
        "meat": 8.0,
        "poultry": 7.5,
        "seafood": 8.5,
        "dairy": 7.0,
        "produce": 6.5,
        "baked goods": 5.0,
        "snacks": 4.0,
        "beverages": 6.0,
        "prepared foods": 5.5,
        "supplements": 4.5,
        "infant formula": 9.5,
        "other": 5.0
    }
    
    def __init__(self):
        """Initialize the Economic Impact Agent."""
        # Ensure data directories exist
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.ANALYZED_DATA_DIR, exist_ok=True)
        
        # Initialize Tavily search tool
        self.tavily_tool = TavilySearchResults(
            max_results=10,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False
        )
        
        # Initialize the analysis chain
        self.analysis_chain = self._create_analysis_chain()
        
        logger.info("Economic Impact Agent initialized")
    
    def run(self, processed_files: List[str] = None) -> List[str]:
        """
        Execute the economic impact analysis for all processed recall files.
        
        Args:
            processed_files: Optional list of processed data files to analyze.
                            If None, all files in the processed data directory will be analyzed.
        
        Returns:
            List of file paths containing the analyzed data
        """
        logger.info("Starting economic impact analysis")
        
        # If no files provided, get all JSON files in the processed data directory
        if processed_files is None:
            processed_files = [
                os.path.join(self.PROCESSED_DATA_DIR, f)
                for f in os.listdir(self.PROCESSED_DATA_DIR)
                if f.endswith('.json')
            ]
        
        # Analyze each file
        analyzed_files = []
        for file_path in processed_files:
            try:
                # Analyze the economic impact
                analyzed_file = self._analyze_file(file_path)
                if analyzed_file:
                    analyzed_files.append(analyzed_file)
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
        
        logger.info(f"Economic impact analysis complete. Analyzed {len(analyzed_files)} recall announcements")
        return analyzed_files
    
    def _create_analysis_chain(self):
        """
        Create the LangChain chain for analyzing economic impact.
        
        Returns:
            A LangChain chain for economic impact analysis
        """
        # Create the system prompt
        system_prompt = """
        You are an expert food industry economist specializing in recall impact analysis. 
        Your task is to analyze the economic impact of food recalls based on the provided information.

        Key Factors to Consider:
        1. Health Risk Level
           - High risk: Serious health consequences, potential fatalities
           - Medium risk: Moderate health effects, hospitalization possible
           - Low risk: Minor health effects, temporary discomfort

        2. Distribution Scope
           - International: Global market impact
           - National: Country-wide distribution
           - Regional: Multiple states/regions
           - Local: Limited geographic area

        3. Product Category
           - Critical products (infant formula, medications)
           - High-value products (meat, seafood)
           - Mass-market products (snacks, beverages)
           - Specialty products (supplements, prepared foods)

        4. Brand Impact
           - Major national brands
           - Regional brands
           - Local brands
           - Private label products

        5. Market Context
           - Market share and position
           - Seasonal factors
           - Industry trends
           - Distribution channels

        Impact Assessment Guidelines:
        1. Impact Score (0.0-10.0)
           - Consider all factors holistically
           - Weight factors based on their relative importance
           - Ensure score reflects overall economic impact

        2. Impact Categories
           - LOW (0.0-3.0): Limited scope, minimal health risk
           - MEDIUM (3.1-6.0): Moderate scope, significant health risk
           - HIGH (6.1-10.0): Wide scope, serious health risk

        3. Cost Estimation
           - Consider direct costs (recall, disposal)
           - Include indirect costs (brand damage, lost sales)
           - Factor in market recovery time
        """
        
        # Create the human template
        human_template = """
        Please analyze the economic impact of the following food recall and provide your analysis in the exact JSON format specified in the system prompt:
        
        Title: {title}
        Product: {product_name}
        Brand: {brand_name}
        Recalling Firm: {recalling_firm}
        Reason for Recall: {reason}
        Health Risk: {health_risk}
        Distribution Scope: {distribution_scope}
        Distribution States: {distribution_states}
        Base Score: {base_score}
        
        Market Context:
        {market_context}
        
        Remember to return your analysis in the exact JSON format with these fields:
        - impactCategory (string: low/medium/high)
        - impactScore (number: 0.0-10.0)
        - reasoning (string: detailed explanation)
        - affectedIndustry (string: specific industry sector)
        - estimatedCostRange (string: e.g., '$10K-$100K')
        - marketContext (string: market analysis)
        """
        
        # Create the output parser
        parser = JsonOutputParser(pydantic_object=ImpactAnalysis)
        
        # Create and return the chain
        return create_structured_llm_chain(
            system_prompt=system_prompt,
            human_template=human_template,
            output_parser=parser
        )
    
    def _search_with_retry(self, search_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Tavily search with retry mechanism for rate limiting.
        
        Args:
            search_query: The search query to execute
            
        Returns:
            List of search results
        """
        retry_count = 0
        current_delay = INITIAL_RETRY_DELAY
        
        while retry_count < MAX_RETRIES:
            try:
                results = self.tavily_tool.invoke(search_query)
                return results
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "Status code 429" in error_msg or "Rate limit exceeded" in error_msg:
                    retry_count += 1
                    
                    # Try to extract wait time from error message
                    wait_match = re.search(r"retry after (\d+)s", error_msg)
                    if wait_match:
                        current_delay = int(wait_match.group(1))
                    else:
                        # Use exponential backoff if no wait time specified
                        current_delay = min(current_delay * 2, MAX_RETRY_DELAY)
                    
                    logger.warning(f"Rate limit hit. Retrying in {current_delay} seconds. Attempt {retry_count}/{MAX_RETRIES}")
                    time.sleep(current_delay)
                else:
                    # For non-rate-limit errors, log and return empty results
                    logger.error(f"Error during Tavily search: {error_msg}")
                    return []
        
        logger.error(f"Max retries ({MAX_RETRIES}) exceeded for search query: {search_query}")
        return []
    
    def _get_market_context(self, product_name: str, brand_name: str) -> str:
        """
        Get market context using Tavily search.
        
        Args:
            product_name: Name of the recalled product
            brand_name: Name of the brand
            
        Returns:
            Market context information
        """
        try:
            # Search for market information
            search_query = f"market size and trends for {product_name} {brand_name} food industry"
            market_results = self._search_with_retry(search_query)
            
            # Extract relevant information
            market_context = []
            for result in market_results:
                if result.get("content"):
                    market_context.append(result["content"])
            logger.info(f"Market context: {market_context}")
            return "\n".join(market_context) if market_context else "No market context available."
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return "Error retrieving market context."
    
    def _analyze_file(self, file_path: str) -> str:
        """
        Analyze the economic impact of a single processed recall file.
        
        Args:
            file_path: Path to the processed recall file
            
        Returns:
            Path to the analyzed data file, or None if analysis failed
        """
        logger.info(f"Analyzing file: {file_path}")
        
        try:
            # Load the processed data
            with open(file_path, 'r') as f:
                recall_dict = json.load(f)
            logger.info(f"Recall dict: {recall_dict}")
            # Get market context using Tavily
            market_context = self._get_market_context(
                recall_dict["product_name"],
                recall_dict.get("brand_name", "")
            )
            
            # Calculate base impact score from product category
            product_name = recall_dict["product_name"].lower()
            base_score = 5.0  # Default for "other" category
            for category, weight in self.PRODUCT_CATEGORIES.items():
                if category in product_name:
                    base_score = weight
                    break
            
            # Prepare the analysis input
            analysis_input = {
                "title": recall_dict["title"],
                "product_name": recall_dict["product_name"],
                "brand_name": recall_dict.get("brand_name", ""),
                "recalling_firm": recall_dict.get("recalling_firm", ""),
                "reason": recall_dict["reason"],
                "health_risk": recall_dict["health_risk"],
                "distribution_scope": recall_dict["distribution_scope"],
                "distribution_states": ", ".join(recall_dict.get("distribution_states", [])),
                "market_context": market_context,
                "base_score": base_score  # Add base score to help guide the LLM
            }
            
            # Analyze using the LLM chain
            impact_analysis = self.analysis_chain.invoke(analysis_input)
            logger.info(f"Impact analysis: {impact_analysis}")
            
            # Ensure impact score is within valid range
            impact_score = max(0.0, min(10.0, impact_analysis.get("impactScore", 0.0)))
            
            # Update the recall with economic impact information
            recall_dict["economic_impact"] = impact_analysis.get("impactCategory", "unknown").lower()
            recall_dict["impact_score"] = impact_score
            recall_dict["impact_analysis"] = {
                "reasoning": impact_analysis.get("reasoning", ""),
                "affected_industry": impact_analysis.get("affectedIndustry", ""),
                "estimated_cost_range": impact_analysis.get("estimatedCostRange", ""),
                "market_context": market_context
            }
            
            # Save the analyzed data
            recall_id = Path(file_path).stem
            analyzed_file_path = os.path.join(self.ANALYZED_DATA_DIR, f"{recall_id}.json")
            
            with open(analyzed_file_path, 'w') as f:
                json.dump(recall_dict, f, indent=2)
            
            logger.info(f"Successfully analyzed recall: {recall_dict['title']} - Impact: {recall_dict['economic_impact']}")
            return analyzed_file_path
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Run the agent standalone
    agent = EconomicImpactAgent()
    analyzed_files = agent.run()
    print(f"Analyzed {len(analyzed_files)} recall announcements")
    for file in analyzed_files:
        print(f"  - {file}") 