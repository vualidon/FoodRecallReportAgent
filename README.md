# Food Recall Report Agent

A multi-agent system for analyzing and reporting on food recall news in the United States.

## Overview

This system utilizes LangChain and Google's Gemini LLM to:

1. Collect food recall data from FDA and USDA websites
2. Extract key information from recall announcements
3. Analyze potential economic impact
4. Generate comprehensive weekly reports

## System Architecture

The system consists of four specialized agents:

- **Data Collection Agent**: Fetches recall data from FDA and USDA websites
- **Information Extraction Agent**: Processes raw data to identify key details
- **Economic Impact Agent**: Estimates financial consequences of recalls
- **Reporting Agent**: Generates weekly reports ranking recalls by severity and impact

These agents are coordinated by an Orchestrator that manages the workflow.

## Data Sources

- **FDA**: Web scraping of the FDA recalls page
- **USDA**: Web scraping of the USDA FSIS recalls page

## Installation

1. Clone this repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=tavily_key
FIRECRAWL_API_KEY=firecrawl_key
```

## Usage

Run the orchestrator to execute the complete pipeline:

```
python main.py
python main.py --days 14
```

Or run individual steps:

```
python main.py --step collect
python main.py --step extract
python main.py --step analyze
python main.py --step report
```

## Project Structure

```
.
├── data/                  # Storage for collected and processed data
│   ├── raw/               # Raw data from FDA API and USDA website
│   ├── processed/         # Structured data after information extraction
│   └── analyzed/          # Data with economic impact analysis
├── reports/               # Output directory for generated reports
├── src/
│   ├── agents/            # Implementation of specialized agents
│   ├── models/            # Data models and schemas
│   ├── utils/             # Helper functions and utilities
│   └── orchestrator.py    # Workflow coordinator
├── main.py                # Main entry point
├── test_fda_api.py        # FDA API test script
└── README.md
```
