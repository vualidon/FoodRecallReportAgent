# Food Recall Report Agent

A multi-agent system for analyzing and reporting on food recall news in the United States.

## Overview

This system utilizes LangChain and Google's Gemini LLM to:

1. Collect food recall data from FDA API and USDA websites
2. Extract key information from recall announcements
3. Analyze potential economic impact
4. Generate comprehensive weekly reports

## System Architecture

The system consists of four specialized agents:

- **Data Collection Agent**: Fetches recall data from FDA API and USDA websites
- **Information Extraction Agent**: Processes raw data to identify key details
- **Economic Impact Agent**: Estimates financial consequences of recalls
- **Reporting Agent**: Generates weekly reports ranking recalls by severity and impact

These agents are coordinated by an Orchestrator that manages the workflow.

## Data Sources

- **FDA**: Uses the [openFDA API](https://open.fda.gov/apis/food/enforcement/) to collect structured food recall data
- **USDA**: Web scraping of the USDA FSIS recalls page

## File Naming Convention

The system uses a consistent file naming convention:

- For FDA recalls: `fda_<report_date>_<unique_id>.json` where `report_date` is the date the recall was reported by the FDA (YYYYMMDD format)
- For USDA recalls: `usda_<timestamp>_<unique_id>.json` where `timestamp` is the time of data collection (YYYYMMDDHHMMSS format)

This approach ensures that FDA recall files can be easily sorted chronologically by their official report date.

## Installation

1. Clone this repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key
FDA_API_KEY=your_fda_api_key
```

## Usage

Run the orchestrator to execute the complete pipeline:

```
python main.py
```

Or run individual steps:

```
python main.py --step collect
python main.py --step extract
python main.py --step analyze
python main.py --step report
```

To test the FDA API connection:

```
python test_fda_api.py
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

## License

MIT
