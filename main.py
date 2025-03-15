#!/usr/bin/env python3
"""
Food Recall Report Agent - Main Entry Point

This script serves as the main entry point for the Food Recall Report Agent,
a multi-agent system for analyzing and reporting on food recalls in the United States.
"""

import sys
import os
import argparse
from src.orchestrator import FoodRecallOrchestrator
from src.utils.init import init_application, create_example_env_file

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Food Recall Report Agent - A multi-agent system for food recall analysis'
    )
    parser.add_argument(
        '--step', 
        choices=['collect', 'extract', 'analyze', 'report', 'all'],
        default='all', 
        help='Step to execute (default: all)'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=7,
        help='Number of days to include in the report (default: 7)'
    )
    parser.add_argument(
        '--input', 
        nargs='*',
        help='Input files for the specified step'
    )
    args = parser.parse_args()
    
    # Initialize the application
    print("Initializing Food Recall Report Agent...")
    if not init_application():
        print("Initialization failed. Please check logs for details.")
        sys.exit(1)
    
    # create_example_env_file()
    
    # Create and run the orchestrator
    print(f"Running {'full pipeline' if args.step == 'all' else args.step + ' step'}...")
    orchestrator = FoodRecallOrchestrator()
    
    try:
        if args.step == 'all':
            report_path = orchestrator.run_pipeline(days=args.days)
            print(f"\nPipeline completed successfully!")
            print(f"Report generated: {os.path.abspath(report_path)}")
            
            # Print the report content
            print("\nReport Preview:")
            with open(report_path, 'r') as f:
                # Read first 20 lines or whole file if smaller
                lines = f.readlines()
                preview = lines[:min(20, len(lines))]
                print("".join(preview))
                if len(lines) > 20:
                    print(f"\n... (Report continues. Open {report_path} to view the full report)")
        else:
            result = orchestrator.run_step(args.step, args.input, args.days)
            if args.step == 'report':
                print(f"\nReport generated: {os.path.abspath(result)}")
                
                # Print the report preview
                print("\nReport Preview:")
                with open(result, 'r') as f:
                    # Read first 20 lines or whole file if smaller
                    lines = f.readlines()
                    preview = lines[:min(20, len(lines))]
                    print("".join(preview))
                    if len(lines) > 20:
                        print(f"\n... (Report continues. Open {result} to view the full report)")
            else:
                print(f"\nStep '{args.step}' completed successfully.")
                print(f"Output files: {result}")
    
    except Exception as e:
        print(f"Error running the system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 