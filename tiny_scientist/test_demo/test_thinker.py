import json
import os
from pathlib import Path
from tiny_scientist.thinker import Thinker
from tiny_scientist.tool import PaperSearchTool

def test_group_discussion():
    # Initialize tools
    tools = [PaperSearchTool()]
    
    # Create output directory if it doesn't exist
    output_dir = Path("/Users/jiaxun/Desktop/emnlp_program/tiny-scientist/tiny_scientist/test_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Thinker with correct parameters
    thinker = Thinker(
        tools=tools,
        iter_num=1,  # Set to 1 for testing
        model="gpt-4o",  # Use GPT-4 for better quality responses
        output_dir=str(output_dir),
        temperature=0.7,
        prompt_template_dir=None  # Use default prompt templates
    )
    
    # Load test data
    test_file = Path(__file__).parent / "test.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Get the first test case
    test_case = test_data[0]
    
    print("\n=== Starting Group Discussion Test ===")
    print(f"Task: {test_case['Task']}")
    print(f"Domain: medicine")  # Since this is a medical research task
    
    # Run the think method with the test case
    result = thinker.think(
        intent=test_case['Task'],
        domain="medicine",  # Specify medicine domain since this is a medical research task
        experiment_type="physical",  # This is a physical experiment
        num_rounds=3  # Set to 3 rounds for testing
    )
    
    # Print the result
    print("\n=== Group Discussion Result ===")
    print(json.dumps(json.loads(result), indent=2))
    
    # Save the result
    output_file = output_dir / "group_discussion_result.json"
    with open(output_file, 'w') as f:
        json.dump(json.loads(result), f, indent=2)
    
    print(f"\nResult saved to: {output_file}")

if __name__ == "__main__":
    test_group_discussion() 