#!/usr/bin/env python3
"""
Simple test for improved diagram tool with TikZ and XML support
"""

import json
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_diagram_tool import ImprovedDiagramTool


def test_single_format():
    """Test single format generation"""
    print("🎨 Testing Improved Diagram Tool")
    print("=" * 40)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.3)
    
    # Test case
    test_case = {
        "section_name": "Method",
        "section_content": """
        Our methodology involves three main steps: 1) Data preprocessing and augmentation, 
        2) Model training with different optimizers, 3) Performance evaluation and comparison.
        The preprocessing includes normalization and data augmentation techniques.
        Training uses cross-validation with early stopping.
        """
    }
    
    # Test TikZ format
    print("\n📊 Testing TikZ format...")
    result = tool.generate_diagram(
        section_name=test_case["section_name"],
        section_content=test_case["section_content"],
        preferred_format="tikz",
        auto_fallback=True
    )
    
    print(f"✅ Success: {result['success']}")
    if result['success']:
        print(f"📁 Files: {result.get('files', {})}")
        print(f"📝 Summary: {result.get('summary', '')}")
        print(f"🎯 Format: {result.get('format', '')}")
    else:
        print(f"❌ Error: {result.get('error', '')}")
    
    # Test XML format
    print("\n📊 Testing XML format...")
    result = tool.generate_diagram(
        section_name=test_case["section_name"],
        section_content=test_case["section_content"],
        preferred_format="xml",
        auto_fallback=True
    )
    
    print(f"✅ Success: {result['success']}")
    if result['success']:
        print(f"📁 Files: {result.get('files', {})}")
        print(f"📝 Summary: {result.get('summary', '')}")
        print(f"🎯 Format: {result.get('format', '')}")
    else:
        print(f"❌ Error: {result.get('error', '')}")


def test_tool_interface():
    """Test the tool interface like other tools in tool.py"""
    print("\n🔧 Testing Tool Interface")
    print("=" * 40)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.3)
    
    # Create query like other tools
    query = json.dumps({
        "section_name": "Experimental_Setup",
        "section_content": """
        The experiment will use a ResNet-50 architecture trained on ImageNet dataset. 
        We will compare three adaptive learning rate methods: Adam, RMSprop, and Adagrad. 
        The training will run for 100 epochs with batch size 32. 
        We will measure accuracy, convergence speed, and computational efficiency.
        """,
        "format": "tikz"
    })
    
    # Test run method
    print("📊 Testing run() method...")
    results = tool.run(query)
    
    print(f"✅ Results: {results}")
    if "diagram" in results:
        diagram = results["diagram"]
        print(f"📁 Files: {diagram.get('files', {})}")
        print(f"📝 Summary: {diagram.get('summary', '')}")
        print(f"🎯 Format: {diagram.get('format', '')}")


if __name__ == "__main__":
    test_single_format()
    test_tool_interface()
    print("\n🎉 All tests completed!") 