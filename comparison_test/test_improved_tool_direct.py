#!/usr/bin/env python3
"""
Direct test for improved diagram tool with better test cases
"""

import json
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_diagram_tool import ImprovedDiagramTool


def test_svg_generation():
    """Test SVG generation with real academic content"""
    print("🎨 Testing SVG Generation")
    print("=" * 40)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.75)
    
    # Test case based on real academic paper
    test_case = {
        "section_name": "Method",
        "section_content": """
        We propose a novel deep learning architecture that combines convolutional neural networks (CNNs) 
        with attention mechanisms. Our model consists of three main components: 1) A feature extraction 
        module using ResNet-50 backbone, 2) A multi-head attention layer that captures spatial dependencies, 
        and 3) A classification head with softmax activation. The model is trained end-to-end using 
        cross-entropy loss and Adam optimizer with learning rate 0.001.
        """
    }
    
    print("📊 Testing SVG format...")
    result = tool.draw_diagram(
        section_name=test_case["section_name"],
        section_content=test_case["section_content"],
        preferred_format="svg"
    )
    
    print(f"✅ Success: {result['success']}")
    if result['success']:
        print(f"📁 Files: {result.get('files', {})}")
        print(f"📝 Summary: {result.get('summary', '')}")
        print(f"🎯 Format: {result.get('format', '')}")
        
        # Show content preview
        content = result.get("content", "")
        if content:
            print(f"\n📝 Content preview (first 300 chars):")
            print(content[:300] + "..." if len(content) > 300 else content)
    else:
        print(f"❌ Error: {result.get('error', '')}")
        print(f"🔍 Full response: {result.get('full_response', '')[:500]}...")


def test_tikz_generation():
    """Test TikZ generation with experimental setup"""
    print("\n🎨 Testing TikZ Generation")
    print("=" * 40)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.75)
    
    # Test case for experimental setup
    test_case = {
        "section_name": "Experimental_Setup",
        "section_content": """
        We evaluate our method on three benchmark datasets: ImageNet, CIFAR-10, and Places365. 
        For each dataset, we split the data into training (80%), validation (10%), and test (10%) sets. 
        We compare against five baseline methods: ResNet-50, VGG-16, DenseNet-121, EfficientNet-B0, 
        and Vision Transformer. All experiments are conducted on NVIDIA V100 GPUs with 32GB memory. 
        Training runs for 100 epochs with batch size 64, and we report top-1 accuracy, top-5 accuracy, 
        and training time as evaluation metrics.
        """
    }
    
    print("📊 Testing TikZ format...")
    result = tool.draw_diagram(
        section_name=test_case["section_name"],
        section_content=test_case["section_content"],
        preferred_format="tikz"
    )
    
    print(f"✅ Success: {result['success']}")
    if result['success']:
        print(f"📁 Files: {result.get('files', {})}")
        print(f"📝 Summary: {result.get('summary', '')}")
        print(f"🎯 Format: {result.get('format', '')}")
        
        # Show content preview
        content = result.get("content", "")
        if content:
            print(f"\n📝 Content preview (first 300 chars):")
            print(content[:300] + "..." if len(content) > 300 else content)
    else:
        print(f"❌ Error: {result.get('error', '')}")
        print(f"🔍 Full response: {result.get('full_response', '')[:500]}...")


def test_tool_interface():
    """Test the tool interface like other tools in tool.py"""
    print("\n🔧 Testing Tool Interface")
    print("=" * 40)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.75)
    
    # Create query like other tools
    query = json.dumps({
        "section_name": "Results",
        "section_content": """
        Our ablation study demonstrates the effectiveness of each component. The baseline ResNet-50 
        achieves 76.2% accuracy on ImageNet. Adding attention mechanisms improves performance to 78.5%, 
        while the full model with all components reaches 81.3% accuracy. We also analyze the impact 
        of different attention heads: 4 heads (77.8%), 8 heads (79.1%), and 16 heads (81.3%). 
        The model converges faster with more attention heads, requiring 45 epochs vs 67 epochs for 
        the baseline to reach 75% accuracy.
        """,
        "format": "svg"
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
    test_svg_generation()
    test_tikz_generation()
    test_tool_interface()
    print("\n🎉 All tests completed!") 