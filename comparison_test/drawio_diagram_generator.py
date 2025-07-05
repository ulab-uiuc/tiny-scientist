#!/usr/bin/env python3
"""
DrawIO Diagram Generator - Convert text descriptions to DrawIO XML code
"""

import json
import os
import sys
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports to avoid dependency issues
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiny_scientist", "utils"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiny_scientist"))

from llm import create_client, get_response_from_llm


class DrawIODiagramGenerator:
    """
    DrawIO XML code generator based on text descriptions
    """
    
    def __init__(self, model: str = "claude-3-7-sonnet-20250219", temperature: float = 0.3):
        """
        Initialize the DrawIO diagram generator
        
        Args:
            model: LLM model to use
            temperature: Generation temperature (lower for more consistent output)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM client
        self.client, self.model_name = create_client(model)
        
        # Create output directory
        self.output_dir = Path("diagram_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # DrawIO system prompt with ID requirements
        self.drawio_prompt = """You are a DrawIO code generator. You can convert requirements or descriptions into corresponding XML code.

## Core Capabilities
1. Generate runnable draw.io code directly from visual descriptions/requirements
2. Validation mechanism ensures code accuracy
3. Standardized code block output
4. Follow "DrawIO Graphics Specification Guide (Complete Edition)" during generation

## Processing Flow
① Receive input → ② Parse elements → ③ Structure modeling → ④ Syntax generation → ⑤ Integrity validation → ⑥ Output result

## Output Specification
```xml
<!-- Validated draw.io code -->
<mxfile>
    [Generated core code]
</mxfile>
```

## CRITICAL ID REQUIREMENTS:
1. Every mxCell element MUST have a unique id attribute
2. IDs must be alphanumeric and start with a letter
3. No empty or duplicate IDs allowed
4. Use descriptive IDs like "start_node", "process_step", "decision_point"
5. Root cells should have IDs "0" and "1"
6. All vertices and edges must have unique IDs

## Interaction Rules
- When receiving image descriptions: "Parsing structural relationships (describing image details)...(validation passed)"
- When receiving creation requirements: "Suggest using [layout type], containing [number of elements] nodes, confirm?"
- Exception handling: "Layer X nodes have missing connections, automatically completed"

## Advantage Features
- Element positioning accuracy: ±5px equivalent coordinates
- Support automatic layout optimization (can be disabled)
- Built-in syntax corrector (error rate <0.3%)

Please provide chart description or creation requirements, I will directly output ready-to-use code.

## Important Rules:
1. Always generate complete, valid DrawIO XML code
2. Use proper mxGraph structure with cells, vertices, and edges
3. Include proper styling and positioning
4. Ensure all elements are properly connected
5. Use academic/technical color schemes
6. Make diagrams clear and professional
7. EVERY mxCell MUST have a unique id attribute
"""

    def generate_drawio_xml(self, description: str, diagram_type: str = "flowchart") -> Dict[str, Any]:
        """
        Generate DrawIO XML from text description
        
        Args:
            description: Text description of the diagram
            diagram_type: Type of diagram (flowchart, architecture, etc.)
            
        Returns:
            Dictionary containing XML content and metadata
        """
        print(f"🎨 Generating DrawIO XML for {diagram_type}...")
        
        # Create specific prompt for the diagram type
        user_prompt = f"""
Generate DrawIO XML code for a {diagram_type} diagram based on the following description:

{description}

Requirements:
- Create a clear, professional diagram
- Use appropriate shapes and colors for academic/technical context
- Ensure proper connections and flow
- Include all key elements mentioned in the description
- Use standard DrawIO mxGraph structure
- EVERY mxCell element MUST have a unique id attribute
- Use descriptive IDs like "start_node", "process_step", "decision_point"

Generate ONLY the complete DrawIO XML code, no additional text or explanations.
"""
        
        try:
            print("🔍 Debug: Calling LLM for DrawIO XML generation...")
            
            # Generate XML using LLM
            llm_response, msg_history = get_response_from_llm(
                user_prompt,
                model=self.model_name,
                client=self.client,
                system_message=self.drawio_prompt,
                temperature=self.temperature,
                task_name="generate_drawio_xml"
            )
            
            print(f"🔍 Debug: LLM response length: {len(llm_response)}")
            print(f"🔍 Debug: LLM response preview: {llm_response[:200]}...")
            
            # Extract XML content
            xml_content = self._extract_xml_content(llm_response)
            
            if not xml_content:
                return {
                    "success": False,
                    "error": "No valid DrawIO XML found in response",
                    "full_response": llm_response
                }
            
            # Fix XML IDs
            xml_content = self._fix_xml_ids(xml_content)
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawio_{diagram_type}_{timestamp}.xml"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            return {
                "success": True,
                "xml_content": xml_content,
                "file_path": str(filepath),
                "diagram_type": diagram_type,
                "timestamp": datetime.now().isoformat(),
                "full_response": llm_response
            }
            
        except Exception as e:
            print(f"❌ Debug: Exception in DrawIO generation: {e}")
            return {
                "success": False,
                "error": str(e),
                "diagram_type": diagram_type,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_xml_content(self, response: str) -> Optional[str]:
        """
        Extract DrawIO XML content from LLM response
        """
        # Try to extract XML code blocks
        xml_patterns = [
            r"```xml\s*(.*?)\s*```",
            r"<mxfile.*?</mxfile>",
            r"<mxGraphModel.*?</mxGraphModel>"
        ]
        
        for pattern in xml_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1) if "```" in pattern else match.group(0)
        
        # If no code blocks found, check if the entire response is XML
        if response.strip().startswith("<") and response.strip().endswith(">"):
            return response.strip()
        
        return None
    
    def _fix_xml_ids(self, xml_content: str) -> str:
        """
        Fix XML IDs to ensure they are unique and valid
        """
        # Find all mxCell elements
        cell_pattern = r'<mxCell([^>]*?)>'
        cells = re.findall(cell_pattern, xml_content)
        
        # Track used IDs
        used_ids = set()
        id_counter = 2  # Start from 2 since 0 and 1 are reserved for root
        
        # Fix each cell
        for i, cell_attrs in enumerate(cells):
            # Check if id attribute exists and is valid
            id_match = re.search(r'id="([^"]*)"', cell_attrs)
            
            if not id_match or not id_match.group(1) or id_match.group(1) in used_ids:
                # Generate new unique ID
                new_id = f"cell_{id_counter}"
                while new_id in used_ids:
                    id_counter += 1
                    new_id = f"cell_{id_counter}"
                
                # Replace or add id attribute
                if id_match:
                    # Replace existing invalid ID
                    xml_content = xml_content.replace(
                        f'id="{id_match.group(1)}"',
                        f'id="{new_id}"',
                        1
                    )
                else:
                    # Add new ID attribute
                    xml_content = re.sub(
                        r'(<mxCell[^>]*?)(>)',
                        rf'\1 id="{new_id}"\2',
                        xml_content,
                        count=1
                    )
                
                used_ids.add(new_id)
            else:
                used_ids.add(id_match.group(1))
        
        return xml_content
    
    def generate_simple_example(self) -> str:
        """
        Generate a simple DrawIO example for testing
        """
        simple_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="5.0" etag="example" version="22.1.16" type="device">
  <diagram name="Simple Flowchart" id="simple-flowchart">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="start_node" value="Start" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
          <mxGeometry x="320" y="80" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="process_step" value="Process Data" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="320" y="200" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="decision_point" value="Valid?" style="rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="320" y="300" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="end_node" value="End" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" vertex="1" parent="1">
          <mxGeometry x="320" y="420" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="edge_1" edge="1" parent="1" source="start_node" target="process_step">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge_2" edge="1" parent="1" source="process_step" target="decision_point">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge_3" edge="1" parent="1" source="decision_point" target="end_node">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
        
        # Save simple example
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_example_{timestamp}.xml"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(simple_xml)
        
        print(f"✅ Simple example saved to: {filepath}")
        return str(filepath)


def main():
    """Test the DrawIO diagram generator"""
    
    print("🎨 DrawIO Diagram Generator Test")
    print("=" * 50)
    
    # Initialize generator
    generator = DrawIODiagramGenerator(model="claude-3-7-sonnet-20250219", temperature=0.3)
    
    # Generate simple example first
    print("\n📊 Generating simple example...")
    simple_file = generator.generate_simple_example()
    
    # Test cases
    test_cases = [
        {
            "name": "machine_learning_pipeline",
            "description": """
            Create a machine learning pipeline diagram with the following components:
            1. Data Collection (input)
            2. Data Preprocessing (cleaning and normalization)
            3. Feature Engineering (extract features)
            4. Model Training (train ML model)
            5. Model Evaluation (test performance)
            6. Model Deployment (deploy to production)
            
            The flow should be linear from top to bottom, with clear connections between each step.
            Use appropriate colors and shapes for each component.
            """,
            "type": "flowchart"
        },
        {
            "name": "neural_network_architecture",
            "description": """
            Create a neural network architecture diagram showing:
            1. Input Layer (3 nodes)
            2. Hidden Layer 1 (5 nodes)
            3. Hidden Layer 2 (4 nodes)
            4. Output Layer (2 nodes)
            
            Show connections between layers with arrows.
            Use different colors for input, hidden, and output layers.
            Include activation functions (ReLU for hidden layers, Softmax for output).
            """,
            "type": "architecture"
        },
        {
            "name": "experimental_workflow",
            "description": """
            Create an experimental workflow diagram showing:
            1. Hypothesis Formulation
            2. Literature Review
            3. Experimental Design
            4. Data Collection
            5. Data Analysis
            6. Results Interpretation
            7. Paper Writing
            8. Publication
            
            Show the iterative nature of research with feedback loops.
            Use appropriate shapes and colors for academic context.
            """,
            "type": "workflow"
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}/{len(test_cases)}: {test_case['name']}")
        
        result = generator.generate_drawio_xml(
            description=test_case["description"],
            diagram_type=test_case["type"]
        )
        
        print(f"✅ Success: {result['success']}")
        if result['success']:
            print(f"📁 File: {result.get('file_path', '')}")
            print(f"🎯 Type: {result.get('diagram_type', '')}")
            
            # Show content preview
            content = result.get("xml_content", "")
            if content:
                print(f"\n📝 Content preview (first 300 chars):")
                print(content[:300] + "..." if len(content) > 300 else content)
        else:
            print(f"❌ Error: {result.get('error', '')}")
            print(f"🔍 Full response: {result.get('full_response', '')[:500]}...")
        
        # Add delay to avoid rate limiting
        if i < len(test_cases):
            import time
            time.sleep(3)
    
    print("\n🎉 All tests completed!")
    print(f"📁 Check the 'diagram_outputs' directory for generated files")


if __name__ == "__main__":
    main() 