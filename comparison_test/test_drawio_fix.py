#!/usr/bin/env python3
"""
Test script for DrawIO ID fix functionality
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawio_diagram_generator import DrawIODiagramGenerator


def test_id_fix():
    """Test the ID fixing functionality"""
    
    print("🧪 Testing DrawIO ID Fix Functionality")
    print("=" * 50)
    
    # Initialize generator
    generator = DrawIODiagramGenerator()
    
    # Test XML with duplicate/missing IDs
    problematic_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="5.0" etag="example" version="22.1.16" type="device">
  <diagram name="Test Diagram" id="test-diagram">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="" value="Start" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
          <mxGeometry x="320" y="80" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="duplicate" value="Process" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="320" y="200" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="duplicate" value="End" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" vertex="1" parent="1">
          <mxGeometry x="320" y="320" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell edge="1" parent="1" source="start" target="process">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell edge="1" parent="1" source="process" target="end">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    
    print("📝 Original XML (with problems):")
    print(problematic_xml)
    print("\n" + "="*50)
    
    # Fix the XML
    fixed_xml = generator._fix_xml_ids(problematic_xml)
    
    print("🔧 Fixed XML:")
    print(fixed_xml)
    print("\n" + "="*50)
    
    # Check for problems
    import re
    
    # Check for empty IDs
    empty_ids = re.findall(r'id=""', fixed_xml)
    print(f"❌ Empty IDs found: {len(empty_ids)}")
    
    # Check for duplicate IDs
    all_ids = re.findall(r'id="([^"]*)"', fixed_xml)
    unique_ids = set(all_ids)
    print(f"📊 Total IDs: {len(all_ids)}")
    print(f"📊 Unique IDs: {len(unique_ids)}")
    print(f"❌ Duplicate IDs: {len(all_ids) - len(unique_ids)}")
    
    if len(empty_ids) == 0 and len(all_ids) == len(unique_ids):
        print("✅ ID fix successful!")
        return True
    else:
        print("❌ ID fix failed!")
        return False


def test_simple_generation():
    """Test simple diagram generation"""
    
    print("\n🎨 Testing Simple Diagram Generation")
    print("=" * 50)
    
    # Initialize generator
    generator = DrawIODiagramGenerator()
    
    # Generate simple example
    try:
        file_path = generator.generate_simple_example()
        print(f"✅ Simple example generated: {file_path}")
        
        # Check if file exists and is valid
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic XML structure
            if '<mxfile' in content and '</mxfile>' in content:
                print("✅ Generated file has valid XML structure")
                return True
            else:
                print("❌ Generated file has invalid XML structure")
                return False
        else:
            print("❌ Generated file does not exist")
            return False
            
    except Exception as e:
        print(f"❌ Error generating simple example: {e}")
        return False


def main():
    """Run all tests"""
    
    print("🧪 DrawIO Generator Test Suite")
    print("=" * 50)
    
    # Test ID fix
    id_fix_success = test_id_fix()
    
    # Test simple generation
    generation_success = test_simple_generation()
    
    # Summary
    print("\n📊 Test Results:")
    print(f"ID Fix Test: {'✅ PASS' if id_fix_success else '❌ FAIL'}")
    print(f"Generation Test: {'✅ PASS' if generation_success else '❌ FAIL'}")
    
    if id_fix_success and generation_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")


if __name__ == "__main__":
    main() 