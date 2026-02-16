#!/usr/bin/env python3
"""
Test DrawerTool with Gemini models to verify SVG generation works correctly.

This tests the actual TinyScientist implementation to ensure Gemini models
can generate diagrams just like GPT-4.

Usage:
    export GOOGLE_API_KEY=your-key-here
    python test_drawer_gemini.py
"""

import json
import os
import sys


def test_drawer_with_gemini():
    """Test DrawerTool with Gemini model."""

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 80)
    print("TINY-SCIENTIST DRAWERTOOL + GEMINI TEST")
    print("=" * 80)
    print()

    try:
        # Import TinyScientist components
        from tiny_scientist.smolagents_tools import DrawerTool
        from tiny_scientist.utils.llm import AVAILABLE_LLMS, create_client

        print("✓ Imports successful")
        print()

        # Check Gemini models are available
        gemini_models = [
            m for m in AVAILABLE_LLMS if "gemini" in m.lower() and "image" in m.lower()
        ]
        print(f"✓ Found {len(gemini_models)} Gemini image models:")
        for model in gemini_models:
            print(f"  - {model}")
        print()

        if not gemini_models:
            print("❌ No Gemini image models found in AVAILABLE_LLMS")
            sys.exit(1)

        # Test client creation
        print("TEST 1: Client Creation")
        print("-" * 80)

        model_name = "gemini-3-pro-image-preview"
        try:
            client, actual_model = create_client(model_name)
            print(f"✓ Client created for: {actual_model}")
            print(f"  Client type: {type(client)}")
            print()
        except Exception as e:
            print(f"❌ Failed to create client: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Test DrawerTool initialization
        print("TEST 2: DrawerTool Initialization")
        print("-" * 80)

        try:
            drawer = DrawerTool(
                model=model_name, output_dir="./test_output", temperature=0.7
            )
            print("✓ DrawerTool initialized")
            print(f"  Model: {drawer.model}")
            print(f"  Is image model: {drawer.is_image_model}")
            print(f"  Output dir: {drawer.output_dir}")
            print()

            if drawer.is_image_model:
                print("  ⚠ WARNING: is_image_model=True")
                print("     This means base64 decoding will be attempted")
                print("     This is WRONG for SVG generation!")
                print()
        except Exception as e:
            print(f"❌ Failed to initialize DrawerTool: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Test diagram generation
        print("TEST 3: Diagram Generation")
        print("-" * 80)

        section_content = """
        Our method consists of three main components:
        1. Input Processing: Raw data is normalized and tokenized
        2. Feature Extraction: A neural network extracts relevant features
        3. Classification: A softmax layer produces the final predictions
        """

        query = json.dumps(
            {"section_name": "Method", "section_content": section_content}
        )

        try:
            print("Calling drawer.forward()...")
            result = drawer.forward(query)
            print("✓ drawer.forward() completed")
            print()

            print("Result structure:")
            print(f"  Type: {type(result)}")
            print(
                f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
            )
            print()

            if "diagram" in result:
                diagram = result["diagram"]
                print("✓ Diagram data found:")
                print(f"  Keys: {list(diagram.keys())}")
                print()

                # Check what format we got
                if "svg" in diagram:
                    svg_content = diagram.get("svg", "")
                    print(f"  ✓ SVG content: {len(svg_content)} characters")

                    if svg_content:
                        print(f"    First 200 chars: {svg_content[:200]}")

                        if "<svg" in svg_content and "</svg>" in svg_content:
                            print("    ✓ Valid SVG tags detected")
                            print()
                            print("    ✅ SUCCESS: Gemini generated SVG as text!")
                        else:
                            print("    ✗ No valid SVG tags found")
                    else:
                        print("    ✗ SVG content is empty")
                    print()

                if "image_path" in diagram:
                    image_path = diagram.get("image_path", "")
                    print(f"  Image path: {image_path}")

                    if image_path:
                        print("    ⚠ WARNING: Image path was generated")
                        print("       This suggests base64 decoding was attempted")
                        print("       This is WRONG for SVG generation!")

                        # Check if file exists
                        if os.path.exists(image_path):
                            size = os.path.getsize(image_path)
                            print(f"       File exists: {size} bytes")
                        else:
                            print("       File does NOT exist (which is expected)")
                    print()

                if "summary" in diagram:
                    summary = diagram.get("summary", "")
                    print(f"  Summary: {summary}")
                    print()

            else:
                print("✗ No 'diagram' key in result")
                print(f"  Full result: {result}")

        except Exception as e:
            print(f"❌ Diagram generation failed: {e}")
            import traceback

            traceback.print_exc()
            print()

        print()
        print("=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)
        print(
            """
If you see:
- ✓ SVG content with valid tags → Implementation works correctly!
- ✗ Image path generated → Base64 logic shouldn't be used
- ✗ Errors about base64 → Model returns text, not base64

Expected behavior:
- Gemini should return SVG code as TEXT (just like GPT-4)
- No base64 decoding needed
- No image file should be created
- Result should have diagram["svg"] with actual SVG XML
"""
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print(
            "Make sure you're in the project directory and dependencies are installed:"
        )
        print("  cd /path/to/tiny-scientist")
        print("  pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    test_drawer_with_gemini()
