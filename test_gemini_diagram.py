#!/usr/bin/env python3
"""
Test script to verify Gemini model output format for diagram generation.

This script tests what Gemini image models actually return when prompted
to generate SVG diagrams vs raster images.

Usage:
    export GOOGLE_API_KEY=your-key-here
    python test_gemini_diagram.py
"""

import json
import os
import sys


def test_gemini_response_format():
    """Test what Gemini models return for diagram generation."""

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY=your-key-here")
        sys.exit(1)

    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ ERROR: google-generativeai package not installed")
        print("   Install with: pip install google-generativeai")
        sys.exit(1)

    print("=" * 80)
    print("GEMINI DIAGRAM OUTPUT FORMAT TEST")
    print("=" * 80)
    print()

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Test 1: Text-based SVG generation (what we actually want)
    print("TEST 1: SVG Code Generation (Text Output)")
    print("-" * 80)

    svg_prompt = """Generate a simple SVG diagram for a neural network architecture.

Respond with a valid JSON object containing:
- "summary": a one-sentence description
- "svg": a complete SVG code string

Format:
{
  "summary": "Neural network with input, hidden, and output layers",
  "svg": "<svg width='400' height='300'>...</svg>"
}

Do not wrap in code blocks or markdown."""

    try:
        # Test with gemini-3-pro-image-preview (text mode)
        model = genai.GenerativeModel("gemini-3-pro-image-preview")
        response = model.generate_content(svg_prompt)

        print("✓ Model: gemini-3-pro-image-preview")
        print(f"✓ Response type: {type(response)}")
        print()

        # Check what we got
        if hasattr(response, "text"):
            print("✓ response.text available: YES")
            print(f"  Length: {len(response.text)} characters")
            print()
            print("First 500 characters of response.text:")
            print("-" * 80)
            print(response.text[:500])
            print("-" * 80)
            print()

            # Try to parse as JSON
            try:
                parsed = json.loads(response.text)
                print("✓ JSON parsing: SUCCESS")
                print(f"  Keys: {list(parsed.keys())}")

                if "summary" in parsed:
                    print(f"  summary: {parsed['summary'][:100]}...")

                if "svg" in parsed:
                    svg_content = parsed["svg"]
                    print(f"  svg length: {len(svg_content)} characters")
                    print(f"  svg starts with: {svg_content[:50]}")

                    # Verify it's actual SVG
                    if "<svg" in svg_content and "</svg>" in svg_content:
                        print("  ✓ Contains valid SVG tags")
                    else:
                        print("  ⚠ Does not contain SVG tags")

            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing: FAILED - {e}")
                print("  Attempting to find SVG in raw text...")
                if "<svg" in response.text:
                    print("  ✓ Found SVG tags in response")
                else:
                    print("  ✗ No SVG tags found")
        else:
            print("✗ response.text: NOT AVAILABLE")

        # Check for other attributes
        print()
        print("Response structure:")
        print(
            f"  candidates: {len(response.candidates) if hasattr(response, 'candidates') else 'N/A'}"
        )

        if hasattr(response, "candidates") and len(response.candidates) > 0:
            candidate = response.candidates[0]
            print(
                f"  content.parts: {len(candidate.content.parts) if hasattr(candidate.content, 'parts') else 'N/A'}"
            )

            if hasattr(candidate.content, "parts"):
                for i, part in enumerate(candidate.content.parts):
                    print(f"  part[{i}]:")
                    if hasattr(part, "text"):
                        print(f"    - text: {len(part.text) if part.text else 0} chars")
                    if hasattr(part, "inline_data"):
                        print(f"    - inline_data: {part.inline_data}")

        print()
        print("=" * 80)
        print()

    except Exception as e:
        print(f"✗ ERROR in Test 1: {e}")
        import traceback

        traceback.print_exc()
        print()

    # Test 2: Image generation mode (what we DON'T want for SVG)
    print("TEST 2: Raster Image Generation (IMAGE modality)")
    print("-" * 80)

    image_prompt = "Generate a diagram of a neural network architecture"

    try:
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=image_prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

        print("✓ Model: gemini-3-pro-image-preview (IMAGE modality)")
        print()

        has_image = False
        has_text = False

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                has_image = True
                print("✓ Found inline_data (raster image)")
                print(f"  MIME type: {part.inline_data.mime_type}")
                print(f"  Data type: {type(part.inline_data.data)}")
                print(f"  Data length: {len(part.inline_data.data)} bytes")
                print()

                # Save sample
                if part.inline_data.mime_type == "image/png":
                    with open("/tmp/gemini_test_image.png", "wb") as f:
                        f.write(part.inline_data.data)
                    print("  Saved to: /tmp/gemini_test_image.png")

            if hasattr(part, "text") and part.text:
                has_text = True
                print(f"✓ Found text: {part.text[:100]}...")

        if not has_image and not has_text:
            print("✗ No image or text data found")

        print()
        print("=" * 80)
        print()

    except Exception as e:
        print(f"⚠ Test 2 skipped or failed: {e}")
        print("  (This is expected if IMAGE modality is not available)")
        print()

    # Test 3: Compare with text-only Gemini model
    print("TEST 3: Standard Gemini (Text-only model for comparison)")
    print("-" * 80)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(svg_prompt)

        print("✓ Model: gemini-1.5-flash")
        print(f"✓ Response has text: {hasattr(response, 'text')}")

        if hasattr(response, "text"):
            print(f"  Length: {len(response.text)} characters")
            print()
            print("First 300 characters:")
            print(response.text[:300])
            print()

            # Check if it's valid JSON with SVG
            try:
                parsed = json.loads(response.text)
                if "svg" in parsed and "<svg" in parsed["svg"]:
                    print("✓ Standard Gemini also returns SVG as text in JSON")
            except:
                pass

        print()
        print("=" * 80)

    except Exception as e:
        print(f"✗ ERROR in Test 3: {e}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        """
For SVG diagram generation in scientific papers:

1. ✓ Use DEFAULT text mode (no IMAGE modality)
2. ✓ Prompt asks for SVG code in JSON format
3. ✓ Response comes as response.text (string)
4. ✓ Parse JSON and extract SVG string
5. ✗ NO base64 decoding needed
6. ✗ NO binary image data handling needed

Gemini image models work EXACTLY like GPT-4 for SVG code generation!
They only differ when using IMAGE modality (which we don't want).
"""
    )


if __name__ == "__main__":
    test_gemini_response_format()
