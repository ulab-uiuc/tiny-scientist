#!/usr/bin/env python3
"""
Test Nano Banana core function directly
"""

import os

# Import and test the core function
from nano_banana_server import generate_diagram


def test_nano_banana_core():
    """Test the core generate_diagram function"""
    print("🍌 Testing Nano Banana core function...")

    # Check if GEMINI_API_KEY is set
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("\n❌ GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-api-key-here'")
        print("\nTesting without API key (should return error gracefully)...")
        result = generate_diagram("Method", "This is a test method section.")
        print(f"Result: {result}")
        return result

    print(f"\n✅ GEMINI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
    print("\n🎨 Testing diagram generation with real API key...")

    # Test with real API key
    result = generate_diagram(
        "Results",
        "This section presents experimental results showing improved accuracy of our machine learning model, achieving 95.2% accuracy on the benchmark dataset.",
    )

    print(f"Summary: {result.get('summary', 'No summary')}")
    print(f"SVG length: {len(result.get('svg', ''))}")

    if result.get("svg"):
        print("\n🖼️ SVG content preview:")
        svg = result["svg"]
        print(svg[:300] + "..." if len(svg) > 300 else svg)
    else:
        print("\n⚠️ No SVG content generated")

    return result


if __name__ == "__main__":
    try:
        results = test_nano_banana_core()
        print("\n✅ Core function test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
