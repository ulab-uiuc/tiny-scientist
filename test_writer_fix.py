#!/usr/bin/env python3
"""
Test script to verify Writer search functionality fixes
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tiny_scientist.scientist import TinyScientist

async def test_writer_search_fix():
    """Test the Writer search functionality with proper error handling"""
    
    print("🧪 Testing Writer search fix...")
    
    # Simple test idea with all required fields
    test_idea = {
        "Title": "Test Adaptive Step Size Research",
        "Problem": "Evaluating different adaptive step size strategies",
        "Importance": "Important for optimization efficiency",
        "Difficulty": "Requires systematic benchmarking",
        "NoveltyComparison": "First comprehensive comparison",
        "Approach": "Systematic evaluation framework",
        "problem": "Evaluating different adaptive step size strategies",  # lowercase version
        "importance": "Important for optimization efficiency",  # lowercase version
        "difficulty": "Requires systematic benchmarking",  # lowercase version
        "novelty": "First comprehensive comparison",  # lowercase version
        "Experiment": {
            "Model": "Adaptive Optimizer",
            "Dataset": "Synthetic Functions",
            "Metric": "Convergence Rate"
        },
        "is_experimental": False  # Non-experimental to avoid needing experiment files
    }
    
    # Create TinyScientist instance
    scientist = TinyScientist(
        model="gpt-4o-mini",
        use_mcp=True
    )
    
    try:
        async with scientist:
            print("✅ MCP servers initialized")
            
            # Test just the writing phase
            print("📝 Testing paper writing (this will test search functionality)...")
            
            try:
                pdf_path = scientist.write(test_idea)
                print(f"✅ Paper writing completed: {pdf_path}")
                
            except Exception as e:
                print(f"❌ Paper writing failed: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Writer Search Fix Test")
    print("="*30)
    
    try:
        asyncio.run(test_writer_search_fix())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc() 