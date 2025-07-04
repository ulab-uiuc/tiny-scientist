#!/usr/bin/env python3
"""
Direct test of paper search API functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the search function directly
from tiny_scientist.mcp.paper_search_server import search_papers, S2_API_KEY, config

async def test_paper_search_directly():
    """Test paper search functionality directly"""
    
    print("🔧 Direct Paper Search API Test")
    print("="*40)
    
    # Test configuration
    print(f"API Key: {'Configured' if S2_API_KEY else 'Missing'}")
    if S2_API_KEY:
        print(f"API Key preview: {S2_API_KEY[:10]}...")
    print(f"Config keys: {list(config.keys())}")
    if 'core' in config:
        print(f"Core config keys: {list(config['core'].keys())}")
    
    # Test a simple search
    print("\n📄 Testing direct search...")
    test_query = "machine learning"
    
    try:
        result = await search_papers(test_query, result_limit=2)
        print(f"✅ Search result: {result}")
        
        # Try to parse the result
        import json
        parsed = json.loads(result)
        print(f"✅ Parsed result type: {type(parsed)}")
        print(f"✅ Parsed result keys: {list(parsed.keys())}")
        
        for key, value in parsed.items():
            print(f"  - {key}: {type(value)} -> {value}")
            
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_paper_search_directly()) 