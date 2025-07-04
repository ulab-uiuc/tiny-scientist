#!/usr/bin/env python3
"""
Test Semantic Scholar API key validity
"""

import asyncio
import httpx

async def test_api_key() -> None:
    """Test API key validity with various methods"""
    
    print("🔧 Testing Semantic Scholar API Key Validity")
    print("="*50)
    
    api_key = "n1gleFbCPq5SMMHPOEsrf5bvU8mgEJ0t5uyJvlqe"
    base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Test 1: Simple search without API key
    print("\n📄 Test 1: Search WITHOUT API key...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{base_url}/paper/search",
                params={"query": "machine learning", "limit": 1},
                timeout=30.0
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: Found {result.get('total', 0)} papers")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 2: Simple search with API key
    print("\n📄 Test 2: Search WITH API key...")
    headers = {"X-API-KEY": api_key}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{base_url}/paper/search",
                params={"query": "machine learning", "limit": 1},
                headers=headers,
                timeout=30.0
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: Found {result.get('total', 0)} papers")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 3: Check API key validity with a simple endpoint
    print("\n📄 Test 3: Testing API key with paper details endpoint...")
    # Use a well-known paper ID
    paper_id = "649def34f8be52c8b66281af98ae884c09aef38b"  # Attention is All You Need
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{base_url}/paper/{paper_id}",
                params={"fields": "title,authors"},
                headers=headers,
                timeout=30.0
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result.get('title', 'Unknown title')}")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_key()) 