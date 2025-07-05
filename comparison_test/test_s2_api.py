#!/usr/bin/env python3
"""
直接测试Semantic Scholar API
"""

import requests
import json
import time

def test_s2_api():
    """测试Semantic Scholar API"""
    print("🔬 Testing Semantic Scholar API")
    print("="*50)
    
    # API配置 - 使用无认证访问
    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    S2_API_KEY = None  # 使用无认证访问
    
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    
    print(f"API Key: {'Yes' if S2_API_KEY else 'No'}")
    print(f"Headers: {headers}")
    
    # 测试查询
    test_queries = [
        "machine learning",
        "adaptive step size optimization",
        "gradient descent",
        "neural network training"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📚 Test {i}: '{query}'")
        
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,paperId",
        }
        
        url = f"{S2_API_BASE}/paper/search"
        
        try:
            print(f"Making request to: {url}")
            print(f"Params: {params}")
            
            response = requests.get(url, headers=headers, params=params, timeout=30.0)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                total = result.get("total", 0)
                data = result.get("data", [])
                
                print(f"✅ Success: Found {total} total papers, {len(data)} returned")
                
                if data:
                    for j, paper in enumerate(data[:2], 1):
                        title = paper.get("title", "Unknown")
                        authors = paper.get("authors", [])
                        author_names = [author.get("name", "Unknown") for author in authors[:3]]
                        print(f"   {j}. {title}")
                        print(f"      Authors: {', '.join(author_names)}")
                        print(f"      Year: {paper.get('year', 'Unknown')}")
                        print(f"      Venue: {paper.get('venue', 'Unknown')}")
                else:
                    print("   No papers found in response")
                time.sleep(5)
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response text: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # 添加延迟
        if i < len(test_queries):
            print("⏳ Waiting 2 seconds...")
            time.sleep(2)

def test_simple_query():
    """测试简单查询"""
    print("\n🧪 Testing Simple Query")
    print("="*50)
    
    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    S2_API_KEY = "n1gleFbCPq5SMMHPOEsrf5bvU8mgEJ0t5uyJvlqe"
    
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    
    # 最简单的查询
    params = {
        "query": "machine learning",
        "limit": 1
    }
    
    url = f"{S2_API_BASE}/paper/search"
    
    try:
        print(f"Making simple request...")
        response = requests.get(url, headers=headers, params=params, timeout=30.0)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_api_key():
    """测试API密钥"""
    print("\n🔑 Testing API Key")
    print("="*50)
    
    S2_API_KEY = "n1gleFbCPq5SMMHPOEsrf5bvU8mgEJ0t5uyJvlqe"
    
    # 测试不带API密钥
    print("Testing without API key...")
    headers = {}
    response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", 
                          headers=headers, 
                          params={"query": "test", "limit": 1}, 
                          timeout=10.0)
    print(f"Without key - Status: {response.status_code}")
    
    # 测试带API密钥
    print("Testing with API key...")
    headers = {"X-API-KEY": S2_API_KEY}
    response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", 
                          headers=headers, 
                          params={"query": "test", "limit": 1}, 
                          timeout=10.0)
    print(f"With key - Status: {response.status_code}")

def main():
    """主函数"""
    print("🔬 Semantic Scholar API Test")
    print("="*60)
    
    # 测试1: API密钥
    test_api_key()
    
    # 测试2: 简单查询
    test_simple_query()
    
    # 测试3: 完整API测试
    test_s2_api()
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

if __name__ == "__main__":
    main() 