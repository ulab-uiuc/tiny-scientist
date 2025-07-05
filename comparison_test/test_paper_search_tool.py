import json
import sys
import os
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tiny_scientist.mcp.tool import PaperSearchTool

def test_paper_search_tool():
    """测试PaperSearchTool的基本功能"""
    print("🧪 Testing PaperSearchTool from tool.py")
    print("="*50)
    
    # 初始化PaperSearchTool
    try:
        tool = PaperSearchTool()
        print("✅ PaperSearchTool initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PaperSearchTool: {e}")
        return False
    
    # 测试查询列表
    test_queries = [
        "adaptive step size optimization",
        "gradient descent variants",
        "neural network training methods",
        "reinforcement learning algorithms",
        "deep learning optimization"
    ]
    
    results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📚 Test {i}: Searching for '{query}'")
        
        try:
            # 调用PaperSearchTool的run方法
            result = tool.run(query)
            
            if result:
                print(f"✅ Search successful")
                print(f"📄 Found {len(result)} papers")
                
                # 显示结果详情
                for j, (title, paper_data) in enumerate(result.items()):
                    print(f"   📄 {j+1}. {title}")
                    if isinstance(paper_data, dict):
                        bibtex_available = paper_data.get("bibtex", "N/A") != "N/A"
                        print(f"      BibTeX: {'✅' if bibtex_available else '❌'}")
                        
                        # 显示其他信息
                        source = paper_data.get("source", "N/A")
                        info = paper_data.get("info", "N/A")
                        print(f"      Source: {source}")
                        print(f"      Info: {info}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # 保存结果
        results[query] = result
        
        # 在查询之间等待1秒
        if i < len(test_queries):
            print("   ⏳ Waiting 1 second before next query...")
            import time
            time.sleep(1.0)
    
    return results

def test_paper_search_with_idea():
    """测试使用研究想法作为查询"""
    print("\n🧪 Testing PaperSearchTool with Research Ideas")
    print("="*50)
    
    # 研究想法列表
    research_ideas = [
        {
            "Title": "Adaptive Learning Rate Methods for Deep Neural Networks",
            "Description": "Investigating novel adaptive learning rate strategies that automatically adjust based on gradient statistics and loss landscape analysis.",
            "Methodology": "We propose a new adaptive learning rate algorithm that combines momentum-based optimization with dynamic step size adjustment."
        },
        {
            "Title": "Gradient Descent Variants for Convex Optimization",
            "Description": "Comparing different gradient descent variants including SGD, Adam, and RMSprop for convex optimization problems.",
            "Methodology": "We implement and benchmark various gradient descent algorithms on standard convex optimization test functions."
        },
        {
            "Title": "Neural Network Training with Regularization",
            "Description": "Exploring regularization techniques for improving neural network generalization and preventing overfitting.",
            "Methodology": "We investigate L1, L2 regularization, dropout, and early stopping methods for neural network training."
        }
    ]
    
    tool = PaperSearchTool()
    results = {}
    
    for i, idea in enumerate(research_ideas, 1):
        print(f"\n📚 Test {i}: Searching with research idea")
        print(f"   Title: {idea['Title']}")
        print(f"   Description: {idea['Description'][:100]}...")
        
        try:
            # 将研究想法转换为JSON字符串
            idea_json = json.dumps(idea)
            result = tool.run(idea_json)
            
            if result:
                print(f"✅ Search successful")
                print(f"📄 Found {len(result)} papers")
                
                for j, (title, paper_data) in enumerate(result.items()):
                    print(f"   📄 {j+1}. {title}")
                    if isinstance(paper_data, dict):
                        bibtex_available = paper_data.get("bibtex", "N/A") != "N/A"
                        print(f"      BibTeX: {'✅' if bibtex_available else '❌'}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        results[f"idea_{i}"] = result
        
        # 在查询之间等待1秒
        if i < len(research_ideas):
            print("   ⏳ Waiting 1 second before next query...")
            import time
            time.sleep(1.0)
    
    return results

def test_paper_search_configuration():
    """测试PaperSearchTool的配置"""
    print("\n🧪 Testing PaperSearchTool Configuration")
    print("="*50)
    
    # 测试不同的配置
    configs = [
        {"s2_api_key": "FfOnoChxCS2vGorFNV4sQB7KdzzRalp9ygKzAGf8", "description": "With API key"},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n📋 Test {i}: {config['description']}")
        
        try:
            # 使用自定义API key初始化
            api_key = config.get("s2_api_key")
            tool = PaperSearchTool(s2_api_key=api_key)
            
            print(f"✅ Tool initialized with API key: {'Yes' if api_key else 'No'}")
            print(f"📋 Engine: {tool.engine}")
            
            # 测试简单查询
            test_query = "machine learning"
            result = tool.run(test_query)
            
            if result:
                print(f"✅ Search successful, found {len(result)} papers")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            import traceback
            traceback.print_exc()

def test_search_for_papers_method():
    """测试search_for_papers方法"""
    print("\n🧪 Testing search_for_papers Method")
    print("="*50)
    
    tool = PaperSearchTool()
    
    test_queries = [
        "optimization algorithms",
        "deep learning",
        "reinforcement learning"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📚 Test {i}: Direct search_for_papers call")
        print(f"   Query: {query}")
        
        try:
            # 直接调用search_for_papers方法
            papers = tool.search_for_papers(query, result_limit=2)
            
            if papers:
                print(f"✅ Found {len(papers)} papers")
                
                for j, paper in enumerate(papers):
                    title = paper.get("title", "No title")
                    authors = paper.get("authors", [])
                    venue = paper.get("venue", "Unknown venue")
                    year = paper.get("year", "Unknown year")
                    paper_id = paper.get("paperId", "No ID")
                    
                    print(f"   📄 {j+1}. {title}")
                    print(f"      Authors: {len(authors)} authors")
                    print(f"      Venue: {venue}")
                    print(f"      Year: {year}")
                    print(f"      Paper ID: {paper_id}")
                    
                    # 测试BibTeX获取
                    if paper_id and paper_id != "No ID":
                        bibtex = tool.fetch_bibtex(paper_id)
                        if bibtex and bibtex != "N/A":
                            print(f"      BibTeX: ✅ Available")
                            print(f"      BibTeX preview: {bibtex[:100]}...")
                        else:
                            print(f"      BibTeX: ❌ Not available")
            else:
                print("❌ No papers found")
                
        except Exception as e:
            print(f"❌ Search failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # 在查询之间等待1秒
        if i < len(test_queries):
            print("   ⏳ Waiting 1 second before next query...")
            import time
            time.sleep(1.0)

def main():
    """主测试函数"""
    print("🔬 PaperSearchTool Test Suite")
    print("="*60)
    
    # 测试1: 基本功能测试
    print("\n1️⃣ Basic Functionality Test")
    basic_results = test_paper_search_tool()
    
    # 测试2: 配置测试
    print("\n2️⃣ Configuration Test")
    test_paper_search_configuration()
    
    # 测试3: 研究想法测试
    print("\n3️⃣ Research Idea Test")
    idea_results = test_paper_search_with_idea()
    
    # 测试4: 直接方法测试
    print("\n4️⃣ Direct Method Test")
    test_search_for_papers_method()
    
    # 保存结果
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    # 统计结果
    total_basic_results = sum(len(result) if result else 0 for result in basic_results.values())
    total_idea_results = sum(len(result) if result else 0 for result in idea_results.values())
    
    print(f"📄 Basic tests: {total_basic_results} papers found")
    print(f"📄 Idea tests: {total_idea_results} papers found")
    print(f"✅ All tests completed")
    
    # 保存详细结果到文件
    timestamp = __import__('time').strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_test/paper_search_tool_results_{timestamp}.json"
    
    all_results = {
        "basic_tests": basic_results,
        "idea_tests": idea_results,
        "test_info": {
            "timestamp": timestamp,
            "total_basic_results": total_basic_results,
            "total_idea_results": total_idea_results
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()