import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tiny_scientist.utils.mcp_client import MCPClient

async def test_paper_search_server():
    """测试paper_search MCP服务器"""
    print("🧪 Testing Paper Search MCP Server")
    print("="*50)
    
    # 初始化MCP客户端
    client = MCPClient()
    
    try:
        # 启动paper_search服务器
        print("🚀 Starting paper_search server...")
        success = await client.start_server("paper_search")
        
        if not success:
            print("❌ Failed to start paper_search server")
            return False
        
        print("✅ Paper search server started successfully")
        
        # 测试搜索论文功能
        test_queries = [
            "adaptive step size optimization",
            "gradient descent variants",
            "neural network training methods"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📚 Test {i}: Searching for '{query}'")
            
            try:
                result = await client.call_tool(
                    "paper_search", 
                    "search_papers", 
                    query=query, 
                    result_limit=2
                )
                
                if result:
                    print(f"✅ Search successful")
                    try:
                        # 尝试解析JSON结果
                        parsed_result = json.loads(result)
                        if isinstance(parsed_result, dict):
                            paper_count = len(parsed_result)
                            print(f"   Found {paper_count} papers")
                            
                            # 显示前几个论文的标题
                            for j, (title, paper_data) in enumerate(parsed_result.items()):
                                if j < 2:  # 只显示前2个
                                    print(f"   {j+1}. {title}")
                                    if isinstance(paper_data, dict) and "bibtex" in paper_data:
                                        bibtex_available = paper_data["bibtex"] != "N/A"
                                        print(f"      BibTeX: {'✅' if bibtex_available else '❌'}")
                        else:
                            print(f"   Unexpected result format: {type(parsed_result)}")
                    except json.JSONDecodeError:
                        print(f"   Raw result: {result[:200]}...")
                else:
                    print("❌ Search failed - no result returned")
                    
            except Exception as e:
                print(f"❌ Search failed with error: {e}")
        
        # 测试获取可用工具
        print(f"\n🔧 Testing available tools...")
        tools = await client.get_available_tools("paper_search")
        if tools:
            print(f"✅ Available tools: {[tool.get('name', 'unknown') for tool in tools]}")
        else:
            print("❌ No tools available")
        
        # 健康检查
        print(f"\n🏥 Health check...")
        health = await client.health_check()
        print(f"Health status: {health}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # 清理：停止所有服务器
        print(f"\n🧹 Cleaning up...")
        await client.stop_all_servers()

async def test_direct_server_communication():
    """直接测试服务器通信"""
    print("\n🔌 Testing Direct Server Communication")
    print("="*50)
    
    import subprocess
    import time
    
    # 启动paper_search服务器进程
    server_script = os.path.join(os.path.dirname(__file__), '..', 'tiny_scientist', 'mcp', 'paper_search_server.py')
    
    print(f"Starting server: {server_script}")
    
    try:
        process = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        time.sleep(2)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ Server process is running")
            
            # 发送简单的初始化请求
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            request_json = json.dumps(init_request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # 读取响应
            response_line = process.stdout.readline()
            if response_line:
                print(f"✅ Server responded: {response_line.strip()}")
            else:
                print("❌ No response from server")
                
        else:
            print("❌ Server process terminated")
            stderr_output = process.stderr.read()
            print(f"Server stderr: {stderr_output}")
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        
    finally:
        # 清理进程
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

async def main():
    """主测试函数"""
    print("🔬 MCP Paper Search Server Test")
    print("="*60)
    
    # 测试1: 使用MCP客户端
    print("\n1️⃣ Testing with MCP Client")
    success1 = await test_paper_search_server()
    
    # 测试2: 直接服务器通信
    print("\n2️⃣ Testing Direct Server Communication")
    await test_direct_server_communication()
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ MCP Client Test: {'PASSED' if success1 else 'FAILED'}")
    print("✅ Direct Server Test: COMPLETED")
    
    return success1

if __name__ == "__main__":
    asyncio.run(main()) 