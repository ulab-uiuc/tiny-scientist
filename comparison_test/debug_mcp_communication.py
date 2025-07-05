#!/usr/bin/env python3
"""
调试MCP通信问题
"""

import asyncio
import json
import sys
import os
import subprocess
import time
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_direct_server_communication():
    """直接测试服务器通信"""
    print("🔌 Testing Direct Server Communication")
    print("="*50)
    
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
            
            # 发送初始化请求
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
            print(f"Sending init request: {request_json.strip()}")
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # 读取响应
            response_line = process.stdout.readline()
            print(f"Init response: {response_line.strip()}")
            
            if response_line:
                try:
                    response = json.loads(response_line.strip())
                    print(f"✅ Init successful: {response}")
                    
                    # 发送initialized通知
                    initialized_notification = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {}
                    }
                    
                    notification_json = json.dumps(initialized_notification) + "\n"
                    print(f"Sending initialized notification: {notification_json.strip()}")
                    process.stdin.write(notification_json)
                    process.stdin.flush()
                    
                    # 等待一下
                    time.sleep(1)
                    
                    # 测试工具调用
                    tool_request = {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "search_papers",
                            "arguments": {
                                "query": "machine learning",
                                "result_limit": 2
                            }
                        }
                    }
                    
                    request_json = json.dumps(tool_request) + "\n"
                    print(f"Sending tool request: {request_json.strip()}")
                    process.stdin.write(request_json)
                    process.stdin.flush()
                    
                    # 读取响应
                    response_line = process.stdout.readline()
                    print(f"Tool response: {response_line.strip()}")
                    
                    if response_line:
                        try:
                            response = json.loads(response_line.strip())
                            print(f"✅ Tool call successful: {response}")
                        except json.JSONDecodeError as e:
                            print(f"❌ Tool response JSON error: {e}")
                            print(f"Raw response: {repr(response_line)}")
                    else:
                        print("❌ No tool response")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Init response JSON error: {e}")
                    print(f"Raw response: {repr(response_line)}")
            else:
                print("❌ No init response")
                
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

def test_server_startup():
    """测试服务器启动"""
    print("\n🚀 Testing Server Startup")
    print("="*50)
    
    server_script = os.path.join(os.path.dirname(__file__), '..', 'tiny_scientist', 'mcp', 'paper_search_server.py')
    
    print(f"Server script: {server_script}")
    print(f"Script exists: {os.path.exists(server_script)}")
    
    # 检查Python模块是否可以导入
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("paper_search_server", server_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Module can be imported")
    except Exception as e:
        print(f"❌ Module import failed: {e}")
    
    # 检查配置文件
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    print(f"Config path: {config_path}")
    print(f"Config exists: {os.path.exists(config_path)}")
    
    if os.path.exists(config_path):
        try:
            import toml
            config = toml.load(config_path)
            print(f"✅ Config loaded successfully")
            print(f"   S2 API Key: {'Yes' if config.get('core', {}).get('s2_api_key') else 'No'}")
            print(f"   Search engine: {config.get('core', {}).get('engine', 'semanticscholar')}")
        except Exception as e:
            print(f"❌ Config load failed: {e}")

async def test_mcp_client():
    """测试MCP客户端"""
    print("\n🔧 Testing MCP Client")
    print("="*50)
    
    try:
        from tiny_scientist.utils.mcp_client import MCPClient
        
        client = MCPClient()
        
        # 启动服务器
        print("Starting paper_search server...")
        success = await client.start_server("paper_search")
        
        if success:
            print("✅ Server started successfully")
            
            # 测试工具调用
            print("Testing tool call...")
            result = await client.call_tool(
                "paper_search", 
                "search_papers", 
                query="machine learning", 
                result_limit=2
            )
            
            if result:
                print(f"✅ Tool call successful")
                print(f"Result: {result[:200]}...")
            else:
                print("❌ Tool call failed")
        else:
            print("❌ Server start failed")
            
    except Exception as e:
        print(f"❌ MCP client test failed: {e}")
        
    finally:
        if 'client' in locals():
            await client.stop_all_servers()

def main():
    """主函数"""
    print("🔬 MCP Communication Debug")
    print("="*60)
    
    # 测试1: 服务器启动
    test_server_startup()
    
    # 测试2: 直接通信
    test_direct_server_communication()
    
    # 测试3: MCP客户端
    asyncio.run(test_mcp_client())
    
    print("\n" + "="*60)
    print("📊 DEBUG SUMMARY")
    print("="*60)

if __name__ == "__main__":
    main() 