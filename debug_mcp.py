#!/usr/bin/env python3
"""
MCP Diagnostics and Troubleshooting Script - Help identify issues in TinyScientist MCP architecture
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tiny_scientist.utils.mcp_client import MCPClient


async def test_mcp_servers() -> None:
    """Test all MCP server startup and functionality"""
    print("🧪 Starting MCP server diagnostics...")
    
    client = MCPClient()
    
    print(f"📋 Configured servers: {list(client.server_configs.keys())}")
    
    # 1. Test server startup
    print("\n🚀 Testing server startup...")
    start_results = await client.start_all_servers()
    
    for server_name, success in start_results.items():
        if success:
            print(f"✅ {server_name}: Started successfully")
        else:
            print(f"❌ {server_name}: Failed to start")
    
    # 2. Health check
    print("\n🏥 Running health checks...")
    health_results = await client.health_check()
    
    for server_name, healthy in health_results.items():
        if healthy:
            print(f"✅ {server_name}: Healthy")
        else:
            print(f"❌ {server_name}: Unhealthy")
    
    # 3. Test available tools
    print("\n🛠️ Getting available tools...")
    for server_name in client.server_configs.keys():
        if client.is_server_running(server_name):
            tools = await client.get_available_tools(server_name)
            if tools:
                tool_names = [tool['name'] for tool in tools]
                print(f"✅ {server_name}: {tool_names}")
            else:
                print(f"❌ {server_name}: Cannot get tool list")
        else:
            print(f"❌ {server_name}: Server not running")
    
    # 4. Test specific functionality
    print("\n🧪 Testing specific tool functions...")
    
    # Test paper search
    if client.is_server_running("paper_search"):
        print("📄 Testing paper search...")
        result = await client.call_tool(
            "paper_search", 
            "search_papers", 
            query="machine learning optimization",
            result_limit=2
        )
        if result:
            print(f"✅ Paper search successful, result length: {len(result)} characters")
            try:
                parsed = json.loads(result)
                print(f"✅ JSON parsing successful, found {len(parsed)} papers")
                
                # Check data format
                for title, meta in parsed.items():
                    print(f"📰 Paper: {title}")
                    if isinstance(meta, dict):
                        print(f"   🔗 Has bibtex: {'bibtex' in meta}")
                        print(f"   📝 Data type: {type(meta)}")
                    else:
                        print(f"   ❌ Format error: expected dict, got {type(meta)}")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"❌ Raw result: {result[:200]}...")
        else:
            print("❌ Paper search failed")
    
    # Test code search
    if client.is_server_running("code_search"):
        print("💻 Testing code search...")
        result = await client.call_tool(
            "code_search", 
            "search_github_repositories", 
            query="machine learning",
            result_limit=2
        )
        if result:
            print(f"✅ Code search successful")
            try:
                parsed = json.loads(result)
                print(f"✅ JSON parsing successful, found {len(parsed)} repositories")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
        else:
            print("❌ Code search failed")
    
    # Test drawer
    if client.is_server_running("drawer"):
        print("🎨 Testing diagram generation...")
        result = await client.call_tool(
            "drawer", 
            "generate_diagram", 
            section_name="method",
            section_content="This is a test method for generating diagrams"
        )
        if result:
            print(f"✅ Diagram generation successful")
            try:
                parsed = json.loads(result)
                print(f"✅ JSON parsing successful")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
        else:
            print("❌ Diagram generation failed")
    
    # 5. Add detailed logging
    print("\n📊 Status summary:")
    print(f"Configured server count: {len(client.server_configs)}")
    print(f"Running servers: {client.get_running_servers()}")
    
    # Cleanup
    print("\n🧹 Cleaning up servers...")
    await client.stop_all_servers()
    print("✅ All servers stopped")

async def debug_writer_search() -> None:
    """Debug search functionality in writer"""
    print("\n🔍 Debugging Writer search format issues...")
    
    client = MCPClient()
    await client.start_server("paper_search")
    
    if client.is_server_running("paper_search"):
        # Simulate writer search
        print("📄 Simulating Writer paper search...")
        result = await client.call_tool(
            "paper_search", 
            "search_papers", 
            query="adaptive step size optimization",
            result_limit=2
        )
        
        if result:
            print(f"🔄 Raw MCP result: {result[:200]}...")
            
            try:
                parsed_result = json.loads(result)
                print(f"✅ MCP parsing successful: {type(parsed_result)}")
                
                # Check if format conversion is needed
                for title, meta in parsed_result.items():
                    print(f"\n📰 Paper title: {title}")
                    print(f"📊 Meta type: {type(meta)}")
                    print(f"📊 Meta content: {meta}")
                    
                    if isinstance(meta, dict):
                        if 'bibtex' in meta:
                            print("✅ Format correct: contains bibtex field")
                        else:
                            print("❌ Format error: missing bibtex field")
                    else:
                        print(f"❌ Format error: expected dict, got {type(meta)}")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
    await client.stop_all_servers()

if __name__ == "__main__":
    print("🔧 TinyScientist MCP Diagnostics Tool")
    print("="*50)
    
    try:
        # Run diagnostics
        asyncio.run(test_mcp_servers())
        asyncio.run(debug_writer_search())
        
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc() 