"""
FastMCP Client Test for Tiny Scientist Tools

This module demonstrates how to use the FastMCP client to interact with
the Tiny Scientist MCP server. It includes examples of:
- In-memory testing (fastest for development)
- Stdio transport testing 
- Multi-server configuration testing
- All available tools, resources, and prompts

Based on the FastMCP client patterns and examples.
"""

import asyncio
import json
import os
from fastmcp import Client

async def test_in_memory_connection():
    """
    Test the MCP server using in-memory transport (fastest method)
    
    This method directly imports the server instance and connects to it
    in memory, eliminating network overhead and separate processes.
    """
    print("=== Testing In-Memory Connection ===")
    
    # Import the server instance from our MCP server module
    try:
        from test_mcp import mcp as tiny_scientist_server
    except ImportError:
        print("❌ Failed to import server. Make sure test_mcp.py is in the same directory.")
        return
    
    async with Client(tiny_scientist_server) as client:
        print("✅ Connected to Tiny Scientist MCP Server")
        
        # 1. Test server connectivity
        try:
            await client.ping()
            print("✅ Server ping successful")
        except Exception as e:
            print(f"❌ Server ping failed: {e}")
            return
        
        # Check configuration status
        await check_server_config(client)
        
        # 2. List and display available tools
        print("\n--- Available Tools ---")
        tools = await client.list_tools()
        for tool in tools:
            print(f"🔧 {tool.name}: {tool.description}")
        
        # 3. List and display available resources
        print("\n--- Available Resources ---")
        resources = await client.list_resources()
        for resource in resources:
            print(f"📄 {resource.uri}: {getattr(resource, 'description', 'No description')}")
        
        # 4. List and display available prompts
        print("\n--- Available Prompts ---")
        prompts = await client.list_prompts()
        for prompt in prompts:
            print(f"💡 {prompt.name}: {prompt.description}")
        
        # 5. Test paper search tool
        await test_paper_search(client)
        
        # 6. Test GitHub repository search
        await test_github_search(client)
        
        # 7. Test resources
        await test_resources(client)
        
        # 8. Test prompts
        await test_prompts(client)

async def check_server_config(client):
    """Check server configuration and API key status"""
    try:
        config_resource = await client.read_resource("research://config/settings")
        if config_resource and len(config_resource) > 0:
            print("\n--- Server Configuration ---")
            config_text = config_resource[0].text
            print(f"📋 {config_text}")
    except Exception as e:
        print(f"⚠️ Could not read server configuration: {e}")

async def test_paper_search(client):
    """Test paper search functionality"""
    print("\n--- Testing Paper Search ---")
    
    try:
        # Search for papers on machine learning
        print("🔍 Calling search_papers tool...")
        result = await client.call_tool("search_papers", {
            "query": "transformer neural networks",
            "result_limit": 2
        })
        
        print("📚 Paper search results:")
        print(f"🔍 Raw result type: {type(result)}")
        print(f"🔍 Raw result: {result}")
        
        if hasattr(result, 'content') and result.content:
            result_text = result.content[0].text
            print(f"🔍 Result text: {result_text}")
            try:
                result_data = json.loads(result_text)
                if "results" in result_data and result_data["results"]:
                    for title, info in result_data["results"].items():
                        print(f"\n  📖 Title: {title}")
                        print(f"     Authors: {info.get('authors', 'Unknown')}")
                        print(f"     Year: {info.get('year', 'Unknown')}")
                        print(f"     Citations: {info.get('citationCount', 0)}")
                        print(f"     Venue: {info.get('venue', 'Unknown')}")
                elif "error" in result_data:
                    print(f"  ❌ Error in paper search: {result_data['error']}")
                else:
                    print(f"  ⚠️ No papers found. Response: {result_data}")
            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse JSON: {e}")
                print(f"  Raw result: {result_text[:500]}...")
        else:
            print(f"  ❌ No content in result. Result object: {result}")
        
        # Test BibTeX retrieval (if we have a paper ID)
        print("\n📄 Testing BibTeX retrieval:")
        bibtex_result = await client.call_tool("get_paper_bibtex", {
            "paper_id": "649def34f8be52c8b66281af98ae884c09aef38b"  # Example paper ID
        })
        
        print(f"🔍 BibTeX result type: {type(bibtex_result)}")
        if hasattr(bibtex_result, 'content') and bibtex_result.content:
            bibtex_text = bibtex_result.content[0].text
            print(f"🔍 BibTeX text: {bibtex_text}")
            try:
                bibtex_data = json.loads(bibtex_text)
                if "bibtex" in bibtex_data:
                    print(f"  ✅ BibTeX retrieved successfully")
                    print(f"     {bibtex_data['bibtex'][:100]}...")
                elif "error" in bibtex_data:
                    print(f"  ❌ BibTeX error: {bibtex_data['error']}")
                else:
                    print(f"  ⚠️ Unexpected BibTeX response: {bibtex_data}")
            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse BibTeX JSON: {e}")
                print(f"  Raw BibTeX result: {bibtex_text[:200]}...")
        else:
            print(f"  ❌ No content in BibTeX result: {bibtex_result}")
                
    except Exception as e:
        print(f"❌ Paper search test failed: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")

async def test_github_search(client):
    """Test GitHub search functionality"""
    print("\n--- Testing GitHub Search ---")
    
    try:
        # Test repository search
        print("🔍 Testing repository search:")
        print("🔍 Calling search_github_repositories tool...")
        repo_result = await client.call_tool("search_github_repositories", {
            "query": "machine learning python",
            "result_limit": 3
        })
        
        print(f"🔍 Repository result type: {type(repo_result)}")
        print(f"🔍 Repository result: {repo_result}")
        
        if hasattr(repo_result, 'content') and repo_result.content:
            repo_text = repo_result.content[0].text
            print(f"🔍 Repository text: {repo_text}")
            try:
                repo_data = json.loads(repo_text)
                if "results" in repo_data and repo_data["results"]:
                    for idx, info in repo_data["results"].items():
                        print(f"\n  🏛️ Repository {idx}: {info['title']}")
                        print(f"     URL: {info['source']}")
                        print(f"     Info: {info['info']}")
                        print(f"     Description: {info.get('description', 'No description')[:100]}...")
                elif "error" in repo_data:
                    print(f"  ❌ Error in repository search: {repo_data['error']}")
                else:
                    print(f"  ⚠️ No repositories found. Response: {repo_data}")
            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse repository JSON: {e}")
                print(f"  Raw repository result: {repo_text[:500]}...")
        else:
            print(f"  ❌ No content in repository result: {repo_result}")
        
        # Test code search
        print("\n💻 Testing code search:")
        print("🔍 Calling search_github_code tool...")
        code_result = await client.call_tool("search_github_code", {
            "query": "neural network implementation",
            "result_limit": 2
        })
        
        print(f"🔍 Code result type: {type(code_result)}")
        print(f"🔍 Code result: {code_result}")
        
        if hasattr(code_result, 'content') and code_result.content:
            code_text = code_result.content[0].text
            print(f"🔍 Code text: {code_text}")
            try:
                code_data = json.loads(code_text)
                if "results" in code_data and code_data["results"]:
                    for idx, info in code_data["results"].items():
                        print(f"\n  📝 Code {idx}: {info['title']}")
                        print(f"     URL: {info['source']}")
                        print(f"     Info: {info['info']}")
                elif "error" in code_data:
                    print(f"  ❌ Error in code search: {code_data['error']}")
                else:
                    print(f"  ⚠️ No code found. Response: {code_data}")
            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse code JSON: {e}")
                print(f"  Raw code result: {code_text[:500]}...")
        else:
            print(f"  ❌ No content in code result: {code_result}")
                
    except Exception as e:
        print(f"❌ GitHub search test failed: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")

async def test_resources(client):
    """Test resource access"""
    print("\n--- Testing Resources ---")
    
    try:
        # Test configuration settings resource
        config_resource = await client.read_resource("research://config/settings")
        print("⚙️ Configuration settings:")
        if config_resource and len(config_resource) > 0:
            print(f"  {config_resource[0].text}")
        
        # Test recent papers resource
        recent_resource = await client.read_resource("research://papers/recent")
        print("\n📚 Recent papers resource:")
        if recent_resource and len(recent_resource) > 0:
            print(f"  {recent_resource[0].text}")
        
        # Test trending repositories resource with parameter
        trending_resource = await client.read_resource("research://github/trending/python")
        print("\n🔥 Trending Python repositories:")
        if trending_resource and len(trending_resource) > 0:
            print(f"  {trending_resource[0].text}")
            
    except Exception as e:
        print(f"❌ Resource test failed: {e}")

async def test_prompts(client):
    """Test prompt generation"""
    print("\n--- Testing Prompts ---")
    
    try:
        # Test research paper search prompt
        paper_prompt = await client.get_prompt("research_paper_search_prompt", {
            "topic": "deep learning",
            "focus_area": "computer vision"
        })
        
        print("📝 Research paper search prompt:")
        if paper_prompt and paper_prompt.messages:
            for message in paper_prompt.messages:
                if hasattr(message, 'content') and message.content:
                    print(f"  {message.content.text[:2000]}...")
        
        # Test code implementation search prompt
        code_prompt = await client.get_prompt("code_implementation_search_prompt", {
            "algorithm": "transformer architecture",
            "language": "python"
        })
        
        print("\n💻 Code implementation search prompt:")
        if code_prompt and code_prompt.messages:
            for message in code_prompt.messages:
                if hasattr(message, 'content') and message.content:
                    print(f"  {message.content.text[:2000]}...")
        
        # Test research idea development prompt
        idea_prompt = await client.get_prompt("research_idea_development_prompt", {
            "domain": "natural language processing",
            "current_gaps": "Limited multilingual understanding in low-resource languages"
        })
        
        print("\n💡 Research idea development prompt:")
        if idea_prompt and idea_prompt.messages:
            for message in idea_prompt.messages:
                if hasattr(message, 'content') and message.content:
                    print(f"  {message.content.text[:2000]}...")
                    
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        import traceback
        print(f"   Debug info: {traceback.format_exc()}")

async def test_stdio_connection():
    """
    Test the MCP server using stdio transport
    
    This method runs the server as a separate process and communicates
    via stdin/stdout, which is how most MCP clients work in production.
    """
    print("\n=== Testing Stdio Connection ===")
    
    try:
        # Connect via stdio to the test_mcp.py script
        async with Client("test_mcp.py") as client:
            print("✅ Connected via stdio transport")
            
            # Simple connectivity test
            await client.ping()
            print("✅ Stdio ping successful")
            
            # List available tools
            tools = await client.list_tools()
            print(f"📊 Found {len(tools)} tools via stdio")
            
            # Quick search test
            result = await client.call_tool("search_papers", {
                "query": "artificial intelligence",
                "result_limit": 1
            })
            
            if hasattr(result, 'content') and result.content:
                print("✅ Stdio tool call successful")
                print(f"   Result preview: {result.content[0].text[:1000]}...")
            else:
                print("✅ Stdio tool call completed (no content)")
                
    except Exception as e:
        print(f"❌ Stdio connection test failed: {e}")
        print("   Note: This might fail if the server script has import issues")

async def test_multi_server_config():
    """
    Test multi-server configuration using MCP config format
    
    This demonstrates how to connect to multiple MCP servers
    and use their tools with server prefixes.
    """
    print("\n=== Testing Multi-Server Configuration ===")
    
    # Example multi-server configuration
    config = {
        "mcpServers": {
            "tiny_scientist": {
                "command": "python",
                "args": [os.path.join(os.path.dirname(__file__), "test_mcp.py")],
                "env": {}
            },
            # Note: Add more servers here when available
            # "weather": {
            #     "url": "https://weather-api.example.com/mcp"
            # }
        }
    }
    
    try:
        async with Client(config) as client:
            print("✅ Connected to multi-server configuration")
            
            # List all available tools (should show server prefixes)
            tools = await client.list_tools()
            print(f"📊 Found {len(tools)} tools across all servers:")
            for tool in tools:
                print(f"  🔧 {tool.name}: {tool.description}")
            
            # Call tools with server prefixes
            # Note: The actual prefix format depends on the MCP client implementation
            # For now, we'll test without prefixes since we only have one server
            result = await client.call_tool("search_papers", {
                "query": "quantum computing",
                "result_limit": 1
            })
            
            if hasattr(result, 'content') and result.content:
                print("✅ Multi-server tool call successful")
                print(f"   Result preview: {result.content[0].text[:1000]}...")
                
    except Exception as e:
        print(f"❌ Multi-server configuration test failed: {e}")
        print("   Note: This requires the server to be properly configured")

async def main():
    """Main test function that runs all test scenarios"""
    print("🚀 Starting FastMCP Client Tests for Tiny Scientist Tools")
    print("=" * 60)
    
    # Test 1: In-memory connection (fastest and most reliable)
    await test_in_memory_connection()
    
    # Test 2: Stdio connection (production-like)
    await test_stdio_connection()
    
    # Test 3: Multi-server configuration (advanced usage)
    await test_multi_server_config()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\nUsage Summary:")
    print("- In-memory testing: Fastest for development and unit tests")
    print("- Stdio testing: Most common production deployment method")
    print("- Multi-server config: For complex applications using multiple MCP servers")
    print("\nNext steps:")
    print("- Add the server to your MCP client configuration")
    print("- Use the tools in your research workflow")
    print("- Extend the server with additional research tools")

if __name__ == "__main__":
    # Run all tests
    asyncio.run(main()) 