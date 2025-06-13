#!/usr/bin/env python3
"""
MCP Agent Test Script

This script is used to test:
1. Configure and initialize MCP Agent
2. Connect to configured MCP servers
3. Execute basic MCP tool calls

Usage:
    python test_mcp_agent.py

Before running, please ensure:
1. Install necessary dependencies
2. Configure config.toml file
3. Related MCP servers are available
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tiny_scientist.mcp_agent import MCPAgent, create_mcp_agent


async def test_mcp_connection():
    """Test MCP connection and basic functionality"""
    print("🧪 Testing MCP Agent connection and basic functionality...")
    
    # Create MCP Agent
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output",
        max_iterations=5,
        temperature=0.7
    )
    
    # Initialize agent
    try:
        print("🔌 Initializing MCP Agent...")
        initialized = await agent.initialize()
        
        if not initialized:
            print("❌ MCP Agent initialization failed")
            return False
        
        print("✅ MCP Agent initialization successful")
        
        # Check if MCP is enabled
        if not agent.is_mcp_enabled():
            print("⚠️ MCP is not enabled, please check configuration file")
            return False
        
        print("🔗 MCP is enabled")
        
        # Display connected server information
        print(f"📡 Number of connected MCP servers: {len(agent.connections)}")
        for server_name, connection in agent.connections.items():
            print(f"  - {server_name}: {len(connection.available_tools)} tools")
            for tool in connection.available_tools[:3]:  # Only show first 3 tools
                print(f"    * {tool.name}: {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    
    finally:
        # Clean up connections
        await agent.cleanup()


async def test_simple_mcp_task():
    """Test simple MCP task execution"""
    print("\n🎯 Testing simple MCP task execution...")
    
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output",
        max_iterations=3,
        temperature=0.7
    )
    
    # Simple test goal
    goal = "Use available MCP tools to perform a simple operation and report results"
    context = {
        "task_type": "simple_test",
        "expected_result": "demonstration"
    }
    
    print(f"🎯 Goal: {goal}")
    
    try:
        result = await agent.run(goal=goal, context=context)
        
        print("\n✅ Task execution completed!")
        print(f"📊 Result summary:")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")
        print(f"  - Total iterations: {result.get('total_iterations', 0)}")
        print(f"  - Total actions: {result.get('total_actions', 0)}")
        print(f"  - Success rate: {result.get('success_rate', 0):.1%}")
        print(f"  - Servers used: {', '.join(result.get('servers_used', []))}")
        print(f"  - Tools used: {', '.join(result.get('tools_used', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        return False


def check_prerequisites():
    """Check runtime prerequisites"""
    print("🔍 Checking runtime prerequisites...")
    
    # Check configuration file
    config_file = project_root / "config.toml"
    if not config_file.exists():
        print("❌ config.toml file not found")
        print("💡 Please copy and configure from config.template.toml")
        return False
    
    # Check output directory
    os.makedirs("test_output", exist_ok=True)
    
    print("✅ Prerequisites check passed")
    return True


async def main():
    """Main test function"""
    print("🚀 MCP Agent Basic Test Suite")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run tests
    tests = [
        ("MCP Connection Test", test_mcp_connection),
        ("Simple MCP Task", test_simple_mcp_task),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
                
        except Exception as e:
            print(f"💥 {test_name} exception: {e}")
            results.append((test_name, False))
        
        # Brief rest between tests
        await asyncio.sleep(1)
    
    # Summarize results
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print(f"  Success rate: {passed/total:.1%}")
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n🎉 Testing completed!")
    
    if passed == total:
        print("🌟 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed, please check output")
        return 1


if __name__ == "__main__":
    # Set event loop policy (may be needed on some systems)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error occurred during test run: {e}")
        sys.exit(1) 