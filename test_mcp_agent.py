#!/usr/bin/env python3
"""
Test Script: Start and Run MCP Agent

This script demonstrates how to:
1. Configure and initialize MCP Agent
2. Run different types of tasks
3. Handle results and errors

Usage:
    python test_mcp_agent.py

Make sure before running:
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


async def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing MCP Agent basic functionality...")
    
    # Create MCP Agent
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output",
        max_iterations=10,
        temperature=0.7
    )
    
    # Simple test goal
    goal = "List the contents of current directory and analyze Python files in it"
    context = {
        "working_directory": os.getcwd(),
        "file_types_of_interest": ["python", "markdown", "yaml"]
    }
    
    print(f"🎯 Goal: {goal}")
    print(f"📝 Context: {json.dumps(context, indent=2, ensure_ascii=False)}")
    
    try:
        result = await agent.run(goal=goal, context=context)
        
        print("\n✅ Execution completed!")
        print(f"📊 Result summary:")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")
        print(f"  - Total iterations: {result.get('total_iterations', 0)}")
        print(f"  - Total actions: {result.get('total_actions', 0)}")
        print(f"  - Success rate: {result.get('success_rate', 0):.1%}")
        print(f"  - Servers used: {', '.join(result.get('servers_used', []))}")
        print(f"  - Tools used: {', '.join(result.get('tools_used', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_research_task():
    """Test research task"""
    print("\n🔬 Testing MCP Agent research functionality...")
    
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output_research",
        max_iterations=15
    )
    
    goal = "Search for latest research information about 'transformer attention mechanism' and summarize key findings"
    context = {
        "research_domain": "machine_learning",
        "focus_area": "attention_mechanisms",
        "time_period": "recent",
        "output_format": "structured_summary"
    }
    
    print(f"🎯 Goal: {goal}")
    
    try:
        result = await agent.run(goal=goal, context=context)
        
        print("\n✅ Research task completed!")
        print(f"📊 Result summary:")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")
        print(f"  - Execution time: {result.get('execution_time', 0):.2f} seconds")
        print(f"  - Action statistics: {result.get('action_summary', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research test failed: {e}")
        return False


async def test_file_operations():
    """Test file operation functionality"""
    print("\n📁 Testing MCP Agent file operation functionality...")
    
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output_files",
        max_iterations=8
    )
    
    goal = "Analyze project structure, find all Python files, and create a project overview document"
    context = {
        "project_root": ".",
        "include_patterns": ["*.py", "*.md", "*.yaml", "*.toml"],
        "exclude_patterns": ["__pycache__", "*.pyc", ".git"],
        "output_file": "project_overview.md"
    }
    
    print(f"🎯 Goal: {goal}")
    
    try:
        result = await agent.run(goal=goal, context=context)
        
        print("\n✅ File operation task completed!")
        print(f"📊 Result summary:")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")
        print(f"  - Total steps: {len(result.get('plan_executed', []))}")
        print(f"  - Current step: {result.get('current_step', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ File operation test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling"""
    print("\n⚠️ Testing MCP Agent error handling...")
    
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./test_output_error",
        max_iterations=5
    )
    
    # Intentionally set a difficult/impossible task to test error handling
    goal = "Access non-existent website http://definitely-does-not-exist-12345.com and get content"
    context = {
        "expected_result": "error_handling_test",
        "should_fail": True
    }
    
    print(f"🎯 Goal (expected to fail): {goal}")
    
    try:
        result = await agent.run(goal=goal, context=context)
        
        print("\n📋 Error handling test results:")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")
        print(f"  - Errors handled: {sum(1 for action in result.get('action_summary', {}).items() if 'error' in action[0].lower())}")
        print(f"  - Success rate: {result.get('success_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"🔧 Error handling test encountered exception (this may be expected): {e}")
        return True  # For error handling tests, exceptions may be expected


def check_prerequisites():
    """Check runtime prerequisites"""
    print("🔍 Checking runtime prerequisites...")
    
    # Check configuration file
    config_file = project_root / "config.toml"
    if not config_file.exists():
        print("❌ config.toml file not found")
        print("💡 Please copy and configure from config.template.toml")
        return False
    
    # Check output directories
    for test_dir in ["test_output", "test_output_research", "test_output_files", "test_output_error"]:
        os.makedirs(test_dir, exist_ok=True)
    
    print("✅ Prerequisites check passed")
    return True


async def main():
    """Main test function"""
    print("🚀 MCP Agent Test Suite")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Research Task", test_research_task),
        ("File Operations", test_file_operations),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
                
        except Exception as e:
            print(f"💥 {test_name} test exception: {e}")
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