#!/usr/bin/env python3
"""
Convenience script to run all MCP servers for testing and debugging.
This script will start all configured MCP servers and keep them running.
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_scientist.utils.mcp_client import MCPClient


class MCPServerManager:
    """Manager for running multiple MCP servers for testing."""
    
    def __init__(self):
        self.client = MCPClient()
        self.running = True
    
    async def start_all_servers(self):
        """Start all configured MCP servers."""
        print("🚀 Starting all MCP servers...")
        results = await self.client.start_all_servers()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"✅ Started {success_count}/{total_count} servers successfully")
        
        if success_count > 0:
            print("\n📋 Server Status:")
            for server_name, success in results.items():
                status = "✅ Running" if success else "❌ Failed"
                print(f"  {server_name}: {status}")
        
        return success_count > 0
    
    async def health_check(self):
        """Perform health check on all servers."""
        print("\n🔍 Performing health check...")
        health = await self.client.health_check()
        
        healthy_count = sum(1 for status in health.values() if status)
        total_count = len(health)
        
        print(f"💚 {healthy_count}/{total_count} servers are healthy")
        
        for server_name, is_healthy in health.items():
            status = "💚 Healthy" if is_healthy else "💔 Unhealthy"
            print(f"  {server_name}: {status}")
    
    async def list_available_tools(self):
        """List all available tools from all servers."""
        print("\n🛠️  Available Tools:")
        
        for server_name in self.client.server_configs.keys():
            if self.client.is_server_running(server_name):
                tools = await self.client.get_available_tools(server_name)
                if tools:
                    print(f"\n  📦 {server_name}:")
                    for tool in tools:
                        print(f"    • {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                else:
                    print(f"  📦 {server_name}: No tools available")
    
    async def run_interactive_mode(self):
        """Run in interactive mode for testing tools."""
        print("\n🎮 Interactive Mode - Type 'help' for commands")
        
        while self.running:
            try:
                command = input("\n> ").strip()
                
                if command == "help":
                    print("""
Available commands:
  help - Show this help message
  health - Check server health
  tools - List available tools
  test <server> <tool> - Test a specific tool
  status - Show server status
  quit - Exit the program
""")
                
                elif command == "health":
                    await self.health_check()
                
                elif command == "tools":
                    await self.list_available_tools()
                
                elif command == "status":
                    running_servers = self.client.get_running_servers()
                    print(f"Running servers: {', '.join(running_servers) if running_servers else 'None'}")
                
                elif command.startswith("test "):
                    parts = command.split()
                    if len(parts) >= 3:
                        server_name = parts[1]
                        tool_name = parts[2]
                        await self.test_tool(server_name, tool_name)
                    else:
                        print("Usage: test <server> <tool>")
                
                elif command == "quit":
                    print("👋 Shutting down...")
                    self.running = False
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                self.running = False
            except EOFError:
                print("\n👋 Shutting down...")
                self.running = False
    
    async def test_tool(self, server_name: str, tool_name: str):
        """Test a specific tool."""
        if not self.client.is_server_running(server_name):
            print(f"❌ Server {server_name} is not running")
            return
        
        print(f"🧪 Testing {server_name}.{tool_name}...")
        
        # Sample test inputs for different tools
        test_inputs = {
            "search_papers": {"query": "machine learning", "result_limit": 2},
            "search_github_repositories": {"query": "pytorch", "result_limit": 3},
            "search_github_code": {"query": "neural network", "result_limit": 3},
            "generate_diagram": {
                "section_name": "Method", 
                "section_content": "Our method uses a neural network to process input data."
            },
            "validate_svg": {"svg_content": "<svg><rect width='100' height='100'/></svg>"},
            "get_supported_sections": {},
        }
        
        kwargs = test_inputs.get(tool_name, {})
        
        if not kwargs and tool_name not in ["get_supported_sections"]:
            print(f"❌ No test input defined for tool: {tool_name}")
            return
        
        try:
            result = await self.client.call_tool(server_name, tool_name, **kwargs)
            if result:
                print(f"✅ Tool executed successfully")
                print(f"Result preview: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
            else:
                print(f"❌ Tool returned empty result")
        except Exception as e:
            print(f"❌ Tool execution failed: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        await self.client.stop_all_servers()
        print("✅ All servers stopped")


async def main():
    """Main function to run the MCP server manager."""
    manager = MCPServerManager()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal")
        manager.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all servers
        success = await manager.start_all_servers()
        
        if not success:
            print("❌ Failed to start any servers. Exiting.")
            return
        
        # Perform initial health check
        await manager.health_check()
        
        # List available tools
        await manager.list_available_tools()
        
        # Run interactive mode
        await manager.run_interactive_mode()
        
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    print("🔧 TinyScientist MCP Server Manager")
    print("=" * 40)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 