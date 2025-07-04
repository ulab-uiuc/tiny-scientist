import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import toml
from rich import print


class MCPClient:
    """Client for managing and communicating with MCP servers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP client with configuration.
        
        Args:
            config_path: Path to configuration file containing MCP server settings
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.servers: Dict[str, subprocess.Popen] = {}
        self.server_configs = self.config.get("mcp", {}).get("servers", {})
        
    def _get_default_config_path(self) -> str:
        """Get default config path."""
        this_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(this_dir, "config.toml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        try:
            with open(self.config_path, 'r') as f:
                return toml.load(f)
        except FileNotFoundError:
            print(f"[WARNING] Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return {}
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            bool: True if server started successfully
        """
        if server_name in self.servers:
            print(f"[MCP] Server {server_name} is already running")
            return True
            
        server_config = self.server_configs.get(server_name)
        if not server_config:
            print(f"[ERROR] No configuration found for server: {server_name}")
            return False
        
        try:
            command = server_config.get("command", "")
            args = server_config.get("args", [])
            working_dir = server_config.get("cwd")
            
            # Build full command
            full_command = [command] + args
            
            # Start the server process
            process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                text=True
            )
            
            self.servers[server_name] = process
            
            # Perform MCP initialization handshake
            init_success = await self._initialize_server(server_name)
            if not init_success:
                await self.stop_server(server_name)
                return False
                
            print(f"[MCP] Started server: {server_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start server {server_name}: {e}")
            return False
    
    async def _initialize_server(self, server_name: str) -> bool:
        """Initialize MCP server with proper handshake.
        
        Args:
            server_name: Name of the server to initialize
            
        Returns:
            bool: True if initialization successful
        """
        if server_name not in self.servers:
            return False
        
        try:
            process = self.servers[server_name]
            
            # Send initialize request
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
                        "name": "tiny-scientist-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            request_json = json.dumps(init_request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read initialization response
            response_line = process.stdout.readline()
            if not response_line:
                print(f"[ERROR] No initialization response from {server_name}")
                return False
            
            response = json.loads(response_line.strip())
            
            # Check for initialization success
            if "error" in response:
                print(f"[ERROR] Server initialization failed: {response['error']}")
                return False
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            notification_json = json.dumps(initialized_notification) + "\n"
            process.stdin.write(notification_json)
            process.stdin.flush()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize server {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            bool: True if server stopped successfully
        """
        if server_name not in self.servers:
            print(f"[WARNING] Server {server_name} is not running")
            return True
        
        try:
            process = self.servers[server_name]
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.servers[server_name]
            print(f"[MCP] Stopped server: {server_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to stop server {server_name}: {e}")
            return False
    
    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all configured MCP servers.
        
        Returns:
            Dict mapping server names to success status
        """
        results = {}
        for server_name in self.server_configs.keys():
            results[server_name] = await self.start_server(server_name)
        return results
    
    async def stop_all_servers(self) -> Dict[str, bool]:
        """Stop all running MCP servers.
        
        Returns:
            Dict mapping server names to success status
        """
        results = {}
        for server_name in list(self.servers.keys()):
            results[server_name] = await self.stop_server(server_name)
        return results
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Optional[str]:
        """Call a tool on a specific MCP server.
        
        Args:
            server_name: Name of the server to call
            tool_name: Name of the tool to call
            **kwargs: Tool parameters
            
        Returns:
            Tool response as string, or None if error
        """
        if server_name not in self.servers:
            print(f"[ERROR] Server {server_name} is not running")
            return None
        
        try:
            process = self.servers[server_name]
            
            # Create tool call request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": kwargs
                }
            }
            
            # Send request to server
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read response
            response_line = process.stdout.readline()
            if not response_line:
                print(f"[ERROR] No response from server {server_name}")
                return None
            
            response = json.loads(response_line.strip())
            
            # Check for errors
            if "error" in response:
                print(f"[ERROR] Tool call failed: {response['error']}")
                return None
            
            # Extract result
            result = response.get("result", {})
            if isinstance(result, dict) and "content" in result:
                return result["content"][0].get("text", "")
            elif isinstance(result, str):
                return result
            else:
                return json.dumps(result)
                
        except Exception as e:
            print(f"[ERROR] Failed to call tool {tool_name} on {server_name}: {e}")
            return None
    
    async def get_available_tools(self, server_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get list of available tools from a server.
        
        Args:
            server_name: Name of the server to query
            
        Returns:
            List of tool definitions, or None if error
        """
        if server_name not in self.servers:
            print(f"[ERROR] Server {server_name} is not running")
            return None
        
        try:
            process = self.servers[server_name]
            
            # Create list tools request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            # Send request to server
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read response
            response_line = process.stdout.readline()
            if not response_line:
                print(f"[ERROR] No response from server {server_name}")
                return None
            
            response = json.loads(response_line.strip())
            
            # Check for errors
            if "error" in response:
                print(f"[ERROR] Failed to list tools: {response['error']}")
                return None
            
            # Extract tools
            result = response.get("result", {})
            return result.get("tools", [])
                
        except Exception as e:
            print(f"[ERROR] Failed to get tools from {server_name}: {e}")
            return None
    
    def is_server_running(self, server_name: str) -> bool:
        """Check if a server is currently running.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            True if server is running
        """
        if server_name not in self.servers:
            return False
        
        process = self.servers[server_name]
        return process.poll() is None
    
    def get_running_servers(self) -> List[str]:
        """Get list of currently running servers.
        
        Returns:
            List of server names
        """
        return [name for name in self.servers.keys() if self.is_server_running(name)]
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all configured servers.
        
        Returns:
            Dict mapping server names to health status
        """
        results = {}
        for server_name in self.server_configs.keys():
            if self.is_server_running(server_name):
                # Try to get tools as a health check
                tools = await self.get_available_tools(server_name)
                results[server_name] = tools is not None
            else:
                results[server_name] = False
        return results
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_all_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_all_servers()


# Convenience functions for common operations
async def search_github_code(query: str, client: MCPClient, result_limit: int = 10) -> Optional[str]:
    """Search GitHub code using MCP client.
    
    Args:
        query: Search query
        client: MCP client instance
        result_limit: Maximum results to return
        
    Returns:
        Search results as JSON string
    """
    if not client.is_server_running("code_search"):
        await client.start_server("code_search")
    
    return await client.call_tool(
        "code_search", 
        "search_github_code", 
        query=query, 
        result_limit=result_limit
    )


async def search_github_repositories(query: str, client: MCPClient, result_limit: int = 10) -> Optional[str]:
    """Search GitHub repositories using MCP client.
    
    Args:
        query: Search query or JSON research idea
        client: MCP client instance
        result_limit: Maximum results to return
        
    Returns:
        Search results as JSON string
    """
    if not client.is_server_running("code_search"):
        await client.start_server("code_search")
    
    return await client.call_tool(
        "code_search", 
        "search_github_repositories", 
        query=query, 
        result_limit=result_limit
    )


async def search_papers(query: str, client: MCPClient, result_limit: int = 3) -> Optional[str]:
    """Search papers using MCP client.
    
    Args:
        query: Search query
        client: MCP client instance
        result_limit: Maximum results to return
        
    Returns:
        Search results as JSON string
    """
    if not client.is_server_running("paper_search"):
        await client.start_server("paper_search")
    
    return await client.call_tool(
        "paper_search", 
        "search_papers", 
        query=query, 
        result_limit=result_limit
    )


async def generate_diagram(section_name: str, section_content: str, client: MCPClient) -> Optional[str]:
    """Generate diagram using MCP client.
    
    Args:
        section_name: Name of the paper section
        section_content: Content of the section
        client: MCP client instance
        
    Returns:
        Diagram data as JSON string
    """
    if not client.is_server_running("drawer"):
        await client.start_server("drawer")
    
    return await client.call_tool(
        "drawer", 
        "generate_diagram", 
        section_name=section_name, 
        section_content=section_content
    ) 