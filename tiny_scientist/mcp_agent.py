# tiny_scientist/mcp_agent.py
import asyncio
import json
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import websockets
from rich import print

from .configs import Config
from .data import MCPConfig, MCPModuleConfig, MCPServerConfig
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import create_client, extract_json_between_markers, get_response_from_llm


class MCPConnectionType(Enum):
    """Types of MCP connections supported"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class ActionType(Enum):
    """Types of actions the MCP Agent can perform"""
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    TOOL_CALL = "tool_call"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class MCPTool:
    """Represents an MCP tool that can be called"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    capabilities: List[str]


@dataclass
class AgentAction:
    """Represents an action taken by the agent"""
    action_type: ActionType
    content: str
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    timestamp: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MCPConnection(ABC):
    """Abstract base class for MCP connections"""
    
    def __init__(self, server_config: MCPServerConfig, server_name: str):
        self.server_config = server_config
        self.server_name = server_name
        self.connected = False
        self.available_tools: List[MCPTool] = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server"""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools on the MCP server"""
        pass


class StdioMCPConnection(MCPConnection):
    """MCP connection using stdio protocol"""
    
    def __init__(self, server_config: MCPServerConfig, server_name: str):
        super().__init__(server_config, server_name)
        self.process = None
    
    async def connect(self) -> bool:
        """Start the MCP server process and establish communication"""
        try:
            if not self.server_config.command:
                raise ValueError(f"No command specified for stdio server {self.server_name}")
            
            # Build command with arguments
            cmd_parts = [self.server_config.command]
            if self.server_config.args:
                cmd_parts.extend(self.server_config.args)
            
            print(f"[MCP] Starting server with command: {' '.join(cmd_parts)}")
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Initialize MCP protocol handshake
            await self._initialize_protocol()
            
            self.connected = True
            self.available_tools = await self.list_tools()
            
            print(f"[MCP] Connected to stdio server: {self.server_name}")
            return True
            
        except Exception as e:
            print(f"[MCP] Failed to connect to {self.server_name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Terminate the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            
            self.connected = False
            print(f"[MCP] Disconnected from stdio server: {self.server_name}")
    
    async def _initialize_protocol(self) -> None:
        """Initialize MCP protocol with handshake"""
        # Simplified MCP initialization
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
                    "name": "TinyScientist",
                    "version": "1.0.0"
                }
            }
        }
        
        await self._send_request(init_request)
        # In real implementation, would wait for and parse response
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server"""
        if not self.process or not self.connected:
            raise RuntimeError(f"Not connected to server {self.server_name}")
        
        request_data = json.dumps(request) + "\n"
        self.process.stdin.write(request_data.encode())
        await self.process.stdin.drain()
        
        # Read response (simplified)
        response_data = await self.process.stdout.readline()
        if response_data:
            return json.loads(response_data.decode())
        
        raise RuntimeError("No response from server")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a tool call request to the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        try:
            response = await self._send_request(request)
            
            if "result" in response:
                return response["result"]
            elif "error" in response:
                raise RuntimeError(f"MCP tool error: {response['error']}")
            else:
                raise RuntimeError("Invalid MCP response")
        
        except Exception as e:
            print(f"[MCP] Tool call failed for {tool_name}: {e}")
            return {"error": str(e), "success": False}
    
    async def list_tools(self) -> List[MCPTool]:
        """Request list of available tools from MCP server"""
        if not self.connected:
            return []
        
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/list"
        }
        
        try:
            response = await self._send_request(request)
            
            if "result" in response and "tools" in response["result"]:
                tools_data = response["result"]["tools"]
                return [
                    MCPTool(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", {}),
                        server_name=self.server_name,
                        capabilities=tool.get("capabilities", [])
                    )
                    for tool in tools_data
                ]
            return []
        
        except Exception as e:
            print(f"[MCP] Failed to list tools from {self.server_name}: {e}")
            return []


class SSEMCPConnection(MCPConnection):
    """MCP connection using Server-Sent Events"""
    
    def __init__(self, server_config: MCPServerConfig, server_name: str):
        super().__init__(server_config, server_name)
        self.session = None
    
    async def connect(self) -> bool:
        """Establish SSE connection to MCP server"""
        try:
            if not self.server_config.url:
                raise ValueError(f"No URL specified for SSE server {self.server_name}")
            
            timeout = aiohttp.ClientTimeout(total=self.server_config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection
            async with self.session.get(self.server_config.url) as response:
                if response.status == 200:
                    self.connected = True
                    self.available_tools = await self.list_tools()
                    print(f"[MCP] Connected to SSE server: {self.server_name}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[MCP] Failed to connect to SSE server {self.server_name}: {e}")
            if self.session:
                await self.session.close()
            return False
    
    async def disconnect(self) -> None:
        """Close SSE connection"""
        if self.session:
            await self.session.close()
            self.connected = False
            print(f"[MCP] Disconnected from SSE server: {self.server_name}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via SSE endpoint"""
        if not self.session or not self.connected:
            raise RuntimeError(f"Not connected to SSE server {self.server_name}")
        
        endpoint = f"{self.server_config.url}/tools/{tool_name}"
        
        try:
            async with self.session.post(endpoint, json=parameters) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}", "success": False}
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def list_tools(self) -> List[MCPTool]:
        """List tools via SSE endpoint"""
        if not self.session or not self.connected:
            return []
        
        endpoint = f"{self.server_config.url}/tools"
        
        try:
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    tools_data = await response.json()
                    return [
                        MCPTool(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            input_schema=tool.get("inputSchema", {}),
                            server_name=self.server_name,
                            capabilities=tool.get("capabilities", [])
                        )
                        for tool in tools_data.get("tools", [])
                    ]
            return []
        
        except Exception as e:
            print(f"[MCP] Failed to list tools from SSE server {self.server_name}: {e}")
            return []


class WebSocketMCPConnection(MCPConnection):
    """MCP connection using WebSocket"""
    
    def __init__(self, server_config: MCPServerConfig, server_name: str):
        super().__init__(server_config, server_name)
        self.websocket = None
    
    async def connect(self) -> bool:
        """Establish WebSocket connection to MCP server"""
        try:
            if not self.server_config.url:
                raise ValueError(f"No URL specified for WebSocket server {self.server_name}")
            
            self.websocket = await websockets.connect(
                self.server_config.url,
                timeout=self.server_config.timeout
            )
            
            self.connected = True
            self.available_tools = await self.list_tools()
            print(f"[MCP] Connected to WebSocket server: {self.server_name}")
            return True
            
        except Exception as e:
            print(f"[MCP] Failed to connect to WebSocket server {self.server_name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print(f"[MCP] Disconnected from WebSocket server: {self.server_name}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via WebSocket"""
        if not self.websocket or not self.connected:
            raise RuntimeError(f"Not connected to WebSocket server {self.server_name}")
        
        request = {
            "type": "tool_call",
            "tool_name": tool_name,
            "parameters": parameters,
            "id": int(time.time() * 1000)
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response_data = await self.websocket.recv()
            return json.loads(response_data)
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def list_tools(self) -> List[MCPTool]:
        """List tools via WebSocket"""
        if not self.websocket or not self.connected:
            return []
        
        request = {
            "type": "list_tools",
            "id": int(time.time() * 1000)
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response_data = await self.websocket.recv()
            response = json.loads(response_data)
            
            return [
                MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.server_name,
                    capabilities=tool.get("capabilities", [])
                )
                for tool in response.get("tools", [])
            ]
        
        except Exception as e:
            print(f"[MCP] Failed to list tools from WebSocket server {self.server_name}: {e}")
            return []


class MCPAgent:
    """
    MCP (Model Context Protocol) Agent that can use multiple MCP servers
    to accomplish complex tasks through planning, execution, and reflection.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str,
        max_iterations: int = 15,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        config_file_path: Optional[str] = None,
    ):
        """Initialize the MCP Agent"""
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Load configuration
        self.config = Config(prompt_template_dir, config_file_path)
        self.mcp_config = self.config.mcp
        self.prompts = self.config.prompt_template.mcp_agent_prompt
        
        # Agent state
        self.action_history: List[AgentAction] = []
        self.working_memory: Dict[str, Any] = {}
        self.goal: str = ""
        self.current_plan: List[str] = []
        self.current_step: int = 0
        
        # MCP connections and tools
        self.connections: Dict[str, MCPConnection] = {}
        self.available_tools: List[MCPTool] = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled for the agent module"""
        return self.mcp_config.agent.enabled
    
    async def initialize(self) -> bool:
        """Initialize MCP connections and discover available tools"""
        if not self.is_mcp_enabled():
            print("[MCP Agent] MCP is disabled for agent module")
            return False
    
        print("[MCP Agent] Initializing MCP connections...")
        
        # Get enabled servers for agent module
        enabled_servers = [
            name for name in self.mcp_config.agent.servers
            if name in self.mcp_config.servers and self.mcp_config.servers[name].enabled
        ]
        
        if not enabled_servers:
            print("[MCP Agent] No enabled servers configured for agent module")
            return False
        
        # Create connections for each enabled server
        success_count = 0
        for server_name in enabled_servers:
            server_config = self.mcp_config.servers[server_name]
            
            try:
                connection = self._create_connection(server_config, server_name)
                if await connection.connect():
                    self.connections[server_name] = connection
                    self.available_tools.extend(connection.available_tools)
                    success_count += 1
                    print(f"[MCP Agent] Connected to {server_name} with {len(connection.available_tools)} tools")
                else:
                    print(f"[MCP Agent] Failed to connect to {server_name}")
            
            except Exception as e:
                print(f"[MCP Agent] Error connecting to {server_name}: {e}")
        
        print(f"[MCP Agent] Connected to {success_count}/{len(enabled_servers)} servers")
        print(f"[MCP Agent] Total available tools: {len(self.available_tools)}")
        
        return success_count > 0
    
    def _create_connection(self, server_config: MCPServerConfig, server_name: str) -> MCPConnection:
        """Create appropriate connection type based on server configuration"""
        connection_type = MCPConnectionType(server_config.type)
        
        if connection_type == MCPConnectionType.STDIO:
            return StdioMCPConnection(server_config, server_name)
        elif connection_type == MCPConnectionType.SSE:
            return SSEMCPConnection(server_config, server_name)
        elif connection_type == MCPConnectionType.WEBSOCKET:
            return WebSocketMCPConnection(server_config, server_name)
        else:
            raise ValueError(f"Unsupported connection type: {server_config.type}")
    
    async def cleanup(self) -> None:
        """Clean up all MCP connections"""
        print("[MCP Agent] Cleaning up MCP connections...")
        
        for connection in self.connections.values():
            try:
                await connection.disconnect()
            except Exception as e:
                print(f"[MCP Agent] Error during cleanup: {e}")
    
        self.connections.clear()
        self.available_tools.clear()
    
    async def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent to achieve the specified goal
        
        Args:
            goal: The high-level goal for the agent to achieve
            context: Optional context information
            
        Returns:
            Dictionary containing the execution results
        """
        if not self.is_mcp_enabled():
            return {"error": "MCP is not enabled for agent module", "success": False}
        
        # Initialize if not already done
        if not self.connections:
            if not await self.initialize():
                return {"error": "Failed to initialize MCP connections", "success": False}
        
        # Reset state
        self.goal = goal
        self.working_memory = context or {}
        self.action_history = []
        self.current_plan = []
        self.current_step = 0
        
        print(f"[MCP Agent] Starting execution with goal: {goal}")
        
        try:
            result = await self._execute_goal()
            return result
        finally:
            await self.cleanup()
    
    async def _execute_goal(self) -> Dict[str, Any]:
        """Main execution loop for achieving the goal"""
        iteration = 0
        goal_achieved = False
        
        # Step 1: Create initial plan
        await self._create_plan()
        
        while iteration < self.max_iterations and not goal_achieved:
            iteration += 1
            print(f"\n[MCP Agent] === Iteration {iteration} ===")
            
            # Execute current step in plan
            if self.current_step < len(self.current_plan):
                step_description = self.current_plan[self.current_step]
                action_result = await self._execute_step(step_description)
                
                # Reflect on the result
                reflection_result = await self._reflect_on_result(action_result)
            
            # Check if goal is achieved
                if self._is_goal_achieved(reflection_result):
                    goal_achieved = True
                    print("[MCP Agent] Goal achieved!")
                else:
                    self.current_step += 1
                    
                    # If we've completed all steps but haven't achieved the goal,
                    # create a new plan
                    if self.current_step >= len(self.current_plan):
                        print("[MCP Agent] Completed current plan, creating new plan...")
                        await self._create_plan()
                        self.current_step = 0
            else:
                # No plan available, try to create one
                await self._create_plan()
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        return self._generate_final_report(goal_achieved, iteration)
    
    @api_calling_error_exponential_backoff(retries=3, base_wait_time=1)
    async def _create_plan(self) -> None:
        """Create a plan to achieve the goal"""
        context_str = json.dumps(self.working_memory, indent=2)
        tools_str = self._format_available_tools()
        
        prompt = self.prompts.planning_prompt.format(
            goal=self.goal,
            context=context_str,
            available_tools=tools_str
        )
        
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.system_prompt,
                temperature=self.temperature
            )
            
            # Extract plan from response (expect a numbered list)
            plan_lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    # Remove numbering/bullets and clean up
                    cleaned = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                                   '-', '*', '•']:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                            break
                    if cleaned:
                        plan_lines.append(cleaned)
            
            self.current_plan = plan_lines[:10]  # Limit to 10 steps
            self.current_step = 0
            
            # Record planning action
            action = AgentAction(
                action_type=ActionType.PLAN,
                content=f"Created plan with {len(self.current_plan)} steps",
                result=self.current_plan
            )
            self.action_history.append(action)
            
            print(f"[MCP Agent] Created plan with {len(self.current_plan)} steps:")
            for i, step in enumerate(self.current_plan, 1):
                print(f"  {i}. {step}")
            
        except Exception as e:
            print(f"[MCP Agent] Error creating plan: {e}")
            self.current_plan = ["Use available tools to work towards the goal"]
    
    @api_calling_error_exponential_backoff(retries=3, base_wait_time=1)
    async def _execute_step(self, step_description: str) -> Any:
        """Execute a single step in the plan"""
        print(f"[MCP Agent] Executing step: {step_description}")
        
        # Use LLM to determine which tool to use for this step
        tools_str = self._format_available_tools()
        context_str = json.dumps(self.working_memory, indent=2)
        
        prompt = self.prompts.tool_selection_prompt.format(
            current_step=step_description,
            available_tools=tools_str,
            context=context_str
        )
        
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.system_prompt,
                temperature=self.temperature
            )
            
            # Try to extract tool call information from response
            tool_info = extract_json_between_markers(response)
            
            if tool_info and "tool_name" in tool_info:
                tool_name = tool_info["tool_name"]
                parameters = tool_info.get("parameters", {})
                
                # Find the tool
                tool = next((t for t in self.available_tools if t.name == tool_name), None)
                if tool:
                    # Execute tool call
                    result = await self._call_mcp_tool(tool, parameters)
                    
                    # Record action
                    action = AgentAction(
                        action_type=ActionType.TOOL_CALL,
                        content=step_description,
                        tool_name=tool_name,
                        parameters=parameters,
                        result=result,
                        success="error" not in result
                    )
                    self.action_history.append(action)
                    
                    # Update working memory
                    self.working_memory[f"step_{self.current_step}_result"] = result
                    
                    return result
                else:
                    error_msg = f"Tool '{tool_name}' not found"
                    print(f"[MCP Agent] {error_msg}")
                    return {"error": error_msg, "success": False}
            else:
                # No specific tool identified, just record the step as executed
                action = AgentAction(
                    action_type=ActionType.EXECUTE,
                    content=step_description,
                    result="Step executed without tool call"
                )
                self.action_history.append(action)
                return {"message": "Step executed", "success": True}
        
        except Exception as e:
            error_msg = f"Error executing step: {e}"
            print(f"[MCP Agent] {error_msg}")
            
            action = AgentAction(
                action_type=ActionType.ERROR_RECOVERY,
                content=step_description,
                success=False,
                error_message=error_msg
            )
            self.action_history.append(action)
            
            return {"error": error_msg, "success": False}
    
    async def _call_mcp_tool(self, tool: MCPTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool on the appropriate server"""
        # Find the connection for this tool
        connection = self.connections.get(tool.server_name)
        if not connection:
            return {"error": f"No connection to server {tool.server_name}", "success": False}
        
        try:
            print(f"[MCP Agent] Calling tool {tool.name} on {tool.server_name}")
            result = await connection.call_tool(tool.name, parameters)
            print(f"[MCP Agent] Tool call result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to call tool {tool.name}: {e}"
            print(f"[MCP Agent] {error_msg}")
            return {"error": error_msg, "success": False}
    
    @api_calling_error_exponential_backoff(retries=3, base_wait_time=1)
    async def _reflect_on_result(self, result: Any) -> Dict[str, Any]:
        """Reflect on the result of an action"""
        progress_summary = self._summarize_progress()
        
        prompt = self.prompts.reflection_prompt.format(
            action=self.action_history[-1].content if self.action_history else "Unknown",
            result=json.dumps(result, indent=2),
            goal=self.goal,
            progress=progress_summary
        )
        
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.system_prompt,
                temperature=self.temperature
            )
            
            # Record reflection
            action = AgentAction(
                action_type=ActionType.REFLECT,
                content=response,
                result={"reflection": response}
            )
            self.action_history.append(action)
            
            return {"reflection": response, "success": True}
        
        except Exception as e:
            print(f"[MCP Agent] Error during reflection: {e}")
            return {"error": str(e), "success": False}
    
    def _is_goal_achieved(self, reflection_result: Dict[str, Any]) -> bool:
        """Determine if the goal has been achieved based on reflection"""
        reflection_text = reflection_result.get("reflection", "").lower()
        
        # Simple heuristic - look for completion indicators in reflection
        completion_indicators = [
            "goal achieved", "goal completed", "task completed", "finished",
            "accomplished", "successful", "done", "objective met"
        ]
        
        return any(indicator in reflection_text for indicator in completion_indicators)
    
    def _format_available_tools(self) -> str:
        """Format available MCP tools for LLM consumption"""
        if not self.available_tools:
            return "No MCP tools available"
        
        tool_descriptions = []
        for tool in self.available_tools:
            tool_descriptions.append(
                f"- {tool.name} (server: {tool.server_name}): {tool.description}\n"
                f"  Input schema: {json.dumps(tool.input_schema, indent=2)}\n"
                f"  Capabilities: {', '.join(tool.capabilities)}"
            )
        
        return "\n\n".join(tool_descriptions)
    
    def _summarize_progress(self) -> str:
        """Summarize the progress made so far"""
        if not self.action_history:
            return "No actions taken yet"
        
        summary_parts = [
            f"Goal: {self.goal}",
            f"Plan: {len(self.current_plan)} steps, currently on step {self.current_step + 1}",
            f"Actions taken: {len(self.action_history)}",
            f"Tools used: {len(set(a.tool_name for a in self.action_history if a.tool_name))}",
        ]
        
        # Add recent actions
        recent_actions = self.action_history[-3:]
        if recent_actions:
            summary_parts.append("Recent actions:")
        for i, action in enumerate(recent_actions, 1):
                status = "✓" if action.success else "✗"
                summary_parts.append(f"  {status} {action.action_type.value}: {action.content[:100]}...")
        
        return "\n".join(summary_parts)
    
    def _generate_final_report(self, goal_achieved: bool, iterations: int) -> Dict[str, Any]:
        """Generate a final report of the agent's execution"""
        report = {
            "goal": self.goal,
            "goal_achieved": goal_achieved,
            "total_iterations": iterations,
            "total_actions": len(self.action_history),
            "execution_time": time.time() - (self.action_history[0].timestamp if self.action_history else time.time()),
            "final_working_memory": self.working_memory,
            "plan_executed": self.current_plan,
            "current_step": self.current_step,
            "servers_used": list(self.connections.keys()),
            "tools_used": list(set(a.tool_name for a in self.action_history if a.tool_name)),
            "action_summary": self._summarize_actions(),
            "success_rate": self._calculate_success_rate(),
        }
        
        # Save report to file
        report_path = osp.join(self.output_dir, "mcp_agent_report.json")
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"[MCP Agent] Final report saved to: {report_path}")
        return report
    
    def _summarize_actions(self) -> Dict[str, int]:
        """Summarize the types of actions taken"""
        action_counts = {}
        for action in self.action_history:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        return action_counts
    
    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of actions"""
        if not self.action_history:
            return 0.0
        
        successful_actions = sum(1 for action in self.action_history if action.success)
        return successful_actions / len(self.action_history)


# Factory function for easy instantiation
def create_mcp_agent(
    model: str = "gpt-4o",
    output_dir: str = "./mcp_agent_output",
    max_iterations: int = 15,
    temperature: float = 0.75,
    prompt_template_dir: Optional[str] = None,
    config_file_path: Optional[str] = None,
) -> MCPAgent:
    """Create an MCP Agent with the specified configuration"""
    return MCPAgent(
        model=model,
        output_dir=output_dir,
        max_iterations=max_iterations,
        temperature=temperature,
        prompt_template_dir=prompt_template_dir,
        config_file_path=config_file_path,
    )


# Example usage function
async def example_usage():
    """Example of how to use the MCP Agent"""
    agent = create_mcp_agent(
        model="gpt-4o",
        output_dir="./mcp_agent_output"
    )
    
    goal = "Research the latest developments in transformer architectures and summarize key findings"
    context = {
        "research_domain": "machine_learning",
        "focus_area": "transformers",
        "time_period": "last_6_months"
    }
    
    result = await agent.run(goal=goal, context=context)
    
    print(f"Goal achieved: {result.get('goal_achieved', False)}")
    print(f"Actions taken: {result.get('total_actions', 0)}")
    print(f"Success rate: {result.get('success_rate', 0):.1%}")
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())