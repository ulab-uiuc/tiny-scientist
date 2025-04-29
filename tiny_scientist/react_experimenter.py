import json
import logging
import os
import os.path as osp
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from .configs import Config
from .tool import BaseTool
from .utils.llm import create_client, get_batch_responses_from_llm_with_tools, get_response_from_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReactExperimenter:
    """
    ReAct Experimenter that conducts experiments by reasoning and taking actions
    using available tools in a specified domain.
    """
    def __init__(
        self,
        model: str,
        output_dir: str,
        domain: str = "general",
        tool_dir: str = "tiny_scientist/tools",
        max_iterations: int = 10,
        prompt_template_dir: Optional[str] = None,
    ):
        """
        Initialize the ReactExperimenter.
        
        Args:
            model: LLM model to use (e.g., "gpt-4o")
            output_dir: Directory to store experiment results and logs
            domain: Domain for the experiment (e.g., "chemistry", "physics")
            tool_dir: Directory containing tool implementations
            max_iterations: Maximum number of reasoning iterations
            prompt_template_dir: Optional custom prompt template directory
        """
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.tool_dir = tool_dir
        self.domain = domain.lower()
        self.max_iterations = max_iterations
        self.config = Config(prompt_template_dir)
        
        try:
            self.prompts = self.config.prompt_template.react_experimenter_prompt
            logger.info("Successfully loaded react_experimenter_prompt templates")
        except AttributeError:
            # If the specific prompts don't exist, log a warning
            logger.warning("react_experimenter_prompt not found in templates. Using fallback prompts.")
            # Define fallback prompts
            from types import SimpleNamespace
            self.prompts = SimpleNamespace()
            self.prompts.react_system_prompt = """You are a research assistant AI conducting experiments. Use step-by-step reasoning and available tools to solve the problem."""
            self.prompts.initial_experiment_prompt = """Conduct an experiment on: {title}. Problem: {problem}. Approach: {approach}."""

        # Setup logging to file
        self.log_file = osp.join(self.output_dir, "experiment_logger.log")
        file_handler = logging.FileHandler(self.log_file, mode='w')  # Overwrite log each time
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Load tools for the specified domain
        self.tools = self._load_tools_for_domain()
        self.tool_schemas = self._get_tool_schemas()
        
        logger.info(f"Initialized ReactExperimenter with domain: {domain}")
        logger.info(f"Loaded tools: {list(self.tools.keys())}")

    def _load_tools_for_domain(self) -> Dict[str, BaseTool]:
        """Load tools specifically for the selected domain."""
        tools: Dict[str, BaseTool] = {}
        
        # Check if tools directory exists
        if not osp.isdir(self.tool_dir):
            logger.warning(f"Tool directory not found: {self.tool_dir}. No tools loaded.")
            return tools
        
        import importlib
        import inspect
        import sys
        
        # Map domains to related tool files (without .py extension)
        domain_to_tools = {
            "chemistry": ["chemical_tool"],
            "physics": ["physical_tool"],
            "general": ["chemical_tool", "physical_tool"],  # General includes all tools
        }
        
        # Get the tool files for the specified domain
        tool_files = domain_to_tools.get(self.domain, [])
        if not tool_files:
            logger.warning(f"No tools defined for domain: {self.domain}. Falling back to general tools.")
            tool_files = domain_to_tools["general"]
        
        # Add tool directory to path if not already there
        tool_dir_abs = osp.abspath(self.tool_dir)
        if tool_dir_abs not in sys.path:
            sys.path.insert(0, tool_dir_abs)
        
        # Import each tool file for the domain
        for tool_file in tool_files:
            tool_path = osp.join(self.tool_dir, f"{tool_file}.py")
            
            if not osp.exists(tool_path):
                logger.warning(f"Tool file not found: {tool_path}")
                continue
            
            try:
                # Import the module
                module_name = tool_file
                if module_name in sys.modules:
                    # Reload if already loaded
                    module = importlib.reload(sys.modules[module_name])
                else:
                    module = importlib.import_module(module_name)
                
                # Find all tool classes in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                        try:
                            # Create an instance of the tool
                            tools[name] = obj()  # Instantiate with default parameters
                            logger.info(f"Loaded tool '{name}' from {tool_file}.py")
                        except Exception as e:
                            logger.error(f"Failed to instantiate tool {name}: {e}")
                            logger.debug(traceback.format_exc())
            
            except ImportError as e:
                logger.error(f"Failed to import module {module_name}: {e}")
                logger.debug(traceback.format_exc())
        
        # If no tools were loaded, log a warning
        if not tools:
            logger.warning("No tools were loaded for the specified domain.")
        
        return tools

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Generate OpenAI-compatible tool schemas from loaded tools."""
        schemas = []
        
        for tool_name, tool_instance in self.tools.items():
            # Get the tool's run method signature
            import inspect
            sig = inspect.signature(tool_instance.run)
            
            # Extract parameter info from the method signature
            parameters = {}
            required_params = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                param_info = {
                    "type": "string"  # Default type
                }
                
                # Try to infer type from type annotation or default value
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        param_info["type"] = "string"
                    elif param.annotation in (int, float) or (
                        hasattr(param.annotation, "__origin__") and 
                        param.annotation.__origin__ is Union and
                        any(t in param.annotation.__args__ for t in (int, float))
                    ):
                        param_info["type"] = "number"
                    elif param.annotation == bool:
                        param_info["type"] = "boolean"
                    elif param.annotation == list or param.annotation == List:
                        param_info["type"] = "array"
                    elif param.annotation == dict or param.annotation == Dict:
                        param_info["type"] = "object"
                
                # Extract description from the docstring if available
                if tool_instance.run.__doc__:
                    doc_lines = tool_instance.run.__doc__.strip().split('\n')
                    for i, line in enumerate(doc_lines):
                        if f'{param_name}:' in line:
                            description = line.split(f'{param_name}:')[1].strip()
                            if i + 1 < len(doc_lines) and not any(p + ':' in doc_lines[i + 1] for p in sig.parameters):
                                # Add the next line if it doesn't contain another parameter
                                description += " " + doc_lines[i + 1].strip()
                            param_info["description"] = description
                            break
                
                # If no description was found, create a basic one
                if "description" not in param_info:
                    param_info["description"] = f"Parameter {param_name} for {tool_name}"
                
                parameters[param_name] = param_info
                
                # If parameter has no default value and is not Optional, mark as required
                if param.default == inspect.Parameter.empty and (
                    param.annotation == inspect.Parameter.empty or 
                    not (hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is Union and
                         type(None) in param.annotation.__args__)
                ):
                    required_params.append(param_name)
            
            # Create the function schema
            doc_summary = ""
            if tool_instance.__doc__:
                doc_summary = tool_instance.__doc__.strip().split('\n')[0]
            elif tool_instance.run.__doc__:
                doc_summary = "Tool for " + tool_instance.run.__doc__.strip().split('\n')[0].lower()
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc_summary or f"Tool for {tool_name}",
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required_params
                    }
                }
            }
            schemas.append(schema)
        
        return schemas

    def _execute_tool(self, tool_name: str, arguments: str) -> str:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: JSON string of arguments to pass to the tool
            
        Returns:
            String representation of the tool's output
        """
        # Check if the tool exists
        if tool_name not in self.tools:
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            logger.error(error_msg)
            return error_msg
        
        tool = self.tools[tool_name]
        
        try:
            # Parse the arguments JSON
            try:
                args_dict = json.loads(arguments)
            except json.JSONDecodeError:
                # Try to extract a JSON object using a regex if the string contains other text
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, arguments, re.DOTALL)
                if match:
                    args_dict = json.loads(match.group(0))
                else:
                    # If still fails, return an error
                    error_msg = f"Error: Failed to parse arguments: {arguments}"
                    logger.error(error_msg)
                    return error_msg
            
            # Log tool execution
            logger.info(f"Executing tool: {tool_name}")
            logger.info(f"Arguments: {json.dumps(args_dict, indent=2)}")
            
            # Execute the tool
            try:
                result = tool.run(**args_dict)
                
                # Convert the result to a string if it's a dict
                if isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)
                else:
                    result_str = str(result)
                
                logger.info(f"Tool {tool_name} execution result: {result_str[:200]}...")
                return result_str
                
            except TypeError as e:
                # This could happen if the arguments don't match the function's parameters
                error_msg = f"Error executing tool {tool_name}: {e}. Check if all required arguments are provided."
                logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return error_msg

    def run(self, idea: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Run the ReAct experiment using reasoning and tools.
        
        Args:
            idea: Dictionary containing experiment details
            baseline_results: Optional dictionary of baseline results for comparison
            
        Returns:
            Tuple of (success, output_directory)
        """
        logger.info(f"Starting ReAct experiment in domain: {self.domain}")
        logger.info(f"Experiment title: {idea.get('Title', 'Untitled')}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Format the initial prompt with experiment details
        formatted_baseline = json.dumps(baseline_results, indent=2) if baseline_results else "No baseline results provided."
        
        # Create initial prompt with experiment details
        try:
            initial_prompt = self.prompts.initial_experiment_prompt.format(
                title=idea.get("Title", "N/A"),
                problem=idea.get("Problem", "N/A"),
                novelty=idea.get("NoveltyComparison", "N/A"),
                approach=idea.get("Approach", "N/A"),
                experiment_details=idea.get("Experiment", "N/A"),
                baseline_results=formatted_baseline,
                available_tools=json.dumps([{
                    "name": name,
                    "description": tool.__doc__ or f"Tool for {name}" 
                } for name, tool in self.tools.items()], indent=2)
            )
        except (KeyError, AttributeError) as e:
            logger.warning(f"Error formatting initial prompt: {e}. Using fallback.")
            initial_prompt = f"""
            Conduct an experiment based on the following idea:
            
            Title: {idea.get('Title', 'N/A')}
            Problem: {idea.get('Problem', 'N/A')}
            Approach: {idea.get('Approach', 'N/A')} 
            Experiment Details: {idea.get('Experiment', 'N/A')}
            
            Available tools: {', '.join(self.tools.keys())}
            
            Think step by step and use the tools to conduct the experiment.
            The goal is to produce a scientific result addressing the problem.
            """
        
        # Create a system prompt that guides the LLM to use ReAct properly
        system_prompt = self.prompts.react_system_prompt
        
        # Initialize message history and state
        message_history = []
        current_prompt = initial_prompt
        final_result = {}
        success = False
        
        # Write experiment idea to the log
        logger.info(f"Experiment idea: {json.dumps(idea, indent=2)}")
        logger.info(f"Starting ReAct loop with {self.max_iterations} max iterations")
        
        # Main ReAct loop
        for i in range(self.max_iterations):
            logger.info(f"--- Iteration {i+1}/{self.max_iterations} ---")
            logger.info(f"Current prompt: {current_prompt[:200]}...")
            
            # Add current prompt to message history
            message_history.append({"role": "user", "content": current_prompt})
            
            # Get response from LLM with tool calling capability
            try:
                responses, histories = get_batch_responses_from_llm_with_tools(
                    msg=current_prompt,
                    client=self.client,
                    model=self.model,
                    system_message=system_prompt,
                    tools=self.tool_schemas,
                    n_responses=1,
                    msg_history=message_history[:-1],
                    temperature=0.7,  # Higher temperature for more creative reasoning
                )
                
                response = responses[0]  # Get first response
                message_history = histories[0]  # Update message history
                
                # Check if the response is a tool call
                if isinstance(response, dict) and "tool_calls" in response:
                    logger.info("LLM decided to use tools")
                    tool_results = []
                    
                    # Process each tool call
                    for tool_call in response["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"]["arguments"]
                        tool_id = tool_call["id"]
                        
                        # Execute the tool
                        logger.info(f"Executing tool: {tool_name}")
                        observation = self._execute_tool(tool_name, tool_args)
                        
                        # Log the tool result
                        logger.info(f"Tool result: {observation[:200]}...")
                        
                        # Add tool result to results list
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": observation
                        })
                    
                    # Add all tool results to message history
                    message_history.extend(tool_results)
                    
                    # Create a new user message for the next iteration
                    result_summary = "\n\n".join([f"Tool: {tr['name']}\nResult: {tr['content'][:1000]}..." 
                                                if len(tr['content']) > 1000 
                                                else f"Tool: {tr['name']}\nResult: {tr['content']}" 
                                                for tr in tool_results])
                    
                    current_prompt = (
                        "The tools have been executed. Here are the results:\n\n"
                        f"{result_summary}\n\n"
                        "Based on these results, continue reasoning about the experiment. "
                        "You can use more tools if needed, or provide a final answer if you've completed the experiment."
                    )
                
                # If the response is text (not a tool call)
                elif isinstance(response, str):
                    logger.info(f"LLM Response: {response[:200]}...")
                    
                    # Check if this is a final answer
                    if "Final Answer:" in response or "FINAL ANSWER:" in response:
                        logger.info("Final answer detected")
                        
                        # Try to extract structured results from the final answer
                        try:
                            # Look for JSON in the response
                            json_pattern = r'\{.*\}'
                            match = re.search(json_pattern, response, re.DOTALL)
                            
                            if match:
                                final_result = json.loads(match.group(0))
                            else:
                                # If no JSON, create a simple result with the full text
                                final_answer_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response.split("FINAL ANSWER:", 1)[1].strip()
                                final_result = {
                                    "conclusion": final_answer_text,
                                    "experiment_complete": True
                                }
                            
                            success = True
                            logger.info(f"Experiment completed successfully with result: {json.dumps(final_result, indent=2)}")
                            
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.warning(f"Could not parse final answer properly: {e}")
                            final_result = {
                                "raw_response": response,
                                "note": "Failed to parse structured results"
                            }
                        
                        break  # Exit the ReAct loop when we get a final answer
                    
                    # If it's just reasoning, prepare for the next iteration
                    current_prompt = (
                        "You gave this reasoning:\n\n"
                        f"{response[:1000]}{'...' if len(response) > 1000 else ''}\n\n"
                        "Continue the experiment. You can use tools to gather data, perform calculations, "
                        "or analyze results. Once you've completed the experiment, provide a final answer "
                        "with your conclusions."
                    )
                
                # Handle unexpected response format
                else:
                    logger.error(f"Unexpected response format: {type(response)}")
                    current_prompt = (
                        "There was an error processing your last response. "
                        "Please continue with the experiment using clear reasoning "
                        "and proper tool usage when needed."
                    )
            
            except Exception as e:
                # Handle any errors in the LLM call
                logger.error(f"Error during LLM interaction: {e}")
                logger.debug(traceback.format_exc())
                
                error_message = f"Error during processing: {str(e)}"
                message_history.append({"role": "assistant", "content": error_message})
                
                current_prompt = (
                    f"There was an error: {str(e)}. Please continue with the experiment "
                    "using clear reasoning and proper tool usage when needed."
                )
            
            # Check if we've reached the maximum number of iterations
            if i == self.max_iterations - 1 and not success:
                logger.warning("Maximum iterations reached without a final answer")
                final_result = {
                    "status": "incomplete",
                    "error": "Max iterations reached without conclusion",
                    "partial_results": "See experiment_logger.log for details",
                }
        
        # Save final results
        result_save_path = osp.join(self.output_dir, "experiment_results.txt")
        try:
            # Include experiment metadata in the results
            final_result["experiment"] = {
                "title": idea.get("Title", "Untitled"),
                "domain": self.domain,
                "model": self.model,
                "iterations": min(i + 1, self.max_iterations)
            }
            
            # Save results
            with open(result_save_path, "w") as f:
                json.dump(final_result, f, indent=2)
            
            logger.info(f"Experiment results saved to {result_save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
            success = False
        
        return success, self.output_dir 