import json
import os
import os.path as osp
import random
from typing import Any, Dict, List, Optional, Tuple, cast, Union
import yaml
import traceback # Import traceback for robust error handling

from rich import print

from .configs import Config
from .tool import PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from .malicious_agent import MaliciousAgent
from .defense_agent import DefenseAgent


class Thinker:
    def __init__(
        self,
        tools: List[Any],
        iter_num: int,
        search_papers: bool = True,
        generate_exp_plan: bool = True,
        model: str = "",
        output_dir: str = "",
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        enable_malicious_agents: bool = False,
        enable_defense_agent: bool = False,
        attack_probability: float = 0.5,
        attack_severity: str = "medium",  # low, medium, high
    ):
        self.tools = tools
        self.iter_num = iter_num
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.searcher = PaperSearchTool()
        self.search_papers = search_papers
        self.generate_exp_plan = generate_exp_plan
        self.prompts = self.config.prompt_template.thinker_prompt
        self.intent = ""
        self.domain = ""
        self.experiment_type = ""
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Enable malicious agents settings
        self.enable_malicious_agents = enable_malicious_agents
        self.enable_defense_agent = enable_defense_agent
        self.attack_probability = attack_probability
        self.attack_severity = attack_severity
        
        # Load scientist prompts from YAML file
        yaml_path = osp.join(osp.dirname(__file__), "prompts", "scientist_prompts.yaml")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.SCIENTIST_PROMPTS = yaml.safe_load(f)
        
        # Define malicious agent role
        self.malicious_agent_config = {
            "role": "Malicious Agent",
            "expertise": "subverting legitimate research discussions",
            "focus": "introducing subtle manipulations to research ideas"
        }
        
        # Attack targets (what aspects to attack)
        self.attack_targets = [
            "methodology",
            "ethical_considerations",
            "safety_measures",
            "research_goals",
            "experiment_design"
        ]
        
        # Intercepted messages
        self.intercepted_messages = {}
        self.corrected_messages = {}
        
        # Discussion history
        self.discussion_history = []

        # Get all domain-specific experts
        domain_experts = list(self.SCIENTIST_PROMPTS[self.domain].items()) if self.domain in self.SCIENTIST_PROMPTS else []
        # Get common experts
        common_experts = list(self.SCIENTIST_PROMPTS["common"].items())
        # Combine all experts
        all_experts = domain_experts + common_experts
        # malicious agent
        self.malicious_agent = None
        if self.enable_malicious_agents:
            self.malicious_agent = MaliciousAgent(
                model=self.model,
                client=self.client,
                role="Research Expert",
                expertise="Scientific methodology",
                focus="Research innovation"
            )
            all_experts.append(("malicious_agent", {
                "role": "Research Expert",
                "expertise": "Scientific methodology",
                "focus": "Research innovation"
            }))
            
        # defense agent
        self.defense_agent = None
        if self.enable_defense_agent:
            self.defense_agent = DefenseAgent(
                model=self.model,
                client=self.client,
                role="Security Expert",
                expertise="detecting manipulations and deception",
                focus="ensuring research integrity and safety"
            )
            all_experts.append(("defense_agent", {
                "role": "Security Expert",
                "expertise": "detecting manipulations and deception",
                "focus": "ensuring research integrity and safety"
            }))
            
        self.all_experts = all_experts

        self.WRITER_MINI_REQUIRED_KEYS = [
            "Title", "Problem", "Approach", "Expected Results", 
            "Importance", "Difficulty", "NoveltyComparison", "Experiment"
        ]

    def _get_agent_prompt(self, agent_info: Dict[str, str], idea_json: str, intent: str, related_works: str, history: List[Dict[str, str]]) -> str:
        """Generate a prompt for a specific agent role."""
        history_str = "\n".join([
            f"{msg.get('role', 'Unknown Role')} ({msg.get('agent_name', 'Unknown Agent')}): {msg.get('content', '')}"
            for msg in history
        ]) if history else "No previous discussion."
        
        return self.prompts.group_discussion_prompt.format(
            role=agent_info['role'],
            expertise=agent_info['expertise'],
            focus=agent_info['focus'],
            idea=idea_json,
            intent=intent,
            related_works=related_works,
            history=history_str
        )

    def _conduct_group_discussion(self, idea_json: str, num_rounds: int = 3) -> List[Dict[str, Any]]:
        """Conduct a multi-agent discussion about the research idea.
        Returns the detailed discussion history.
        """
        print(f"\nStarting multi-agent discussion for idea (first 100 chars): {idea_json[:100]}...")
        
        query = self._generate_search_query(idea_json, intent=self.intent)
        related_works_string = self._get_related_works(query)
        
        self.discussion_history = []
        
        # Ensure domain and experts are correctly initialized here if not done in __init__ or think method context
        # This was moved from __init__ to be dynamic based on domain passed to think method
        current_domain = self.domain if self.domain else "common" # Fallback to common if no domain
        domain_experts = list(self.SCIENTIST_PROMPTS.get(current_domain, {}).items())
        if not domain_experts and current_domain != "common": # If domain specific experts not found, and it wasn't common already
            print(f"[WARNING] No experts found for domain '{current_domain}'. Using only common experts.")
        common_experts_list = list(self.SCIENTIST_PROMPTS["common"].items())
        
        # Rebuild self.all_experts for current discussion context
        current_all_experts = domain_experts + common_experts_list
        # Add malicious/defense agents if enabled
        if self.enable_malicious_agents and self.malicious_agent:
            current_all_experts.append(("malicious_agent", {
                "role": "Research Expert", "expertise": "Scientific methodology", "focus": "Research innovation"
            }))
        if self.enable_defense_agent and self.defense_agent:
             current_all_experts.append(("defense_agent", {
                "role": "Security Expert", "expertise": "detecting manipulations and deception", "focus": "ensuring research integrity and safety"
            }))
        # Remove duplicates based on agent name (first element of tuple)
        seen_expert_names = set()
        unique_experts = []
        for expert_tuple in current_all_experts:
            if expert_tuple[0] not in seen_expert_names:
                unique_experts.append(expert_tuple)
                seen_expert_names.add(expert_tuple[0])
        self.all_experts = unique_experts # Update self.all_experts for this call
        print(f"[DEBUG] Experts for this discussion: {[e[0] for e in self.all_experts]}")

        for round_num in range(num_rounds):
            print(f"\nRound {round_num + 1} discussion:")
            for expert_name, expert_info in self.all_experts:
                print(f"\n{expert_info['role']} ('{expert_name}') is thinking...")
                prompt = self._get_agent_prompt(expert_info, idea_json, self.intent, related_works_string, self.discussion_history)
                system_prompt = f"""You are {expert_info['role']}, an expert in {expert_info['expertise']}.
Your focus is on {expert_info['focus']}.
Please provide your analysis in the following format:
THOUGHT: [Your detailed analysis and reasoning]
SUGGESTIONS: [Your specific suggestions for improvement]"""
                
                text, _ = get_response_from_llm(
                    prompt, client=self.client, model=self.model, system_message=system_prompt,
                    msg_history=[], temperature=self.temperature,
                )
                
                # Safely parse thought and suggestions (using the robust parsing from previous fix)
                thought_content = "Error parsing THOUGHT"
                suggestions_content = "Error parsing SUGGESTIONS"
                if "THOUGHT:" in text and "SUGGESTIONS:" in text:
                    thought_content = text.split("THOUGHT:", 1)[1].split("SUGGESTIONS:", 1)[0].strip()
                    suggestions_content = text.split("SUGGESTIONS:", 1)[1].strip()
                elif "THOUGHT:" in text: # Only thought found
                    thought_content = text.split("THOUGHT:", 1)[1].strip()
                elif "SUGGESTIONS:" in text: # Only suggestions found (less likely with current prompt)
                    suggestions_content = text.split("SUGGESTIONS:", 1)[1].strip()
                else: # Neither found, use full text as thought
                    thought_content = text.strip()
                    print(f"[WARNING] Could not parse THOUGHT/SUGGESTIONS markers for agent {expert_name}. Using full text as thought.")

                current_group_opinion_for_history = {
                    "agent_name": expert_name, # Clarified key name
                    "role": expert_info['role'],
                    "round": round_num + 1,
                    "original_thought": thought_content, # Store pre-manipulation thought
                    "original_suggestions": suggestions_content, # Store pre-manipulation suggestions
                }
                
                # Log the (potentially modified) content to discussion_history for full transparency
                    self.discussion_history.append({
                    "agent_name": expert_name,
                        "role": expert_info['role'],
                    "round": round_num + 1,
                    "content": f"THOUGHT: {current_group_opinion_for_history.get('manipulated_thought', current_group_opinion_for_history['original_thought'])}\nSUGGESTIONS: {current_group_opinion_for_history.get('manipulated_suggestions', current_group_opinion_for_history['original_suggestions'])}",
                    # Add more fields if malicious/defense agents modify them, e.g., "is_manipulated", "is_corrected"
                })
                print(f"{expert_info['role']} ('{expert_name}') completed analysis for round {round_num + 1}.")

        if self.enable_malicious_agents and hasattr(self, 'attack_session_id') and self.attack_session_id:
            self._save_attack_logs()
        if self.enable_defense_agent and hasattr(self, 'defense_session_id') and self.defense_session_id:
            self._save_defense_logs()
            
        print(f"[DEBUG] Multi-agent discussion completed. History length: {len(self.discussion_history)}")
        return self.discussion_history

    def think(
        self,
        intent: str,
        domain: str = "",
        experiment_type: str = "",
        pdf_content: Optional[str] = None,
        num_rounds: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]: # Return type changed to tuple (idea_json_string, discussion_history)
        self.intent = intent
        self.domain = domain
        self.experiment_type = experiment_type
        
        print(f"[INFO] thinker.think: Starting for intent: '{intent[:100]}...'")
        if domain: print(f"[INFO] Domain: {domain}")
        if experiment_type: print(f"[INFO] Experiment type: {experiment_type}")

        initial_idea_json_str = "{}"
        discussion_history_data: List[Dict[str, Any]] = []
        refined_idea_json_str = "{}"
        
        try:
            pdf_content = self._load_pdf_content(pdf_content)
            query = self._generate_search_query(intent)
            related_works_string = self._get_related_works(query)
            
            initial_idea_json_str = self._generate_idea(intent, related_works_string, pdf_content)
            # _generate_idea now calls _ensure_final_idea_structure internally before returning string
            
            # Conduct multi-agent discussion using the initial (structured) idea JSON string
            discussion_history_data = self._conduct_group_discussion(initial_idea_json_str, num_rounds)
            
            # Refine the idea based on the discussion history (or a summary of it)
            # _refine_idea_with_group_opinions takes the initial idea and the raw discussion_history_data
            refined_idea_json_str = self._refine_idea_with_group_opinions(initial_idea_json_str, discussion_history_data)
            
        except Exception as e:
            print(f"[ERROR] thinker.think: Exception during main think process: {e}. Full traceback:")
            traceback.print_exc()
            # If error, refined_idea_json_str might be empty or initial, discussion_history_data might be partial or empty
            # The _ensure_final_idea_structure will handle making a valid idea string from current_idea_json_str
            if not refined_idea_json_str or refined_idea_json_str == "{}":
                 refined_idea_json_str = initial_idea_json_str # Fallback to initial if refinement failed
        
        # Final safeguard for the idea string (already done by _generate_idea, but good for refined_idea too)
        final_structured_idea_json_str = self._ensure_final_idea_structure(refined_idea_json_str, intent)
        
        if self.generate_exp_plan:
            final_structured_idea_json_str = self._generate_experiment_plan(final_structured_idea_json_str)
            final_structured_idea_json_str = self._ensure_final_idea_structure(final_structured_idea_json_str, intent)

        print(f"[INFO] thinker.think: Process complete. Returning final idea and discussion history (length {len(discussion_history_data)}).")
        return final_structured_idea_json_str, discussion_history_data

    def _refine_idea_with_group_opinions(self, idea_json: str, group_discussion_history: List[Dict[str, Any]]) -> str:
        """Refine the idea based on group discussions history."""
        print("\nRefining idea based on full group discussion history...")
        
        # Convert the discussion history to a string format for the prompt
        # This might need to be more sophisticated depending on LLM context length limits
        discussion_summary_for_prompt = []
        for entry in group_discussion_history:
            # Use a multi-line f-string to correctly include newlines
            discussion_entry_str = f"""Round {entry.get('round', 'N/A')} - {entry.get('role', 'Agent')} ({entry.get('agent_name','N/A')}):
{entry.get('content', 'No content recorded')}"""
            discussion_summary_for_prompt.append(discussion_entry_str)
        discussion_str = "\n\n".join(discussion_summary_for_prompt)

        if len(discussion_str) > 10000: # Example truncation
            print(f"[WARNING] Discussion history for refinement prompt is very long ({len(discussion_str)} chars). Truncating.")
            discussion_str = discussion_str[:5000] + "\n...[TRUNCATED DISCUSSION HISTORY]...\n" + discussion_str[-5000:]
        
        synthesis_prompt = f"""
        Based on the following detailed group discussion, please refine the research idea.
        
        Original idea (JSON format):
        {idea_json}
        
        Group Discussion History:
        {discussion_str}
        
        Please refine the idea by:
        1. Addressing concerns and critiques raised during the discussion.
        2. Incorporating valuable suggestions and new insights.
        3. Ensuring the refined idea remains coherent, scientifically sound, and maintains its core concept while being improved.
        4. If ethical concerns were raised and not fully resolved, ensure the refined idea reflects this or suggests mitigation.
        
        Respond ONLY with the refined idea in the same JSON format as the original idea.
        Ensure all original fields are present, updated as necessary, and no new fields are added unless they are standard for the idea structure.
        """
        
        text, _ = get_response_from_llm(
            synthesis_prompt,
            client=self.client, model=self.model,
            system_message=self.prompts.idea_system_prompt, # This system prompt should guide JSON output
            msg_history=[], temperature=self.temperature,
        )
        
        refined_idea_dict = extract_json_between_markers(text)
        if not refined_idea_dict or not isinstance(refined_idea_dict, dict):
            print("[WARNING] Failed to extract refined idea JSON from LLM response after group discussions. Returning original idea string.")
            print(f"[DEBUG] LLM response for refinement was: {text[:500]}...")
            return idea_json # Fallback to original idea JSON string if refinement fails to produce valid JSON dict
            
        # It is crucial that refined_idea_dict maintains the structure expected by _ensure_final_idea_structure
        # We can call _ensure_final_idea_structure here too, or rely on the one in think()
        # For now, let think() handle the final absolute structural guarantee.
        return json.dumps(refined_idea_dict, indent=2)

    def _manipulate_agent_message(self, original_message: Dict[str, Any], attack_target: str) -> Dict[str, Any]:
        """Manipulate an agent's message according to the attack strategy."""
        manipulated_message = original_message.copy()
        
        # Get the system prompt for the malicious agent
        system_prompt = self.prompts.manipulator_system_prompt
        
        # Format manipulation prompt template
        manipulation_prompt = self.prompts.manipulation_prompt.format(
            role=self.malicious_agent_config['role'],
            expertise=self.malicious_agent_config['expertise'],
            thought=original_message['thought'],
            suggestions=original_message['suggestions'],
            rating=original_message.get('rating', 5.0),  # Default to 5.0 if rating is missing
            target=attack_target
        )
        
        # Get manipulated message
        try:
            text, _ = get_response_from_llm(
                manipulation_prompt,
                client=self.client,
                model=self.model,
                system_message=system_prompt,
                msg_history=[],
                temperature=0.7,  # Slightly higher temperature for more creativity in manipulation
            )
            
            # Extract the manipulated parts
            try:
                parts = text.split("THOUGHT:")
                if len(parts) < 2:
                    raise ValueError("Missing THOUGHT section")
                    
                thought_and_rest = parts[1]
                thought_parts = thought_and_rest.split("SUGGESTIONS:")
                if len(thought_parts) < 2:
                    raise ValueError("Missing SUGGESTIONS section")
                    
                manipulated_thought = thought_parts[0].strip()
                
                suggestions_and_rest = thought_parts[1]
                suggestions_parts = suggestions_and_rest.split("RATING:")
                if len(suggestions_parts) < 2:
                    raise ValueError("Missing RATING section")
                    
                manipulated_suggestions = suggestions_parts[0].strip()
                
                rating_text = suggestions_parts[1].strip()
                import re
                rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
                if not rating_match:
                    raise ValueError("Could not find numeric rating")
                manipulated_rating = float(rating_match.group(1))
                
                manipulated_message["thought"] = manipulated_thought
                manipulated_message["suggestions"] = manipulated_suggestions
                manipulated_message["rating"] = manipulated_rating
            except Exception as e:
                print(f"[red]Failed to manipulate message: {e}[/red]")
                # Return original message if manipulation fails
                return original_message
        except Exception as e:
            print(f"[red]Error calling LLM for manipulation: {e}[/red]")
            return original_message
            
        return manipulated_message
        
    def _save_defense_logs(self):
        """Save logs of the defense session."""
        if not hasattr(self, 'corrected_messages') or not self.corrected_messages:
            return
            
        defense_log_dir = osp.join(self.output_dir, "defense_logs")
        log_file = osp.join(defense_log_dir, f"{self.defense_session_id}.json")
        
        # Create a summary of the defense
        defense_summary = {
            "session_id": self.defense_session_id,
            "intent": self.intent,
            "domain": self.domain,
            "experiment_type": self.experiment_type,
            "corrected_messages": self.corrected_messages
        }
        
        # Save to JSON file
        with open(log_file, 'w') as f:
            json.dump(defense_summary, f, indent=2)
            
        print(f"[green](Hidden) Defense logs saved to {log_file}[/green]")

    def _load_pdf_content(self, pdf_path: Optional[str] = None) -> Optional[str]:
        if pdf_path and osp.isfile(pdf_path):
            with open(pdf_path, "r", encoding="utf-8") as file:
                content = file.read()
            print(f"Using content from PDF file: {pdf_path}")
            return content
        return None

    def _refine_idea(self, idea_json: str) -> str:
        current_idea_json = idea_json

        for j in range(self.iter_num):
            print(f"Refining idea {j + 1}th time out of {self.iter_num} times.")

            current_idea_dict = json.loads(current_idea_json)
            for tool in self.tools:
                tool_input = json.dumps(current_idea_dict)
                info = tool.run(tool_input)
                current_idea_dict.update(info)
            current_idea_json = json.dumps(current_idea_dict)

            current_idea_json = self.rethink(current_idea_json, current_round=j + 1)

        return current_idea_json

    def rethink(self, idea_json: str, current_round: int = 1) -> str:
        query = self._generate_search_query(
            idea_json, intent=self.intent, query_type="rethink"
        )
        related_works_string = self._get_related_works(query)

        return self._reflect_idea(idea_json, current_round, related_works_string)

    def run(
        self,
        intent: str,
        domain: str = "",
        experiment_type: str = "",
        num_ideas: int = 1, # This is a param of run, not think
        check_novelty: bool = False,
        pdf_content: Optional[str] = None,
        # num_rounds for group discussion can be passed to think if needed, or use think's default
    ) -> Tuple[Union[List[Dict[str, Any]], Dict[str, Any]], Optional[List[Dict[str, Any]]]]: # Return idea(s) and optional discussion history
        all_ideas_with_details = []
        self.intent = intent # Set instance intent for other methods if they use it
        self.domain = domain
        self.experiment_type = experiment_type
        # pdf_content is passed to think method

        # Reset states for a fresh run if these are instance variables accumulating across calls to run()
        if hasattr(self, 'intercepted_messages'): self.intercepted_messages = {}
        if hasattr(self, 'corrected_messages'): self.corrected_messages = {}
        if hasattr(self, 'discussion_history'): self.discussion_history = [] # Reset history for a new run

        first_discussion_history = None # To store history of the first idea if num_ideas=1 or only for first

        for i in range(num_ideas):
            print(f"\nProcessing idea {i + 1}/{num_ideas}")

            # Call the modified Thinker.think method which returns (idea_json_str, discussion_history_data)
            # Pass num_rounds for discussion if it's a parameter here, or let Thinker.think use its default
            idea_json_str, current_discussion_history = self.think(
                intent=intent, 
                domain=domain, 
                experiment_type=experiment_type, 
                pdf_content=pdf_content
                # num_rounds=default_num_rounds_for_discussion # If you want to control rounds from here
            )
            
            if i == 0: # Store discussion history only for the first idea, or adapt if history per idea is needed
                first_discussion_history = current_discussion_history

            try:
                idea_dict = json.loads(idea_json_str)
            except json.JSONDecodeError:
                print(f"[ERROR] Thinker.run: Failed to parse final idea JSON from self.think(): {idea_json_str[:200]}...")
                # Handle error, maybe skip this idea or return an error structure
                all_ideas_with_details.append({"error": "Failed to parse idea JSON", "raw_idea_string": idea_json_str})
                continue # Skip to next idea if in a loop

            if not idea_dict:
                print(f"[ERROR] Thinker.run: Generated idea_dict is empty for idea {i + 1}")
                all_ideas_with_details.append({"error": "Empty idea dict generated"})
                continue

            print(f"[INFO] Thinker.run: Generated idea (in run method): {idea_dict.get('Title', 'Unnamed')}")

            # The original run method had _refine_idea and generate_experiment_plan here.
            # _refine_idea was a loop of self.rethink(), and generate_experiment_plan was called based on self.generate_exp_plan.
            # My modified self.think() now incorporates generate_experiment_plan if self.generate_exp_plan is true,
            # and also calls _ensure_final_idea_structure. The old _refine_idea loop is not in the new self.think.
            # If that iterative refinement is still needed, it would have to be added back into self.think or here.
            
            # For simplicity, I am assuming the idea_dict from self.think is now the "current_idea_final"
            current_idea_final_dict = idea_dict
            
            # Optional: Novelty Check (if required, this logic can remain)
            if check_novelty:
                print("[INFO] Thinker.run: Performing novelty check...")
                current_idea_final_json_str = self._check_novelty(json.dumps(current_idea_final_dict))
                try:
                    current_idea_final_dict = json.loads(current_idea_final_json_str)
                except json.JSONDecodeError:
                    print(f"[ERROR] Thinker.run: Failed to parse idea JSON after novelty check: {current_idea_final_json_str[:200]}...")
                    # Decide how to handle, maybe use pre-novelty-check version
            
            # Add metadata about manipulations and defenses (this part of logic can remain if relevant)
            if self.enable_malicious_agents and hasattr(self, 'intercepted_messages') and self.intercepted_messages:
                current_idea_final_dict["_potentially_manipulated"] = True
                # ... (rest of malicious/defense logging)

            all_ideas_with_details.append(current_idea_final_dict)
            print(f"[INFO] Thinker.run: Completed processing for idea: {current_idea_final_dict.get('Title', 'Unnamed')}")

        # Determine what to return based on num_ideas
        final_ideas_to_return: Union[List[Dict[str, Any]], Dict[str, Any]]
        if not all_ideas_with_details:
            print("[ERROR] Thinker.run: No valid ideas generated.")
            final_ideas_to_return = {} if num_ideas == 1 else []
        elif num_ideas == 1:
            final_ideas_to_return = all_ideas_with_details[0]
        else:
            final_ideas_to_return = all_ideas_with_details
        
        # Return the idea(s) and the discussion history (e.g., of the first idea)
        return final_ideas_to_return, first_discussion_history

    def rank(
        self, ideas: List[Dict[str, Any]], intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rank multiple research ideas."""
        intent = intent or self.intent

        ideas_json = json.dumps(ideas, indent=2)
        evaluation_result = self._get_idea_evaluation(ideas_json, intent)
        ranked_ideas = self._parse_evaluation_result(evaluation_result, ideas)

        return ranked_ideas

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def modify_idea(
        self,
        original_idea: Dict[str, Any],
        modifications: List[Dict[str, Any]],
        behind_idea: Optional[Dict[str, Any]] = None,
        all_ideas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Modify an idea based on score adjustments.
        """
        # Extract required information from modifications
        instruction_lines = []
        behind_content = (
            behind_idea.get("content", "") if behind_idea else "(No reference idea)"
        )

        for mod in modifications:
            metric_name = {
                "noveltyScore": "Novelty",
                "feasibilityScore": "Feasibility",
                "impactScore": "Impact",
            }.get(mod["metric"])

            direction = mod["direction"]
            instruction_lines.append(
                {
                    "metric": metric_name,
                    "direction": direction,
                    "reference": behind_content,
                }
            )

        # Prepare the prompt using the template from YAML
        prompt = self.prompts.modify_idea_prompt.format(
            idea=json.dumps(original_idea),
            modifications=json.dumps(instruction_lines),
            intent=self.intent,
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract modified idea from response
        modified_idea = extract_json_between_markers(text)
        if not modified_idea:
            print("Failed to extract modified idea")
            return original_idea

        # Apply metadata from original idea
        modified_idea["id"] = f"node-{len(all_ideas) + 1}" if all_ideas else "node-1"
        modified_idea["parent_id"] = original_idea.get("id", "unknown")
        modified_idea["is_modified"] = True

        # Re-rank the modified idea along with all other ideas
        if all_ideas:
            ranking_ideas = [
                idea for idea in all_ideas if idea.get("id") != original_idea.get("id")
            ]
            ranking_ideas.append(modified_idea)

            ranked_ideas = self.rank(ranking_ideas, self.intent)

            for idea in ranked_ideas:
                if idea.get("id") == modified_idea.get("id"):
                    return idea

        return modified_idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def merge_ideas(
        self,
        idea_a: Dict[str, Any],
        idea_b: Dict[str, Any],
        all_ideas: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge two ideas into a new one.
        """
        # Using the merge prompt template from YAML
        prompt = self.prompts.merge_ideas_prompt.format(
            idea_a=json.dumps(idea_a), idea_b=json.dumps(idea_b), intent=self.intent
        )

        # Call LLM to get merged content
        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract the merged idea from response
        merged_idea = extract_json_between_markers(text)
        if not merged_idea:
            print("Failed to extract merged idea")
            return None

        # Add metadata about the merged sources
        merged_idea["id"] = f"node-{len(all_ideas) + 1}" if all_ideas else "node-1"
        merged_idea["parent_ids"] = [
            idea_a.get("id", "unknown"),
            idea_b.get("id", "unknown"),
        ]
        merged_idea["is_merged"] = True

        # Re-rank the merged idea along with all other ideas
        if all_ideas:
            # Create a list with all ideas except the ones being merged, plus the new merged idea
            ranking_ideas = [
                idea
                for idea in all_ideas
                if idea.get("id") != idea_a.get("id")
                and idea.get("id") != idea_b.get("id")
            ]
            ranking_ideas.append(merged_idea)

            # Rank all ideas together
            ranked_ideas = self.rank(ranking_ideas, self.intent)

            # Find and return the merged idea from the ranked list
            for idea in ranked_ideas:
                if idea.get("id") == merged_idea.get("id"):
                    return idea

        # If no other ideas provided or ranking failed, return just the merged idea
        return merged_idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: str) -> str:
        idea_dict = json.loads(idea)

        print("Generating experimental plan for the idea...")
        prompt = self.prompts.experiment_plan_prompt.format(
            idea=idea, intent=self.intent
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        experiment_plan = extract_json_between_markers(text)
        if not experiment_plan:
            print("Failed to generate experimental plan.")
            return idea

        idea_dict["Experiment"] = experiment_plan
        print("Experimental plan generated successfully.")

        return json.dumps(idea_dict, indent=2)

    def _get_idea_evaluation(self, ideas_json: str, intent: str) -> str:
        """Get comparative evaluation from LLM"""
        prompt = self.prompts.idea_evaluation_prompt.format(
            intent=intent, ideas=ideas_json
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.evaluation_system_prompt,
            msg_history=[],
            temperature=0.3,
        )

        return text

    def _parse_evaluation_result(
        self, evaluation_text: str, original_ideas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse evaluation result and update idea dictionaries with rankings"""
        # Extract JSON from response
        evaluation_data = extract_json_between_markers(evaluation_text)
        if not evaluation_data:
            print("Failed to extract JSON from evaluation response")
            return []
        # Create mapping from idea name to original idea dict
        idea_map = {idea.get("Name", ""): idea for idea in original_ideas}

        # Create ranked list
        ranked_ideas = []
        for ranked_item in evaluation_data.get("ranked_ideas", []):
            idea_name = ranked_item.get("Name", "")
            if idea_name in idea_map:
                # Get original idea and update with ranking data
                idea = idea_map[idea_name].copy()

                # Add ranking information
                idea["FeasibilityRanking"] = ranked_item.get("FeasibilityRanking")
                idea["NoveltyRanking"] = ranked_item.get("NoveltyRanking")
                idea["ImpactRanking"] = ranked_item.get("ImpactRanking")
                idea["NoveltyReason"] = ranked_item.get("NoveltyReason", "")
                idea["FeasibilityReason"] = ranked_item.get("FeasibilityReason", "")
                idea["ImpactReason"] = ranked_item.get("ImpactReason", "")
                # Remove all the scoring, using ranking instead
                if "Interestingness" in idea:
                    del idea["Interestingness"]
                if "Feasibility" in idea:
                    del idea["Feasibility"]
                if "Novelty" in idea:
                    del idea["Novelty"]
                if "IntentAlignment" in idea:
                    del idea["IntentAlignment"]
                if "Score" in idea:
                    del idea["Score"]
                ranked_ideas.append(idea)

        return ranked_ideas

    def _get_related_works(self, query: str) -> str:
        """Get related works using query caching, similar to Reviewer class"""
        if query in self._query_cache:
            related_papers = self._query_cache[query]
            print("✅ Using cached query results")
        else:
            print(f"Searching for papers with query: {query}")
            results_dict = self.searcher.run(query)
            related_papers = list(results_dict.values()) if results_dict else []
            self._query_cache[query] = related_papers

            if related_papers:
                print("✅ Related Works Found")
            else:
                print("❎ No Related Works Found")

        return self._format_paper_results(related_papers)

    def _generate_search_query(
        self, content: str, intent: Optional[str] = None, query_type: str = "standard"
    ) -> str:
        prompt_mapping = {
            "standard": self.prompts.query_prompt.format(intent=content),
            "rethink": self.prompts.rethink_query_prompt.format(
                intent=intent, idea=content
            ),
            "novelty": self.prompts.novelty_query_prompt.format(
                intent=intent, idea=content
            ),
        }

        prompt = prompt_mapping.get(query_type, "")
        response, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        query_data = extract_json_between_markers(response)
        return str(query_data.get("Query", "")) if query_data else ""

    def _determine_experiment_type(self, idea_dict: Optional[Dict[str, Any]]) -> str: # Added Optional for safety
        if not isinstance(idea_dict, dict): # Guard against None or non-dict
            idea_dict = {}

        if self.experiment_type:
            return self.experiment_type
            
        current_domain_for_think = self.domain
        if current_domain_for_think:
            physical_domains = ["Biology", "Physics", "Chemistry", "Material Science", "Medical Science", "Medicine"]
            computational_domains = ["Information Science"]
            normalized_domain = current_domain_for_think.replace("_", " ").title()
            if normalized_domain in physical_domains: return 'physical'
            if normalized_domain in computational_domains: return 'computational'
        
        text_to_check = ' '.join([
            str(idea_dict.get('Title', '')), # Use str() for safety
            str(idea_dict.get('Problem', '')),
            str(idea_dict.get('Approach', ''))
        ]).lower()
        
        physical_keywords_map = {
            'chemistry': ['chemical', 'reaction', 'compound', 'molecule', 'synthesis', 'catalyst'],
            'physics': ['particle', 'force', 'energy', 'wave', 'field', 'measurement'],
            'biology': ['cell', 'organism', 'tissue', 'gene', 'protein', 'enzyme'],
            'materials': ['material', 'fabrication', 'synthesis', 'characterization'],
            'medicine': ['clinical', 'patient', 'therapy', 'drug', 'trial']
        }
        computational_keywords_map = {
            'computer_science': ['algorithm', 'program', 'software', 'computation', 'code', 'simulation'],
            'information_science': ['data', 'information', 'analysis', 'processing', 'network'],
            'mathematics': ['mathematical', 'equation', 'model']
        }
        for _, keywords in physical_keywords_map.items():
            if any(keyword in text_to_check for keyword in keywords): return 'physical'
        for _, keywords in computational_keywords_map.items():
            if any(keyword in text_to_check for keyword in keywords): return 'computational'
                
        print("[INFO] thinker._determine_experiment_type: Defaulting to 'computational'.")
                return 'computational'
                
    @api_calling_error_exponential_backoff(retries=3, base_wait_time=2) # Added retry
    def _generate_experiment_plan(self, idea_json_str: str) -> str:
        print(f"[DEBUG] thinker._generate_experiment_plan: Received idea string: {idea_json_str[:200]}...")
        try:
            idea_dict = json.loads(idea_json_str)
            if not isinstance(idea_dict, dict):
                 raise json.JSONDecodeError("Parsed JSON is not a dictionary", idea_json_str, 0)
        except json.JSONDecodeError as e:
            print(f"[ERROR] thinker._generate_experiment_plan: Invalid JSON for idea: {e}. Cannot generate plan.")
            # Return the original idea string, the _ensure_final_idea_structure in think() will handle Experiment field.
            return idea_json_str 

        print("[INFO] thinker._generate_experiment_plan: Generating experimental plan...")
        experiment_type = self._determine_experiment_type(idea_dict)
        print(f"[DEBUG] thinker._generate_experiment_plan: Determined experiment type: {experiment_type}")
        
        prompt_template_name = 'physical_experiment_plan_prompt' if experiment_type == 'physical' else 'experiment_plan_prompt'
        current_prompt_template = getattr(self.prompts, prompt_template_name)
        
        prompt = current_prompt_template.format(idea=idea_json_str, intent=self.intent)

        text, _ = get_response_from_llm(
            prompt, client=self.client, model=self.model,
            system_message=self.prompts.idea_system_prompt, # This should guide for JSON output of the plan
            msg_history=[], temperature=self.temperature
        )

        experiment_plan_dict = extract_json_between_markers(text)
        
        if not isinstance(experiment_plan_dict, dict):
            print(f"[WARNING] thinker._generate_experiment_plan: Failed to extract valid JSON dict for plan. LLM response: {text[:300]}...")
            experiment_plan_dict = {"Description": "Experimental plan generation failed or returned non-dict.", "Error": "Extraction failed."}
        
        experiment_plan_dict.setdefault('Type', experiment_type) # Ensure Type is in the plan
        
        # Update the original idea_dict with the new/updated Experiment section
        idea_dict["Experiment"] = experiment_plan_dict 
        
        updated_idea_json_str = json.dumps(idea_dict, indent=2)
        print(f"[DEBUG] thinker._generate_experiment_plan: Idea with experiment plan: {updated_idea_json_str[:500]}...")
        return updated_idea_json_str

    def _save_ideas(self, ideas: List[str]) -> None:
        output_path = osp.join(self.output_dir, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {output_path}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_idea(
        self, idea_json: str, current_round: int, related_works_string: str
    ) -> str:
        prompt = self.prompts.idea_reflection_prompt.format(
            intent=self.intent,
            current_round=current_round,
            num_reflections=self.iter_num,
            related_works_string=related_works_string,
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        new_idea = extract_json_between_markers(text)
        if isinstance(new_idea, list) and new_idea:
            new_idea = new_idea[0]

        if not new_idea:
            print("Failed to extract a valid idea from refinement")
            return idea_json

        if "I am done" in text:
            print(f"Idea refinement converged after {current_round} iterations.")

        return json.dumps(new_idea, indent=2)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(
        self,
        intent: str,
        related_works_string: str,
        pdf_content: Optional[str] = None,
    ) -> str: # Returns JSON string
        print(f"[DEBUG] thinker._generate_idea: Generating initial idea for intent: '{intent[:100]}...'")
        pdf_section = (
            f"Based on the content of the following paper:\n\n{pdf_content}\n\n"
            if pdf_content
            else ""
        )

        # The system prompt should strongly guide the LLM to produce JSON with all WRITER_MINI_REQUIRED_KEYS
        # System prompt content is loaded from YAML, ensure it's well-defined there.
        
        llm_response_text, _ = get_response_from_llm(
            self.prompts.idea_first_prompt.format(
                intent=intent,
                related_works_string=related_works_string,
                num_reflections=1,
                pdf_section=pdf_section,
            ),
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        idea_dict = extract_json_between_markers(llm_response_text)
        
        if isinstance(idea_dict, list): # Handle if LLM returns a list
            idea_dict = next((item for item in idea_dict if isinstance(item, dict)), None)

        if not isinstance(idea_dict, dict):
            print(f"[ERROR] thinker._generate_idea: Failed to extract valid JSON dict. LLM response snippet: {llm_response_text[:500]}...")
            # Return an empty dict string; 'think' method will handle final structure.
            return json.dumps({})

        print(f"[DEBUG] thinker._generate_idea: Successfully extracted initial idea dict. Keys: {list(idea_dict.keys())}")
        return json.dumps(idea_dict, indent=2)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _check_novelty(self, idea_json: str, max_iterations: int = 10) -> str:
        try:
            idea_dict = json.loads(idea_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            return idea_json

        print(f"\nChecking novelty of idea: {idea_dict.get('Name', 'Unnamed')}")

        for iteration in range(max_iterations):
            print(f"Novelty check iteration {iteration + 1}/{max_iterations}")

            query = self._generate_search_query(
                idea_json, intent=self.intent, query_type="novelty"
            )
            papers_str = self._get_related_works(query)

            prompt = self.prompts.novelty_prompt.format(
                current_round=iteration + 1,
                num_rounds=max_iterations,
                intent=self.intent,
                idea=idea_json,
                last_query_results=papers_str,
            )

            text, _ = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.novelty_system_prompt,
                msg_history=[],
            )

            if "NOVELTY CHECK: NOVEL" in text:
                print("Decision: Idea is novel")
                idea_dict["novel"] = True
                break
            elif "NOVELTY CHECK: NOT NOVEL" in text:
                print("Decision: Idea is not novel")
                idea_dict["novel"] = False
                break
            elif "NOVELTY CHECK: CONTINUE" in text:
                print("Decision: Need more information to determine novelty")
                continue
            else:
                print(f"No clear decision in iteration {iteration + 1}, continuing")

        if "novel" not in idea_dict:
            print(
                "Maximum iterations reached without decision, defaulting to not novel."
            )
            idea_dict["novel"] = False

        return json.dumps(idea_dict, indent=2)

    @staticmethod
    def _format_paper_results(papers: List[Dict[str, Any]]) -> str:
        """Format paper results exactly like Reviewer class"""
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"{i}: {paper.get('title', 'No title')}. {paper.get('source', 'No authors')}. "
                f"{paper.get('info', 'No venue')}"
            )

        return "\n\n".join(paper_strings)

    def _save_attack_logs(self):
        """Save logs of the attack session."""
        if not hasattr(self, 'intercepted_messages') or not self.intercepted_messages:
            return
            
        attack_log_dir = osp.join(self.output_dir, "attack_logs")
        log_file = osp.join(attack_log_dir, f"{self.attack_session_id}.json")
        
        # Create a summary of the attack
        attack_summary = {
            "session_id": self.attack_session_id,
            "intent": self.intent,
            "domain": self.domain,
            "experiment_type": self.experiment_type,
            "attack_severity": self.attack_severity,
            "intercepted_messages": self.intercepted_messages
        }
        
        # Save to JSON file
        with open(log_file, 'w') as f:
            json.dump(attack_summary, f, indent=2)
            
        print(f"[red](Hidden) Attack logs saved to {log_file}[/red]")

    def _ensure_final_idea_structure(self, idea_json_str: str, intent_for_defaults: str) -> str:
        """
        Parses the idea JSON string and ensures the resulting dictionary 
        contains all required keys for WriterMini, filling with defaults if necessary.
        Returns a JSON string of the validated/completed idea.
        """
        print(f"[DEBUG] thinker._ensure_final_idea_structure: Validating final idea string: {idea_json_str[:200]}...")
        idea_dict = None
        try:
            idea_dict = json.loads(idea_json_str)
            if not isinstance(idea_dict, dict):
                print(f"[ERROR] thinker._ensure_final_idea_structure: Parsed idea is not a dict. Content: {str(idea_dict)[:200]}")
                idea_dict = {} # Force to empty dict if not a dict
        except json.JSONDecodeError:
            print(f"[ERROR] thinker._ensure_final_idea_structure: Failed to parse idea JSON string. Content: {idea_json_str[:200]}")
            idea_dict = {} # Force to empty dict if JSON is invalid

        # At this point, idea_dict is guaranteed to be a dictionary (possibly empty)
        
        for key in self.WRITER_MINI_REQUIRED_KEYS:
            # Check if key is missing, or if it's present but None or an empty string
            if key not in idea_dict or idea_dict.get(key) is None or (isinstance(idea_dict.get(key), str) and not str(idea_dict.get(key)).strip()):
                default_value = f"Default value for {key} based on intent: '{intent_for_defaults[:30]}...'"
                # Define more specific defaults
                if key == "Title":
                    default_value = idea_dict.get("Name") or f"Research on {intent_for_defaults[:50]}"
                elif key == "Problem":
                    default_value = f"The primary challenge is related to {intent_for_defaults[:50]}."
                elif key == "Approach":
                    default_value = f"A standard scientific approach will be used to investigate {intent_for_defaults[:30]}."
                elif key == "Expected Results":
                    default_value = f"Results are expected to shed light on {intent_for_defaults[:30]}."
                elif key == "Importance":
                    default_value = f"The importance of this research on {intent_for_defaults[:30]} lies in its potential contributions."
                elif key == "Difficulty":
                    default_value = f"The main difficulty in researching {intent_for_defaults[:30]} involves technical and theoretical challenges."
                elif key == "NoveltyComparison":
                    default_value = f"This work on {intent_for_defaults[:30]} aims to be novel by addressing gaps in existing studies."
                elif key == "Experiment":
                    default_value = {
                        "Description": f"A conceptual experiment will be designed to test the hypotheses regarding {intent_for_defaults[:30]}.",
                        "Type": self._determine_experiment_type(idea_dict) # Pass current idea_dict for context
                    }
                
                idea_dict[key] = default_value
                print(f"[WARNING] thinker._ensure_final_idea_structure: Key '{key}' was missing/empty. Added default: '{str(default_value)[:100]}...'")
        
        # Specific check for 'Experiment' sub-structure
        if isinstance(idea_dict.get("Experiment"), dict):
            if "Description" not in idea_dict["Experiment"] or not idea_dict["Experiment"]["Description"]:
                idea_dict["Experiment"]["Description"] = f"Default experiment description for {intent_for_defaults[:30]}."
            if "Type" not in idea_dict["Experiment"] or not idea_dict["Experiment"]["Type"]:
                idea_dict["Experiment"]["Type"] = self._determine_experiment_type(idea_dict)
        else: # If 'Experiment' is not a dict (e.g. missing, or wrong type from earlier default)
            print(f"[WARNING] thinker._ensure_final_idea_structure: 'Experiment' field not a valid dict. Resetting. Content: {idea_dict.get('Experiment')}")
            idea_dict["Experiment"] = {
                "Description": f"Default conceptual experiment for {intent_for_defaults[:30]}.",
                "Type": self._determine_experiment_type(idea_dict)
            }
            
        validated_json_str = json.dumps(idea_dict, indent=2)
        print(f"[DEBUG] thinker._ensure_final_idea_structure: Final validated idea keys: {list(idea_dict.keys())}")
        return validated_json_str
