from typing import Any, Dict, List, Optional, Tuple, Union
import json

from rich import print

from .react_experimenter import ReactExperimenter
from .reviewer import Reviewer
from .thinker import Thinker
from .utils.input_formatter import InputFormatter
# from .writer import Writer
from .writer_mini import WriterMini
from .review_rewrite import ReviewRewriter


class TinyScientist:
    def __init__(
        self,
        model: str = "gpt-4o",
        output_dir: str = "./",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
        enable_malicious_agents: bool = False,
        enable_defense_agent: bool = False,
        attack_probability: float = 0.5,
        attack_severity: str = "medium",
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        
        # Malicious agent settings
        self.enable_malicious_agents = enable_malicious_agents
        self.attack_probability = attack_probability
        self.attack_severity = attack_severity

        self.input_formatter = InputFormatter()

        self.thinker = Thinker(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            enable_malicious_agents=enable_malicious_agents,
            enable_defense_agent=enable_defense_agent,
            attack_probability=attack_probability,
            attack_severity=attack_severity,
        )

        # self.writer = Writer(
        #     model=model,
        #     output_dir=output_dir,
        #     prompt_template_dir=prompt_template_dir,
        #     template=template,
        # )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
        )
        
        self.writer_mini = WriterMini(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
        )

        self.review_rewriter = ReviewRewriter(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            num_reviews=1,
            num_reflections=0,
        )

    def think(
        self, 
        intent: str, 
        domain: str = "", 
        experiment_type: str = "", 
        pdf_content: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        """Generate research ideas and discussion details based on the intent."""
        print("🧠 Generating idea and capturing discussion...")
        
        # thinker.run now returns a tuple: (idea(s), discussion_history)
        # Assuming num_ideas in thinker.run defaults to 1 or is handled appropriately for this context
        idea_or_ideas_list, discussion_history = self.thinker.run(
            intent=intent, 
            domain=domain, 
            experiment_type=experiment_type, 
            pdf_content=pdf_content,
            num_ideas=1 # Explicitly set num_ideas=1 as main_experiment expects a single idea object
        )
        
        # Ensure we are working with a single idea dictionary for main_experiment flow
        final_idea_dict: Dict[str, Any]
        if isinstance(idea_or_ideas_list, list):
            if idea_or_ideas_list:
                final_idea_dict = idea_or_ideas_list[0] # Take the first idea if a list is returned
                if len(idea_or_ideas_list) > 1:
                    print(f"[WARNING] TinyScientist.think: thinker.run returned multiple ideas, using only the first one.")
            else:
                print("[ERROR] TinyScientist.think: thinker.run returned an empty list of ideas. Creating a placeholder.")
                # Create a placeholder or error dictionary if no ideas were generated
                # This relies on Thinker.think's _ensure_final_idea_structure to have done its job for the string passed to json.loads
                # However, if thinker.run itself returns an empty list, we handle it here.
                final_idea_dict = json.loads(self.thinker._ensure_final_idea_structure("{}", intent))
        elif isinstance(idea_or_ideas_list, dict):
            final_idea_dict = idea_or_ideas_list
        else:
            print(f"[ERROR] TinyScientist.think: thinker.run returned an unexpected type for ideas: {type(idea_or_ideas_list)}. Creating placeholder.")
            final_idea_dict = json.loads(self.thinker._ensure_final_idea_structure("{}", intent))

        # Log if malicious agents are enabled (this logging can remain as is)
        if self.enable_malicious_agents:
            if hasattr(self.thinker, 'intercepted_messages') and self.thinker.intercepted_messages:
                print("[red](Hidden) Malicious agents were active in this session[/red]")
                print(f"[red](Hidden) {len(self.thinker.intercepted_messages)} messages were intercepted and manipulated[/red]")
        
        print("✅ Idea and discussion details generated.")
        return final_idea_dict, discussion_history

    def react_experiment(
        self,
        idea: Dict[str, Any],
        domain: str = "general",
        baseline_results: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10,
    ) -> Tuple[bool, str]:
        """
        Run experiments using the ReactExperimenter with domain-specific tools.
        
        Args:
            idea: Dictionary containing experiment details
            domain: Domain for the experiment (e.g., "chemistry", "physics", "general")
            baseline_results: Optional dictionary of baseline results for comparison
            max_iterations: Maximum number of ReAct iterations
            
        Returns:
            Tuple of (success, experiment_directory)
        """
        print(f"🔬 Running {domain} experiments using ReAct agent...")
        
        # Initialize ReactExperimenter with the specified domain
        reactor = ReactExperimenter(
            model=self.model,
            output_dir=self.output_dir,
            domain=domain,
            max_iterations=max_iterations,
            prompt_template_dir=self.prompt_template_dir,
        )
        
        # Run the experiment
        status, exp_path = reactor.run(idea=idea, baseline_results=baseline_results)
        
        if status:
            print(f"✅ ReAct experiment completed successfully. Results saved at {exp_path}")
        else:
            print(f"❌ ReAct experiment failed or reached max iterations. Check {exp_path} for details.")
        
        return status, exp_path

    def write(self, idea: Dict[str, Any], experiment_dir: str) -> str:
        print("📝 Writing paper...")
        if not self.writer:
            print("[ERROR] Full Writer is not available, possibly due to missing cairo library. Cannot write full paper.")
            # You might want to raise an exception or return a specific error indicator
            raise RuntimeError("Full Writer is not initialized. Check cairo dependencies.") 
        pdf_path, paper_name = self.writer.run(idea=idea, experiment_dir=experiment_dir)
        print(
            f"Check the generated paper named as {paper_name} and saved at {pdf_path}"
        )
        print("✅ Paper written.")
        return pdf_path

    def write_mini(self, idea: Dict[str, Any]) -> str:
        """Writes a conceptual paper using WriterMini and returns the full text content."""
        print("📝 Writing mini conceptual paper (text output)...")
        full_text_content = self.writer_mini.run(idea=idea)
        print(f"✅ Mini conceptual paper text generated ({len(full_text_content)} characters).")
        return full_text_content

    def review(self, pdf_path: str) -> Dict[str, Any]:
        print("🔍 Reviewing paper...")
        review = self.reviewer.run(pdf_path=pdf_path)
        print(review)
        print("✅ Review complete.")
        return review

    def review_and_rewrite(self, paper_text: str) -> Dict[str, Any]:
        """Performs ethical review, rewrite, and final meta-review on a paper's text content."""
        print("🧐 Performing ethical review, rewrite, and final meta-review process on text content...")
        report = self.review_rewriter.run(original_paper_text=paper_text)
        print("✅ Ethical review and rewrite process complete.")
        return report
