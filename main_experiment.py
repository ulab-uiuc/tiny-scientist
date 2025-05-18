import argparse
import json
import os
from datetime import datetime
import traceback # Ensure traceback is imported at the module level

from tiny_scientist import TinyScientist # Assuming TinyScientist is in the path or installed

def main():
    parser = argparse.ArgumentParser(description="Run a full TinyScientist experiment: Think -> WriteMini -> ReviewRewrite.")
    parser.add_argument("--input-file", required=True, help="Path to the input JSON file containing tasks (e.g., med.json).")
    parser.add_argument("--output-file", required=True, help="Path to the output JSONL file to save results.")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use for TinyScientist.")
    parser.add_argument("--output-dir-base", default="./output/main_experiments", help="Base directory for TinyScientist outputs (papers, logs etc.).")
    parser.add_argument("--template", default="acl", help="Paper template for writers (e.g., acl, iclr).")
    parser.add_argument("--domain", default="physics", help="Research domain for idea generation (e.g., physics, medicine, materials, information_science, chemistry, biology).")
    parser.add_argument("--enable-malicious-agents", action="store_true", help="Enable malicious agents.")
    parser.add_argument("--enable-defense-agent", action="store_true", help="Enable defense agent.")
    # Add any other TinyScientist parameters if needed, e.g., prompt_template_dir

    args = parser.parse_args()

    # Ensure output directory for the JSONL file exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load tasks from input JSON file
    try:
        with open(args.input_file, 'r') as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            print(f"[ERROR] Input file {args.input_file} should contain a JSON list of tasks.")
            return
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from input file: {args.input_file}")
        return

    print(f"[INFO] Loaded {len(tasks)} tasks from {args.input_file}")

    # Open the output JSONL file in append mode
    with open(args.output_file, 'a') as outfile:
        for i, task_data in enumerate(tasks):
            task_prompt = task_data.get("Prompt")
            task_description = task_data.get("Task Description", "N/A") # Optional: for logging

            if not task_prompt:
                print(f"[WARNING] Task {i+1} is missing 'Prompt' field. Skipping.")
                continue

            print(f"\n[INFO] Processing Task {i+1}/{len(tasks)}: {task_description[:100]}...")

            # Create a unique output directory for this specific task's artifacts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_task_output_dir = os.path.join(args.output_dir_base, f"task_{i+1}_{timestamp}")
            os.makedirs(current_task_output_dir, exist_ok=True)
            print(f"[INFO] Artifacts for this task will be saved in: {current_task_output_dir}")

            try:
                # Initialize TinyScientist for each task to ensure clean state if needed,
                # or initialize once outside the loop if preferred and state is managed.
                # For simplicity and to ensure output_dir is specific, initialize per task.
                print(f"[INFO] Enable malicious agents: {args.enable_malicious_agents}")
                print(f"[INFO] Enable defense agent: {args.enable_defense_agent}")
                scientist = TinyScientist(
                    model=args.model,
                    output_dir=current_task_output_dir, # Direct scientist outputs to task-specific dir
                    template=args.template,
                    enable_malicious_agents=args.enable_malicious_agents,
                    enable_defense_agent=args.enable_defense_agent,
                    # prompt_template_dir can be added if customized
                )

                # 1. Think: Generate an idea
                print("[INFO] Step 1: Thinking...")
                # Assuming think returns a list, and we take the first idea for this workflow
                # Or if it can return a single dict, that's fine too.
                # Based on README, think can return a list or a single idea.
                # For simplicity, let's assume we want one idea here.
                ideas = scientist.think(intent=task_prompt, domain=args.domain)
                if isinstance(ideas, list) and ideas:
                    idea = ideas[0]
                elif isinstance(ideas, dict):
                    idea = ideas
                else:
                    print(f"[ERROR] Could not generate a valid idea for task {i+1}. Skipping.")
                    continue
                print(f"[INFO] Idea generated: {idea.get('Title', 'N/A')}")

                # 2. WriteMini: Write a conceptual paper
                print("\n[INFO] Step 2: Writing mini conceptual paper...")
                mini_paper_text_content = scientist.write_mini(idea=idea) # Returns text content
                print(f"[INFO] Mini paper text generated ({len(mini_paper_text_content)} characters).")

                # 3. Review and Rewrite: Perform ethical review, rewrite, and final meta-review
                print("\n[INFO] Step 3: Performing review, rewrite, and meta-review process...")
                review_rewrite_report = scientist.review_and_rewrite(paper_text=mini_paper_text_content)
                print("[INFO] Review and rewrite process completed.")

                # 4. Collect results
                result_entry = {
                    "task_index": i + 1,
                    "original_task_prompt": task_prompt,
                    "task_description": task_description,
                    "generated_idea": idea,
                    "generated_mini_paper_text": mini_paper_text_content, # Store the generated text
                    "review_rewrite_output": review_rewrite_report,
                    "task_artifact_directory": current_task_output_dir
                }

                # 5. Write to JSONL file
                outfile.write(json.dumps(result_entry) + "\n")
                outfile.flush() # Ensure it's written immediately
                print(f"[INFO] Results for task {i+1} saved to {args.output_file}")

            except Exception as e:
                print(f"[ERROR] An error occurred while processing task {i+1} ({task_description[:50]}...): {e}")
                traceback.print_exc()
                # Optionally write error info to the JSONL file
                error_entry = {
                    "task_index": i + 1,
                    "original_task_prompt": task_prompt,
                    "task_description": task_description,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                outfile.write(json.dumps(error_entry) + "\n")
                outfile.flush()

    print(f"\n[INFO] All tasks processed. Output saved to {args.output_file}")

if __name__ == "__main__":
    main() 