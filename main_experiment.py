import argparse
import json
import os
from datetime import datetime
import traceback
from multiprocessing import Pool, Manager, Lock # Added Lock
from functools import partial
# import time # No longer needed for writer sleep

from tiny_scientist import TinyScientist

# Worker function to process a single task and write its result
def process_task_and_write(task_data_tuple, common_args, overall_output_dir_base, file_lock):
    i, task_data = task_data_tuple
    task_prompt = task_data.get("Prompt")
    task_description = task_data.get("Task Description", "N/A")
    
    entry_to_write = None # This will hold the dict to be written to file

    if not task_prompt:
        print(f"[WARNING] Task {i+1} (original index) is missing 'Prompt' field. Skipping.")
        entry_to_write = {"task_index": i + 1, "error": "Missing Prompt field", "status": "skipped"}
    else:
        print(f"\n[INFO] Worker {os.getpid()}: Processing Task {i+1} (original index): {task_description[:100]}...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_task_output_dir = os.path.join(overall_output_dir_base, f"task_{i+1}_{timestamp}")
        try:
            os.makedirs(current_task_output_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Worker {os.getpid()}: Failed to create task output dir {current_task_output_dir}: {e}")
            entry_to_write = {
                "task_index": i + 1, "original_task_prompt": task_prompt, "task_description": task_description,
                "error": f"Failed to create task output directory: {e}", "status": "failed_setup"
            }

        if not entry_to_write: # Proceed only if setup was successful
            # print(f"[INFO] Worker {os.getpid()}: Artifacts for Task {i+1} in: {current_task_output_dir}")
            try:
                scientist = TinyScientist(
                    model=common_args.model,
                    output_dir=current_task_output_dir,
                    template=common_args.template,
                    enable_malicious_agents=common_args.enable_malicious_agents,
                    enable_defense_agent=common_args.enable_defense_agent,
                )
                idea, discussion_history = scientist.think(intent=task_prompt, domain=common_args.domain)
                
                if not idea:
                    error_detail = "Idea generation returned empty or invalid idea object"
                    print(f"[ERROR] Worker {os.getpid()}: Task {i+1} - {error_detail}.")
                    entry_to_write = {
                        "task_index": i + 1, "original_task_prompt": task_prompt, "task_description": task_description,
                        "error": error_detail, "status": "failed_think"
                    }
                else:
                    mini_paper_text_content = scientist.write_mini(idea=idea)
                    review_rewrite_report = scientist.review_and_rewrite(paper_text=mini_paper_text_content)
                    
                    entry_to_write = {
                        "task_index": i + 1, "original_task_prompt": task_prompt, "task_description": task_description,
                        "generated_idea": idea, "discussion_history": discussion_history,
                        "generated_mini_paper_text": mini_paper_text_content,
                        "review_rewrite_output": review_rewrite_report,
                        "task_artifact_directory": current_task_output_dir, "status": "success"
                    }
                    print(f"[SUCCESS] Worker {os.getpid()}: Task {i+1} completed.")

            except Exception as e:
                print(f"[ERROR] Worker {os.getpid()}: Unhandled exception processing task {i+1}: {e}")
                entry_to_write = {
                    "task_index": i + 1, "original_task_prompt": task_prompt, "task_description": task_description,
                    "error": str(e), "traceback": traceback.format_exc(), "status": "failed_processing"
                }

    # Write the result to the shared output file using a lock
    if entry_to_write is not None:
        try:
            file_lock.acquire()
            # print(f"[DEBUG] Worker {os.getpid()}: Acquired lock for task {entry_to_write.get('task_index')}")
            with open(common_args.output_file, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(entry_to_write) + "\n")
                outfile.flush()
            # print(f"[DEBUG] Worker {os.getpid()}: Released lock for task {entry_to_write.get('task_index')}")
        except Exception as e:
            print(f"[ERROR] Worker {os.getpid()}: Failed to write result for task {entry_to_write.get('task_index')} to file: {e}")
        finally:
            file_lock.release() # Ensure lock is always released
    return entry_to_write.get("status", "unknown") if entry_to_write else "no_entry_created"

def main():
    parser = argparse.ArgumentParser(description="Run TinyScientist experiment with parallel processing and resume capability.")
    parser.add_argument("--input-file", required=True, help="Path to the input JSON file containing tasks.")
    parser.add_argument("--output-file", required=True, help="Path to the output JSONL file to save results.")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use.")
    parser.add_argument("--output-dir-base", default="./output/main_experiments", help="Base directory for task artifacts.")
    parser.add_argument("--template", default="acl", help="Paper template.")
    parser.add_argument("--domain", default="physics", help="Research domain.")
    parser.add_argument("--enable-malicious-agents", action="store_true", help="Enable malicious agents.")
    parser.add_argument("--enable-defense-agent", action="store_true", help="Enable defense agent.")
    parser.add_argument("--parallel-num", type=int, default=1, help="Number of parallel processes (default: 1 for sequential).")

    args = parser.parse_args()

    # Ensure output directory for the JSONL file exists, but DO NOT truncate the file here.
    # The file will be appended to by workers.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # Base directory for individual task artifacts
    os.makedirs(args.output_dir_base, exist_ok=True)

    processed_prompts_set = set()
    if os.path.exists(args.output_file):
        print(f"[INFO] Output file {args.output_file} exists. Reading previously processed prompts...")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f_read:
                for line_number, line in enumerate(f_read):
                    try:
                        # Basic check for non-empty line before parsing
                        if line.strip(): 
                            data = json.loads(line)
                            if "original_task_prompt" in data:
                                processed_prompts_set.add(data["original_task_prompt"])
                        else:
                            print(f"[WARNING] Skipped empty line {line_number + 1} in existing output file.")
                    except json.JSONDecodeError:
                        print(f"[WARNING] Could not parse line {line_number + 1} in {args.output_file} as JSON. Skipping this line for resume check.")
            print(f"[INFO] Found {len(processed_prompts_set)} unique prompts in existing output file.")
        except Exception as e:
            print(f"[ERROR] Failed to read or parse existing output file {args.output_file}: {e}. Will process all tasks.")
            processed_prompts_set.clear() # Ensure it's empty if reading failed
    else:
        print(f"[INFO] Output file {args.output_file} does not exist. Will process all tasks.")

    try:
        with open(args.input_file, 'r') as f:
            all_tasks_from_input = json.load(f)
        if not isinstance(all_tasks_from_input, list):
            print(f"[ERROR] Input file {args.input_file} should be a JSON list of tasks.")
            return
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from input file: {args.input_file}")
        return

    # Filter tasks: only process those not already in processed_prompts_set
    tasks_to_process = []
    for task_data in all_tasks_from_input:
        if task_data.get("Prompt") not in processed_prompts_set:
            tasks_to_process.append(task_data)
        else:
            print(f"[INFO] Skipping task with prompt (already processed): {task_data.get('Prompt')[:70]}...")
            
    print(f"[INFO] Loaded {len(all_tasks_from_input)} total tasks from {args.input_file}.")
    print(f"[INFO] Skipping {len(all_tasks_from_input) - len(tasks_to_process)} tasks already processed.")
    print(f"[INFO] {len(tasks_to_process)} tasks remaining to process.")

    if not tasks_to_process:
        print("[INFO] No new tasks to process.")
        print(f"\n[INFO] All tasks (including previously completed) processed. Output at {args.output_file}")
        return

    # Enumerate only the tasks that need processing, but keep their original index from all_tasks_from_input for consistency in output
    # This requires finding the original index for each task_to_process.
    # For simplicity in process_task_and_write, we pass the original index `i` from the full enumerated list.
    # So, we build tasks_with_original_indices for the pool/loop.
    
    tasks_for_processing_with_original_indices = []
    for original_idx, task_data_from_full_list in enumerate(all_tasks_from_input):
        if task_data_from_full_list.get("Prompt") not in processed_prompts_set:
            tasks_for_processing_with_original_indices.append((original_idx, task_data_from_full_list))
    
    num_tasks_to_run = len(tasks_for_processing_with_original_indices)
    print(f"[INFO] Actual number of tasks to run in this session: {num_tasks_to_run}")

    print(f"[INFO] Running with parallelism: {args.parallel_num}")
    print(f"[INFO] Malicious agents: {'Enabled' if args.enable_malicious_agents else 'Disabled'}")
    print(f"[INFO] Defense agent: {'Enabled' if args.enable_defense_agent else 'Disabled'}")

    if args.parallel_num > 1 and num_tasks_to_run > 0:
        manager = Manager()
        file_lock = manager.Lock()
        worker_func = partial(process_task_and_write, common_args=args, overall_output_dir_base=args.output_dir_base, file_lock=file_lock)
        print(f"[INFO] Starting parallel processing of {num_tasks_to_run} tasks with {args.parallel_num} workers...")
        with Pool(processes=args.parallel_num) as pool:
            pool.map(worker_func, tasks_for_processing_with_original_indices)
        print("[INFO] Parallel processing finished.")
    elif num_tasks_to_run > 0: # Sequential processing for remaining tasks
        print("[INFO] Starting sequential processing for remaining tasks...")
        class DummyLock:
            def acquire(self): pass
            def release(self): pass
        dummy_lock = DummyLock()
        for task_tuple in tasks_for_processing_with_original_indices:
            process_task_and_write(task_tuple, common_args=args, overall_output_dir_base=args.output_dir_base, file_lock=dummy_lock)
        print("[INFO] Sequential processing finished.")
    else:
        print("[INFO] No tasks were scheduled to run in this session (either all done or input was empty after filtering).")

    print(f"\n[INFO] Task processing session complete. Results (if any new) appended to {args.output_file}")

if __name__ == "__main__":
    main() 