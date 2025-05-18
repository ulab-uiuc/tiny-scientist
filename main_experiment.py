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
    parser = argparse.ArgumentParser(description="Run TinyScientist experiment with parallel processing.")
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

    # Critical: For append mode with multiple processes, ensure the output file is created (or truncated) once at the beginning.
    # Otherwise, each process appending might be to a non-existent file if it's the first, or to old content.
    with open(args.output_file, 'w', encoding='utf-8') as f:
        pass # Create/Truncate the file
    print(f"[INFO] Output file {args.output_file} created/truncated for appending.")

    os.makedirs(args.output_dir_base, exist_ok=True)

    try:
        with open(args.input_file, 'r') as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            print(f"[ERROR] Input file {args.input_file} should be a JSON list of tasks.")
            return
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON: {args.input_file}")
        return

    print(f"[INFO] Loaded {len(tasks)} tasks from {args.input_file}")
    print(f"[INFO] Running with parallelism: {args.parallel_num}")
    print(f"[INFO] Malicious agents: {'Enabled' if args.enable_malicious_agents else 'Disabled'}")
    print(f"[INFO] Defense agent: {'Enabled' if args.enable_defense_agent else 'Disabled'}")

    tasks_with_indices = list(enumerate(tasks))
    num_tasks = len(tasks_with_indices)

    if num_tasks == 0:
        print("[INFO] No tasks to process.")
        print(f"\n[INFO] All tasks processed (0 tasks).")
        return

    if args.parallel_num > 1:
        manager = Manager() # Manager is needed for Lock if not using default Lock
        file_lock = manager.Lock() # Create a lock to synchronize file access
        
        worker_func = partial(process_task_and_write, common_args=args, overall_output_dir_base=args.output_dir_base, file_lock=file_lock)
        
        print(f"[INFO] Starting parallel processing of {num_tasks} tasks with {args.parallel_num} workers...")
        with Pool(processes=args.parallel_num) as pool:
            # pool.map will collect all return values, though we don't strictly need them now
            # as writing happens in the worker. We could use apply_async if we don't care about return values.
            # However, map is fine and will also propagate exceptions if any worker fails catastrophically.
            results_statuses = pool.map(worker_func, tasks_with_indices)
        
        print("[INFO] Parallel processing finished.")
        # You can optionally inspect results_statuses here to count successes/failures if needed
        # success_count = sum(1 for status in results_statuses if status == "success")
        # print(f"[INFO] {success_count}/{num_tasks} tasks reported success.")

    else: # Sequential processing
        print("[INFO] Starting sequential processing...")
        # For sequential, no actual lock is needed, but we can pass a dummy lock-like object or None
        # For simplicity, we will just call the worker directly. It will open/close file each time.
        # The file was already truncated at the start of main().
        # Create a dummy lock object for sequential execution that does nothing.
        class DummyLock:
            def acquire(self): pass
            def release(self): pass
        
        dummy_lock = DummyLock()
        
        for task_tuple in tasks_with_indices:
            process_task_and_write(task_tuple, common_args=args, overall_output_dir_base=args.output_dir_base, file_lock=dummy_lock)
        print("[INFO] Sequential processing finished.")

    print(f"\n[INFO] All tasks processed. Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 