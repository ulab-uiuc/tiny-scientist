import os
import json
from tiny_scientist import TinyScientist
from datetime import datetime
from test_writer_mini import run_test_writer_mini # Correctly imports the text-generating function

def run_test_review_rewrite(input_text_content: str): # Parameter changed to input_text_content
    print("\n--- Test: ReviewRewriter ---")
    if not input_text_content: # Check if the text content is valid (e.g., not empty)
        print(f"[ERROR] Input text content is empty or invalid.")
        print("Please ensure test_writer_mini.py runs successfully first or provide valid text content.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = f"./output/test_review_rewrite_{timestamp}"
    os.makedirs(test_output_dir, exist_ok=True)

    # Initialize TinyScientist
    scientist = TinyScientist(
        model="gpt-4o", 
        output_dir=test_output_dir, 
        template="acl" 
    )

    print(f"[INFO] Using input text content (length: {len(input_text_content)} chars).")
    print(f"[INFO] Output artifacts from ReviewRewriter will be in: {test_output_dir}")

    try:
        report = scientist.review_and_rewrite(paper_text=input_text_content) # Pass the text content directly
        print("[SUCCESS] ReviewRewriter test completed.")
        print("Full Report:")
        # The report dictionary should contain the rewritten text, e.g., report["rewritten_paper_content"]
        print(json.dumps(report, indent=2)) 
        
        report_file_path = os.path.join(test_output_dir, "review_rewrite_report.json")
        with open(report_file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Full report also saved to: {report_file_path}")
        
        # Additionally, explicitly print the rewritten paper content if available
        rewritten_text = report.get("rewritten_paper_content", "")
        if rewritten_text:
            print("\n--- Rewritten Paper Text ---")
            print(rewritten_text)
            rewritten_file_path = os.path.join(test_output_dir, "rewritten_paper.txt")
            with open(rewritten_file_path, 'w', encoding='utf-8') as f:
                f.write(rewritten_text)
            print(f"[INFO] Rewritten paper text also saved to: {rewritten_file_path}")
        else:
            print("[INFO] No rewritten paper content found in the report.")

    except Exception as e:
        print(f"[ERROR] ReviewRewriter test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generated_text_content = run_test_writer_mini() # This now returns a string
    
    if generated_text_content:
        run_test_review_rewrite(generated_text_content) # Pass the string content
    else:
        print("[FAIL] Could not run ReviewRewriter test because WriterMini test failed to produce text content.") 