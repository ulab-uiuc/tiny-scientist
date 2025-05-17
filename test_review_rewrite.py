import os
import json
from tiny_scientist import TinyScientist
from datetime import datetime
from test_writer_mini import run_test_writer_mini # Import the function from the first test script

def run_test_review_rewrite(input_pdf_path: str):
    print("\n--- Test: ReviewRewriter ---")
    if not input_pdf_path or not os.path.exists(input_pdf_path):
        print(f"[ERROR] Input PDF path is invalid or file does not exist: {input_pdf_path}")
        print("Please ensure test_writer_mini.py runs successfully first or provide a valid PDF path.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # It's good practice for ReviewRewriter to have its own output directory for its artifacts
    # separate from where the input PDF might be located.
    test_output_dir = f"./output/test_review_rewrite_{timestamp}"
    os.makedirs(test_output_dir, exist_ok=True)

    # Initialize TinyScientist
    # The output_dir for TinyScientist here will be used by ReviewRewriter for its logs/temp files if any.
    scientist = TinyScientist(
        model="gpt-4o", # Or your preferred model
        output_dir=test_output_dir, 
        template="acl" # Template might be used by parts of ReviewRewriter (e.g., if it formats output)
    )

    print(f"[INFO] Using input PDF: {input_pdf_path}")
    print(f"[INFO] Output artifacts from ReviewRewriter will be in: {test_output_dir}")

    try:
        report = scientist.review_and_rewrite(pdf_path=input_pdf_path)
        print("[SUCCESS] ReviewRewriter test completed.")
        print("Full Report:")
        print(json.dumps(report, indent=2)) # Pretty print the JSON report
        
        # Save the report to a file for inspection
        report_file_path = os.path.join(test_output_dir, "review_rewrite_report.json")
        with open(report_file_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Full report also saved to: {report_file_path}")

    except Exception as e:
        print(f"[ERROR] ReviewRewriter test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # First, run the WriterMini test to generate a PDF
    # This assumes run_test_writer_mini() from the other script will create its own output dir for the PDF
    generated_pdf_path = run_test_writer_mini()
    
    if generated_pdf_path:
        # Then, run the ReviewRewrite test using the generated PDF
        run_test_review_rewrite(generated_pdf_path)
    else:
        print("[FAIL] Could not run ReviewRewriter test because WriterMini test failed to produce a PDF.") 