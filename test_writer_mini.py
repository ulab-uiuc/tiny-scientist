import os
import json
from tiny_scientist import TinyScientist
from datetime import datetime

def run_test_writer_mini():
    print("--- Test: WriterMini ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = f"./output/test_writer_mini_{timestamp}"
    os.makedirs(test_output_dir, exist_ok=True)

    # Initialize TinyScientist
    scientist = TinyScientist(
        model="gpt-4o", # Or your preferred model
        output_dir=test_output_dir,
        template="acl" # Or your preferred template
    )

    # Example Idea (modify as needed)
    example_idea = {
        "Title": "A Conceptual Framework for AI-Driven Music Generation with Emotional Control",
        "Problem": "Existing AI music generation models lack fine-grained control over the emotional expression of the generated music, making it difficult to create pieces that evoke specific feelings.",
        "Importance": "Music has a profound impact on human emotions. Providing tools for precise emotional control in AI-generated music would benefit artists, composers, game developers, and content creators.",
        "Difficulty": "Quantifying and modeling emotional expression in music is complex. Mapping high-level emotional concepts to low-level musical features (harmony, melody, rhythm, timbre) in a controllable way is a significant challenge.",
        "NoveltyComparison": "Unlike previous models that focus on style imitation or general composition, this framework proposes a novel hierarchical approach incorporating affective computing principles and user-in-the-loop feedback for emotional fine-tuning.",
        "Experiment": "The conceptual experiment involves designing a system with three main components: 1) An emotion-to-feature mapping module trained on a large dataset of music annotated with emotions. 2) A generative model (e.g., a Transformer or GAN) conditioned on these musical features. 3) An interactive interface allowing users to specify desired emotional trajectories and iteratively refine the generated music.",
        # Add other fields if your idea structure is different or if WriterMini expects them
    }

    print(f"[INFO] Using example idea: {example_idea.get('Title')}")
    print(f"[INFO] Output will be in: {test_output_dir}")

    try:
        pdf_path = scientist.write_mini(idea=example_idea)
        print(f"[SUCCESS] WriterMini test completed. PDF generated at: {pdf_path}")
        return pdf_path # Return for the next test
    except Exception as e:
        print(f"[ERROR] WriterMini test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_test_writer_mini() 