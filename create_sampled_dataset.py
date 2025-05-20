import os
import json
import random
import shutil

def create_sampled_dataset(input_dir="data/ScienceSafetyData/Dataset", output_dir_suffix="_20", num_samples=20):
    """
    Reads JSON files from input_dir, randomly samples a specified number of items
    from each, and writes them to a new directory.

    Args:
        input_dir (str): Path to the directory containing original JSON files.
        output_dir_suffix (str): Suffix to append to the input_dir name to create the output directory name.
                                 For example, if input_dir is "Dataset" and suffix is "_20", output_dir will be "Dataset_20".
        num_samples (int): Number of items to randomly select from each JSON file.
    """
    
    # Construct the output directory path
    # It should be at the same level as input_dir, e.g., data/ScienceSafetyData/Dataset_20
    parent_dir = os.path.dirname(input_dir)
    base_input_dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(parent_dir, base_input_dir_name + output_dir_suffix)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
        # Optional: Clear the directory if it exists and you want to start fresh
        # for item in os.listdir(output_dir):
        #     item_path = os.path.join(output_dir, item)
        #     if os.path.isfile(item_path) or os.path.islink(item_path):
        #         os.unlink(item_path)
        #     elif os.path.isdir(item_path):
        #         shutil.rmtree(item_path)
        # print(f"Cleared existing output directory: {output_dir}")


    print(f"Processing files from: {input_dir}")
    # Iterate over files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            print(f"Processing file: {filename}...")

            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    print(f"  Warning: Content of {filename} is not a list. Skipping.")
                    continue

                if not data:
                    print(f"  Warning: {filename} is empty. Creating an empty list in output.")
                    sampled_data = []
                elif len(data) <= num_samples:
                    sampled_data = data  # Take all if less than or equal to num_samples
                    print(f"  Took all {len(data)} items (as it's <= {num_samples}).")
                else:
                    sampled_data = random.sample(data, num_samples)
                    print(f"  Randomly selected {num_samples} items.")

                # Write to the new file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(sampled_data, f, indent=2, ensure_ascii=False)
                print(f"  Saved sampled data to: {output_file_path}")

            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from {filename}. Skipping.")
            except Exception as e:
                print(f"  An unexpected error occurred with {filename}: {e}. Skipping.")
    print("\nAll files processed.")

if __name__ == "__main__":
    # Workspace root is /Users/zhukunlun/Documents/GitHub/tiny-scientist
    # The script will be in this root.
    source_directory = "data/ScienceSafetyData/Dataset"
    
    # The output directory will be data/ScienceSafetyData/Dataset_20
    create_sampled_dataset(input_dir=source_directory, output_dir_suffix="_20", num_samples=20)
    
    print("\nSampling script finished.") 