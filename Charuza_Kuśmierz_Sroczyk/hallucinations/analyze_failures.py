import json
from typing import List, Dict, Any

def analyze_failures(file_paths: List[str]):
    """
    Analyzes multiple evaluation JSON files to find questions that were
    consistently answered incorrectly across all runs.
    """
    # {sample_id: {"prompt": str, "expected": str, "correct_in_any_file": bool}}
    questions: Dict[str, Dict[str, Any]] = {}

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON, skipping: {file_path}")
            continue

        results = data.get("results", [])
        print(f"Processing {len(results)} results from {file_path}...")

        for item in results:
            sample_id = item.get("sample_id")
            if not sample_id:
                continue

            # Initialize question info if first time seeing it
            if sample_id not in questions:
                questions[sample_id] = {
                    "prompt": item.get("prompt", ""),
                    "expected_response": item.get("expected_response", ""),
                    "correct_in_any_file": False
                }

            # If score is true, mark it as correctly answered in at least one file
            if item.get("score") is True:
                questions[sample_id]["correct_in_any_file"] = True

    # Filter for questions that were never answered correctly
    failed_questions = {
        sid: info for sid, info in questions.items() if not info["correct_in_any_file"]
    }

    print(f"\nFound {len(failed_questions)} questions that failed in all {len(file_paths)} files.")
    
    return failed_questions


def main():
    """Main function to run analysis on a predefined list of files."""
    # --- Configuration ---
    # Specify the paths to the evaluation JSON files you want to analyze.
    EVALUATION_FILES = [
        "/home/dawid/repos/NLP_2025W/Charuza_Kuśmierz_Sroczyk/hallucinations/gpt5-wikipedia_dataset_evaluation.json",
        "/home/dawid/repos/NLP_2025W/Charuza_Kuśmierz_Sroczyk/hallucinations/gpt4_mini_wikipedia_dataset_evaluation.json",
    ]
    OUTPUT_FILE = "/home/dawid/repos/NLP_2025W/Charuza_Kuśmierz_Sroczyk/hallucinations/common_failures.json"
    # ---------------------

    failed_questions = analyze_failures(EVALUATION_FILES)

    # Count failures by category
    category_counts = {}
    for sample_id in failed_questions.keys():
        parts = sample_id.split('-')
        category = "factual"  # Default for format like 'wiki-123'
        if len(parts) > 2:
            category = parts[1]
        
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\nFailure counts by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")
    print("-" * 30)

    # Format results for JSON output
    output_results = []
    for sample_id, info in failed_questions.items():
        output_results.append({
            "sample_id": sample_id,
            "prompt": info["prompt"],
            "expected_response": info["expected_response"]
        })

    # Save to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"results": output_results}, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output_results)} common failures to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
