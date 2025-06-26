import os
import json
from tqdm import tqdm
import argparse
import random

# You might need to install BeautifulSoup for HTML parsing: pip install beautifulsoup4
from bs4 import BeautifulSoup

random.seed(13370)  # for reproducibility

def convert_crag_to_target_format(crag_instance):
    """
    Converts a single CRAG data instance to the target format required by the project.
    """
    target_instance = {}
    target_instance["dataset"] = "crag"
    target_instance["question_id"] = crag_instance.get("interaction_id", "")
    target_instance["question_text"] = crag_instance.get("query", "")

    # Create answers_objects from the answer field
    answer = crag_instance.get("answer", "")
    answers_object = {
        "number": "",
        "date": {"day": "", "month": "", "year": ""},
        "spans": [answer],
    }
    target_instance["answers_objects"] = [answers_object]

    # Create contexts from search_results
    target_instance["contexts"] = []
    search_results = crag_instance.get("search_results", [])
    for i, search_result in enumerate(search_results):
        # Extract plain text from HTML. If BeautifulSoup is not available, it will use the raw HTML.
        html_content = search_result.get("page_result", "")
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            paragraph_text = soup.get_text()
        except Exception:
            paragraph_text = html_content

        # This is a simple heuristic to determine if a context is supporting.
        # It checks if the answer appears in the page's text.
        # You might want to develop a more sophisticated logic for your use case.
        is_supporting = answer.lower() in paragraph_text.lower() if answer else False

        context = {
            "idx": i,
            "title": search_result.get("page_name", "No Title").strip(),
            "paragraph_text": paragraph_text.strip(),
            "is_supporting": is_supporting,
        }
        target_instance["contexts"].append(context)

    return target_instance

def process_crag_file(input_path, output_path):
    """
    Reads a CRAG .jsonl file, converts each line, and writes to an output .jsonl file.
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in tqdm(infile, desc="Processing CRAG data"):
            if not line.strip():
                continue
            crag_data = json.loads(line)
            converted_data = convert_crag_to_target_format(crag_data)
            outfile.write(json.dumps(converted_data) + '\n')

def split_crag_file(input_path, output_dir, dev_ratio=0.8):
    """
    Splits the single CRAG JSONL file into dev and test sets.
    """
    print(f"Splitting {input_path} into dev and test sets...")
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Filter out empty lines
    lines = [line for line in lines if line.strip()]
    
    random.shuffle(lines)

    split_index = int(len(lines) * dev_ratio)
    dev_lines = lines[:split_index]
    test_lines = lines[split_index:]

    os.makedirs(output_dir, exist_ok=True)
    dev_path = os.path.join(output_dir, 'dev.jsonl')
    test_path = os.path.join(output_dir, 'test.jsonl')

    with open(dev_path, 'w') as f:
        f.writelines(dev_lines)
    print(f"Wrote {len(dev_lines)} lines to {dev_path}")

    with open(test_path, 'w') as f:
        f.writelines(test_lines)
    print(f"Wrote {len(test_lines)} lines to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and convert the CRAG dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands: split, convert")

    # Subparser for the 'split' command
    parser_split = subparsers.add_parser('split', help='Split the original CRAG data file into dev and test sets.')
    parser_split.add_argument("--input_file", type=str, required=True, help="Path to the original, unsplit crag_data.jsonl file.")
    parser_split.add_argument("--output_dir", type=str, default="raw_data/crag", help="Directory to save the split dev.jsonl and test.jsonl files.")
    parser_split.add_argument("--dev_ratio", type=float, default=0.8, help="Ratio of the data to use for the dev set (the rest will be for test).")

    # Subparser for the 'convert' command
    parser_convert = subparsers.add_parser('convert', help="Convert a raw split file to the project's processed format.")
    parser_convert.add_argument("set_name", type=str, choices=['dev', 'test'], help="The data split to process (dev or test).")
    parser_convert.add_argument("--input_dir", type=str, default="raw_data/crag", help="Directory containing the raw CRAG data files.")
    parser_convert.add_argument("--output_dir", type=str, default="processed_data/crag", help="Directory to save the processed data files.")

    args = parser.parse_args()

    if args.command == "split":
        if not os.path.exists(args.input_file):
             print(f"Error: Input file for splitting not found at {args.input_file}")
        else:
            split_crag_file(args.input_file, args.output_dir, args.dev_ratio)
            print("Splitting complete.")

    elif args.command == "convert":
        input_file = os.path.join(args.input_dir, f"{args.set_name}.jsonl")
        output_dir = args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.set_name}.jsonl")

        if not os.path.exists(input_file):
            print(f"Error: Input file for conversion not found at {input_file}")
            print("Please run the 'split' command first, e.g.:")
            print("python process_crag.py split --input_file <path_to_your_crag_data.jsonl>")
        else:
            process_crag_file(input_file, output_file)
            print("Conversion complete.")

