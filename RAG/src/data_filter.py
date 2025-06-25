import json


def data_filter(input_file, output_file, attribute, value):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if data.get(attribute) == value:
                    outfile.write(line)
            except json.JSONDecodeError:
                # Handle cases where a line is not valid JSON
                print(f"Skipping invalid JSON line: {line.strip()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter JSON lines based on an attribute value.")
    parser.add_argument("input_file", type=str, help="Input file containing JSON lines")
    parser.add_argument("output_file", type=str, help="Output file to write filtered JSON lines")
    parser.add_argument("attribute", type=str, help="Attribute to filter on")
    parser.add_argument("value", type=str, help="Value of the attribute to filter by")

    args = parser.parse_args()

    data_filter(args.input_file, args.output_file, args.attribute, args.value)
