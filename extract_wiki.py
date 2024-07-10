import pandas as pd
import argparse


def extract_wiktionary(input_file, output_file):
	with open(input_file, 'r', encoding='utf-8') as file:
		for i, line in enumerate(file):
		    print(line.strip())
		    if i >= 76: break


def main():
    parser = argparse.ArgumentParser(description="Extract wiktionary data from a TTL dump file.")
    parser.add_argument('--input', required=True, help='Path to the input TTL dump file.')
    parser.add_argument('--output', required=True, help='Path to the output TSV file.')

    args = parser.parse_args()

    extract_wiktionary(args.input, args.output)

if __name__ == "__main__":
    main()
