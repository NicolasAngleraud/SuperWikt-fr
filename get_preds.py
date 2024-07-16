import pandas as pd


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generating wiktionary_preds.tsv using input Wiktionary tsv file and input Wiktionary examples tsv file.")
	parser.add_argument('--input_wiktionary', required=True, help='Path to the input TSV file.')
	parser.add_argument('--input_examples', required=True, help='Path to the input TSV file.')
	parser.add_argument('--output', required=True, help='Path to the output TSV file.')
	parser.add_argument('--model_dir', required=True, help='Path to the output TSV file.')

	args = parser.parse_args()
