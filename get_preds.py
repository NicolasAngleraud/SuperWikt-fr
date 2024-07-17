import pandas as pd
import argparse
import dataEncoder as data
import lexicalClf as clf


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generating wiktionary_preds.tsv using input Wiktionary tsv file and input Wiktionary examples tsv file.")
	parser.add_argument('--input_wiktionary', required=True, help='Path to the input TSV file.')
	parser.add_argument('--input_examples', required=True, help='Path to the input TSV file.')
	parser.add_argument('--output', required=True, help='Path to the output TSV file.')
	parser.add_argument('--model_dir', required=True, help='Path to the output TSV file.')

	args = parser.parse_args()
	
	
	wiki_def_file = args.input_wiktionary
	wiki_example_file = args.input_examples
	wiki_pred_file = args.output
	
	wiki_encoder = data.wikiEncoder(def_datafile=wiki_def_file, ex_datafile=wiki_example_file, tokenizer=tokenizer, use_sample=False, sample_size=1000)
	
	coeff_ex = 0.68
	coeff_def = 0.80
	
	lex_clf = clf.lexicalClf_V1(params_def, params_ex, DEVICE, coeff_ex, coeff_def)
	lex_clf.load_clf(def_lem_clf_file, ex_clf_file)
	
	wiktionary_predictions = lex_clf.predict_wiki(wiki_encoder)
	
	wiki_df = pd.DataFrame(wiktionary_predictions)
	wiki_df.to_csv(wiki_pred_file, sep='\t', index=False)
