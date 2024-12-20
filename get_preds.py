import pandas as pd
import argparse
import torch
import dataEncoder as data
import lexicalClf as clf
from transformers import AutoModel, AutoTokenizer, AutoConfig



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Predicts the most likely supersense and each supersense scores for each sense of the input Wiktionary resource.")
	parser.add_argument('--input_wiktionary', required=True, help='Path to the input TSV file containing Wiktionary sense data.')
	parser.add_argument('--input_examples', required=True, help='Path to the input TSV file containing Wiktionary example data for each sense.')
	parser.add_argument('--output', required=True, help='Path to the output folder to save produced files.')
	parser.add_argument('--model_dir', required=True, help='Path to the folder where the saved parameters of the trained classifiers are stored.')
	parser.add_argument('--device_id', required=True, help='ID of the GPU or CPU used for the computation of the models calculations.')

	args = parser.parse_args()
	
	device_id = args.device_id
	if device_id != 'cpu':
		if torch.cuda.is_available():
			DEVICE = torch.device("cuda:" + args.device_id)
	else:
		DEVICE = 'cpu'
	
	params = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.000005,
	"weight_decay": 0.001,
	"frozen": False,
	"max_seq_length": 100
	}
	
	MODEL_NAME = "flaubert/flaubert_large_cased"
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	def_lem_clf_file = args.model_dir + "/def_lem_clf.params"
	ex_clf_file = args.model_dir + "/ex_clf.params"
	
	wiki_def_file = args.input_wiktionary
	wiki_example_file = args.input_examples
	wiki_pred_file = args.output
	
	wiki_encoder = data.wikiEncoder(def_datafile=wiki_def_file, ex_datafile=wiki_example_file, tokenizer=tokenizer, use_sample=False, sample_size=1000)
	
	coeff_ex = 0.65
	coeff_def = 0.80
	
	lex_clf = clf.lexicalClf_V1(params, params, DEVICE, coeff_ex, coeff_def)
	lex_clf.load_clf(def_lem_clf_file, ex_clf_file)
	
	wiktionary_predictions = lex_clf.predict_wiki(wiki_encoder)
	
	wiki_df = pd.DataFrame(wiktionary_predictions)
	wiki_df.to_csv(wiki_pred_file, sep='\t', index=False, encoding='utf-8')
