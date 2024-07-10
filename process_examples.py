import spacy
import pandas as pd
import argparse
from collections import Counter, defaultdict
from random import shuffle
import numpy as np
from matplotlib import pyplot as plt


# compound words link token
sp_sym = '##'


def data_analysis(input_file):
	df_senses = pd.read_csv(input_file, sep='\t')
	
	examples = []
	for i in range(1, 24):
		examples.append(df_senses[(df_senses[f'example_{i}'] != "") & (df_senses[f'example_{i}'].notna())][f'example_{i}'].tolist())
	examples_ = [item for sublist in examples for item in sublist]
	
	lemmas = []
	for i in range(1, 24):
		lemmas.append(df_senses[(df_senses[f'example_{i}'] != "") & (df_senses[f'example_{i}'].notna())]['lemma'].tolist())
	lemmas_ = [item for sublist in lemmas for item in sublist]
	
	sense_ids = []
	for i in range(1, 24):
		sense_ids.append(df_senses[(df_senses[f'example_{i}'] != "") & (df_senses[f'example_{i}'].notna())]['sense_id'].tolist())
	sense_ids_ = [item for sublist in sense_ids for item in sublist]

	nums = []
	for i in range(1, 24):
		nums.append([f"{i}"]*len(sense_ids[i-1]))
	nums_ = [item for sublist in nums for item in sublist]

	print('number of examples:', len(examples_))

	return examples_, lemmas_, sense_ids_, nums_



def compound_lemma(lemma, sentence, special_token=sp_sym):
	lemma = str(lemma)
	# Split the lemma into individual words
	lemma_words = lemma.split()

	# Join the lemma words with the special token
	compound_lemma = special_token.join(lemma_words).replace("'", f"'{sp_sym}")

	# Replace spaces in the sentence with the compound lemma
	result_sentence = sentence.replace(' '.join(lemma_words), compound_lemma)

	return result_sentence

def lemmatize_spacy(text, lemma):
	lemma = str(lemma)
	doc = nlp(str(text))
	lemmatized_words = [token.lemma_ for token in doc]
	lemmatized_text = ' '.join(lemmatized_words).replace(' - ', '-')
	for punc in [',', ';', ':', '.', '!', '?']: 
		lemmatized_text = lemmatized_text.replace(f'{punc}-', f'{punc} -')
		lemmatized_text = lemmatized_text.replace(f'{punc}–', f'{punc} –')

	if '_' in lemma or ' ' in lemma:
		if lemma.replace('_', ' ').replace("'", "' ") in lemmatized_text:
			lemmatized_text = compound_lemma(lemma.replace('_', ' '), lemmatized_text, sp_sym)
			
	lemmatized_words = lemmatized_text.split()
	# lemmatized_words = [tok.replace(sp_sym, ' ') for tok in lemmatized_words]
	return lemmatized_words

def tokenize_spacy(text, lemma):
	lemma = str(lemma)
	doc = nlp(str(text))
	tokens = [token.text for token in doc]
	tokenized_text = ' '.join(tokens).replace(' - ', '-')
	for punc in [',', ';', ':', '.', '!', '?']: 
		tokenized_text = tokenized_text.replace(f'{punc}-', f'{punc} -')
		tokenized_text = tokenized_text.replace(f'{punc}–', f'{punc} –')
		
	if '_' in lemma or ' ' in lemma:
		if lemma.replace('_', ' ').replace("'", "' ") in tokenized_text:
			tokenized_text = compound_lemma(lemma.replace('_', ' '), tokenized_text, sp_sym)
			
	tokens = tokenized_text.split()
	# tokens = [tok.replace(sp_sym, ' ') for tok in tokens]
	return tokens 


def examples_iterator(*example_lists):
	for examples in example_lists:
	        for example in examples:
	        	yield example
            
            

def find_rank(lemma, tokenized_words):
	lemma = str(lemma)
	for i, word in enumerate(tokenized_words):
		if (word == lemma) or (word.lower() == lemma.lower()): return i
	return -1

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Process wiktionary examples from a TSV file and outputs the resultings processed examples in a TSV file.")
	parser.add_argument('--input', required=True, help='Path to the input TSV file.')
	parser.add_argument('--output', required=True, help='Path to the output TSV file.')

	args = parser.parse_args()
	
	
	print('WIKTIONARY EXAMPLES EXTRACTION...')	
	examples_, lemmas_, sense_ids_, nums_ = data_analysis(args.input)
	print('WIKTIONARY EXAMPLES EXTRACTED.')
	print()
	
	example_iter = examples_iterator(
		examples_
	)

	lemma_iter = examples_iterator(
		lemmas_
	)

	sense_id_iter = examples_iterator(
		sense_ids_
	)

	num_iter = examples_iterator(
		nums_
	)



	print("LOADING SPACY fr_core_news_lg...")
	nlp = spacy.load("fr_core_news_lg")
	print("LOADED SPACY fr_core_news_lg.")
	print()

	lemmas_not_found = 0

	examples_data = []



	print("FINDING TARGET WORD RANK IN EXAMPLES...")

	for example, lemma, sense_id, num in zip(example_iter, lemma_iter, sense_id_iter, num_iter):
		
		result_spacy = tokenize_spacy(example, lemma)
		result_spacy_lower = [tok.lower() for tok in result_spacy]
		
		example_data = {'sense_id': sense_id, 'lemma': lemma, 'num_ex': num, 'word_rank': -1, 'example': ' '.join(result_spacy)}
		
		if lemma.replace('_', sp_sym).replace("'", f"'{sp_sym}") in result_spacy or lemma.replace('_', sp_sym).replace("'", f"'{sp_sym}").lower() in result_spacy_lower: example_data['word_rank'] = find_rank(lemma.replace('_', sp_sym).replace("'", f"'{sp_sym}"), result_spacy)
		else:
			lemmatized_example = lemmatize_spacy(example, lemma)
			if lemma.replace('_', sp_sym).replace("'", f"'{sp_sym}") in lemmatized_example: example_data['word_rank'] = find_rank(lemma.replace('_', sp_sym).replace("'", f"'{sp_sym}"), lemmatized_example)
			else: lemmas_not_found += 1
			
		examples_data.append(example_data)

	print("FINISHED FINDING TARGET WORD RANK IN EXAMPLES.")
	print("Number of examples where the target word was not found ", lemmas_not_found)

	examples_data_df = pd.DataFrame(examples_data)
	examples_data_df.to_csv(args.output, sep='\t', index=False)
