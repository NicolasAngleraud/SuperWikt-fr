import argparse
import torch
import lexical_classifier as lclf
import pandas as pd


SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon", "act*cognition"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person", "groupxperson"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance", "artifact*cognition"],
               "informational_object": ["cognition", "communication", "act*cognition", "artifact*cognition"],
               "quantification": ["quantity", "part", "group", "groupxperson"],
               "other": ["institution", "possession", "time"]
               }
               
LPARAMETERS = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 128,
	"patience": 2,
	"lr": 0.001,
	"frozen": True,
	"dropout": 0.1,
	"max_seq_length": 100
}

params_keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "dropout", "max_seq_length"]


def parse_clf_name(clf_name):
	params = {}
	run = clf_name.split('-')[-1]
	clf_name = ''.join(clf_name.split('-')[:-1])
	str_params = {el.split('=')[0] : el.split('=')[1] for el in clf_name.strip(".params").split(';')}
	str_params['run'] = run
	for param in str_params:
		if param == 'lr':
			continue
	
		if str_params[param] in ['True', 'False']: 
			if str_params[param] == 'True':
				params[param] = True
			else:
				params[param] = False
				
		elif any(char.isdigit() for char in str_params[param]):
			if str_params[param].isdigit():
				params[param] = int(str_params[param])
			else:
				params[param] = float(str_params[param])
				
		else:
			params[param] = str_params[param]
			
	return params


def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-mode", choices=['train', 'evaluate'], help="Sets the mode of the program, either train classifiers or evaluate trained models.")
	parser.add_argument("-lexical_data_file", default="./donnees_stage_wiktionnaire_supersenses.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-batch_size", choices=['8', '16', '32', '64'], help="batch_size for the classifier.")
	parser.add_argument("-nb_runs", choices=['1', '2', '3', '4', '5'], default='1', help="number of runs for each classifier.")
	parser.add_argument("-dropout", choices=['0.1', '0.3'], help="dropout rate for the classifier.")
	parser.add_argument("-trained_model_name", help="name of the trained classifier to load and evaluate.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()

	if args.mode == 'train':
		
		df_dev = []

		# DEVICE setup
		device_id = args.device_id
		if torch.cuda.is_available():
			DEVICE = torch.device("cuda:" + args.device_id)
			

		nb_runs = int(args.nb_runs)
		patience = 2
		batch_size = args.batch_size
		frozen = False
		lrs = [0.00001, 0.000001] #, 0.000005, 0.000001, 0.0000005, 0.0000001]
		hidden_layer_sizes = [256]
		dropout = float(args.dropout)

		for i in range(nb_runs):
			train_examples, freq_dev_examples, rand_dev_examples = lclf.encoded_examples(datafile=args.lexical_data_file)
			
			for lr in lrs:

				for hidden_layer_size in hidden_layer_sizes:
		    
					dev_data = {}

					print()
					print(f"run {i+1} : lr = {lr}")
					print(f"dropout :  {dropout} ; hidden layer size : {hidden_layer_size}; batch_size : {batch_size}")
					print()


					params = {key: value for key, value in LPARAMETERS.items()}
					params['lr'] = lr
					params['patience'] = patience
					params['frozen'] = frozen
					params['dropout'] = dropout
					params['hidden_layer_size'] = hidden_layer_size
					
					classifier_name = ';'.join([f'{key}={params[key]}' for key in params_keys]).strip(';')
					
					dev_data['clf_name'] = f'{classifier_name}-{i+1}'
					dev_data["run"] = i + 1

					classifier = lclf.SupersenseTagger(params, DEVICE)
					lclf.training(params, train_examples, freq_dev_examples, rand_dev_examples, classifier, DEVICE, dev_data)
					lclf.evaluation(freq_dev_examples, classifier, params, DEVICE, f"freq-dev", dev_data)
					lclf.evaluation(rand_dev_examples, classifier, params, DEVICE, f"rand-dev", dev_data)

					print(f"CLASSIFIER TRAINED ON {len(train_examples)} EXAMPLES.")

					sequoia_baseline = lclf.MostFrequentSequoia()
					train_baseline = lclf.MostFrequentTrainingData()
					wiki_baseline = lclf.MostFrequentWiktionary()

					sequoia_baseline.training()
					train_baseline.training()
					wiki_baseline.training()

					dev_data["freq_dev_sequoia_baseline"] = sequoia_baseline.evaluation(freq_dev_examples)
					dev_data["freq_dev_train_baseline"] =train_baseline.evaluation(freq_dev_examples)
					dev_data["freq_dev_wiki_baseline"] = wiki_baseline.evaluation(freq_dev_examples)

					dev_data["rand_dev_sequoia_baseline"] = sequoia_baseline.evaluation(rand_dev_examples)
					dev_data["rand_dev_train_baseline"] =train_baseline.evaluation(rand_dev_examples)
					dev_data["rand_dev_wiki_baseline"] = wiki_baseline.evaluation(rand_dev_examples)

					print("BASELINES COMPUTED.")

					df_dev.append(dev_data)

		print("CREATION OF THE EVALUATION FILE...")
		df = pd.DataFrame(df_dev)
		excel_filename = f'./lexical_classifier_results.xlsx'
		df.to_excel(excel_filename, index=False)
		
		print("PROCESS DONE.")

	elif args.mode == 'evaluate':
	
		device_id = args.device_id
		if torch.cuda.is_available():
			DEVICE = torch.device("cuda:" + args.device_id)
		
		eval_data = {}
		clf_name = args.trained_model_name
		params = {'batch_size': args.batch_size, 'frozen': True, 'hidden_layer_size': 256, 'dropout': 0.1}
		# params = parse_clf_name(clf_name)
		
		# for param in params: eval_data[param] = params[param]
		
		loaded_model = lclf.SupersenseTagger(params, DEVICE)
		loaded_model.load_state_dict(torch.load(f'./lexical_classifiers/{clf_name}'))
		train_examples, freq_dev_examples, rand_dev_examples = lclf.encoded_examples(datafile=args.lexical_data_file)
		# loaded_model.deep_analysis(train_examples, freq_dev_examples, rand_dev_examples, eval_data)
		
		lclf.evaluation(freq_dev_examples, loaded_model, params, DEVICE, f"freq-dev", eval_data)
		lclf.evaluation(rand_dev_examples, loaded_model, params, DEVICE, f"rand-dev", eval_data)
		
		sequoia_baseline = lclf.MostFrequentSequoia()
		train_baseline = lclf.MostFrequentTrainingData()
		wiki_baseline = lclf.MostFrequentWiktionary()

		sequoia_baseline.training()
		train_baseline.training()
		wiki_baseline.training()

		eval_data["freq_dev_sequoia_baseline"] = sequoia_baseline.evaluation(freq_dev_examples)
		eval_data["freq_dev_train_baseline"] =train_baseline.evaluation(freq_dev_examples)
		eval_data["freq_dev_wiki_baseline"] = wiki_baseline.evaluation(freq_dev_examples)

		eval_data["rand_dev_sequoia_baseline"] = sequoia_baseline.evaluation(rand_dev_examples)
		eval_data["rand_dev_train_baseline"] =train_baseline.evaluation(rand_dev_examples)
		eval_data["rand_dev_wiki_baseline"] = wiki_baseline.evaluation(rand_dev_examples)
		
		df_eval = pd.DataFrame(list(eval_data))
		# df.to_excel(f"./{clf_name.strip('.params')}.xlsx", index=False)
		df.to_excel("./test_eval_pretrained_model.xlsx", index=False)
		
		
