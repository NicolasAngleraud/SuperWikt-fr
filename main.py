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
	"nb_epochs": 25,
	"batch_size": 32,
	"hidden_layer_size": 128,
	"patience": 2,
	"lr": 0.001,
	"frozen": True,
	"dropout": 0.2,
	"max_seq_length": 100
}


def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-lexical_eval_type", choices=['rand', 'freq', 'train'], help="The dataset on which the lexical classifier will be evaluated.")
	parser.add_argument("-lexical_data_file", default="./donnees_stage_wiktionnaire_supersenses.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-inference_data_file", default=None, help="File containing the data for inference.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = get_parser_args()

	df_dev = []
	df_test = []

	# DEVICE setup
	device_id = args.device_id
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda:" + args.device_id)
		
	eval_prefix = args.lexical_eval_type

	nb_runs = 1
	patiences = [2]
	frozen = False
	lrs = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005]
	hidden_layer_sizes = [128, 256, 512]
	dropouts = [0.2, 0.5]

	for i in range(nb_runs):
		train_examples, dev_examples, test_examples = lclf.encoded_examples(datafile=args.lexical_data_file, eval_prefix=eval_prefix)
		
		
		for lr in lrs:
			for patience in patiences:
				for dropout in dropouts:
					for hidden_layer_size in hidden_layer_sizes:
			    
						dev_data = {}
						test_data = {}

						print("")
						print(f"run {i+1} : lr = {lr}")
						print(f"dropout :  {dropout} ; hidden layer size : {hidden_layer_size}")
						print(f"eval set : {eval_prefix}")
						print("")


						params = {key: value for key, value in LPARAMETERS.items()}
						params['lr'] = lr
						params['patience'] = patience
						params['frozen'] = frozen
						params['dropout'] = dropout
						params['hidden_layer_size'] = hidden_layer_size

						dev_data["run"] = i + 1
						test_data["run"] = i + 1

						classifier = lclf.SupersenseTagger(params, DEVICE)
						lclf.training(params, train_examples, dev_examples, classifier, DEVICE, dev_data, test_data)
						lclf.evaluation(dev_examples, classifier, params, DEVICE,  i+1, f"{eval_prefix}-dev", dev_data)
						lclf.evaluation(test_examples, classifier, params, DEVICE, i+1, f"{eval_prefix}-test", test_data)

						print(f"CLASSIFIER TRAINED ON {len(train_examples)} EXAMPLES.")

						sequoia_baseline = lclf.MostFrequentSequoia()
						train_baseline = lclf.MostFrequentTrainingData()
						wiki_baseline = lclf.MostFrequentWiktionary()

						sequoia_baseline.training()
						train_baseline.training()
						wiki_baseline.training()

						dev_data["sequoia_baseline"] = sequoia_baseline.evaluation(dev_examples)
						test_data["sequoia_baseline"] = sequoia_baseline.evaluation(test_examples)

						dev_data["train_baseline"] =train_baseline.evaluation(dev_examples)
						test_data["train_baseline"] = train_baseline.evaluation(test_examples)

						dev_data["wiki_baseline"] = wiki_baseline.evaluation(dev_examples)
						test_data["wiki_baseline"] = wiki_baseline.evaluation(test_examples)

						print("BASELINES COMPUTED.")

						df_dev.append(dev_data)
						df_test.append(test_data)

	print("CREATION OF THE EVALUATION FILES...")
	# dev
	df = pd.DataFrame(df_dev)
	excel_filename = f'./results_{eval_prefix}-dev.xlsx'
	df.to_excel(excel_filename, index=False)

	# test
	df = pd.DataFrame(df_test)
	excel_filename = f'./results_{eval_prefix}-test.xlsx'
	df.to_excel(excel_filename, index=False)
	
	print("PROCESS DONE.")

