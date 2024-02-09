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
	"lr": 0.0001,
	"frozen": True,
	"max_seq_length": 100
}


def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="donnees_stage_wiktionnaire_supersenses.xlsx", help="")
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
		
	eval_prefix = 'freq'

	nb_runs = 1
	patiences = [2]
	frozen = True
	lrs = [0.0005]
	def_errors = []

	for i in range(nb_runs):
		train_examples, dev_examples, test_examples = lclf.encoded_examples(datafile=args.data_file, eval_prefix=eval_prefix)
 
		for lr in lrs:
			for patience in patiences:

			    dev_data = {}
			    test_data = {}


			    print("")
			    print(f"run {i+1} : lr = {lr}")
			    print(f"eval set : {eval_prefix}")
			    print("")


			    params = {key: value for key, value in LPARAMETERS.items()}
			    params['lr'] = lr
			    params['patience'] = patience
			    params['frozen'] = frozen

			    dev_data["run"] = i + 1
			    test_data["run"] = i + 1

			    classifier = lclf.SupersenseTagger(params, DEVICE)
			    lclf.training(params, train_examples, dev_examples, classifier, DEVICE, dev_data, test_data)
			    lclf.evaluation(dev_examples, classifier, params, DEVICE,  i+1, f"{eval_prefix}-dev", dev_data)
			    lclf.evaluation(test_examples, classifier, params, DEVICE, i+1, f"{eval_prefix}test", test_data)

			   
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

			    df_dev.append(dev_data)
			    df_test.append(test_data)


	# dev
	df = pd.DataFrame(df_dev)
	excel_filename = f'results_{eval_prefix}-dev.xlsx'
	df.to_excel(excel_filename, index=False)

	# test
	df = pd.DataFrame(df_test)
	excel_filename = 'results_{eval_prefix}-test.xlsx'
	df.to_excel(excel_filename, index=False)

