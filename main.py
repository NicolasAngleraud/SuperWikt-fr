import argparse
import torch
import lexical_classifier as lclf
import pandas as pd


# supersenses used
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
               
class Parameters:
    def __init__(self, nb_epochs=25, batch_size=32, hidden_layer_size=256, patience=2, lr=0.00025, frozen=True, max_seq_length=100, window_example=100):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.patience = patience
        self.lr = lr
        self.frozen = frozen
        self.max_seq_length = max_seq_length
        self.window_example = window_example
        self.keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "max_seq_length", "window_example"]
        
        
def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="donnees_stage_wiktionnaire_supersenses.xlsx", help="")
	parser.add_argument("-def_errors", action="store_true", help="Writes a xlsx file containing the description of the examples wrongly predicted by the classifier during evalutation.")
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

	nb_runs = 1
	patiences = [2]
	frozen = True
	lrs = [0.000024]
	def_errors = []

	for i in range(nb_runs):
		train_examples, dev_examples, test_examples = lclf.encoded_examples(datafile=args.data_file, eval_prefix='freq')
 
		for lr in lrs:
			for patience in patiences:

			    dev_data = {}
			    test_data = {}


			    print("")
			    print(f"run {i+1} : lr = {lr}")
			    print("")

			    hypersense_dist_dev = {hypersense: 0 for hypersense in HYPERSENSES}
			    hypersense_correct_dev = {hypersense: 0 for hypersense in HYPERSENSES}
			    supersense_dist_dev = {supersense: 0 for supersense in SUPERSENSES}
			    supersense_correct_dev = {supersense: 0 for supersense in SUPERSENSES}

			    hypersense_dist_test = {hypersense: 0 for hypersense in HYPERSENSES}
			    hypersense_correct_test = {hypersense: 0 for hypersense in HYPERSENSES}
			    supersense_dist_test = {supersense: 0 for supersense in SUPERSENSES}
			    supersense_correct_test = {supersense: 0 for supersense in SUPERSENSES}


			    params = Parameters(lr=lr, patience=patience, frozen=frozen)

			    dev_data["run"] = i + 1
			    test_data["run"] = i + 1

			    classifier = clf.SupersenseTagger(params, DEVICE)
			    clf.training(params, train_examples, dev_examples, classifier, DEVICE, dev_data, test_data)
			    clf.evaluation(dev_examples, classifier, DEVICE, supersense_dist_dev,
				            supersense_correct_dev, hypersense_dist_dev, hypersense_correct_dev, def_errors, i+1, 
				            "dev", dev_data)
			    clf.evaluation(test_examples, classifier, DEVICE, supersense_dist_test,
				            supersense_correct_test, hypersense_dist_test, hypersense_correct_test, def_errors, i+1, 
				            "test", test_data)

			   
			    sequoia_baseline = clf.MostFrequentSequoia()
			    train_baseline = clf.MostFrequentTrainingData()
			    wiki_baseline = clf.MostFrequentWiktionary()

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
	excel_filename = 'results_dev.xlsx'
	df.to_excel(excel_filename, index=False)

	# test
	df = pd.DataFrame(df_test)
	excel_filename = 'results_test.xlsx'
	df.to_excel(excel_filename, index=False)


	if args.def_errors:
		df = pd.DataFrame(def_errors)
		excel_filename = 'descriptions_errors.xlsx'
		df.to_excel(excel_filename, index=False)
