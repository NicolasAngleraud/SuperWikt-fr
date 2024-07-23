import pandas as pd
import argparse




allowed_relations = ["rdf:type",
                     "lexinfo:partOfSpeech",
                     "ontolex:canonicalForm",
                     "ontolex:sense",
                     "dbnary:describes",
                     "dbnary:synonym",
                     "lexinfo:gender",
                     "skos:definition",
                     "skos:example",
                     "rdfs:label",
                     "dbnary:partOfSpeech"]

allowed_rdf_types = ["ontolex:LexicalSense",
                     "ontolex:Form",
                     "ontolex:LexicalEntry",
                     "dbnary:Page",
                     "ontolex:Word",
                     "ontolex:MultiWordExpression"]

allowed_categories = ["lexinfo:noun", '"-nom-"']

labels_to_ignore = ["vieilli", "archaïque", "désuet", "archaïque, orthographe d’avant 1835"]

lang = "fra"


def extract_wiki_paragraphs(input_file):
	print("Extracting paragraphs of Wiktionary data from ttl file...")
	paragraphs = []
	with open(input_file, 'r', encoding='utf-8') as file:
		paragraph = []

		for i, line in enumerate(file):

			if line.startswith('@prefix'): continue

			elif not line.strip() and len(paragraph)>0:
				paragraphs.append(paragraph)
				paragraph = []

			else:
				line = line.strip()
				paragraph.append(line)

			#if i >= 76: break

	print(f"Extracted {len(paragraphs)} paragraphs of Wiktionary data from ttl file.")

	return paragraphs
	

def filter_paragraphs(paragraphs):
	print("Filtering irrelevant paragraphs from Wiktionary data...")
	
	filtered_paragraphs = []
	while len(paragraphs) > 0:
		
		par = paragraphs.pop(0)
		for line in par:
			if "rdf:type" in line:
				for rdf_type in allowed_rdf_types:
					if rdf_type in line:
						filtered_paragraphs.append(par)
						break
	
	print("Filtered irrelevant paragraphs from Wiktionary data.")
	print(f"{len(filtered_paragraphs)} relevant paragraphs left to process for the extraction of the Wiktionary.")
	return filtered_paragraphs


def parse_paragraphs(paragraphs_list, output_file):
	pass


def main():
    parser = argparse.ArgumentParser(description="Extract wiktionary data from a TTL dump file.")
    parser.add_argument('--input', required=True, help='Path to the input TTL dump file.')
    parser.add_argument('--output', required=True, help='Path to the output TSV file.')

    args = parser.parse_args()

    paragraphs = extract_wiki_paragraphs(args.input)
    
    paragraphs = filter_paragraphs(paragraphs)
    
    parse_paragraphs(paragraphs, args.output)

if __name__ == "__main__":
    main()
