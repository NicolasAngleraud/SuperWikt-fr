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

	with open(input_file, 'r', encoding='utf-8') as file:
		nb_paragraphs = 0
		add_paragraph = False
		paragraph = []
		for i, line in enumerate(file):
		
			

			if line.startswith('@prefix'): continue

			elif not line.strip() and len(paragraph)>0:
			
				if add_paragraph:
					parse_paragraph(paragraph)
					nb_paragraphs += 1
					
				add_paragraph = False
				paragraph = []

			else:
				line = line.strip()
				
				if "rdf:type" in line:
					for rdf_type in allowed_rdf_types:
						if rdf_type in line:
							add_paragraph = True
							break
							
				paragraph.append(line)

			#if i >= 76: break

	print(f"Extracted {len(paragraphs)} paragraphs of Wiktionary data from ttl file.")

	return paragraphs
	

def parse_paragraph(paragraph):
	pass


def main():
    parser = argparse.ArgumentParser(description="Extract wiktionary data from a TTL dump file.")
    parser.add_argument('--input', required=True, help='Path to the input TTL dump file.')
    parser.add_argument('--output', required=True, help='Path to the output TSV file.')

    args = parser.parse_args()

    paragraphs = extract_wiki_paragraphs(args.input)
    
    parse_paragraphs(paragraphs, args.output)

if __name__ == "__main__":
    main()
