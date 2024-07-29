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
                     "ontolex:LexicalEntry",
                     "dbnary:Page",
                     "ontolex:Form",
                     "ontolex:Word",
                     "ontolex:MultiWordExpression"]

allowed_categories = ["lexinfo:noun", '"-nom-"']

cat2pos = {"lexinfo:noun": "noun", '"-nom-"': "noun"}

labels_to_ignore = ["vieilli", "archaïque", "désuet"]

lang = "fra"

def extract_labels_definition(text):
    if not text.strip():
        return None, None
    
    text = text.strip()
    labels = set()
    i = 0
    n = len(text)

    while i < n and text[i] == '(':
        end = text.find(')', i)
        if end == -1:
            definition = text[i:].replace('(', '').strip()
            return labels, definition
        label = text[i+1:end].strip().lower()
        labels.add(label)
        i = end + 1
        
        while i < n and text[i].isspace():
            i += 1

    definition = text[i:].strip()
    return labels, definition
    
        
def normalization_id(full_id, lang=lang):
    if full_id.startswith(lang + ":"):
        return full_id.removeprefix(lang + ":").strip('_')
    elif full_id.startswith("<http://kaiko.getalp.org/dbnary/"):
        return full_id.removeprefix("<http://kaiko.getalp.org/dbnary/" + lang + "/").removesuffix(">").strip('_')
    else:
        return full_id


def parse_page(paragraph, wiki_data):
	page = None
	entry_ids = None
	for i, line in enumerate(paragraph):
		line = line.strip()
		if i == 0:
			page = normalization_id(line.split('\t')[0])
		if "dbnary:describes" in line:
			entry_ids = [normalization_id(el.strip().strip(' .').strip().strip('.')) for el in line.split('dbnary:describes')[1].strip().split(" .")[0].strip().split(' , ') if el.strip('.').strip()]
	if page: 
		wiki_data['pages'][page] = {'entry_ids':entry_ids}
	else:
		raise TypeError("Page not found.")
	if not entry_ids: raise TypeError(f"Page '{page}' seems empty.")
		
		
def parse_entry(paragraph, wiki_data):
	entry = None
	pos = None
	form_id = None
	sense_ids = None
	for i, line in enumerate(paragraph): print(f"LINE {i}:", line)
	
	for i, line in enumerate(paragraph):
		line = line.strip()
		if i == 0:
			print("LINE:", line)
			print("ENTRY:" , line.split('\t')[0])
			entry = normalization_id(line.split('\t')[0])
		if "partOfSpeech" in line:
			for cat in allowed_categories:
				if cat in line: pos = cat2pos[cat]
		if "ontolex:canonicalForm" in line:
			form_id = normalization_id(line.split("ontolex:canonicalForm")[1].strip(";").strip())
		if "ontolex:sense" in line:
			sense_ids = [normalization_id(el.strip().strip(' .').strip().strip('.')) for el in line.split('ontolex:sense')[1].strip().split(" .")[0].strip().split(' , ') if el.strip('.').strip()]
	if entry:
		wiki_data['entries'][entry] = {'pos':pos, 'form_id':form_id, 'sense_ids':sense_ids}
	else:
		raise TypeError("Entry not found.")
	if not sense_ids: raise TypeError(f"Entry '{entry}' seems empty.")
	
	
def parse_sense(paragraph, wiki_data):
	sense = None
	labels = None
	definition = None
	examples = []
	end_mark = "@fr"
	for i, line in enumerate(paragraph):
		line = line.strip()
		if i == 0:
			sense = normalization_id(line.split('\t')[0])
		if "skos:definition" in line and "rdf:value" in line:
			text = line.split('rdf:value')[1].strip().strip('"')
			end_index = text.find(end_mark)
			if index != -1: 
				definition = text[:index].strip().strip('"').strip()
			else:
				definition = text.strip().strip(';').strip(']').strip('[').strip().strip('"').strip()
			labels, definition = extract_labels_definition(definition)
				
		if "skos:example" in line and "rdf:value" in line:
			text = line.split('rdf:value')[1].strip().strip('"')
			end_index = text.find(end_mark)
			if index != -1: 
				example = text[:index].strip().strip('"').strip()
			else:
				example = text.strip().strip(';').strip(']').strip('[').strip().strip('"').strip()
			examples.append(example)
					
	if sense:
		wiki_data['senses'][sense] = {'definition':definition, 'labels':labels, 'examples':examples}
	else:
		raise TypeError("Sense not found.")
					
				
	
def parse_form(paragraph, wiki_data):
	form = None
	gender = None
	for i, line in enumerate(paragraph):
		line = line.strip()
		if i == 0:
			form = normalization_id(line.split('\t')[0])
		if "lexinfo:gender" in line:
			gender = line.split("lexinfo:gender")[1].split('lexinfo:')[1].strip(';').strip()
	if form:
		wiki_data['forms'][form] = {'gender':gender}
	else:
		raise TypeError("Form not found.")
	if not gender: raise TypeError(f"Form '{form}' seems empty.")
			
def parse_paragraph(paragraph, rdf_type, wiki_data):
	if rdf_type == "dbnary:Page":
		parse_page(paragraph, wiki_data)
	elif rdf_type in ["ontolex:LexicalEntry", "ontolex:Word", "ontolex:MultiWordExpression"]:
		parse_entry(paragraph, wiki_data)
	elif rdf_type == "ontolex:LexicalSense":
		parse_sense(paragraph, wiki_data)
	elif rdf_type == "ontolex:Form":
		parse_form(paragraph, wiki_data)
	else:
		print(f"ERROR WITH RDF TYPE: {rdf_type}")


def extract_wiki_data(input_file):
	print("Extracting paragraphs of Wiktionary data from ttl file...")
	
	columns = ['page', 'entry_id', 'sense_id', 'supersenses', 'hypersenses', 'labels', 'synonyms', 'gender', 'definition'] + [f'example_{i}' for i in range(1, 26)]
	df = pd.DataFrame(columns=columns)

	with open(input_file, 'r', encoding='utf-8') as file:
		nb_paragraphs = 0
		add_paragraph = False
		rdf_type = None
		paragraph = []
		wiki_data = {"pages":{}, "entries":{}, "senses":{}, "forms":{}}
		
		for i, line in enumerate(file):
		
			if line.startswith('@prefix'): continue

			elif not line.strip() and len(paragraph)>0:
			
				if add_paragraph:
					parse_paragraph(paragraph, rdf_type, wiki_data)
					nb_paragraphs += 1
					
				add_paragraph = False
				paragraph = []
				rdf_type = None

			else:
				line = line.strip()
				
				if "rdf:type" in line:
					for rdf in allowed_rdf_types:
						if rdf in line:
							add_paragraph = True
							rdf_type = rdf
							break
							
				if line.strip(): paragraph.append(line)

			if i >= 76: break
	return wiki_data
	
	
def data2df(wiki_data, output_file):
	for key, value in wiki_data.items():
		print("KEY: ", key)
		print("VALUE: ", value)
	


def main():
    parser = argparse.ArgumentParser(description="Extract wiktionary data from a TTL dump file.")
    parser.add_argument('--input', required=True, help='Path to the input TTL dump file.')
    parser.add_argument('--output', required=True, help='Path to the output TSV file.')
    args = parser.parse_args()

    wiktionary_data = extract_wiki_data(args.input)
    
    data2df(wiktionary_data, args.output)

if __name__ == "__main__":
    main()
