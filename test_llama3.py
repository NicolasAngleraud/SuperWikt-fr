from transformers import AutoTokenizer, AutoModelForCausalLM


## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct


API_TOKEN = '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'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=API_TOKEN)


def_ = "Auteur de lettres ou de coups de téléphone anonymes."

prompt = """Tu es un annotateur en sémantique lexciale qui doit attribuer à une définition la classe sémantique qui correspond le plus à ce qui est décrit par la définition. Le choix des classes est restreint aux quatre classes suivantes: person, animal, mineral, plant.

Répond au format de la manière suivante, en ne donnant aucun autre information ou mot: {{'définition': définition, 'classe sémantique': classe sémantique}}

Voici quelques exemples:
{{'définition': 'riz : Céréale que l’on cultive dans les terres humides et marécageuses des pays chauds.', 'classe sémantique': 'plant'}},
{{'définition': 'oisillon : Petit oiseau.', 'classe sémantique': 'animal'}},
{{'définition': 'grand-mère : Mère du père (grand-mère paternelle) ou de la mère (grand-mère maternelle) d’une personne.', 'classe sémantique': 'person'}}

{{'définition: {BODY}', 'classe sémantique': """.format(BODY=def_)


inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 11, num_return_sequences=1, temperature=0.2)

generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)


print("Generated Classification:", generated_classification)

