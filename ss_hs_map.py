import pandas as pd



SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']
               
HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance"],
               "informational_object": ["cognition", "communication"],
               "quantification": ["quantity", "part", "group"],
               "other": ["institution", "possession", "time"]
               }
               
                       
class_mapping = {
    'act': {'supersense': 'act', 'hypersense': 'dynamic_situation'},
    'animal': {'supersense': 'animal', 'hypersense': 'animate_entity'},
    'artifact': {'supersense': 'artifact', 'hypersense': 'inanimate_entity'},
    'attribute': {'supersense': 'attribute', 'hypersense': 'stative_situation'},
    'body': {'supersense': 'body', 'hypersense': 'inanimate_entity'},
    'cognition': {'supersense': 'cognition', 'hypersense': 'informational_object'},
    'communication': {'supersense': 'communication', 'hypersense': 'informational_object'},
    'event': {'supersense': 'event', 'hypersense': 'dynamic_situation'},
    'feeling': {'supersense': 'feeling', 'hypersense': 'stative_situation'},
    'food': {'supersense': 'food', 'hypersense': 'inanimate_entity'},
    'institution': {'supersense': 'institution', 'hypersense': 'institution'},
    'act*cognition': {'supersense': 'act*cognition, act, cognition', 'hypersense': 'dynamic_situation*informational_object, dynamic_situation, informational_object'},
    'object': {'supersense': 'object', 'hypersense': 'inanimate_entity'},
    'possession': {'supersense': 'possession', 'hypersense': 'possession'},
    'person': {'supersense': 'person', 'hypersense': 'animate_entity'},
    'phenomenon': {'supersense': 'phenomenon', 'hypersense': 'dynamic_situation'},
    'plant': {'supersense': 'plant', 'hypersense': 'inanimate_entity'},
    'artifact*cognition': {'supersense': 'artifact*cognition, artifact, cognition', 'hypersense': 'inanimate_entity*informational_object, inanimate_entity, informational_object'},
    'quantity': {'supersense': 'quantity', 'hypersense': 'quantification'},
    'relation': {'supersense': 'state', 'hypersense': 'stative_situation'},
    'state': {'supersense': 'state', 'hypersense': 'stative_situation'},
    'substance': {'supersense': 'substance', 'hypersense': 'inanimate_entity'},
    'time': {'supersense': 'time', 'hypersense': 'time'},
    'groupxperson': {'supersense': 'groupxperson, group, person', 'hypersense': 'quantification*animate_entity, quantification, animate_entity'}
}


wiktionnaire = "./wiktionnaire.xlsx"
wiktionnaire_preds = "./wiktionary_predictions.xlsx"

df_wiki = pd.read_excel(wiktionnaire, engine='openpyxl')
df_wiki_preds = pd.read_excel(wiktionnaire_preds, engine='openpyxl')

# Convert the class_mapping dictionary to a DataFrame
class_mapping_df = pd.DataFrame(class_mapping).T.reset_index()
class_mapping_df.columns = ['pred', 'supersense', 'hypersense']

# Merge the predictions DataFrame with the class mapping DataFrame
df_merged = pd.merge(df_wiki_preds, class_mapping_df, on='pred', how='left')

# Merge the resulting DataFrame with the info DataFrame on sense_id
df_final = pd.merge(df_wiki, df_merged, on='sense_id', how='left')

# Update supersense and hypersense columns
df_final['supersense'] = df_final['supersense_y']
df_final['hypersense'] = df_final['hypersense_y']
df_final['lemma'] = df_final['lemma_y']


columns_order = ['sense_id', 'entry_id', 'lemma', 'supersense', 'hypersense', 'labels','definition'] + [f'example_{i}' for i in range(1, 24)] + ['pred']

columns_to_convert = ['lemma', 'definition'] + [f'example_{i}' for i in range(1, 24)]

df_final = df_final[columns_order]
df_final = df_final.fillna('')
df_final[columns_to_convert] = df_final[columns_to_convert].astype(str)


df_final.to_excel('./Wiktionnaire_fr_noms_sem.xlsx', index=False)
df_final.to_csv('./Wiktionnaire_fr_noms_sem.tsv', sep='\t', index=False, encoding='utf-8')

