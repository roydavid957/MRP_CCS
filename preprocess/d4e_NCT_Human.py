import pandas as pd
from tqdm import tqdm
import sys
import spacy

def get_prot(doc):
    nsubj = [str(tok) for tok in doc if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
    ents = [str(ent) for ent in doc.ents if ent.label_ == "PERSON"]
    if len(set(nsubj).intersection(set(ents))) > 0:
        return True
    else:
        return False

'''
Filter stories containing human protagonist
in object or subject position
from NCT Full stories
'''
def get_data(src_path: str,spacy_model="en_core_web_sm"):
    data = pd.read_csv(src_path,sep='\t')
    nlp = spacy.load(spacy_model)
    
    story_list_dict = []
    for idx, row in data.iterrows():
        add=False
        story = list(row[1:5])
        if int(row[-1]) == 1:       # add correct continuation
            story.append(row[-3])
        elif int(row[-1]) == 2:
            story.append(row[-2])
        # story = [row['Sentence1'],row['Sentence2'],row['Sentence3'],row['Sentence4'],row['Sentence5']]
        for sentence in story:
            doc = nlp(sentence)
            if get_prot(doc):
                add=True
                break

        if add:
            story_list_dict.append(row)

    return story_list_dict

if __name__ == '__main__':
    df = pd.DataFrame(get_data(sys.argv[1]))
    df.to_csv(sys.argv[2],index=False,sep='\t')