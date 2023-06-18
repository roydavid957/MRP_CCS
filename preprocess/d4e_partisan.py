import sys
import json
import spacy
import crosslingual_coreference
import numpy as np
import pandas as pd
import torch

def get_prot(doc):
    nsubj = [str(tok) for tok in doc if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
    ents = [str(ent) for ent in doc.ents if ent.label_ == "PERSON"]
    if len(set(nsubj).intersection(set(ents))) > 0:
        return True
    else:
        return False

'''
Extract data following a protagonist
in subject or object position.
Using crosslingual coreference and spacy
'''

# 1. Check if at least 6 sentences
# 2. Do coreference
# 3. Check if cluster heads are protagonist
def get_data(src_path):
    if torch.cuda.is_available():
        device = "cuda"
        model_name = 'info_xlm'
    else:
        device = "cpu"
        model_name = 'minilm'
    print('\nUsing device:', device)
    print('\nUsing model:', model_name)
    
    story_list_dict = []
    n = 6

    spacy.prefer_gpu()
    nlp = spacy.load("nl_core_news_sm") # ("nl_core_news_lg")
    nlp.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2
                                        , "model_name": model_name
                                        , "device": device
                                        })
    
    with open(src_path,'r') as f:
        
        for line in f:
            line = json.loads(line)
            try:
                doc = nlp(line['text'])
                if len(list(doc.sents)) >= n:

                    # print('\nCoref clusters:',doc._.coref_clusters)
                    # print('\nCoref text:',doc._.resolved_text)
                    # print('\nCoref heads:',doc._.cluster_heads)

                    doc2 = nlp(doc._.resolved_text)
                    nsubj = [str(tok) for tok in doc2 if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
                    ents = [str(ent) for ent in doc2.ents if ent.label_ == "PERSON"]
                    # print('\nEntities:',ents)
                    # print('\nsubj/obj:',nsubj)
                    overlap = list(set([ent for subj in nsubj for ent in ents if subj in ent]))
                    # print('\nOverlap:',overlap)
                    over_sents = list(set([idx for idx, sent in enumerate(list(doc2.sents)) for ent in overlap if ent in str(sent)]))
                    # print('\nSents:',over_sents)
                    if len(set(over_sents).intersection(set([0,1,2,3,4]))) > 0 and str(list(doc.sents)[n-2]).endswith('.'):
                        story_id = line['id']
                        sentences = [str(sent) for sent in doc.sents]
                        # cutoff = n-1 if sentences[n-2].endswith('.') else n

                        last_sents = sentences[n-1:]
                        rand_sent = np.random.choice(last_sents)
                        label = np.random.choice([1,2])
                        if label == 1:
                            opt1 = sentences[n-2]
                            opt2 = rand_sent
                        elif label == 2:
                            opt2 = sentences[n-2]
                            opt1 = rand_sent

                        story_list_dict.append({'StoryID':story_id,
                                                'Sentence1':sentences[0],
                                                'Sentence2':sentences[1],
                                                'Sentence3':sentences[2],
                                                'Sentence4':sentences[3],
                                                'Continuation1':opt1,
                                                'Continuation2':opt2,
                                                'Label':label
                                                })
            except:
                pass
    return story_list_dict


if __name__ == '__main__':
    data = get_data(sys.argv[1])
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2],sep='\t')