import sys
import json
import spacy
import crosslingual_coreference
import numpy as np
import pandas as pd
import torch

def get_sentences(sentences):
    punct_list = ['.', '?', '!']
    for idx, sent in enumerate(sentences):
        if sent[-1] not in punct_list:
            new_sent = (' ').join([sent,sentences[idx+1]]) if len(sentences) >= idx+1 else sent+'.'
            sentences[idx] = new_sent.replace('  ', ' ')
            del sentences[idx+1]
    return sentences

'''
Extract data following a protagonist
in subject or object position.
Using (loose) (crosslingual) coreference
'''

# 1. Check if at least 6 sentences
# 2. Do coreference
# 3. Check if cluster heads are protagonist
def get_data(src_path):
    if torch.cuda.is_available():
        device = 0
        model_name = 'info_xlm'
        spacy_model = "nl_core_news_sm"
    else:
        device = -1
        model_name = 'minilm'
        spacy_model = "nl_core_news_lg"
        
    print('\nUsing device:', device)
    print('\nUsing model:', model_name)
    print('\nUsing spacy model:', spacy_model)
    
    story_list_dict = []
    n = 6

    spacy.prefer_gpu()
    nlp = spacy.load(spacy_model)
    nlp.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2
                                        , "model_name": model_name
                                        , "device": device
                                        })
    with open(src_path,'r') as f:
        
        for line in f:
            line = json.loads(line)
            if line['text'] != '':
                doc = nlp(line['text'])
                if len(list(doc.sents)) >= n:
                    doc_sents = [str(sent) for sent in doc.sents]
                    sents = get_sentences(doc_sents)
                    if len(sentences) >= n:

                        # print('\nCoref clusters:',doc._.coref_clusters)
                        # print('\nCoref text:',doc._.resolved_text)
                        # print('\nCoref heads:',doc._.cluster_heads)

                        coref_doc = nlp(doc._.resolved_text)
                        nsubj = [str(tok) for tok in coref_doc if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
                        ents = [str(ent) for ent in coref_doc.ents if ent.label_ == "PERSON"]
                        # print('\nEntities:',ents)
                        # print('\nsubj/obj:',nsubj)
                        overlap = list(set([ent for subj in nsubj for ent in ents if subj in ent]))
                        # print('\nOverlap:',overlap)
                        coref_sentences = get_sentences(coref_doc)
                        over_sents = list(set([idx for idx, sent in enumerate(coref_sentences) for ent in overlap if ent in str(sent)]))
                        # print('\nSents:',over_sents)
                        if len(set(over_sents).intersection(set([0,1,2,3,4]))) > 0:
                            story_id = line['id']
                            sentences = [str(sent) for sent in doc.sents]
                            # cutoff = n-1 if sentences[n-2].endswith('.') else n

                            last_sents = sentences[n-1:]
                            rand_sent = np.random.choice(last_sents)
                            while len(rand_sent.split(' ')) < 2:
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
                            # print(story_list_dict)
    return story_list_dict


if __name__ == '__main__':
    data = get_data(sys.argv[1])
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2],sep='\t', index=False)