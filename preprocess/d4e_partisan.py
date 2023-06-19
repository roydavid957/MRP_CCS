import sys
import spacy
import crosslingual_coreference
import pandas as pd
import torch

def get_sentences(sentences):
    punct_list = ['.', '?', '!']
    for idx, sent in enumerate(sentences):
        if sent[-1] not in punct_list:
            if len(sentences) >= idx+1:
                new_sent = (' ').join([sent,sentences[idx+1]])
                del sentences[idx+1]
            else:
                new_sent = sent+'.'
            sentences[idx] = new_sent.replace('  ', ' ')
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
        # model_name = 'info_xlm'
        model_name = 'minilm'
        # spacy_model = "nl_core_news_lg"
        spacy_model = "nl_core_news_sm"
    else:
        device = -1
        model_name = 'minilm'
        spacy_model = "nl_core_news_sm"

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
    data = pd.read_csv(src_path,sep='\t')
        
    for idx,row in data.iterrows():
        sents = [row[1],row[2],row[3],row[4]]
        text = (' ').join(sents)

        try:
            coref_doc = nlp(text._.resolved_text)
        except IndexError as e:
            print(f'\n{e}\n{row}')
            next(data)

        nsubj = [str(tok) for tok in coref_doc if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
        ents = [str(ent) for ent in coref_doc.ents if ent.label_ == "PERSON"]
        overlap = list(set([ent for subj in nsubj for ent in ents if subj in ent]))

        coref_sents = [str(sent) for sent in coref_doc.sents]
        over_sents = list(set([idx for idx, sent in enumerate(coref_sents) for ent in overlap if ent in str(sent)]))

        if len(set(over_sents).intersection(set([0,1,2,3,4]))) > 0:
            story_list_dict.append(row)
            
    return story_list_dict


if __name__ == '__main__':
    data = get_data(sys.argv[1])
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2],sep='\t', index=False)