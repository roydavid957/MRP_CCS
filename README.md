# Master Research/Graduation Project MSc Computational Cognitive Science
Probing Pre-Trained (Large) Language Models for Narrative Coherence: an analysis of large language models in a zero-shot, multilingual setting using per layer probes using a Support Vector Machine on Cloze task datasets.

## Abstract
Probing tasks can be used to explore the capabilities of large language models (LLMs) in terms of their ability to encode linguistic knowledge and how they process (coherent) sequences of text, by using the models' representations to solve a task (proxied by a dataset). Transformer-based LLMs, such as BERT, have shown to be able to encode linguistic knowledge and dominate the state-of-the-art in a variety of NLP tasks. The extend to which these pre-trained large language models (PTLLMs) capture narrative coherence, given (coherent) sequences of text and a set of possible ending/follow-up sequences, in a zero-shot, multilingual setting has not been explored yet. This research presents an extensive study of the abilities of six PTLLMs, two multi-lingual (mDeBERTaV3 and XML-RoBERTa) and four monolingual language models (English: BERT, RoBERTa; Dutch: BERTje, RobBERTV2), to encode narrative coherence across sixteen datasets, consisting of either: short fictional stories or short news article narratives, with each several alternative variations, with varying narrativity types and coherence complexity. In addition we introduce a (small) language specific dataset for Dutch.

Our results show that these PTLLMs can capture narrative coherence mostly when having access to the full text and in simple cases, namely when the possible follow-up sequences do not present subtle linguistic differences and do not require complex commonsense reasoning. In most of these instances, the higher layers (8-12) yield the best performance. Moreover, when the data presented consists of short, coherent sentences with subtle linguistic differences between the possible ending-sequences, the models' performance tends to drop (≈0.2 points) compared to the simple(r) cases, however still capturing (some) coherence. However, the models tend to struggle capturing coherence when the data presented consists of long(er) format sentences and subtle linguistic differences are present between the possible follow-up sequences. At the same time, simple probes show competitive results when compared to state-of-the-art systems on the same task and outperform all our baselines.

## Code
- probing.py: per layer probing of the PTLLMs using a linear SVM.
- baseline.py: lexical overlap and SVM+tfidf baselines.

## Datasets
- SCT; Story Cloze Test
- NCT: Narrative Cloze Task
- NCT-Dutch: Narrative Cloze Task (Dutch version)
