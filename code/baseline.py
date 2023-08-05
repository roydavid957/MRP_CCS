from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import argparse
from utils import load_all_samples, get_y_proba_dict, eval_proba
import numpy as np
import spacy
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def train_eval(X_train, y_train, X_test, y_test):

    vectorizer = TfidfVectorizer()
    vectorized_X_train = vectorizer.fit_transform(X_train)
    
    scaler = MaxAbsScaler()
    scaled_X_train = scaler.fit_transform(vectorized_X_train)

    clf = LinearSVC(max_iter=50000, dual=False)
    clf = CalibratedClassifierCV(clf)   # enables LinearSVC to output probabilities

    if X_test:
      vectorized_X_test = vectorizer.transform(X_test)
      clf.fit(vectorized_X_train, y_train)
      y_pred = clf.predict(vectorized_X_test)
      y_pred_proba = clf.predict_proba(vectorized_X_test)
    #   print(clf.classes_)
      return classification_report(y_test, y_pred, output_dict=True), y_pred_proba
    else:
      folds = []
      acc_score_list = []
      skf = StratifiedKFold(n_splits=5)
      for train_index, test_index in skf.split(scaled_X_train, np.array(y_train)):
        x_train_fold, y_train_fold = scaled_X_train[train_index], np.array(y_train)[train_index]
        x_test_fold, y_test_fold = scaled_X_train[test_index], np.array(y_train)[test_index]

        clf.fit(x_train_fold, y_train_fold)
        y_pred = clf.predict(x_test_fold)

        acc_score_list.append(accuracy_score(y_test_fold, y_pred))
        y_pred_proba = clf.predict_proba(x_test_fold)
        folds.append({'X': test_index, 'Y': y_pred_proba})

    #   print(acc_score_list)
    #   print(cross_val_score(clf, vectorized_X_train, y_train, cv=5))
      return acc_score_list, folds

def get_lemmas(nlp, text):
    return [token.lemma_ for doc in nlp.pipe(text) for token in doc if not token.is_punct]

'''
Simple baseline measuring lexical overlap
based on percentage of words (lemma) that appear both in the context 
and in the possible continuations
'''    
def simple_baseline(samples,labels,spacy_model, args):
    np.random.seed(2023)
    nlp = spacy.load(spacy_model)
    nlp.get_pipe("lemmatizer")
    sample_dict = dict()
    y_true = [sample.label for sample in samples]
    tokenizer = RegexpTokenizer(r'\w+')
    lang = 'english' if args.lang == 'en' else 'dutch'


    for sample in samples:
        input_sentences = [tokenizer.tokenize(sent.text) for sent in sample.input]
        input_sentences = [(' ').join([word for word in tokenizer.tokenize(sent.text) if word not in stopwords.words(lang)]) for sent in sample.input]
        input_lemmas = Counter(get_lemmas(nlp, input_sentences))
        target_sentence = (' ').join([word for word in tokenizer.tokenize(sample.target.text) if word not in stopwords.words(lang)])
        # target_lemmas = Counter(get_lemmas(nlp, [sample.target.text]))
        target_lemmas = Counter(get_lemmas(nlp, [target_sentence]))
        # print(input_lemmas)
        # print(target_lemmas)
        diff = input_lemmas-target_lemmas
        # print(list(diff.elements()))
        if sample.id in sample_dict.keys():
            sample_dict[sample.id][str(sample.label)] = len(list(diff.elements()))
        else:
            sample_dict[sample.id] = dict()
            sample_dict[sample.id][str(sample.label)] = len(list(diff.elements()))

    y_pred_list = []
    for sample, label in zip(sample_dict.values(), labels):
        # print(sample,label)
        true_label = 1 if int(label) == 1 else 2
        false_label = 2 if true_label == 1 else 1

        if sample['1'] == sample['0']:
            # y_pred = np.random.choice([1,2])
            y_pred = false_label
        else:
            y_pred = true_label if sample['1'] < sample['0'] else false_label
        y_pred_list.append(y_pred)
    
    return classification_report(labels, y_pred_list)




def main():
    parser = argparse.ArgumentParser(description='Baseline using TF-IDF and lexical overlap')
    parser.add_argument('-tr', '--train_file')
    parser.add_argument('-ts', '--test_file', default='', help='Leave empty for k-fold cv')
    parser.add_argument('-ds', '--data_set', help='Specify dataset for dataloader: "SCT" (Story Cloze Task), "NCT" (Narrative Cloze Task), "CMCNC" (Coherent Multiple Choice Narrative Cloze)')
    parser.add_argument('-o', '--output_file',
                      help='absolute path to output file for classification report')
    parser.add_argument('-l', '--lang',
                      help='en/nl for spacy model for event extraction', default='en')
    
    args = parser.parse_args()

    spacy_model = "en_core_web_sm" if args.lang == 'en' else "nl_core_news_sm"

    print('\nLoading samples...')
    train_samples, labels_list, train_labels = load_all_samples(args.train_file, args, spacy_model)        # Loading of train and test samples from
    if args.test_file == '':
       valid_samples, test_label_list = [], labels_list # empty for k-fold cv
       sb_report = simple_baseline(train_samples, train_labels, spacy_model, args)
    else:
      valid_samples, test_label_list, valid_labels = load_all_samples(args.test_file, args, spacy_model)   # train and test source files
      
      sb_report = simple_baseline(valid_samples, valid_labels, spacy_model, args)
    
    print('\nLexical overlap baseline...')
    print(sb_report)
    # exit()

    all_labels = list(set(test_label_list).union(set(labels_list)))

    all_labels += ['macro avg', 'weighted avg', 'accuracy']
    cls_scores = {label:[] for label in all_labels}
    cls_scores_proba = {f'{label}_proba':[] for label in all_labels}

    print('\nSVM inputs...')
    X_train, y_train = [f'{sample.input_sent} {sample.target.text}' for sample in train_samples], [sample.label for sample in train_samples]      # Creation of SVM inputs
    if valid_samples:
        X_test, y_test = [f'{sample.input_sent} {sample.target.text}' for sample in valid_samples], [sample.label for sample in valid_samples]
    else:
        X_test, y_test = [],[]  # empty for k-fold cv

    print('\nSVM training and evaluation...')
    cls_report, y_pred_proba = train_eval(X_train, y_train, X_test, y_test)        # SVM training and evaluation
    print(cls_report)

    # Change the predicted output format
    if args.test_file == '':        # per fold for cv
        cls_report_proba = []
        for fold in y_pred_proba:
            y_pred_dict = get_y_proba_dict(np.array(train_samples)[fold['X']],fold['Y'])

            y_pred_list, y_true = eval_proba(y_pred_dict)

            cls_report_proba.append(accuracy_score(y_true, y_pred_list))
    elif args.data_set.lower() != 'cmcnc':
        y_pred_dict = get_y_proba_dict(valid_samples,y_pred_proba)
        y_pred_list, y_true = eval_proba(y_pred_dict)

        print(classification_report(y_true, y_pred_list))
        cls_report_proba = classification_report(y_true, y_pred_list, output_dict=True)

        y_pred_list_og, y_true = eval_proba(y_pred_dict, valid_labels)
        
        print(classification_report(valid_labels, y_pred_list_og))
    
    print(cls_report_proba)
    
    for label in all_labels:
        if label in cls_report:
            if label == 'accuracy':
                cls_scores[label].append(str(cls_report[label]))
            else:
                cls_scores[label].append(str(cls_report[label]['f1-score']))
        elif label == 'accuracy' and valid_samples == []:       # for cross validation scores
            cls_scores[label].append(str(np.array(cls_report).mean()))  # computes mean of k-fold cv per layer
        else:
            cls_scores[label].append(str(None))

    for label in all_labels:
        if label in cls_report_proba:
            if label == 'accuracy':
                cls_scores_proba[f'{label}_proba'].append(str(cls_report_proba[label]))
            else:
                cls_scores_proba[f'{label}_proba'].append(str(cls_report_proba[label]['f1-score']))
        elif label == 'accuracy' and valid_samples == []:       # for cross validation scores
            cls_scores_proba[f'{label}_proba'].append(str(np.array(cls_report_proba).mean()))
        else:
            cls_scores_proba[f'{label}_proba'].append(str(None))
    
    for label in cls_scores:
        print(f'{label}\t{cls_scores[label]}\n')
    
    for label in cls_scores_proba:
        print(f'{label}\t{cls_scores_proba[label]}\n')
    
    # with open(args.output_file, 'w+') as out_file:                  # The output file will contain a row for each class f1-score and a row for 
    #     for label in cls_scores:                                 # macro avg, weighted avg and accuracy
    #         out_file.write(f'{label}\t{cls_scores[label]}\n')  # each row will contain a value for each layer

if __name__ == '__main__':
    main()