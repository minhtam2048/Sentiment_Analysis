import pandas as pd
import copy
import os
import numpy as np
import re
import keras.backend as K

from tqdm import tqdm
from collections import defaultdict
from os.path import abspath
from spacy.lang.vi import Vietnamese
from .constant import DEFAULT_MAX_LENGTH
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import f1_score
import string
import io
from janome.tokenizer import Tokenizer

def split_array(arr, condition):
    if len(arr) == 0:
        return []
    result = []
    accumulated = [arr[0]]
    for ele in arr[1:]:
        if condition(ele):
            result.append(copy.deepcopy(accumulated))
            accumulated = [copy.deepcopy(ele)]
        else:
            accumulated.append(copy.deepcopy(ele))
    result.append(copy.deepcopy(accumulated))
    return result


def read_file(file_path,type= 'pos'):
    label = 0
    if type == 'neg':
        label = 1
    texts = []
    input_file = io.open(file_path, 'r', encoding='utf-8')
    line_list = input_file.readlines()
    index = 0
    while index < len(line_list):
        line_list[index] = line_list[index]
        texts.append(line_list[index].strip())
        index += 2
    labels =[]
    for k in range(5000):
        labels.append(label)
    return texts,np.array(labels)



def tokenize(texts):
    nlp = Tokenizer()
    docs = []
    for text in texts:
        tokens = np.array([postprocess_token(token.base_form) for token in nlp.tokenize(text)])
        docs.append(tokens)
    return docs

def postprocess_token(token):
    if token in string.punctuation:
        return '<punct>'
    elif token.isdigit():
        return '<number>'
    else:
        return token

def make_embedding(texts, embedding_path, max_features):
    embedding_path = abspath(embedding_path)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if embedding_path.endswith('.vec'):
        embedding_index = dict(get_coefs(*o.strip().split(" "))
                               for o in open(embedding_path))
        mean_embedding = np.mean(np.array(list(embedding_index.values())),axis=0)
    elif embedding_path.endswith('bin'):
        embedding_index = KeyedVectors.load_word2vec_format(
            embedding_path, binary=True)
        mean_embedding = np.mean(embedding_index.vectors, axis=0)
    embed_size = mean_embedding.shape[0]
    word_index = sorted(list({word for sentence in texts for word in sentence}))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    i = 1
    word_map = defaultdict(lambda: nb_words)
    for word in word_index:
        if i >= max_features:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = mean_embedding
        word_map[word] = i
        i += 1
    
    embedding_matrix[-1] = mean_embedding
    return embed_size, word_map, embedding_matrix

def text_to_sequences(texts, word_map, max_len=DEFAULT_MAX_LENGTH, checkmap=False):
    texts_id = []
    for sentence in texts:
        if(checkmap):
            smap = []
            for i in sentence :
            #    print(i)
            #    print("-----------")
               if (i in word_map):
                   smap.append(i)
            sentence = smap
        sentence = [word_map[word] for word in sentence][:max_len]
        padded_setence = np.pad(
            sentence, (0, max(0, max_len - len(sentence))), 'constant', constant_values=0)
        texts_id.append(padded_setence)
    return np.array(texts_id)

def find_threshold(pred_proba, y_true, metric = f1_score):
    cur_acc = 0
    cur_thres = 0
    for ind in range(len(pred_proba) - 1):
        threshold = (pred_proba[ind][0] + pred_proba[ind + 1][0]) / 2
        pred = (pred_proba > threshold).astype(np.int8)
        acc = metric(pred, y_true)
        if acc > cur_acc:
            cur_thres = threshold
            cur_acc = acc

    return cur_thres

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def predictions_to_submission(test_data, predictor):
    tqdm.pandas()
    submission = test_data[['id']]
    submission['label'] = test_data['text'].progress_apply(predictor)
    return submission


# HELPERS FOR HIERARCHICAL MODEL:
def sent_tokenize(texts):
    nlp = Tokenizer()
    docs = []
    for text in texts:
        text_tokenized = []
        if (len(text) > 3):
            for sentence in text.split('。'):
                sent_tokens = np.array([postprocess_token(token.base_form) for token in nlp.tokenize(sentence)])
                text_tokenized.append(sent_tokens)
        else:
            text_tokenized.append([])
        docs.append(text_tokenized)

    return docs

def sent_embedding(tokenized_texts, embedding_path, max_features):
    embedding_path = abspath(embedding_path)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if embedding_path.endswith('.vec'):
        embedding_index = dict(get_coefs(*o.strip().split(" "))
                               for o in open(embedding_path))
        mean_embedding = np.mean(np.array(list(embedding_index.values())), axis=0)
    elif embedding_path.endswith('bin'):
        embedding_index = KeyedVectors.load_word2vec_format(
            embedding_path, binary=True)
        mean_embedding = np.mean(embedding_index.vectors, axis=0)
    embed_size = mean_embedding.shape[0]
    word_index = {word for text in tokenized_texts for sentence in text for word in sentence}
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))

    i = 1
    word_map = defaultdict(lambda: nb_words)
    for word in word_index:
        if i >= max_features:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = mean_embedding
        word_map[word] = i
        i += 1
    embedding_matrix[-1] = mean_embedding
    return embed_size, word_map, embedding_matrix

def text_sents_to_sequences(texts, word_map, max_nb_sent, max_sent_len):
    ret = []
    for i in range(len(texts)):
        text_vecs = []
        for j in range(len(texts[i])):
            if (j < max_nb_sent):
                sent_vecs = []
                for k in range(len(texts[i][j])):
                    if (k < max_sent_len):
                        sent_vecs.append(word_map[texts[i][j][k]])
                if (len(sent_vecs) < max_sent_len):
                    sent_vecs = np.pad(
                        sent_vecs,
                        (0, max(0, max_sent_len - len(sent_vecs))),
                        'constant',
                        constant_values=0
                    )
                text_vecs.append(sent_vecs)


        if (len(text_vecs) < max_nb_sent):
            text_vecs = np.pad(
                text_vecs,
                ((0, max_nb_sent - len(text_vecs)), (0, 0)),
                'constant',
                constant_values=0
            )

        ret.append(text_vecs)

    return np.array(ret)
