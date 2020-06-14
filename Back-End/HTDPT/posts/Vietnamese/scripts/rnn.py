from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    CuDNNGRU, GRU, LSTM, Bidirectional, CuDNNLSTM, \
    GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, \
    Lambda, Concatenate, TimeDistributed
from .util import f1
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.activations import softmax
from keras_layer_normalization import LayerNormalization
from .net_components import AttLayer, AdditiveLayer
from keras.utils.vis_utils import plot_model

def SARNNKerasCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, use_fasttext = False, trainable = True, use_additive_emb = False):
    if use_fasttext:
        inp = Input(shape=(maxlen, embed_size))
        x = inp
    else:
        inp = Input(shape = (maxlen, ))
        x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix], trainable = trainable)(inp)

    if use_additive_emb:
        x = AdditiveLayer()(x)
        x = Dropout(0.5)(x)


    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = SeqSelfAttention(
        # attention_type = SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = SeqWeightedAttention()(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def HARNNCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40, use_fasttext = False, trainable = True, use_additive_emb = False):
    if use_fasttext:
        sent_inp = Input(shape = (max_sent_len, embed_size))
        embed = sent_inp
    else:
        sent_inp = Input(shape = (max_sent_len, ))
        embed = Embedding(
            input_dim = max_features,
            output_dim = embed_size,
            weights = [embeddingMatrix],
            trainable = trainable
        )(sent_inp)

    if use_additive_emb:
        embed = AdditiveLayer()(embed)
        embed = Dropout(0.5)(embed)


    word_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(embed)
    word_att = SeqWeightedAttention()(word_lstm)
    sent_encoder = Model(sent_inp, word_att)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(doc_encoder)
    sent_att = SeqWeightedAttention()(sent_lstm)
    preds = Dense(1, activation = "sigmoid")(sent_att)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model



