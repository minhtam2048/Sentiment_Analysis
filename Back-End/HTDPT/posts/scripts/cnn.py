from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    Conv1D, MaxPool1D, \
    Dropout, BatchNormalization, \
    Concatenate, Flatten, Add
    
from posts.scripts.util import f1
from posts.scripts.net_components import AdditiveLayer


def TextCNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, filter_sizes = {2, 3, 4, 5}, use_fasttext = False, trainable = True, use_additive_emb = False):
    if use_fasttext:
        inp = Input(shape=(maxlen, embed_size))
        x = inp
    else:
        inp = Input(shape = (maxlen, ))
        x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix], trainable = trainable)(inp)

    if use_additive_emb:
        x = AdditiveLayer()(x)
        x = Dropout(0.5)(x)


    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation = 'relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis = 1)(conv_ops)
    # concat = Dropout(0.1)(concat)
    concat = BatchNormalization()(concat)


    conv_2 = Conv1D(128, 5, activation = 'relu')(concat)
    conv_2 = MaxPool1D(5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Dropout(0.1)(conv_2)

    conv_3 = Conv1D(128, 5, activation = 'relu')(conv_2)
    conv_3 = MaxPool1D(5)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    # conv_3 = Dropout(0.1)(conv_3)


    flat = Flatten()(conv_3)

    op = Dense(64, activation = "relu")(flat)
    # op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model


def VDCNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, filter_sizes = {2, 3, 4, 5}, use_fasttext = False, trainable = True, use_additive_emb = False):
    if use_fasttext:
        inp = Input(shape=(maxlen, embed_size))
        x = inp
    else:
        inp = Input(shape = (maxlen, ))
        x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix], trainable = trainable)(inp)

    if use_additive_emb:
        x = AdditiveLayer()(x)
        x = Dropout(0.5)(x)

    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation = 'relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis = 1)(conv_ops)
    # concat = Dropout(0.1)(concat)
    concat = BatchNormalization()(concat)


    conv_2_main = Conv1D(128, 5, activation = 'relu', padding='same')(concat)
    conv_2_main = BatchNormalization()(conv_2_main)
    conv_2_main = Conv1D(128, 5, activation = 'relu', padding='same')(conv_2_main)
    conv_2_main = BatchNormalization()(conv_2_main)
    conv_2 = Add()([concat, conv_2_main])
    conv_2 = MaxPool1D(pool_size = 2, strides = 2)(conv_2)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Dropout(0.1)(conv_2)

    conv_3_main = Conv1D(128, 5, activation = 'relu', padding='same')(conv_2)
    conv_3_main = BatchNormalization()(conv_3_main)
    conv_3_main = Conv1D(128, 5, activation = 'relu', padding='same')(conv_3_main)
    conv_3_main = BatchNormalization()(conv_3_main)
    conv_3 = Add()([conv_2, conv_3_main])
    conv_3 = MaxPool1D(pool_size = 2, strides = 2)(conv_3)
    # conv_3 = BatchNormalization()(conv_3)
    # conv_3 = Dropout(0.1)(conv_3)


    flat = Flatten()(conv_3)

    op = Dense(64, activation = "relu")(flat)
    # op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

