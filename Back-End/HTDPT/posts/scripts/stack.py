import numpy as np
from sklearn.model_selection import KFold
from scripts.util import f1


from keras.callbacks import EarlyStopping, ModelCheckpoint

from numpy import save,load
import pickle
import json
import os


class StackedGeneralizerWithHier:
    def __init__(self,model_path,
                 models, hier_models,meta_models,meta_hier_models,
                 meta_learner_model,
                 models_name,hier_models_name):
        self._model_path = model_path
        self._models = models
        self._hier_models = hier_models
        self._meta_models = meta_models
        self._meta_hier_models = meta_hier_models
        self._meta_learner_model = meta_learner_model
        #name
        self._models_name = models_name
        self._hier_models_name = hier_models_name
        return

    def load_pretrained_model(self):
        for i in range(len(self._models)):
            # load parameter
            embedding_mat = load(self._model_path + '/embedding_mat.npy')
            with open(self._model_path + '/embed_size.dat', "rb") as f:
                embed_size = pickle.load(f)
            word_map = json.load(open(self._model_path + '/word_map.csv'))
            self._models[i] = self._models[i](
                embeddingMatrix=embedding_mat,
                embed_size=embed_size,
                max_features=embedding_mat.shape[0]
            )
            exists = os.path.isfile('{}/'.format(self._model_path) + self._models_name[i] + '-models.hdf5')
            if exists:
                self._models[i].load_weights('{}/'.format(self._model_path) + self._models_name[i] + '-models.hdf5')
                print(self._models_name[i] + "-model loaded")

        for i in range(len(self._hier_models)):
            # load parameter
            embedding_mat_sent = load(self._model_path + '/embedding_mat_sent.npy')
            with open(self._model_path + '/embed_size_sent.dat', "rb") as f:
                embed_size_sent = pickle.load(f)
            word_map_sent = json.load(open(self._model_path + '/word_map_sent.csv'))
            with open(self._model_path + '/max_nb_sent.dat', "rb") as f:
                max_nb_sent = pickle.load(f)
            with open(self._model_path + '/max_sent_len.dat', "rb") as f:
                max_sent_len = pickle.load(f)
            self._hier_models[i] = self._hier_models[i](
                embeddingMatrix=embedding_mat_sent,
                embed_size=embed_size_sent,
                max_features=embedding_mat_sent.shape[0],
                max_nb_sent=max_nb_sent,
                max_sent_len=max_sent_len
            )
            exists = os.path.isfile('{}/'.format(self._model_path)+self._hier_models_name[i]+'-models.hdf5')
            if exists:
                self._hier_models[i].load_weights('{}/'.format(self._model_path)+self._hier_models_name[i]+'-models.hdf5')
                print(self._hier_models_name[i] + "-model loaded")

    def load_pretrained_meta_model(self):
        for i in range(len(self._meta_models)):
            # load parameter
            embedding_mat = load(self._model_path + '/embedding_mat.npy')
            with open(self._model_path + '/embed_size.dat', "rb") as f:
                embed_size = pickle.load(f)
            word_map = json.load(open(self._model_path + '/word_map.csv'))
            self._meta_models[i] = self._meta_models[i](
                embeddingMatrix=embedding_mat,
                embed_size=embed_size,
                max_features=embedding_mat.shape[0]
            )
            exists = os.path.isfile('{}/'.format(self._model_path) + self._models_name[i] + '-meta-models.hdf5')
            if exists:
                self._meta_models[i].load_weights('{}/'.format(self._model_path) + self._models_name[i] + '-meta-models.hdf5')
                print(self._models_name[i] + "-meta-model loaded")

        for i in range(len(self._hier_models)):
            # load parameter
            embedding_mat_sent = load(self._model_path + '/embedding_mat_sent.npy')
            with open(self._model_path + '/embed_size_sent.dat', "rb") as f:
                embed_size_sent = pickle.load(f)
            word_map_sent = json.load(open(self._model_path + '/word_map_sent.csv'))
            with open(self._model_path + '/max_nb_sent.dat', "rb") as f:
                max_nb_sent = pickle.load(f)
            with open(self._model_path + '/max_sent_len.dat', "rb") as f:
                max_sent_len = pickle.load(f)
            self._meta_hier_models[i] = self._meta_hier_models[i](
                embeddingMatrix=embedding_mat_sent,
                embed_size=embed_size_sent,
                max_features=embedding_mat_sent.shape[0],
                max_nb_sent=max_nb_sent,
                max_sent_len=max_sent_len
            )
            exists = os.path.isfile('{}/'.format(self._model_path) + self._hier_models_name[i] + '-meta-models.hdf5')
            if exists:
                self._meta_hier_models[i].load_weights('{}/'.format(self._model_path) + self._hier_models_name[i] + '-meta-models.hdf5')
                print(self._hier_models_name[i] + "-meta-model loaded")

    def train_models(self, X, y, X_val, y_val, X_hier, X_hier_val, model_path, epochs, batch_size,
                     patience,):


        for ind in range(len(self._models)):
            checkpoint = ModelCheckpoint(
                filepath='{}/'.format(model_path)+self._models_name[ind]+'-models.hdf5',
                monitor='val_f1', verbose=1,
                mode='max',
                save_best_only=True
            )
            early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
            callbacks_list = [checkpoint, early]
            self._models[ind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
            self._models[ind].fit(
                X, y,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                epochs=epochs,
                batch_size=batch_size
            )
            self._models[ind].load_weights('{}/'.format(model_path)+self._models_name[ind]+'-models.hdf5')

        for ind in range(len(self._hier_models)):
            checkpoint = ModelCheckpoint(
                filepath='{}/'.format(model_path)+self._hier_models_name[ind]+'-models.hdf5',
                monitor='val_f1', verbose=1,
                mode='max',
                save_best_only=True
            )
            early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
            callbacks_list = [checkpoint, early]
            self._hier_models[ind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
            self._hier_models[ind].fit(
                X_hier, y,
                validation_data=(X_hier_val, y_val),
                callbacks=callbacks_list,
                epochs=epochs,
                batch_size=batch_size
            )
            self._hier_models[ind].load_weights('{}/'.format(model_path)+self._hier_models_name[ind]+'-models.hdf5')

    def train_meta_learner_model(self, X, y, X_val, y_val, X_hier, X_hier_val, model_path, epochs,
                         batch_size, patience):
        self._meta_hier_models = self._hier_models
        self._meta_models = self._models
        # Obtain level-1 input from each model:
        meta_input = np.zeros((len(X), len(self._meta_models) + len(self._meta_hier_models)))

        for ind in range(len(self._meta_hier_models)):
            pred = np.zeros(len(X))
            kf = KFold(n_splits=5, shuffle=False)
            # # model = self._hier_models[ind]
            # weights =  self._meta_hier_models[ind].get_weights()


            for train_index, test_index in kf.split(X_hier):
                X_train, X_test = X_hier[train_index], X_hier[test_index]
                y_train, y_test = y[train_index], y[test_index]

                checkpoint = ModelCheckpoint(
                    filepath='{}/'.format(model_path)+self._hier_models_name[ind]+'-meta-models.hdf5',
                    monitor='val_f1', verbose=1,
                    mode='max',
                    save_best_only=True
                )
                early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
                callbacks_list = [checkpoint, early]
                self._meta_hier_models[ind].fit(
                    X_train, y_train,
                    validation_data=(X_hier_val, y_val),
                    callbacks=callbacks_list,
                    epochs=epochs,
                    batch_size=batch_size
                )
                self._meta_hier_models[ind].load_weights('{}/'.format(model_path)+self._hier_models_name[ind]+'-meta-models.hdf5')
                pred[test_index] =  self._meta_hier_models[ind].predict(X_test).reshape(-1)


            meta_input[:, len(self._meta_models) + ind] = pred


        for ind in range(len(self._meta_models)):
            pred = np.zeros(len(X))
            kf = KFold(n_splits=5, shuffle=False)
            # # model = self._models[ind]
            # weights = self._meta_models[ind].get_weights()

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                checkpoint = ModelCheckpoint(
                    filepath='{}/'.format(model_path)+self._models_name[ind]+'-meta-models.hdf5',
                    monitor='val_f1', verbose=1,
                    mode='max',
                    save_best_only=True
                )
                early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
                callbacks_list = [checkpoint, early]
                self._meta_models[ind].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    epochs=epochs,
                    batch_size=batch_size
                )
                self._meta_models[ind].load_weights('{}/'.format(model_path)+self._models_name[ind]+'-meta-models.hdf5')
                pred[test_index] = self._meta_models[ind].predict(X_test).reshape(-1)


            meta_input[:, ind] = pred


        self._meta_learner_model.fit(meta_input, y)
        pkl_filename = "{}/meta-learner-models.pkl".format(model_path)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self._meta_learner_model, file)

    def predict(self, X, X_hier):
        meta_input = self.compute_meta_data(X, X_hier)
        return (self._meta_learner_model.predict(meta_input) > 0.5).astype(np.int8)

    def compute_meta_data(self, X, X_hier):
        prediction = np.zeros((len(X), len(self._models) + len(self._hier_models)))
        for ind in range(len(self._models)):
            pred = self._models[ind].predict(X).reshape(len(X), 1).reshape(-1)
            prediction[:, ind] = pred

        for ind in range(len(self._hier_models)):
            pred = self._hier_models[ind].predict(X_hier).reshape(len(X_hier), 1).reshape(-1)
            prediction[:, len(self._models) + ind] = pred

        return prediction



