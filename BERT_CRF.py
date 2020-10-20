import numpy as np
import pandas as pd
import pickle as pkl
import os
import tqdm
import json

import bert
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from CRF import CRF


class MyBERTCRF:

    def __init__(self,
                 preModelPath="chinese_L-12_H-768_A-12",  # 预训练模型路径
                 learning_rate=0.00001,  # 学习率
                 maxLen=5,  # 句子最大长度
                 yList=["PAD", "B", "I", "O", "START", "END"]  # tag的list
                 ):
        self.preModelPath = preModelPath
        self.learning_rate = learning_rate
        self.maxLen = maxLen
        self.buildVocab()
        self.XTokenizer = bert.bert_tokenization.FullTokenizer(
            os.path.join(self.preModelPath, "vocab.txt"), do_lower_case=True)
        self.yList = yList

        self.buildModel()

    def buildVocab(self):
        print("building vocabulary ...")
        with open(os.path.join(self.preModelPath, "vocab.txt"), "r", encoding="utf8") as vocabFile:
            self.XVocabList = [row.strip() for row in tqdm.tqdm(vocabFile)]
            self.XVocabSize = len(self.XVocabList)

    def removeUNK(self, seqList):
        return [[wItem for wItem in row if wItem in self.XVocabList] for row in seqList]

    def buildModel(self):

        inputLayer = tf.keras.layers.Input(shape=(self.maxLen,), dtype='int32')

        bert_params = bert.params_from_pretrained_ckpt(self.preModelPath)
        bertLayer = bert.BertModelLayer.from_params(
            bert_params, name="bert")(inputLayer)

        crf = CRF(len(self.yList), name='crf_layer')
        crfLayer = crf(bertLayer)

        self.model = tf.keras.models.Model(inputLayer, crfLayer)
        self.model.compile(loss=crf.get_loss,
                           optimizer=SGD(learning_rate=self.learning_rate))

    def fit(self, X, y, epochs=1, preEpochs=3, batch_size=64):
        '''
        X:cutted seq
        y:cutted y
        '''

        X = np.array([self.XTokenizer.convert_tokens_to_ids(
            [wItem for wItem in row if wItem in self.XVocabList]) for row in X])
        X = np.array([row[:self.maxLen]+[0]*max(0, (self.maxLen-len(row)))
                      for row in X.tolist()])

        y = [[self.yList.index(tagItem) for tagItem in row] for row in y]
        y = np.array(
            [row[:self.maxLen]+[0]*max(0, (self.maxLen-len(row))) for row in y])

        print("fine tuning ...")
        self.model.fit(X, y, epochs=preEpochs, batch_size=batch_size)
        self.model.layers[1].trainable = False

        print("fitting ...")
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
        self.model.fit(X, y, epochs=epochs,
                       batch_size=3*batch_size, callbacks=[es])

    def predict(self, X):

        X = np.array([self.XTokenizer.convert_tokens_to_ids(
            [wItem for wItem in row if wItem in self.XVocabList]) for row in X])
        X = np.array([row+[0]*(self.maxLen-len(row)) if len(row) <
                      self.maxLen else row[:self.maxLen] for row in X.tolist()])

        preY = self.model.predict(X)
        preYTags = [[self.yList[tagItem] for tagItem in row] for row in preY]

        return preY, preYTags


if __name__ == "__main__":

    myDf = pd.read_csv("data/test.csv")

    print("preprocessing ...")
    myDf["text"] = myDf["text"].apply(lambda row: "[START] "+row)
    myDf["text"] = myDf["text"].apply(lambda row: row+" [END]")
    myDf["tag"] = myDf["tag"].apply(lambda row: "START "+row)
    myDf["tag"] = myDf["tag"].apply(lambda row: row+" END")

    print("building X and y ...")
    X = [row.split(" ") for row in myDf["text"].values]
    maxLen = max(len(row) for row in X)
    y = [row.split(" ") for row in myDf["tag"].values]

    print("training model ...")
    myBERTCRF = MyBERTCRF(learning_rate=0.00001, maxLen=maxLen)
    myBERTCRF.fit(X, y, epochs=150)

    print("predicting ...")
    preY, preYTags = myBERTCRF.predict(X)

    print("evaluating ...")
    print("preY:\n", preY)
    print("preYTags:\n", preYTags)
    print("f1:", f1_score(
        np.array([[myBERTCRF.yList.index(tagItem) for tagItem in row[:maxLen]+["PAD"]*max(maxLen-len(row), 0)] for row in y]).flatten(), preY.flatten(), average="macro"))

    print("saving model ...")
    myBERTCRF.model.save_weights("model/BERTCRF")
    with open("model/hyperParams.pkl", "wb+") as hpFile:
        hp = {
            "learning_rate": myBERTCRF.learning_rate,
            "maxLen": maxLen,
            "yList": myBERTCRF.yList
        }
        pkl.dump(hp, hpFile)
