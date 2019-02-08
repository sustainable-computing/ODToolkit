from odtk.model.superclass import *
import numpy as np


class LSTM(NormalModel):
    def __init__(self, train, test):

        self.train = train
        # if self.train.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.train)
        self.test = test
        # if self.test.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.test)
        self.hm_epochs = 10
        self.batch_size = 60
        self.cell = 75
        self.learn_rate = 0.0001

    def run(self):

        from keras.models import Sequential
        from keras.layers import LSTM, TimeDistributed, Dense
        from numpy import zeros, reshape

        model = Sequential()
        model.add(LSTM(batch_input_shape=(self.batch_size, 1, self.train.data.shape[1]),
                       units=self.cell,
                       return_sequences=True,  # True: output at all steps. False: output as last step.
                       stateful=True))

        # add output layer
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')

        for _ in range(self.hm_epochs):
            epoch_loss = 0
            for i in range(int(self.train.data.shape[0] / self.batch_size)):
                epoch_x = self.train.data[i * self.batch_size:
                                          (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                             self.train.data.shape[1])
                epoch_y = self.train.occupancy[i * self.batch_size:
                                               (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                                  self.train.occupancy.shape[1])
                cost = model.train_on_batch(epoch_x, epoch_y)
                epoch_loss += cost

        final = zeros(self.test.occupancy.shape, dtype=float)

        for i in range(int(self.test.data.shape[0] / self.batch_size)):
            predict = model.predict(self.test.data[i * self.batch_size:
                                                   (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                                      self.test.data.shape[1]),
                                    self.test.data.shape[0])
            predict = predict.reshape((-1, self.train.occupancy.shape[1]))
            final[i * self.batch_size:(i + 1) * self.batch_size, :] = predict

        if self.test.data.shape[0] % self.batch_size:
            predict = model.predict(self.test.data[-self.batch_size:].reshape(self.batch_size, 1,
                                                                              self.test.data.shape[1]),
                                    self.test.data.shape[0])
            predict = predict.reshape((-1, self.train.occupancy.shape[1]))
            final[-self.batch_size:, :] = predict

        return reshape(final.argmax(axis=1), (-1, 1))


class DALSTM(DomainAdaptiveModel):
    def __init__(self, source, target_retrain, target_test):
        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test

        self.hm_epochs = 10
        self.batch_size = 60
        self.cell = 75
        self.learn_rate = 0.00001

    def run(self):

        from keras.models import Sequential
        from keras.layers import LSTM, TimeDistributed, Dense
        from keras.optimizers import Adam
        from numpy import zeros

        model = Sequential()
        model.add(LSTM(batch_input_shape=(self.batch_size, 1, self.source.data.shape[1]),
                       units=self.cell,
                       return_sequences=True,  # True: output at all steps. False: output as last step.
                       stateful=True))

        # add output layer
        model.add(TimeDistributed(Dense(self.source.occupancy.shape[1])))
        model.compile(optimizer=Adam(self.learn_rate), loss='mse')

        for _ in range(self.hm_epochs):
            epoch_loss = 0
            for i in range(int(self.source.data.shape[0] / self.batch_size)):
                epoch_x = self.source.data[i * self.batch_size:
                                           (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                              self.source.data.shape[1])
                epoch_y = self.source.occupancy[i * self.batch_size:
                                                (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                                   self.source.occupancy.shape[1])
                cost = model.train_on_batch(epoch_x, epoch_y)
                epoch_loss += cost

        if self.target_retrain is None:
            self.hm_epochs = 0
        for _ in range(self.hm_epochs):
            epoch_loss = 0
            for i in range(int(self.target_retrain.data.shape[0] / self.batch_size)):
                epoch_x = self.target_retrain.data[i * self.batch_size:
                                                   (i + 1) * self.batch_size].reshape(self.batch_size, 1,
                                                                                      self.target_retrain.data.shape[1])
                epoch_y = self.target_retrain.occupancy[i * self.batch_size:
                                                        (i + 1) *
                                                        self.batch_size].reshape(self.batch_size, 1,
                                                                                 self.target_retrain.occupancy.shape[1])
                cost = model.train_on_batch(epoch_x, epoch_y)
                epoch_loss += cost

        final = zeros(self.target_test.occupancy.shape, dtype=float)

        for i in range(int(self.target_test.data.shape[0] / self.batch_size)):
            predict = model.predict(self.target_test.data[i * self.batch_size:
                                                          (i + 1) *
                                                          self.batch_size].reshape(self.batch_size, 1,
                                                                                   self.target_test.data.shape[1]),
                                    self.target_test.data.shape[0])
            predict = predict.reshape((predict.shape[0], self.source.occupancy.shape[1]))
            final[i * self.batch_size:(i + 1) * self.batch_size, :] = predict

        if self.target_test.data.shape[0] % self.batch_size:
            predict = model.predict(self.target_test.data[-self.batch_size:].reshape(self.batch_size, 1,
                                                                                     self.target_test.data.shape[1]),
                                    self.target_test.data.shape[0])
            predict = predict.reshape((predict.shape[0], self.source.occupancy.shape[1]))
            final[-self.batch_size:, :] = predict

        return final
