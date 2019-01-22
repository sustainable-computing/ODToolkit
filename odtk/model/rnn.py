from odtk.model.superclass import *


class RNN(NormalModel):
    def __init__(self, train, test):
        self.train = train
        self.train.remove_feature(self.train.header_info[self.train.time_column])
        self.test = test
        self.test.remove_feature(self.test.header_info[self.train.time_column])

        self.hm_epochs = 1000
        self.batch_size = 60
        self.cell = 75
        self.learn_rate = 0.0001

    def run(self):

        from keras.models import Sequential
        from keras.layers import LSTM, TimeDistributed, Dense
        from keras.optimizers import Adam
        from numpy import zeros
        from tqdm import tqdm

        model = Sequential()
        model.add(LSTM(batch_input_shape=(self.batch_size, 1, self.train.data.shape[1]),
                       units=self.cell,
                       return_sequences=True,  # True: output at all steps. False: output as last step.
                       stateful=True))

        # add output layer
        model.add(TimeDistributed(Dense(self.train.occupancy.shape[1])))
        model.compile(optimizer=Adam(self.learn_rate), loss='mse')

        for epoch in tqdm(range(self.hm_epochs)):
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
            predict = predict.reshape((predict.shape[0], self.train.occupancy.shape[1]))
            final[i * self.batch_size:(i + 1) * self.batch_size, :] = predict

        if self.test.data.shape[0] % self.batch_size:
            predict = model.predict(self.test.data[-self.batch_size:].reshape(self.batch_size, 1,
                                                                              self.test.data.shape[1]),
                                    self.test.data.shape[0])
            predict = predict.reshape((predict.shape[0], self.train.occupancy.shape[1]))
            final[-self.batch_size:, :] = predict

        return final
