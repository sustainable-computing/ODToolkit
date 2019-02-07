from odtk.model.superclass import *
import numpy as np


class LSTM(NormalModel):
    def __init__(self, train, test):
        from odtk.modifier.change import change_to_one_hot
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
        from keras.optimizers import Adam
        from numpy import zeros, reshape
        from tqdm import tqdm

        model = Sequential()
        model.add(LSTM(batch_input_shape=(self.batch_size, 1, self.train.data.shape[1]),
                       units=self.cell,
                       return_sequences=True,  # True: output at all steps. False: output as last step.
                       stateful=True))

        # add output layer
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')

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
            predict = predict.reshape((-1, self.train.occupancy.shape[1]))
            final[i * self.batch_size:(i + 1) * self.batch_size, :] = predict

        if self.test.data.shape[0] % self.batch_size:
            predict = model.predict(self.test.data[-self.batch_size:].reshape(self.batch_size, 1,
                                                                              self.test.data.shape[1]),
                                    self.test.data.shape[0])
            predict = predict.reshape((-1, self.train.occupancy.shape[1]))
            final[-self.batch_size:, :] = predict

        print("LSTM done.")

        # return reshape(final.argmax(axis=1), (-1, 1))
        return final


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
        from tqdm import tqdm

        model = Sequential()
        model.add(LSTM(batch_input_shape=(self.batch_size, 1, self.source.data.shape[1]),
                       units=self.cell,
                       return_sequences=True,  # True: output at all steps. False: output as last step.
                       stateful=True))

        # add output layer
        model.add(TimeDistributed(Dense(self.source.occupancy.shape[1])))
        model.compile(optimizer=Adam(self.learn_rate), loss='mse')

        for epoch in tqdm(range(self.hm_epochs)):
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
        for epoch in tqdm(range(self.hm_epochs)):
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

        print("DA-LSTM done.")

        return final


class LSTM2(NormalModel):

    def __init__(self, train, test):
        from odtk.modifier.change import change_to_one_hot
        self.train = train
        # if self.train.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.train)
        self.test = test
        # if self.test.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.test)
        self.hm_epochs = 10
        self.batch_size = 60
        self.hidden_dim = 75
        self.data_dim = len(self.train.header)
        self.learn_rate = 0.0001

        self.whi, self.wxi, self.bi = None, None, None
        self.whf, self.wxf, self.bf = None, None, None
        self.who, self.wxo, self.bo = None, None, None
        self.wha, self.wxa, self.ba = None, None, None
        self.wy, self.by = None, None

    @staticmethod
    def softmax(x):
        x = np.array(x)
        max_x = np.max(x)
        return np.exp(x - max_x) / np.sum(np.exp(x - max_x))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def run(self):
        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy, self.by = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                                             (self.data_dim, self.hidden_dim)), \
                           np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                                             (self.data_dim, 1))

        losses = []
        num_examples = 0

        for epoch in range(self.hm_epochs):
            for i in range(len(self.train.occupancy)):
                self.sgd_step(self.train.data[i, :], self.train.occupancy[i], self.learn_rate)
                num_examples += 1

            loss = self.loss(self.train.data, self.train.occupancy)
            losses.append(loss)
            print('epoch {0}: loss = {1}'.format(epoch + 1, loss))
            if len(losses) > 1 and losses[-1] > losses[-2]:
                self.learn_rate *= 0.5
                print('decrease learning_rate to', self.learn_rate)

    # 初始化 wh, wx, b
    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                               (self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                               (self.hidden_dim, self.data_dim))
        b = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                              (self.hidden_dim, 1))

        return wh, wx, b

    # 初始化各个状态向量
    def _init_s(self, T):
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # input gate
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # forget gate
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # output gate
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # current inputstate
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # hidden state
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # cell state
        ys = np.array([np.zeros((self.data_dim, 1))] * T)  # output value

        return {'iss': iss, 'fss': fss, 'oss': oss,
                'ass': ass, 'hss': hss, 'css': css,
                'ys': ys}

    # 前向传播，单个x
    def forward(self, x):
        # 向量时间长度
        T = len(x)
        # 初始化各个状态向量
        stats = self._init_s(T)

        for t in range(T):
            # 前一时刻隐藏状态
            ht_pre = np.array(stats['hss'][t - 1]).reshape(-1, 1)

            # input gate
            stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], self.sigmoid)
            # forget gate
            stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], self.sigmoid)
            # output gate
            stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], self.sigmoid)
            # current inputstate
            stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], self.tanh)

            # cell state, ct = ft * ct_pre + it * at
            stats['css'][t] = stats['fss'][t] * stats['css'][t - 1] + stats['iss'][t] * stats['ass'][t]
            # hidden state, ht = ot * tanh(ct)
            stats['hss'][t] = stats['oss'][t] * self.tanh(stats['css'][t])

            # output value, yt = softmax(self.wy.dot(ht) + self.by)
            stats['ys'][t] = self.softmax(self.wy.dot(stats['hss'][t]) + self.by)

        return stats

    # 计算各个门的输出
    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        print(x)
        return activation(wh.dot(ht_pre) + wx[:, x].reshape(-1, 1) + b)

    # 预测输出，单个x
    def predict(self, x):
        stats = self.forward(x)
        pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
        return pre_y

    # 计算损失， softmax交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            stats = self.forward(x[i])
            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = stats['ys'][range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss

    # 初始化偏导数 dwh, dwx, db
    def _init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)

        return dwh, dwx, db

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_grad()
        dwho, dwxo, dbo = self._init_wh_wx_grad()
        dwha, dwxa, dba = self._init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.hidden_dim, 1))

        # 前向计算
        stats = self.forward(x)
        # 目标函数对输出 y 的偏导数
        delta_o = stats['ys']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])

            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * self.tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1 - self.tanh(stats['css'][t]) ** 2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t - 1]
            delta_at = delta_ct * stats['iss'][t]

            delta_at_net = delta_at * (1 - stats['ass'][t] ** 2)
            delta_it_net = delta_it * stats['iss'][t] * (1 - stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1 - stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1 - stats['oss'][t])

            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t - 1], x[t])
            dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t - 1], x[t])
            dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t - 1], x[t])
            dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t - 1], x[t])

        return [dwhf, dwxf, dbf,
                dwhi, dwxi, dbi,
                dwha, dwxa, dba,
                dwho, dwxo, dbo,
                dwy, dby]

    # 更新各权重矩阵的偏导数
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby = self.bptt(x, y)

        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf,
                                                         dbf)
        self.whi, self.wxi, self.bi = self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi,
                                                         dbi)
        self.wha, self.wxa, self.ba = self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa,
                                                         dba)
        self.who, self.wxo, self.bo = self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo,
                                                         dbo)

        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db

        return wh, wx, b


class LSTM3(NormalModel):
    def __init__(self, train, test):
        from odtk.modifier.change import change_to_one_hot
        self.train = train
        # if self.train.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.train)
        self.test = test
        # if self.test.occupancy.shape[1] == 1:
        #     change_to_one_hot(self.test)
        self.hm_epochs = 10
        self.batch_size = 1
        self.cell = 10

    def run(self):

        from keras.models import Sequential
        from keras.layers import LSTM, TimeDistributed, Dense, Dropout
        from keras.optimizers import Adam
        from numpy import zeros, reshape
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        train_X = self.train.data.reshape((self.train.data.shape[0], 1, self.train.data.shape[1]))
        train_Y = self.train.occupancy
        test_X = self.test.data.reshape((self.test.data.shape[0], 1, self.test.data.shape[1]))
        test_Y = self.test.occupancy

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

        model = Sequential()
        model.add(LSTM(self.cell,
                       input_shape=(train_X.shape[1], train_X.shape[2]),
                       recurrent_dropout=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(self.train.occupancy.shape[1], activation='sigmoid'))

        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(train_X, train_Y,
                            epochs=self.hm_epochs,
                            batch_size=self.batch_size,
                            validation_data=(train_X, train_Y),
                            verbose=2,
                            shuffle=False)

        final = model.predict(test_X)
        print(final)

        print("LSTM done.")

        return reshape(final.argmax(axis=1), (-1, 1))
