from odtk.model.superclass import *
import tensorflow as tf


class NN():
    def __init__(self, train, test):
        self.train = train
        self.train.remove_feature(self.train.header_info[self.train.time_column])
        self.test = test
        self.test.remove_feature(self.test.header_info[self.train.time_column])
        self.batch_size = 24
        self.hm_epochs = 1000
        self.layer_levels = 1
        self.layer_nodes = [75] * self.layer_levels

        self.input_nodes = self.train.data.shape[1]

        if self.train.occupancy.shape[1] == 1:
            self.n_classes = int(self.train.occupancy.max())
        else:
            self.n_classes = int(self.train.occupancy.shape[1])

    def run(self):

        x = tf.placeholder('float', [None, self.input_nodes])
        y = tf.placeholder('float')

        def neural_network_model(data):
            hidden_layer = []
            layer_result = []
            self.layer_nodes.append(self.input_nodes)

            for hidden_layer_idx in range(self.layer_levels):
                hidden_layer.append({'weights': tf.Variable(
                    tf.zeros([self.layer_nodes[hidden_layer_idx - 1], self.layer_nodes[hidden_layer_idx]])),
                    'biases': tf.Variable(tf.random_normal([self.layer_nodes[hidden_layer_idx]]))})

            output_layer = {'weights': tf.Variable(tf.random_normal([self.layer_nodes[-2], self.n_classes])),
                            'biases': tf.Variable(tf.random_normal([self.n_classes]))}

            layer_result.append(tf.add(tf.matmul(data, hidden_layer[0]['weights']), hidden_layer[0]['biases']))

            layer_result[0] = tf.nn.relu(layer_result[0])

            for hidden_layer_idx in range(1, self.layer_levels):
                layer_result.append(
                    tf.add(tf.matmul(layer_result[hidden_layer_idx - 1], hidden_layer[hidden_layer_idx]['weights']),
                           hidden_layer[hidden_layer_idx]['biases']))

                layer_result[hidden_layer_idx] = tf.nn.relu(layer_result[hidden_layer_idx])

            output = tf.matmul(layer_result[-1], output_layer['weights']) + output_layer['biases']

            return output

        def train_neural_network(x):
            prediction = neural_network_model(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
            optimizer = tf.train.AdamOptimizer().minimize(cost)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch in range(self.hm_epochs):
                    epoch_loss = 0
                    for i in range(int(self.train.data.shape[0] / self.batch_size)):
                        epoch_x = self.train.data[i * self.batch_size:(i + 1) * self.batch_size]
                        epoch_y = self.train.occupancy[i * self.batch_size:(i + 1) * self.batch_size]
                        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                        epoch_loss += c

                    # print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                pred, truth = sess.run([prediction, y], feed_dict={x: self.test.data, y: self.test.occupancy})

                return pred

        return train_neural_network(x)
