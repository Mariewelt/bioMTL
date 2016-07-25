import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import batch_norm, DropoutLayer
from sklearn.metrics import roc_auc_score, roc_curve

class DNN(object):
    def __init__(self, input_X, target, X_train, y_train, X_test, y_test):
        self.input_X = input_X
        self.target = target
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def iterate_minibatches(self, X, y, batchsize):
        n = X.shape[0]
        ind = np.random.permutation(n)
        for start_index in range(0, n, batchsize):
            X_batch = X[ind[start_index:start_index + batchsize], :]
            y_batch = y[ind[start_index:start_index + batchsize], :]
            yield (X_batch, y_batch)

    def build_ae(self, l_input, HU_enc, HU_dec, dimZ, n_out):
        l_enc = lasagne.layers.DenseLayer(l_input, num_units=HU_enc,
                                          nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_z = lasagne.layers.DenseLayer(l_enc, num_units=dimZ, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_dec = lasagne.layers.DenseLayer(l_z, num_units=HU_dec, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_out = lasagne.layers.DenseLayer(l_dec, num_units=n_out, nonlinearity=lasagne.nonlinearities.linear)
        return l_z, l_out

    def build_nn(self, l_obs):
        dense_layer = batch_norm(lasagne.layers.DenseLayer(l_obs,
                                                           num_units=64, nonlinearity=lasagne.nonlinearities.tanh))
        dense_layer = DropoutLayer(dense_layer)
        output_layer = batch_norm(lasagne.layers.DenseLayer(dense_layer, num_units=self.y_test.shape[1],
                                                            nonlinearity=lasagne.nonlinearities.sigmoid))
        return output_layer

    def build_ae_loss(self, l_out, learning_rate=0.01):
        # create prediction variable
        prediction = lasagne.layers.get_output(l_out)

        # create loss function
        loss = lasagne.objectives.squared_error(prediction, self.input_X).mean()

        # create parameter update expressions
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

        # compile training function that updates parameters and returns training loss
        # this will take a while
        train_fn = theano.function([self.input_X], loss, updates=updates)
        test_fn = theano.function([self.input_X], loss)
        pred_fn = theano.function([self.input_X], prediction)
        return train_fn, test_fn, pred_fn

    def build_nn_loss(self, l_out, learning_rate=0.1):
        # create prediction variable
        output = lasagne.layers.get_output(l_out, deterministic=False)
        prediction = lasagne.layers.get_output(l_out, deterministic=True)

        # create loss function
        loss = T.switch(T.neq(self.target,999),lasagne.objectives.binary_crossentropy(output,self.target),0).sum()

        # create parameter update expressions
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        updates = lasagne.updates.adadelta(loss, params, learning_rate=learning_rate)

        # compile training function that updates parameters and returns training loss
        # this will take a while
        train_fn = theano.function([self.input_X, self.target], loss, updates=updates)
        test_fn = theano.function([self.input_X, self.target], loss)
        pred_fn = theano.function([self.input_X], prediction)
        return train_fn, test_fn, pred_fn

    def fit_ae(self, train_fn, test_fn, X_train=None, y_train=None, X_test=None, y_test=None,
               train_loss_log=[], val_loss_log=[], num_epochs=100, batch_size=1000):
        import time

        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            #if epoch % 10 == 0:
            #    print train_fn(self.X_train)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, batch_size):
                inputs, targets = batch
                train_err_batch = train_fn(inputs)
                train_err += train_err_batch
                train_batches += 1

            # And a full pass over the validation data:
            val_loss = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_test, y_test, batch_size):
                inputs, targets = batch
                val_loss += test_fn(inputs)
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))

            train_loss_log.append(train_err / train_batches / batch_size)
            val_loss_log.append(val_loss / val_batches / batch_size)
            print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches / batch_size))
            # print("  train accuracy:\t\t{:.2f} %".format(
            #    train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.2f}".format(
                val_loss / val_batches / batch_size))


        return train_loss_log, val_loss_log

    def fit_nn(self, train_fn, test_fn, X_train=None, y_train=None, X_test=None, y_test=None,
               train_loss_log=[], val_loss_log=[], num_epochs=100, batch_size=1000):
        import time

        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        for epoch in range(num_epochs):
            #if epoch % 10 == 0:
            #    print train_fn(X_train, y_train)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, batch_size):
                inputs, targets = batch
                train_err_batch = train_fn(inputs, targets)
                train_err += train_err_batch
                train_batches += 1

            # And a full pass over the validation data:
            val_loss = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_test, y_test, batch_size):
                inputs, targets = batch
                val_loss += test_fn(inputs, targets)
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))

            train_loss_log.append(train_err / train_batches / batch_size)
            val_loss_log.append(val_loss / val_batches / batch_size)
            print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches / batch_size))
            # print("  train accuracy:\t\t{:.2f} %".format(
            #    train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.2f}".format(
                val_loss / val_batches / batch_size))


        return train_loss_log, val_loss_log

    def cross_val(self,  X, y, nn_train, nn_test, ae_train, ae_test, pred_fn, k=5):
        score = []
        n = X.shape[0]
        start_ind = 0
        step = int(n / k)
        end_ind = step
        input_shape = (None, X.shape[1])
        l_input = lasagne.layers.InputLayer(shape=input_shape, input_var=self.input_X)
        for j in xrange(k):
            score_cur = np.zeros(y.shape[1])
            test_ind = np.arange(start_ind,end_ind)
            train_ind = np.concatenate((np.arange(0, start_ind), np.arange(end_ind, n)))
            X_train = X[train_ind, :]
            y_train = y[train_ind, :]
            X_test = X[test_ind, :]
            y_test = y[test_ind, :]
            l_z, l_ae_out = self.build_ae(l_input=l_input, HU_enc=128, HU_dec=128, dimZ=32, n_out=X.shape[1])
            _, _ = self.fit_ae(ae_train, ae_test, X_train, y_train, X_test, y_test, num_epochs=100)
            l_nn_out = self.build_nn(l_z)
            _, _ = self.fit_nn(nn_train, nn_test, X_train, y_train, X_test, y_test, num_epochs=100)
            y_pred = pred_fn(X_test)
            for i in range(y.shape[1]):
                ind = np.where(y_test[:, i] != 999)
                fpr, tpr, _ = roc_curve(y_test[ind, i][0], y_pred[ind, i][0])
                score_cur[i] =  roc_auc_score(y_test[ind, i][0], y_pred[ind, i][0])
            score.append(score_cur)
            start_ind = end_ind
            end_ind = end_ind + step
            if end_ind > n:
                end_ind = n
        return score




