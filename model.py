import tensorflow as tf
from tensorflow import keras

from cells import *


# © Nicolas Vecoven
# Redefinition of the Mean Squared Error to avoid bugs with ragged vectors.
def MSE(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# © Nicolas Vecoven
class RaggedModel(keras.Model):
    """eras model used to deal with the ragged vectors."""
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = y_pred[0]
            loss = MSE(y_pred, y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        update = tf.constant(True)
        for g in gradients:
            if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                update = tf.constant(False)

        if update:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def propagate_data(self, inputs, length, rnn, cell):
        """
        Give some sequences of inputs to the cell and then feed the cell
        with a fixed entry for a given number of timesteps. Finally returns the
        mean variance of the hidden states obtained at the end.
        """
        # First give the inputs to the cell
        states = rnn(inputs, cell.get_initial_state())
        states = states[1:]

        mean_vars = []
        inp = tf.constant([1.0, 0.0, 1.0], shape=(1, inputs.shape[2]))

        # Then give a fixed entry for a certain number of timesteps
        for i in range(length):
            out, states = cell(inp, states)

        states = [states[0]]
        # Finally return the mean variance of the hidden states
        variances = [tf.math.reduce_std(el, axis=0)**2 for el in states]
        mean_vars.append(tf.reduce_mean(variances))
        return tf.reduce_mean(mean_vars)

    def warmup_step(self, inputs, length, rnn, cell, optimizer, lr=1e-1):
        """
        Perform a warmup step: compute the mean variance of the hidden states
        after a certain number of timesteps and then perform a gradient descent
        step on the MSE between 1 and the mean variance.
        """
        with tf.GradientTape() as tape:
            mean_var = self.propagate_data(inputs, length, rnn, cell)

            loss = tf.math.squared_difference(1., mean_var)

        variables = cell.variables
        grad = tape.gradient(loss, variables)

        update = tf.constant(True)
        for g in grad:
            if tf.reduce_any(tf.math.is_nan(g)) \
                or tf.reduce_any(tf.math.is_inf(g)) \
                or tf.reduce_max(tf.abs(g)) > 10.0:

                update = tf.constant(False)

        if update:
            optimizer.apply_gradients(zip(grad, variables))

        return mean_var


class CustomModel:
    """
    Custom model used in the experiments. Contains a single recurrent cell.
    Works with all the cells defined in 'cells.py'.
    """
    def __init__(self, num_states, num_actions, cell, nb_units=32,
                 weights_load_file=None, lr=0.001, return_state=False):
        """Initialize a new model."""
        self.num_states = num_states
        self.num_actions = num_actions
        self.cell = cell
        self.nb_units = nb_units
        self.weights_load_file = weights_load_file
        self.lr = lr
        self.return_state = return_state

        self.define_model()

    def define_model(self):
        """Create and compile the model."""
        input = keras.layers.Input(shape=[None, self.num_states],
                                   dtype=tf.float32,
                                   ragged=True)

        self.cell = self.cell(self.nb_units)
        self.rnn = keras.layers.RNN(self.cell,
                                    return_sequences=True,
                                    return_state=True)

        out = self.rnn(input)
        x = out[0]
        h = out[1:]

        fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
        x = tf.keras.layers.TimeDistributed(fc1)(x)

        fc = tf.keras.layers.Dense(self.num_actions)
        x = tf.keras.layers.TimeDistributed(fc)(x)

        self.model = RaggedModel(inputs=input, outputs=[x, h])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
        )

        if self.weights_load_file is not None:
            self.model.load_weights(self.weights_load_file).expect_partial()

    def reset_state(self):
        """Reset the hidden state."""
        self.cell.hidden_state = None

    def train(self, x, y, epochs=1, batch_size=32):
        """Train the model."""
        self.reset_state()
        x_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in x], axis=0)

        y_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in y], axis=0)

        self.cell.batch_size = batch_size
        self.model.fit(x_ragged, y_ragged, epochs=epochs, batch_size=batch_size)
        self.reset_state()

    def predict(self, x):
        """Predict action probabilities for a single state."""
        self.cell.batch_size = 1
        out, h = self.model(tf.convert_to_tensor([[x]]))

        self.cell.hidden_state = h

        if self.return_state:
            return out.numpy()[0][0], h

        return out.numpy()[0][0]

    def predict_batch(self, x):
        """Predict action probabilities for a batch of states sequences."""
        self.reset_state()
        self.cell.batch_size = len(x)

        x_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in x], axis=0)

        out = self.model(x_ragged)[0]
        self.reset_state()
        return out

    def save_weights(self, file):
        """Save the weights in a file."""
        self.model.save_weights(file)

    def change_hidden(self, ind, val):
        """Modify one value of the hidden state."""
        tmp = self.cell.hidden_state[0].numpy()
        tmp[0, ind] = val
        self.cell.hidden_state[0] = tf.convert_to_tensor(tmp)

    def warmup_multistability(self, memory, batch_size, it, length, lr=1e-1):
        """Pretrain the model using the multistability warmup algorithm."""
        optimizer = tf.keras.optimizers.Adam(lr=lr)

        vars = []
        for cnt in range(1, it+1):
            # Sample some sequences of inputs
            samples = memory.sample(batch_size)
            data = [val[0][:10] for val in samples]

            self.reset_state()
            self.cell.batch_size = len(data)

            data = tf.convert_to_tensor(data, dtype=tf.float32)

            # Choose a random length between 1 and the current upper limit
            rand_leng = np.random.randint(0, min(cnt * 10, length))
            # Do a warmup step
            mean_var = self.model.warmup_step(data, rand_leng, self.rnn,
                                              self.cell, optimizer, lr).numpy()

            vars.append(mean_var)
            print('{:>4}/{} - Variance: {:>4.3f}'.format(cnt, it, mean_var))

        return vars

    def check_bistability(self, inputs, length=5000):
        """
        Compute the mean variance of the hidden states after a certain length
        given a set of input sequences.
        """
        self.reset_state()
        self.cell.batch_size = len(inputs)

        inputs = [i[:9] for i in inputs]
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        return self.model.propagate_data(inputs, length, self.rnn, self.cell)


class DoubleLayerModel:
    """
    Double layer model used in the experiments. Contains two recurrent cells
    placed in parallel, as described in the report. Works with all the cells
    defined in 'cells.py'.
    """
    def __init__(self, num_states, num_actions, cell, nb_units=32,
                 weights_load_file=None, lr=0.001, return_state=False):
        """Initialize a new model."""
        self.num_states = num_states
        self.num_actions = num_actions
        self.type_cell = cell
        self.nb_units = nb_units
        self.weights_load_file = weights_load_file
        self.lr = lr
        self.return_state = return_state

        self.define_model()

    def define_model(self):
        """Create and compile the model."""
        input = keras.layers.Input(shape=[None, self.num_states],
                                   dtype=tf.float32,
                                   ragged=True)

        self.dyn_cell = self.type_cell(self.nb_units)
        self.dyn_rnn = keras.layers.RNN(self.dyn_cell,
                                        return_sequences=True,
                                        return_state=True)

        self.warmup_cell = self.type_cell(self.nb_units)
        self.warmup_rnn = keras.layers.RNN(self.warmup_cell,
                                           return_sequences=True,
                                           return_state=True)

        dyn_out = self.dyn_rnn(input)
        warmup_out = self.warmup_rnn(input)

        dyn_x = dyn_out[0]
        dyn_h = dyn_out[1:]
        warmup_x = warmup_out[0]
        warmup_h = warmup_out[1:]

        # 2 * nb_units outputs
        x = tf.keras.layers.concatenate([dyn_x, warmup_x], axis=2)

        # Hidden state of this model is a tuple containing the hidden states
        # of the two cells.
        h = (dyn_h, warmup_h)

        fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
        x = tf.keras.layers.TimeDistributed(fc1)(x)

        fc = tf.keras.layers.Dense(self.num_actions)
        x = tf.keras.layers.TimeDistributed(fc)(x)

        self.model = RaggedModel(inputs=input, outputs=[x, h])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
        )

        if self.weights_load_file is not None:
            self.model.load_weights(self.weights_load_file).expect_partial()

    def reset_state(self):
        """Reset the hidden state."""
        self.dyn_cell.hidden_state = None
        self.warmup_cell.hidden_state = None

    def train(self, x, y, epochs=1, batch_size=32):
        """Train the model."""
        self.reset_state()
        x_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in x], axis=0)

        y_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in y], axis=0)

        self.dyn_cell.batch_size = batch_size
        self.warmup_cell.batch_size = batch_size
        self.model.fit(x_ragged, y_ragged, epochs=epochs, batch_size=batch_size)
        self.reset_state()

    def predict(self, x):
        """Predict action probabilities for a single state."""
        self.dyn_cell.batch_size = 1
        self.warmup_cell.batch_size = 1
        out, h = self.model(tf.convert_to_tensor([[x]]))

        self.dyn_cell.hidden_state = h[0]
        self.warmup_cell.hidden_state = h[1]

        if self.return_state:
            return out.numpy()[0][0], h

        return out.numpy()[0][0]

    def predict_batch(self, x):
        """Predict action probabilities for a batch of states sequences."""
        self.reset_state()
        self.dyn_cell.batch_size = len(x)
        self.warmup_cell.batch_size = len(x)

        x_ragged = tf.concat([tf.RaggedTensor.from_row_splits(
            values=el, row_splits=[0, len(el)]) for el in x], axis=0)

        out = self.model(x_ragged)[0]
        self.reset_state()
        return out

    def save_weights(self, file):
        """Save the weights in a file."""
        self.model.save_weights(file)

    def change_hidden(self, ind, val):
        """Modify one value of the hidden state."""
        tmp = self.warmup_cell.hidden_state[0].numpy()
        tmp[0, ind] = val
        self.warmup_cell.hidden_state[0] = tf.convert_to_tensor(tmp)

    def warmup_multistability(self, memory, batch_size, it, length, lr=1e-1):
        """Pretrain the model using the multistability warmup algorithm."""
        optimizer = tf.keras.optimizers.Adam(lr=lr)

        vars = []
        for cnt in range(1, it+1):
            # Sample some sequences of inputs
            samples = memory.sample(batch_size)
            data = [val[0][:10] for val in samples]

            self.reset_state()
            self.dyn_cell.batch_size = len(data)
            self.warmup_cell.batch_size = len(data)

            data = tf.convert_to_tensor(data, dtype=tf.float32)

            # Choose a random length between 1 and the current upper limit
            rand_leng = np.random.randint(0, min(cnt * 10, length))
            # Do a warmup step on the cell that has to be pretrained
            mean_var = self.model.warmup_step(
                data, rand_leng, self.warmup_rnn,
                self.warmup_cell, optimizer, lr
            ).numpy()

            vars.append(mean_var)
            print('{:>4}/{} - Variance: {:>4.3f}'.format(cnt, it, mean_var))

        return vars

    def check_bistability(self, inputs, length=5000):
        """
        Compute the mean variance of the hidden states after a certain length
        given a set of input sequences.
        """
        self.reset_state()
        self.warmup_cell.batch_size = len(inputs)

        inputs = [i[:9] for i in inputs]
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        return self.model.propagate_data(inputs, length, self.warmup_rnn,
                                         self.warmup_cell)
