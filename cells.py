import numpy as np
import tensorflow as tf


# Definitions of the different cells used in the experiments:
# - GRU and LSTM
# - BRC and NBRC (© Nicolas Vecoven, "A bio-inspired bistable recurrent cell
#   allows for long-lasting memory")
# - JANET (© "The Unreasonable Effectiveness of the Forget Gate")

# For each cell, redefinition of the 'get_initial_state' method, called at the
# beginning of the prediction of every sequence. This implementation allows to
# retain the hidden state between two calls.


class GruCellLayer(tf.keras.layers.GRUCell):
    def __init__(self, nb_units):
        super(GruCellLayer, self).__init__(nb_units)

        self.nb_units = nb_units
        self.hidden_state = None
        self.batch_size = None

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if self.hidden_state is None:
            size = self.batch_size if self.batch_size is not None else 1
            return tf.zeros((size, self.nb_units), dtype=dtype)

        return self.hidden_state


class LSTMCellLayer(tf.keras.layers.LSTMCell):
    def __init__(self, nb_units):
        super(LSTMCellLayer, self).__init__(nb_units)

        self.nb_units = nb_units
        self.hidden_state = None
        self.batch_size = None

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if self.hidden_state is None:
            size = self.batch_size if self.batch_size is not None else 1
            return tf.zeros((size, self.nb_units), dtype=dtype), \
                   tf.zeros((size, self.nb_units), dtype=dtype)

        return self.hidden_state


class BistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(BistableRecurrentCellLayer, self).__init__(output_dim, **kwargs)

        self.nb_units = output_dim
        self.hidden_state = None
        self.batch_size = None

    def build(self, input_shape):
        self.kernelz = self.add_weight(
            name="kz",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.kernelr = self.add_weight(
            name="kr",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.kernelh = self.add_weight(
            name="kh",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.memoryz = self.add_weight(
            name="mz",
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(1.0)
        )

        self.memoryr = self.add_weight(
            name="mr",
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(1.0)
        )

        self.br = self.add_weight(
            name="br",
            shape=(self.output_dim,),
            dtype = tf.float32,
            initializer='zeros'
        )

        self.bz = self.add_weight(
            name="bz",
            shape=(self.output_dim,),
            dtype = tf.float32,
            initializer='zeros'
        )

        super(BistableRecurrentCellLayer, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + prev_out * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + prev_out * self.memoryz + self.bz)
        output = z * prev_out + (1.0 - z) * tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)

        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if self.hidden_state is None:
            size = self.batch_size if self.batch_size is not None else 1
            return [tf.zeros((size, self.nb_units), dtype=dtype)]

        return self.hidden_state

    def get_config(self):
        cfg = super().get_config()
        return cfg


class NeuromodulatedBistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        # The nBRC has only one vector for the hidden state, the two other
        # vectors added to the hidden state are the values of the gates 'z'
        # and 'r', which are only used for the experiment.
        self.state_size = [output_dim, output_dim, output_dim]

        super(NeuromodulatedBistableRecurrentCellLayer, self).__init__(output_dim, **kwargs)

        self.nb_units = output_dim
        self.hidden_state = None
        self.batch_size = None

    def build(self, input_shape):
        self.kernelz = self.add_weight(
            name="kz",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.kernelr = self.add_weight(
            name="kr",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.kernelh = self.add_weight(
            name="kh",
            shape=(input_shape[1], self.output_dim),
            dtype=tf.float32,
            initializer='glorot_uniform'
        )

        self.memoryz = self.add_weight(
            name="mz",
            shape=(self.output_dim, self.output_dim),
            dtype=tf.float32,
            initializer='orthogonal'
        )

        self.memoryr = self.add_weight(
            name="mr",
            shape=(self.output_dim, self.output_dim),
            dtype=tf.float32,
            initializer='orthogonal'
        )

        self.br = self.add_weight(
            name="br",
            shape=(self.output_dim,),
            dtype = tf.float32,
            initializer='zeros'
        )

        self.bz = self.add_weight(
            name="bz",
            shape=(self.output_dim,),
            dtype = tf.float32,
            initializer='zeros'
        )

        super(NeuromodulatedBistableRecurrentCellLayer, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + tf.matmul(prev_out, self.memoryz) + self.bz)
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + tf.matmul(prev_out, self.memoryr) + self.br)+1
        h = tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out

        # For the experiment, returns also the values of 'z' and 'r'
        return output, [output, z, r]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if self.hidden_state is None:
            size = self.batch_size if self.batch_size is not None else 1
            return tf.zeros((size, self.nb_units), dtype=dtype), \
                   tf.zeros((size, self.nb_units), dtype=dtype), \
                   tf.zeros((size, self.nb_units), dtype=dtype)

        return self.hidden_state


class JANETCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = [output_dim, output_dim]
        super(JANETCellLayer, self).__init__(output_dim, **kwargs)

        self.nb_units = output_dim
        self.hidden_state = None
        self.batch_size = None

    def build(self, input_shape):
        self.kernelf = self.add_weight(name="kf", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelc = self.add_weight(name="kc", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryf = self.add_weight(name="mf", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='orthogonal')
        self.memoryc = self.add_weight(name="mc", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='orthogonal')

        self.bf = self.add_weight(name="bf", shape=(self.output_dim,), dtype = tf.float32, initializer=self.bias_initializer())
        self.bc = self.add_weight(name="bc", shape=(self.output_dim,), dtype = tf.float32, initializer=self.bias_initializer())

        super(JANETCellLayer, self).build(input_shape)

    def call(self, input, states):
        inp = input
        hprev = states[0]
        cprev = states[1]
        beta = 1
        s = tf.matmul(inp, self.kernelf) + tf.matmul(hprev, self.memoryf) + self.bf
        cbar = tf.nn.tanh(tf.matmul(inp, self.kernelc) + tf.matmul(hprev, self.memoryc) + self.bc)
        c = tf.nn.sigmoid(s) * cprev + (1 - tf.nn.sigmoid(s-beta)) * cbar
        h = c
        return h, [h, c]

    def get_config(self):
        return {"output_dim": self.output_dim}

    def bias_initializer(self):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            p = np.zeros(shape)
            num_units = int(shape[0] // 2)
            p[-num_units:] = np.ones(num_units)
            return tf.constant(p, dtype)

        return _initializer

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if self.hidden_state is None:
            size = self.batch_size if self.batch_size is not None else 1
            return tf.zeros((size, self.nb_units), dtype=dtype), \
                   tf.zeros((size, self.nb_units), dtype=dtype)

        return self.hidden_state
