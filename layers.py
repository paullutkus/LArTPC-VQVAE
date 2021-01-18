import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

###############################
##### Vector Quantization #####
###############################
class VectorQuantizer(keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = keras.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init, trainable=True)
        
    def call(self, inputs):
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        return k_index

    def sample(self, k_index):
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q


####################
##### PixelCNN #####
####################
def gate(inputs):
    x, y = tf.split(inputs, 2, axis=-1)
    return K.tanh(x) + K.sigmoid(y)


class MaskedConv2D(keras.layer.Layer):
    def __init__(self, kernel_size, out_dim, direction, mode, **kwargs):
        self.direction = direction # horizontal or vertical
        self.mode = mode # mask type 'a' or 'b'
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        super(MaskedConv2D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        filter_mid_y = self.kernel_size[0] // 2
        filter_mid_x = self.kernel_size[1] // 2
        in_dim = int(input_shape[-1])
        w_shape = [self.kernel_size[0], self.kernel_size[1], in_dim, self.out_dim]
        mask_filter = np.ones(w_shape, dtype=np.float32)
        # build mask:
        if self.direction == "h":
            mask_filter[(filter_mid_y + 1):, :, :, :] = 0.0
            mask_filter[filter_mid_y, (filter_mid_x + 1):, :, :] = 0.0
        elif self.direction == "v":
            if self.mode == "a":
                mask_filter[filter_mid_y:, :, :, :] = 0.0
            elif self.mode == "b":
                mask_filter[(filter_mid_y + 1):, :, :, :] = 0.0
        if self.mode == "a":
            mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        # build convolution using mask
        self.W = mask_filter * self.add_weight("W_{}".format(self.direction), w_shape, trainable=True)
        self.b = self.add_weight("v_b", [self.out_dim,], trainable=True)

    def call(self, inputs):
        return K.conv2d(inputs, self.W, strides=(1, 1)) + self.b


def gated_masked_conv2d(v_stack_in, h_stack_in, out_dim, kernel, mask="b", residual=True, i=0):
    kernel_size = (kernel // 2 + 1, kernel)
    padding = (kernel // 2, kernel // 2)

    v_stack = keras.layers.ZeroPadding(padding=padding, name="v_pad_{}".format(i))(v_stack_in)
    v_stack = MaskedConv2D(kernel_size, out_dim*2, "v", mask, name="v_masked_conv_{}".format(i))(v_stack)
    v_stack = v_stack[:, :int(v_stack_in.get_shape()[-3]), :, :]
    v_stack_out = keras.layers.Lambda(lambda inputs: gate(inputs), name="v_gate_{}".format(i))(v_stack)

    kernel_size = (1, kernel // 2 + 1)
    padding = (0, kernel // 2)
    h_stack = keras.layers.ZeroPadding2D(padding=padding, name="h_pad_{}".format(i))(h_stack_in)
    h_stack = keras.MaskedConv2D(kernel_size, out_dim*2, "h", mask, name="h_masked_conv_{}".format(i))(h_stack)
    h_stack = h_stack[:, :, :int(h_stack_in.get_shape()[-2]), :]
    h_stack_1 = keras.layers.Conv2D(filter=out_dim*2, kernel_size=1, strides=(1, 1), name="v_to_h_{}".format(i))(v_stack)
    h_stack_out = keras.layers.Lambda(lambda inputs: gate(inputs), name="h_gate_{}".format(i))(h_stack + h_stack_1)

    h_stack_out = keras.layers.Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), name="res_conv_{}".format(i))(h_stack_out)
    if residual:
        h_stack_out += h_stack_in
    return v_stack_out, h_stack_out

