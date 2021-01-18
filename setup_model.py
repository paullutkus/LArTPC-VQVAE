import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from layers import *
from metrics import mse_loss, custom_loss, zq_norm, ze_norm 

##############################
##### Encoder Definition #####
##############################
def encoder_pass(inputs, d, layers=[16, 32]):
    x = inputs
    for i, filters in enumerate(layers):
        x = keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', activation='relu', 
                                strides=(2, 2), name='conv{}'.format(i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
    z_e = keras.layers.Conv2D(filters=d, kernel_size=3, padding='SAME', activation='relu',
                              strides=(1, 1), name='z_e')(x)
    return z_e

##############################
##### Decoder Definition #####
##############################
def decoder_pass(inputs, layers=[32, 16]):
    y = inputs
    for i, filters in enumerate(layers):
        y = keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2),
                                         padding='SAME', activation='relu', 
                                         name='convT{}'.format(i + 1))(y)
    decoder_out = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), 
                                               padding='SAME', activation='relu',
                                               name='output')(y)
    return decoder_out

#################################
##### VQ-VAE  Functionality #####
#################################
def build_vqvae(config):
    k = config['k'] 
    d = config['d']
    img_dim = config['dataset']
    input_shape = (img_dim, img_dim, 1)

    # define encoder
    encoder_inputs = keras.layers.Input(shape=input_shape, name='encoder_inputs')
    z_e = encoder_pass(encoder_inputs, d, layers=config['vqvae_layers'])
    size = int(z_e.get_shape()[1]) 
    config['size'] = size 
    
    # define vq
    vector_quantizer = VectorQuantizer(k, name="vector_quantizer")
    codebook_indices = vector_quantizer(z_e)
    encoder = keras.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

    # define decoder
    decoder_inputs = keras.layers.Input(shape=(size, size, d), name='decoder_inputs')
    decoder_out = decoder_pass(decoder_inputs, layers=config['vqvae_layers'][::-1])
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_out, name='decoder')

    # training layers
    sampling_layer = keras.layers.Lambda(lambda x: vector_quantizer.sample(x), name='sample_from_codebook')
    z_q = sampling_layer(codebook_indices)
    codes = tf.stack([z_e, z_q], axis=-1)
    codes = keras.layers.Lambda(lambda x: x, name='latent_codes')(codes)
    straight_through = keras.layers.Lambda(lambda x: x[1] + tf.stop_gradient(x[0] - x[1]), name='straight_through_estimator') 
    straight_through_zq = straight_through([z_q, z_e])
    reconstructed = decoder(straight_through_zq)
    vqvae = keras.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')

    # inference layers
    codebook_indices = keras.layers.Input(shape=(size, size), name='discrete_codes', dtype=tf.int32)
    z_q = sampling_layer(codebook_indices)
    generated = decoder(z_q)
    vqvae_sampler = keras.Model(inputs=codebook_indices, outputs=generated, name='vq-vae_sampler')

    # what is this for???
    indices = keras.layers.Input(shape=(size, size), name='codes_sampler_inputs', dtype='int32')
    z_q = sampling_layer(indices)
    codes_sampler = keras.Model(inputs=indices, outputs=z_q, name='codes_sampler')

    # get codebook
    indices = keras.layers.Input(shape=(), dtype='int32')
    vector_model = keras.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]),
                               name='get_codebook')
    def get_vqvae_codebook():
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook
    
    vqvae.compile(loss=[mse_loss, custom_loss(config['beta'])], metrics={'latent_codes': [zq_norm, ze_norm]},
                  optimizer=keras.optimizers.Adam(config['vqvae_lr']))
    return vqvae, vqvae_sampler, encoder, decoder, codes_sampler, get_vqvae_codebook

##########################
##### Build PixelCNN #####
##########################
def build_pixelcnn(config, codes_sampler):
    size = config['size']
    pixelcnn_prior_inputs = K.layers.Input(shape=(size, size), name='pixelcnn_prior_inputs', dtype=tf.int32)
    z_q = codes_sampler(pixelcnn_prior_inputs) # maps indices to the actual codebook
    
    v_stack_in, h_stack_in = z_q, z_q
    for i in range(config['pcnn_blocks']):
        mask = 'b' if i > 0 else 'a'
        kernel_size = 3 if i > 0 else 7
        residual = True if i > 0 else False
        v_stack_in, h_stack_in = gated_masked_conv2d(v_stack_in, h_stack_in, num_feature_maps,
                                                     kernel=kernel_size, residual=residual, i=i + 1)

    fc1 = K.layers.Conv2D(filters=config['pcnn_features'], kernel_size=1, name="fc1")(h_stack_in)
    fc2 = K.layers.Conv2D(filters=config['k'], kernel_size=1, name="fc2")(fc1) 
    # activity_regularizer=K.regularizers.l1(l1=7.5e-4)
    # outputs logits for probabilities of codebook indices for each cell

    pixelcnn_prior = K.Model(inputs=pixelcnn_prior_inputs, outputs=fc2, name='pixelcnn-prior')

    # Distribution to sample from the pixelcnn
    dist = tfp.distributions.Categorical(logits=fc2)
    sampled = dist.sample()
    prior_sampler = K.Model(inputs=pixelcnn_prior_inputs, outputs=sampled, name='pixelcnn-prior-sampler')
    return pixelcnn_prior, prior_sampler
