import tensorflow as tf
from tensorflow.keras import backend as K


def mse_loss(ground_truth, predictions):
    mse_loss = tf.reduce_mean((ground_truth - predictions)**2, name='mse_loss')
    return mse_loss

def custom_loss(beta):
    def latent_loss(dummy_ground_truth, outputs):
        del dummy_ground_truth
        z_e, z_q = tf.split(outputs, 2, axis=-1)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
        commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
        latent_loss = tf.identity(vq_loss + beta * commit_loss, name='latent_loss')
        return latent_loss

    return latent_loss

def zq_norm(y_true, y_pred):
    del y_true
    _, z_q = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_q, axis=-1))

def ze_norm(y_true, y_pred):
    del y_true
    z_e, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_e, axis=-1))

def accuracy(y_true, y_pred):
    size = int(y_pred.get_shape()[-2])
    k = int(y_pred.get_shape()[-1])
    y_true = tf.reshape(y_true, (-1, size * size))
    y_pred = tf.reshape(y_pred, (-1, size * size, k))
    return K.cast(K.equal(y_true, K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx())
