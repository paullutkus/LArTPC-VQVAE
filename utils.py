import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import datasets.larcv
import matplotlib.pyplot as plt

from setup_model import build_vqvae, build_pixelcnn


def nle_ckpt(config, ds_train, ds_test, codes_sampler, interval):
    pixelcnn_prior, prior_sampler = build_pixelcnn(config, codes_sampler) 
    for i in range(config["pcnn_epochs"] // interval):
        if i != 0:
            pixelcnn_prior = keras.models.load_model("pcnn_save_{}".format(i-1), compile=False)
            pixelcnn_prior.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[accuracy],
                                   optimizer=keras.optimizers.Adam(config['pcnn_lr']))

        prior_history = pixelcnn_prior.fit(ds_train, ds_train, epochs=interval, 
                                           batch_size=config['pcnn_batch_size'], verbose=1)
        pixelcnn_prior.save("nle_ckpt_{}".format(i))
        likelihoods = get_likelihoods(pixelcnn_prior, ds_test)
        _,_,_ = nle_hist(likelihoods, "test", save=True)
    return pixelcnn_prior, prior_sampler 

def get_likelihoods(pixelcnn_prior, ds):
    likelihoods = []
    for example in ds[:1000]:
        logits = pixelcnn_prior.predict(example)
        logits_flat = np.reshape(logits, (256, 256))
        true_flat = np.reshape(example, (256))
        log_likelihood = 0
        for idx, logit in enumerate(logits_flat):
            soft_out = tf.nn.log_softmax(logit)
            log_likelihood += soft_out[true_flat[idx]]
        likelihoods.append(-log_likelihood.numpy())
    print("mean likelihood:", np.mean(likelihoods))
    return likelihoods

def get_max_count(n):
    max_counts = 0; idx = 0
    for i in range(len(n)):
        if n[i] > max_counts:
            max_counts = n[i]
            idx = i
    return idx

def nle_hist(likelihoods, ds_name, save=False, fname="unnamed_nle"):
    n, bins, patches = plt.hist(likelihoods, bins=100)
        
    plt.suptitle("{} Log-Likelihood Distribution".format(ds_name))
   
    mode = bins[np.argmax(n)] 
    plt.title("min: {} | mean: {} | mode: {} | max: {}".format(round(float(min(likelihoods)), 2),
                                                    round(float(np.mean(likelihoods)), 2),
                                                    round(float(mode), 2),
                                                    round(float(max(likelihoods)), 1)))
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Counts')

    max_count = get_max_count(n)
    plt.axvline(x = bins[-1], color = 'r', label = "max")
    plt.axvline(x = mode, color = 'r', label = "mode")
    plt.axvline(x = bins[0], color = 'r', label = "min")

    plt.savefig(fname, dpi=300)
    return bins[max_count - 50], bins[max_count + 50], [bins[-3], bins[-2], bins[-1]]
