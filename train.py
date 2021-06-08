import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import datasets.larcv

from setup_model import build_vqvae, build_pixelcnn
from argparser import train_parser
from metrics import accuracy
from utils import nle_ckpt

# import dataset
# import config, set mnist option
def train(config): 
    (ds_train, ds_test) = tfds.load('larcv', split=['train', 'test'],
                                    as_supervised=True, 
                                    shuffle_files=config['shuffle'])

    ds_train = np.array(list(ds_train))[:, 0]
    ds_test = np.array(list(ds_test))[:, 0]

    ds_train = np.array([i.numpy() for i in ds_train])
    ds_train = np.array([i / 255 * 10 for i in ds_train])
    ds_train = np.reshape(ds_train, [-1, 64, 64, 1]).astype(float)    

    ds_test = np.array([i.numpy() for i in ds_test])
    ds_test = np.array([i / 255 * 10 for i in ds_test])
    ds_test = np.reshape(ds_test, [-1, 64, 64, 1]).astype(float)    

    vqvae, vqvae_sampler, encoder, decoder, codes_sampler, get_vqvae_codebook = build_vqvae(config)
    vqvae.summary()
    history = vqvae.fit(x=ds_train, y=ds_train, epochs=config['vqvae_epochs'], 
                        batch_size=config['vqvae_batch_size'], 
                        validation_data=(ds_test, ds_test), verbose=2)

    z_train = encoder.predict(ds_train)
    z_test = encoder.predict(ds_test)

    pixelcnn_prior, prior_sampler = nle_ckpt(config, z_train, z_test, codes_sampler, 2)

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpu']) 
    train(config)

if __name__ == '__main__': main() 
