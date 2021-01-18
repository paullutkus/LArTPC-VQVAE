############################################################################
# argparser.py
# Author: Kai Stewart
# Organization: Tufts University
# Department: Physics
# Date: 10.08.2019
# Purpose: - This python script is designed to provide model-agnostic command
#            line argument parsing capabilities for PyTorch models associated
#            with the particle generator project.
#          - Functionality includes instantiation of argument parser objects
#            both for training and deploying PyTorch models.
############################################################################

# System Imports
from argparse import ArgumentParser

# Training argument parser function
def train_parser():
    '''
        Argument Parser Function for model training
        Does: prepares a command line argument parser object with
              functionality for training a generative model
        Args: None
        Returns: argument parser object
    '''
    usage = "Command line arguement parser for set up " + \
            "of PyTorch particle generator model training."
    parser = ArgumentParser(description=usage)

    # model: string that selects the type of model to be trained
    #        options: GAN, AE, EWM
    parser.add_argument('--model', type=str, default='res',
                        help='String that selects the model - options: \
                            fc, conv, res | (default: &(default)s)')
    # checkpoint: string path to saved model checkpoint. If used with
    #             train function, model will resume training from checkpoint
    #             If used with deploy function, model will used saved weights.
    parser.add_argument('--checkpoint', type=str, default='',
                        help='String path to saved model checkpoint. If used \
                            with training function, model will resume trainig \
                                from that checkpoint. If used with deploy \
                                    function, model will deploy with save weights. \
                                        | (default: &(default)s) -- not implmented \
                                            yet.')

    ###################################################### 
    ################## Data Loading ######################
    ######################################################
    
    # MNIST: For proof-of-concept purposes. Overrides other data loading
    #        options and uses the built-in torch MNIST dataset loading function
    parser.add_argument('--MNIST', type=bool, default=False,
                        help='Toggle to train model on MNIST dataset. Overrides \
                            other data loading options and uses the built-in \
                                torch MNIST dataset loading functionality | \
                                    (default: &(default)s)')
    # save_root: path where training output is saved
    parser.add_argument('--save_root', type=str, default='/train_save',
                        help='Path where training output should be saved \
                            | (default: &(default)s)')
    # dataset: - which LArCV1 dataset to use (512, 256, 128, 64, 32)
    parser.add_argument('--dataset', type=int, default=64,
                        help='Which crop size of the LArCV1 dataset to use, or \
                            | (default: &(default)s) -- currently only \
                            supports 64')
    
    #########################################
    ## Environment and Hyperparameter Args ##
    #########################################

    # sample_size: number of samples to generate during training
    parser.add_argument('--sample_size', type=int, default=8,
                        help='Number of image samples to be generated during\
                            training (progress check) | (default: &(default)s) \
                            -- not implemented yet.')
    # gpu: which GPU to train the model(s) on
    parser.add_argument('--gpu', type=int, default=0,
                        help='Select gpu to use for training. If multi-gpu \
                            option is selected, then this option is ignored \
                                | (default: &(default)s)')
    # multi_gpu: toggle whether to train on multiple GPU's (if available)
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Select whether to use multiple GPUs to train \
                            model. This model overrides the --gpu flag \
                                | (default: &(default)s) -- not implemented \
                                    yet.')
    # shuffle: toggle shuffle on/off
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Toggle dataloader batch shuffle on/off \
                            | (default: &(default)s)')
    # drop_last: toggle drop last batch on/off if dataset
    #            size not divisible by batch size
    parser.add_argument('--drop_last', type=bool, default=False,
                        help='Toggle whether the dataloader should drop \
                            the last batch, if the dataset size is not \
                                divisible by the batch size \
                                    | (default: &(default)s)')
    # num_workers: number of worker threads for data io
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Set number of worker threads for data io \
                            | (default: &(default)s) -- note implemented \
                                yet')

    ######################################################
    ################# Model settings #####################
    ######################################################
    
    ### VQ-AE Network
    # k: number of embedding vectors
    parser.add_argument('--k', type=int, default=256,
                        help='Number of embedding vectors \
                            | (default: &(default)s)')
    # d: dimension of embedding vectors
    parser.add_argument('--d', type=int, default=8,
                        help='Dimension of embedding vectors \
                            | (default: &(default)s)')
    # beta: coefficient of commitment loss
    parser.add_argument('--beta', type=float, default=1,
                        help='Coefficient of commitment loss \
                            | (default: &(default)s)')
    # vqvae_batch_size: batch size for vq-vae training
    parser.add_argument('--vqvae_batch_size', type=int, default=512,
                        help='Batch size for VQ-VAE training \
                            | (default: &(default)s)')
    # vqvae_epochs: number of epochs in vq-vae training
    parser.add_argument('--vqvae_epochs', type=int, default=50,
                        help='Number of epochs in VQ-VAE training \
                            | (default: &(default)s)')
    # vqvae_lr: learning rate for vq-vae optimizer
    parser.add_argument('--vqvae_lr', type=float, default=3e-4,
                        help='Learning rate of VQ-VAE optimizer \
                            | (default: &(default)s)')
    # vqvae_layers: layer filters for conv. and res. autoencoders
    parser.add_argument('--vqvae_layers', nargs="*", type=int, default=[16, 32],
                        help='Filter sizes for conv. and res. encoder \
                            and decoder | (default: &(default)s)')

    ### PixelCNN Network
    # pcnn_batch_size: batch size for pixelcnn training
    parser.add_argument('--pcnn_batch_size', type=int, default=256,
                        help='Batch size for PixelCNN training \
                            | (default: &(default)s)')
    # pcnn_epochs: number of epochs in vq-vae training
    parser.add_argument('--pcnn_epochs', type=int, default='15',
                        help='Number of epochs in PixelCNN training \
                            | (default: &(default)s)')
    # pcnn_lr: learning rate for pixelcnn optimizer
    parser.add_argument('--pcnn_lr', type=float, default=1e-3,
                        help='Learning rate of PixelCNN optimizer \
                            | (default: &(default)s)')
    # pcnn_blocks: number of blocks in pixelcnn
    parser.add_argument('--pcnn_blocks', type=int, default=3,
                        help='Number of blocks in PixelCNN optimizer \
                            | (default: &(default)s)')
    # pcnn_features: number of features in pixelcnn blocks
    parser.add_argument('--pcnn_features', type=int, default=512,
                        help='Number of features in PixelCNN blocks \
                            | (default: &(default)s)')

    return parser

# Deploy model argument parser function
def deploy_parser():
    '''
        Argument Parser Function for model training
        Does: prepares a command line argument parser object with
              functionality for deploying a trained generative model.
        Args: None
        Returns: argument parser object
    '''
    usage = "Command line arguement parser for deploying " + \
            "trained PyTorch particle generator models."
    parser = ArgumentParser(description=usage)

    # TODO: Write the parser arguments after deciding on a convention for how
    # model outputs, checkpoints, and experiments will be saved.

    return parser
