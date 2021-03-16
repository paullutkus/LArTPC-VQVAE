# SSNet study

This folder contains code to 1) train a track/shower semantic segmentation network
2) apply it to generated/reconstructed images.

We use the scores of the semantic segmentation network (ssnet),
to compare the similarity of training images to generated/reconstructed images.

Track and showers are the two major classes of patterns found in LArTPC images.
We compare the distribution of track and shower scores
as a metric for the similarity of patterns found in the images.

Loads ssnet network.
Re-scaling of ADC distribution probably necessary, so pre-study needed here.


## Training a ssnet

We provide tools to train SSNet using DeepLearnPhysics open data.
We make a C++ class to load larcv images and truth, `ssnet::SSNetDataLoader`.
The class will take the ROOT file and provide a minibatch of images with their truth.
We use ROOT's tools to provide us python bindings for our C++ class.

## Weights

`ssnet.iclr2021.simdl.checkpoint.run3.18000th.tar`: this was used for the simdl workshop.
It was trained on the deeplearnphysics multi-particle semantic segmentation public data set.
Trained with data augmentation.
Pixel values were clamped from [0,500). These values were then map between [0,10).

Can find a copy on mayer in the `share_folder` directory in TMW's home folder.