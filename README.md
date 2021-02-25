# Installation

1: Create a GPU container with the following Dockerfile:

```
FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update
RUN apt-get install -y vim
RUN python3 -m pip install tfds-nightly
RUN python3 -m pip install tensorflow-probability
```

2: Create or choose a ```.sh``` training script. 
```conv_vqvae_larcv.sh``` in the ```launch_scripts``` 
directory is a good template. Arguments are documented
in the ```argparser.py``` file.


3: Run a training script using ```source [script_name].sh```. 

When loading VQVAE, the following list of custom objects must
be included:

```
vqvae = keras.models.load_model("vqvae_save", 
				custom_objects={"custom_loss":custom_loss,
                                                "latent_loss":custom_loss(config["beta"]),
                                                "zq_norm":zq_norm,
                                                "ze_norm":ze_norm,
                                                "mse_loss":mse_loss,
                                                "accuracy":accuracy})
```
