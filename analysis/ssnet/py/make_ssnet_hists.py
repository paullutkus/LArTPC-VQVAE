import os,sys

"""
run ssnet on test images and save output to a histogram.
"""

import numpy as np
import torch
import uresnet
from uresnet.models.uresnet_dense import UResNet
from uresnet.flags import URESNET_FLAGS

import ROOT as rt
from ROOT  import std
rt.gSystem.Load("../libssnetdata")
from ROOT import ssnet

DATA_DIR="/home/twongj01/data/larcv1_ssnet_data/test_images_for_vqvae/"
data_to_run = {"k256":DATA_DIR+"/samples_k_256.npy",
               "k512_p1":DATA_DIR+"/samples_k_512.npy",
               "k512_p2":DATA_DIR+"/samples_k_512_2.npy",
               "k512_p3":DATA_DIR+"/samples_k_512_3.npy",
               "final_k512_p1":DATA_DIR+"/final_samples_k_512.npy",
               "final_k512_p2":DATA_DIR+"/final_samples_k_512_2.npy",
               "final_k512_p3":DATA_DIR+"/final_samples_k_512_3.npy",
               "final_k512_p4":DATA_DIR+"/final_samples_k_512_4.npy",               
               "test":DATA_DIR+"/test_images.npy"}
               

# Configure network and load it
flags = URESNET_FLAGS()
flags.DATA_DIM = 2
flags.URESNET_FILTERS = 16
flags.URESNET_NUM_STRIDES = 4
flags.SPATIAL_SIZE = 256
flags.NUM_CLASS = 3 # bg, shower, track
flags.LEARNING_RATE = 1.0e-3
flags.WEIGHT_DECAY = 1.0e-4
flags.BATCH_SIZE = 16
flags.checkpoint_file = "../run3/checkpoint.18000th.tar" # 16 features, images conditioned"
flags.DEVNAME = "cuda:0"

DEVICE = torch.device(flags.DEVNAME)

# MODEL
model = UResNet(flags).to(DEVICE)
device_map = {"cuda:0":flags.DEVNAME,
              "cuda:1":flags.DEVNAME}
checkpoint = torch.load( flags.checkpoint_file, map_location=device_map )
model.load_state_dict( checkpoint["state"] )
model.eval()
#print model


def run_sample( input_file, output_file ):

    # LOAD SAMPLES
    images  = np.load(input_file)
    print images.shape
    
    # switch to pytorch form
    images = images.reshape( (images.shape[0],1,64,64) ).astype(np.float32)
    THRESHOLD = 0.2
    #images *= 255.0/10.0
    print images.shape,images.dtype

    # SETUP OUTPUT
    fout = rt.TFile(output_file,"recreate")

    filler = ssnet.FillScoreHist() # c++ routine to speed up filling histogram
    filler.define_hists()
    PIXCLASSES = ["bg","shower","track"]

    # DEFINE LOOP PARAMS
    NIMAGES = images.shape[0]
    NITERS = int(NIMAGES/flags.BATCH_SIZE)
    softmax = torch.nn.Softmax( dim=1 )
    
    for iiter in range(NITERS):

        print "[ITER ",iiter," of ",NITERS,"]"
        
        start_index = flags.BATCH_SIZE*iiter
        end_index = flags.BATCH_SIZE*(iiter+1)

        if end_index>NIMAGES:
            end_index = NIMAGES

        bsize = end_index-start_index

        # prep input tensor
        in_images = images[start_index:end_index,:,:,:]
        images_t = torch.from_numpy( in_images ).float().to(DEVICE)
        with torch.no_grad():
            out_t = model(images_t)
        pred_t = softmax(out_t)
        pred_t[ pred_t>0.999 ] = 0.999
        print "pred_t shape: ",pred_t.shape
        pred_t = pred_t.cpu().numpy()

        # fill histograms via the fillter class
        nabove_tot = 0
        for ib in range(bsize):
            npix = filler.fillInternalHists( in_images[ib,0,:,:],
                                             pred_t[ib,:,:,:],
                                             THRESHOLD )
            nabove_tot += npix
            print "Num above thresh in batch: ",nabove_tot," per image: ",nabove_tot/float(bsize)

    # Write histograms to disk
    fout.Write()
    fout.Close()
    
    return


for sample_name, data_file in data_to_run.items():

    print "====================================="
    print " RUN ",sample_name,": ",data_file
    print "====================================="    
    outfile = "out_ssnet_hists_samples_%s.root"%(sample_name)
    run_sample( data_file, outfile )


    
