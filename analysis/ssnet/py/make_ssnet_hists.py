import os,sys

"""
run ssnet on test images and save output to a histogram.
"""

import numpy as np
import torch
import uresnet
from uresnet.models.uresnet_dense import UResNet
from uresnet.flags import URESNET_FLAGS

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
flags.checkpoint_file = "../checkpoint.18000th.tar" # 16 features, images conditioned"
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

# LOAD SAMPLES
images  = np.load("/home/twongj01/data/larcv1_ssnet_data/test_images_for_vqvae/test_images.npy")
print images.shape
# switch to pytorch form
images = images.reshape( (images.shape[0],1,64,64) ).astype(np.float32)
images *= 255.0/10.0
print images.shape,images.dtype

# SETUP OUTPUT
import ROOT as rt
from ROOT  import std
rt.gSystem.Load("../libssnetdata")
from ROOT import ssnet
fout = rt.TFile("out_ssnet_hists.root","recreate")

filler = ssnet.FillScoreHist() # c++ routine to speed up filling histogram
PIXCLASSES = ["bg","shower","track"]
hsample_v = std.vector("TH1D")(3)
for n,classname in enumerate(PIXCLASSES):
    hsample_v[n] = rt.TH1D("hsample_%s"%(classname),"",1000,0,1.0)

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
    for ib in range(bsize):
        npix = filler.fillHist( in_images[ib,0,:,:], pred_t[ib,:,:,:], hsample_v, 10.0 )
   

# Write histograms to disk
for i in range(hsample_v.size()):
    hsample_v[i].Write()

fout.Close()
    
        
    
