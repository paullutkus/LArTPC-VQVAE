import os,sys
sys.path.append("uresnet_pytorch")
import uresnet

from uresnet.models.uresnet_dense import UResNet
from uresnet.models.uresnet_dense import SegmentationLoss
from uresnet.flags import URESNET_FLAGS

import torch
from torch.utils.tensorboard import SummaryWriter
import ROOT as rt
import numpy as np
from larcv import larcv
rt.gSystem.Load("libssnetdata.so")
from ROOT import ssnet

print "Train UResNet"

flags = URESNET_FLAGS()
flags.DATA_DIM = 2
flags.URESNET_FILTERS = 16
flags.URESNET_NUM_STRIDES = 4
flags.SPATIAL_SIZE = 256
flags.NUM_CLASS = 3 # bg, shower, track
flags.LEARNING_RATE = 1.0e-3
flags.WEIGHT_DECAY = 1.0e-4
flags.BATCH_SIZE = 16

DEVICE = torch.device("cuda:0")

# MODEL
model = UResNet(flags).to(DEVICE)
print model

# LOSS FUNCTION
lossfunc = SegmentationLoss(flags)

# DATA LOADER
train_loader = ssnet.SSNetDataLoader()
train_loader.set_read_mode( ssnet.SSNetDataLoader.kRandomSubBatch )
train_loader.setup( "~/data/larcv1_ssnet_data/train_15k.root", 0, True )

valid_loader = ssnet.SSNetDataLoader()
valid_loader.set_read_mode( ssnet.SSNetDataLoader.kRandomSubBatch )
valid_loader.setup( "~/data/larcv1_ssnet_data/test_10k.root", 0, True )

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(),
                             lr=flags.LEARNING_RATE,
                             weight_decay=flags.WEIGHT_DECAY)

# TENSORBOARD
writer = SummaryWriter()

NUMITERS = 25000
ITERS_PER_VALID = 10
ITERS_PER_TRAIN_STDOUT = 5
ITERS_PER_EPOCH = 40e3/flags.BATCH_SIZE
ITERS_PER_CHECKPOINT = 1000
weight_t = None
def calc_learning_rate( startlr, epoch ):
    lr = startlr*(0.5**(epoch//4))
    return lr

for iiter in range(NUMITERS):

    # adjust learning rate
    epoch = iiter/float(ITERS_PER_EPOCH)
    lr = calc_learning_rate( flags.LEARNING_RATE, epoch )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # training forward pass
    traindata = train_loader.makeTrainingDataDict( flags.BATCH_SIZE, 2 )
    # make weights
    image_t = torch.from_numpy( traindata['image_t'] ).to(DEVICE)
    label_t = torch.from_numpy( traindata['label_t'] ).long().to(DEVICE)
    w_t = torch.ones( label_t.shape ).to(DEVICE)
    num_zeros  = (label_t == 0).sum().float()
    num_shower = (label_t == 1).sum().float()
    num_track  = (label_t == 2).sum().float()
    w_t[ label_t==0 ] *= 1.0/num_zeros
    w_t[ label_t==1 ] *= 1.0/num_shower
    w_t[ label_t==2 ] *= 1.0/num_track   
    #print "  num shower: ",num_shower
    #print "  num track: ",num_track

    model.train()
    optimizer.zero_grad()
    
    out_t = model.forward( image_t )

    loss,acc = lossfunc.forward( out_t, image_t, label_t, w_t )
    loss /= float(flags.BATCH_SIZE)
    if iiter%ITERS_PER_TRAIN_STDOUT==0:
        print "[TRAIN ITER: ",iiter,"]"        
        print "  loss: ",loss
        print "  acc: ",acc/float(flags.BATCH_SIZE)
        writer.add_scalars('data/train_loss', {"loss":loss.item()}, epoch )
        writer.add_scalars('data/train_acc', {"acc":acc}, epoch )

    loss.backward()
    optimizer.step()

    
    if iiter%ITERS_PER_VALID==0:
        print "[VALID ITER: ",iiter,"]"
        model.eval()
        validdata = valid_loader.makeTrainingDataDict( flags.BATCH_SIZE, 2 )
        vimage_t = torch.from_numpy( validdata['image_t'] ).to(DEVICE)
        vlabel_t = torch.from_numpy( validdata['label_t'] ).to(DEVICE)
        with torch.no_grad():
            vout_t = model.forward( vimage_t )
            vw_t = torch.ones( vlabel_t.shape ).to(DEVICE)
            vnum_zeros  = (vlabel_t == 0).sum().float()
            vnum_shower = (vlabel_t == 1).sum().float()
            vnum_track  = (vlabel_t == 2).sum().float()
            vw_t[ vlabel_t==0 ] *= 1.0/vnum_zeros
            vw_t[ vlabel_t==1 ] *= 1.0/vnum_shower
            vw_t[ vlabel_t==2 ] *= 1.0/vnum_track   
            
            vloss, vacc = lossfunc.forward( vout_t, vimage_t, vlabel_t, vw_t )
            vloss /= float(flags.BATCH_SIZE)
            vacc /= float(flags.BATCH_SIZE)
            print "  valid-loss: ",vloss
            print "  valid-acc: ",vacc
            writer.add_scalars('data/valid_loss', {"loss":vloss.item()}, epoch )
            writer.add_scalars('data/valid_acc', {"acc":vacc}, epoch )
            

    if iiter%ITERS_PER_CHECKPOINT==0 and iiter>0:
        filename = "checkpoint.%dth.tar"%(iiter)
        state = {"iter":iiter,
                 "epoch":iiter/ITERS_PER_EPOCH,
                 "state":model.state_dict(),
                 "optimizer":optimizer.state_dict() }
        torch.save(state, filename)
    
