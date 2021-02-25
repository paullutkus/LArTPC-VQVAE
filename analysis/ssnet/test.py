import os
import ROOT as rt
from larcv import larcv

rt.gSystem.Load("libssnetdata.so")

from ROOT import ssnet

print ssnet
print ssnet.SSNetDataLoader
