# numeric packages
import numpy as np
import scipy
import scipy.io
import pandas as pd

# filesystem and OS
import sys, os, time
import glob

# images
import skimage
import skimage
import skimage.io
from scipy.ndimage.morphology import distance_transform_cdt
from skimage import morphology

# plotting
from matplotlib import pyplot as plt
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

class RibskyModel(object):

	def __init__(L, TRE=5, 
		n_urban_seeds=1, n_rural_seeds=100,
		Mu=None, Mr=None):
		self.TRE = TRE
		self.L = L
		if Mu is None:
			self.Mr = initialize_matrix(L, nseeds=n_urban_seeds)
		if Mr is None:
			self.Mu = initialize_matrix(L, nseeds=n_rural_seeds)

	def forward(gu, gr, Ue, niter=1, save_every=1):
		Ur = 100 - Ue
		history = updated_ribsky_model(self.Mu, self.Mr, gu, gr, Ue, Ur, 
			self.T, self.TRE, niter=niter)

	def generate_samples():
		pass


def updated_ribsky_model(Mu0, Mr0, gu, gr, Ue, Ur, T, TRE, niter=1):
    Mu = Mu0.copy()
    Mr = Mr0.copy()
    L = Mu.shape[0]
    history = {}
    for t in range(niter):
    	Mu, Mr = updated_ribsky_iteration(Mu, Mr, gu, gr, Ue, Ur, T, TRE)
            
        # add rural and urban together
        M = (Mu + Mr > 0).astype(int)
        
        # save snapshots
        history[t]  = {"M":M, "Mu":Mu, "Mr":Mr}

        # terminate if map is filled above threshold
        q = M.sum() / float(L**2)
        if q > T:
            break

    return history       


def updated_ribsky_iteration(Mu0, Mr0, gu, gr, Ue, Ur, T, TRE):
    Mu = Mu0.copy()
    Mr = Mr0.copy()
    L = Mu.shape[0]
    M_dict = {}
    for _ in range(Ue):
        Mu = ribsky_model_iteration(Mu, gu)
    for _ in range(Ur):
        Mr = ribsky_model_iteration(Mr, gr)
        
    # transfer rural to urban 
    areas_r, mask = compute_patch_areas(Mr)
    to_urban= [x[0] for x in areas_r if x[1]>TRE]
    for label in to_urban:
        Mu[mask==label] = Mu[mask==label] + Mr[mask==label]
        Mr[mask==label] = 0
         
    return Mu, Mr   


def initialize_matrix(L, nseeds=1):
    M = np.zeros((L,L))
    if nseeds > 1:
        ivec = np.random.randint(0, L, nseeds)
        jvec = np.random.randint(0, L, nseeds)
        M[ivec,jvec] = 1
    else:
        M[L/2,L/2] = 1
    return M


def ribsky_iteration(M, g):
    L = M.shape[0]
    # change D to Euclidean! not available in SciPy 0.19??
    D = distance_transform_cdt(M==0, metric="taxicab") 
    mask = M==1
    D = D ** (-g)
    D[mask] = 0
    norm = D[M<1].sum()
    D[M<1] = D[M<1]/float(norm)
    R = np.random.random((L,L))
    M[R<D] = 1
    return M


def compute_patch_areas(M):
    mask = morphology.label(M)
    areas = []
    for i in np.arange(1,mask.max()):
        areas.append((i,(mask==i).sum()))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas, mask


def generate_samples(params, savepath=""):
    Ue, Gu, Gr, T, n = params
    Ue = int(Ue)
    Ur = 100 - Ue
    nseed = 100
    max_niter = 100
    L = 300  # map size
    TRE=5 # threshold of area from rural to urban NOW IN NUMBER OF PIXEL
    iter_save = 3
    
    basedir = savepath + \
        "/Hu%d_Gu%2.2f_Gr%2.2f_T%2.2f"%(Ue, Gu, Gr, T)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    Mu0 = initialize_matrix(L, 1)
    Mr0 = initialize_matrix(L, nseed)
    M, M_dict = updated_ribsky_model(Mu0, Mr0, Gu, Gr, Ue, Ur, T, TRE, \
                             niter=max_niter)

    if iter_save is not None:
        ret = []
        for it, M in M_dict.iteritems():
            if it % iter_save == 0:
                filename = "%s/%d_iter%d.png"%(basedir,n,it)
                skimage.io.imsave(filename, M)
                ret.append(filename)
    else:
        filename = "%s/%d_iter%d.png"%(basedir,n,max(M_dict.keys()))
        skimage.io.imsave(filename, M)
        ret = [filename]
        
    return ret
