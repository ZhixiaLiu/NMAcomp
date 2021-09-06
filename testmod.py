import sys
import os
from prody import parsePDB,calcTempFactors,parseMMCIF,confProDy
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import *
from multiscale import MultiscaleGNM,MultiscaleANM,multiscale_gamma
from os import listdir
from collections import defaultdict
import pickle
import multiprocessing as mp
from functools import partial
import logging
ion()


# downloading pdb files from pdbid.txt
with open('pdbid.txt','r') as f:
    pdblist = f.read().split()
pdbl = PDBList()
for i in pdblist:
    pdbl.retrieve_pdb_file(i,pdir='PDB')
f_path = "./PDB/"
nameset = sort(list(set([l.split(".")[0] for l in listdir(f_path)])))

def getBfactor(structure):
    '''
    get Bfactor from Biopython structure
    input:
        structure: structure loaded by Biopython parser
    output:
        bfactor: a list of bfactors
    '''
    bfactor = []
    for i in structure.get_atoms():
        if i.get_name() == 'CA':
            bfactor.append(i.get_bfactor())
    bfactor = np.array(bfactor)
    return bfactor

def getMatrix(calpha,kernel,model,eta):
    '''
    helper function to calculate single K or H from parametes
    input:
        calpha: calpha coordinate information from prody
        eta: 
    '''
    if model == 'mGNM': 
        nm = MultiscaleGNM()
        nm.buildKirchhoff(calpha,cutoff=9999, gamma=multiscale_gamma(eta=eta,v=3,kernel=kernel))
        return nm.getKirchhoff()
    elif model == 'mANM':
        nm = MultiscaleANM()
        nm.buildHessian(calpha,cutoff=9999, gamma=multiscale_gamma(eta=eta,v=3,kernel=kernel))
        return nm.getHessian()

# parrelization with multiproccess
def pccFromModel(calpha,bfactor,test_range,kernel,model):
    '''
    calculate pcc from model, used to unit test, parrelize the matrix building
    input:
        calpha: calpha structure
        bfactor: bfactors to fit
        test_range: the range of eta
        model: name of model, in the set ('mGNM','mANM')
        kernel: name of kernel, in the set ('exponential','lorenz','ilf')
    output:
        pcc_grid: a single matrix of pcc
    '''
    limit = len(list(test_range))
    pcc_grid = np.zeros((limit,limit))
    func = partial(getMatrix, calpha, kernel,model)
    with mp.Pool(min(limit,mp.cpu_count())) as pool:
        pre_build=pool.map(func,test_range)       
    #print(pre_build[0].shape)
    #print(size(bfactor))
    
    for i in range(limit):
        for j in range(limit):
            if i != j:
                mat = [pre_build[i],pre_build[j]]
            else:
                mat = [pre_build[i]]
            if model=='mGNM':
                nm = MultiscaleGNM()
                nm.buildMultiscaleKirchhoffFromKernels(mat,bfactor)
                nm.calcModes('all')
            elif model =='mANM':
                nm = MultiscaleANM()
                nm.buildMultiscaleHessianFromKernels(mat,bfactor)
                nm.calcModes('all')
            tempfluc = calcTempFactors(nm,calpha)
            pcc_grid[i,j] = pcc_grid[j,i] = np.corrcoef(tempfluc,bfactor)[0,1]
    return pcc_grid


def pcc_pool(name,model,kernel,test_limit = 30):
    '''
    calculate pcc from pdbids
    input:
        name: pdbid
        model: name of model, in the set ('mGNM','mANM')
        kernel: name of kernel, in the set ('exponential','lorenz','ilf')
        test_limit: the uplimit of tested eta.
    output:
        pcc: a single matrix of pcc
    '''
    #print(name)
    parser = MMCIFParser()
    structure = parser.get_structure(name,f_path+name+'.cif')
    bfactor = getBfactor(structure)
    cur_str = parseMMCIF(f_path+name+'.cif')
    calpha = cur_str.select('name CA')
    test_range = range(1,test_limit+1)
    if kernel == 'ilf':
        test_range = range(5,test_limit+5)
    pcc=pccFromModel(calpha,bfactor,test_range=test_range,kernel=kernel,model=model)
    return pcc


def calc_save(model,kernel,nameset,logfile = './logfile',exclude_long = 1000):
    '''
    input:
        model: name of model, in the set ('mGNM','mANM')
        kernel: name of kernel, in the set ('exponential','lorenz','ilf')
        nameset: list of pdbid, the pdbid of pre-downloaded proteins
        logfile: the saved path of logfile
        exclude_long: the cut off size, above which the structure is excluded
    putput:
        pcc_total: a list of pcc matrices, order is the same as nameset
    save_path is the path to save precalculated data.
    file_path is the path individual matrix is saved.
    
    '''
    save_path = f'./result/{model}_{kernel}/'
    try:
        os.mkdir(save_path)
    except FileExistsError as err:
        print(err)
    pcc_total = []
    for i,name in enumerate(nameset):
        print(i,name)
        if exclude_long != None:
            parser = MMCIFParser()
            structure = parser.get_structure(name,f_path+name+'.cif')
            bfactor = getBfactor(structure)
            cur_str = parseMMCIF(f_path+name+'.cif')
            calpha = cur_str.select('name CA')
            if len(calpha) > exclude_long:
                print('skipped')
                continue
        filepath = save_path+f'saved_{name}'
        # todo: ammend try-except
        try:
            if os.path.isfile(filepath):
                with open(filepath,'rb') as f:
                    pcc_temp=pickle.load(f)
            else:
                pcc_temp=pcc_pool(name,model,kernel)
                with open(filepath,'wb') as f:
                    pickle.dump(pcc_temp,f)
            pcc_total.append(pcc_temp)
        except Exception as err:
            print(err)
    return pcc_total
