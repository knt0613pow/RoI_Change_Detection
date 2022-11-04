
import torch
import pygmtools as pygm
import pickle
import pandas as pd
from pathlib import Path
import json
import numpy as np
import os
from collections import OrderedDict
import scipy.spatial as spa
import itertools
from model.delaunay2D import Delaunay2D

def Path_img2npy(imgpath):
    return imgpath + '.npy'

def Path_img2json(imgpath):
    return imgpath + '.json'

def read_json(fname):
    with open(fname, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def read_npy(fname):
    return np.load(fname)

def delaunay_triangulation(kpt):
    breakpoint()
    dd = spa.Delaunay(kpt)
    AA = torch.zeros(len(kpt[0]), len(kpt[0]))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)

data_path  = 'data/Daesan2'
relation_path = 'data/Daesan2/relation.pkl'
with open(relation_path,"rb") as fr:
    data = pickle.load(fr)

path_img1 = os.path.join(data_path, data[109][0])
path_img2 = os.path.join(data_path, data[109][1])

path_npy1 = Path_img2npy(path_img1)
path_npy2 = Path_img2npy(path_img2)
path_json1 = Path_img2json(path_img1)
path_json2 = Path_img2json(path_img2)

j1 = read_json(path_json1)
j2 = read_json(path_json2)

feature1 = read_npy(path_npy1)
feature2 = read_npy(path_npy2)


img1_label = []
img1_points = []
for j1_label in j1['shapes']:
    img1_label.append(j1_label['label'])
    img1_x_list =np.array(j1_label['points'])[:,0]
    img1_y_list = np.array(j1_label['points'])[:,1]
    img1_x = (np.max(img1_x_list) +np.min(img1_x_list))/2
    img1_y = (np.max(img1_y_list) +np.min(img1_y_list))/2
    pt = [img1_x, img1_y]
    img1_points.append(pt)

img2_label = []
img2_points = []
for j2_label in j2['shapes']:
    img2_label.append(j2_label['label'])
    img2_x_list =np.array(j2_label['points'])[:,0]
    img2_y_list = np.array(j2_label['points'])[:,1]
    img2_x = (np.max(img2_x_list) +np.min(img2_x_list))/2
    img2_y = (np.max(img2_y_list) +np.min(img2_y_list))/2
    img2_points.append([img2_x, img2_y])
breakpoint()
img1_tri = Delaunay2D()
[img1_tri.addPoint(img1_pt) for img1_pt in img1_points]
img2_tri = Delaunay2D()
[img2_tri.addPoint(img2_pt) for img2_pt in img2_points]
