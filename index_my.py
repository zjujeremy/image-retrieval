# -*- coding: utf-8 -*-

import h5py
import argparse
import numpy as np
from service.vggnet import VGGNet
import os
import sys
from os.path import dirname
BASE_DIR = dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=os.path.join(BASE_DIR, '../BLIP/mm_representation'), help="train data path.")
    parser.add_argument("--index_file", type=str, default=os.path.join(BASE_DIR, 'index', 'coco.h5'), help="index file path.")
    args = parser.parse_args()
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    ff = [os.path.join(args.train_data, f) for f in os.listdir(args.train_data) if f.endswith(".txt")]
    feats = []
    names = []
    captions = []
    for i, ff_path in enumerate(ff):
        print("extract {}...".format(ff_path))
        with open("ff_path", "r") as fin:
            cont = fin.readlines()
            if len(cont) != 3:
                continue
            names.append(cont[0].strip())
            captions.append(cont[1].strip())
            feats.append([float(ele) for ele in cont[2].strip().split(",")])
    feats = np.array(feats)
    print("--------------------------------------------------")
    print("         writing feature extraction results")
    print("--------------------------------------------------")
    h5f = h5py.File(args.index_file, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.create_dataset('dataset_3', data = np.string_(captions))
    h5f.close()
