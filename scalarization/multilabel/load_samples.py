"""
Auxiliar file to data class to gather and organize Machine Learning datasets
"""
"""
Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause

import numpy as np
import pandas as pd
import yaml
import os
import tarfile

__FILE_NAME = os.path.basename(__file__)
__FILE_FOLDER = os.path.abspath(__file__).replace(__FILE_NAME, "")


def pandas_read(filename):
    return np.array(pd.read_csv(filename))


def load_samples(name):
    if not os.path.exists(__FILE_FOLDER+'bases/'):
        tar = tarfile.open(__FILE_FOLDER+'bases.tar.gz')
        tar.extractall(__FILE_FOLDER)
    
    with open(__FILE_FOLDER+'bases/'+name+'/info.yaml', 'r') as stream:
        geral = yaml.load(stream)

    print(geral)
    data = pandas_read(__FILE_FOLDER+'bases/'+name+'/data.csv')
    X = data[:, :-geral['outputs']]
    nonzero = np.where(X.std(axis=0)!=0)[0]
    X = (X-X.mean(axis=0))[:,nonzero]/X.std(axis=0)[nonzero]
    return X, data[:, -geral['outputs']:]
