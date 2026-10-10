''' Here implement the feature extraction functions for different input features and import this file in dataloader Python file '''
import os
import random
import numpy as np
import torch
import dgl
import logging
from rdkit import Chem
from rdkit.Chem import AllChem


CHARPROTSET = { "A": 1,   "C": 2,   "B": 3,   "E": 4,   "D": 5,   "G": 6,   "F": 7,   "I": 8,   "H": 9,   "K": 10,   "M": 11,   "L": 12,   "O": 13,   "N": 14,   "Q": 15,   "P": 16,   "S": 17,   "R": 18,   "U": 19,   "T": 20,   "W": 21,   "V": 22,   "Y": 23,   "X": 24,   "Z": 25}

CHARPROTLEN = 25

def integer_encoding_protein(sequence, max_length=1400):
    '''
    Integer encoding for protein string sequence.
    Input parameters:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of the input sequence.
    '''
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


def generate_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            mol, useChirality=False, radius=2, nBits=2048
        )
    )
    return fp


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)        
    
    
def graph_collate_func(x):
    smiles, seq, d, fp, p, bert, y = zip(*x)
    d = dgl.batch(d)
    #torch.stack(bert)
    return smiles, seq, d, torch.tensor(fp).float(), torch.tensor(np.array(p)), torch.tensor(bert), torch.tensor(y)


def graph_collate_inference_func(x):
    smiles, seq, d, fp, p, bert = zip(*x)
    d = dgl.batch(d)
    #torch.stack(bert)
    return smiles, seq, d, torch.tensor(fp).float(), torch.tensor(np.array(p)), torch.tensor(bert)