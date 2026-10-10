''' Import essential libraries such as torch, numpy, utils, etc. '''
import torch.utils.data as data
from functools import partial
import torch
import pickle
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from helpers import integer_encoding_protein, generate_morgan_fp


class PreparedDataset(data.Dataset):
    #Load BERT-based features of sequences
    BERT_features_dict = pickle.load(open('../datasets/BERT_Learned_Features_8012_LAID.pickle','rb'))
    #BERT_features_dict = pickle.load(open('../datasets/BERT_Learned_Features_Perturbations.pickle','rb'))
    ''' __init__ can also accept other parameters based on your needs '''
    ''' "ids_list" contains the indexes of the dataframe "data_df" '''
    #Previously: max_drug_atoms=290
    def __init__(self, ids_list, data_df, max_drug_atoms=60, max_seq_length=1400):
        self.ids_list = ids_list
        self.data_df = data_df
        self.max_drug_atoms = max_drug_atoms
        self.max_seq_length = max_seq_length
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.get_atom_graph = partial(smiles_to_bigraph, add_self_loop=True)
        
    ''' __len__ overwrites __len__ of data.Dataset '''
    def __len__(self):
        return len(self.ids_list)
    
    ''' __getitem__ overwrites __getitem__ of data.Dataset '''
    ''' This function shows how to retrieve each element (record) of the dataframe using an index '''
    def __getitem__(self, index):
        #To get real index in the dataframe
        idx = self.ids_list[index]
        ''' DRUG FEATURES '''
        canonical_smiles = self.data_df.iloc[idx]['Canonical SMILES']
        atom_features_interactions= self.get_atom_graph(smiles=canonical_smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_atom_features = atom_features_interactions.ndata.pop('h')
        num_actual_atoms = actual_atom_features.shape[0]
        num_virtual_atoms = self.max_drug_atoms - num_actual_atoms
        virtual_node_bit = torch.zeros([num_actual_atoms, 1])
        actual_atom_features = torch.cat((actual_atom_features, virtual_node_bit), 1)
        atom_features_interactions.ndata['h'] = actual_atom_features
        virtual_node_features = torch.cat((torch.zeros(num_virtual_atoms, 74), torch.ones(num_virtual_atoms, 1)), 1)
        atom_features_interactions.add_nodes(num_virtual_atoms, {"h": virtual_node_features})
        atom_features_interactions = atom_features_interactions.add_self_loop()
       #Load Morgan FP
        morgan_fp = generate_morgan_fp(canonical_smiles)
        ''' PROTEIN FEATURES '''
        sequence = self.data_df.iloc[idx]['Sequence']
        protein_integer_encodes = integer_encoding_protein(sequence, self.max_seq_length)
        BERT_features = PreparedDataset.BERT_features_dict[sequence]
        ''' OUTPUT LABELS '''
        y = self.data_df.iloc[idx]['bind']
        return canonical_smiles, sequence, atom_features_interactions, morgan_fp, protein_integer_encodes, BERT_features, y


class PreparedInferenceDataset(data.Dataset):
    #Load BERT-based features of sequences
    BERT_features_dict = pickle.load(open('../datasets/BERT_Learned_Features_8012_LAID.pickle','rb'))
    ''' __init__ can also accept other parameters based on your needs '''
    ''' "ids_list" contains the indexes of the dataframe "data_df" '''
    #max_drug_atoms=290
    def __init__(self, ids_list, data_df, max_drug_atoms=60, max_seq_length=1400):
        self.ids_list = ids_list
        self.data_df = data_df
        self.max_drug_atoms = max_drug_atoms
        self.max_seq_length = max_seq_length
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.get_atom_graph = partial(smiles_to_bigraph, add_self_loop=True)
        
    ''' __len__ overwrites __len__ of data.Dataset '''
    def __len__(self):
        return len(self.ids_list)
    
    ''' __getitem__ overwrites __getitem__ of data.Dataset '''
    ''' This function shows how to retrieve each element (record) of the dataframe using an index '''
    def __getitem__(self, index):
        #To get real index in the dataframe
        idx = self.ids_list[index]
        ''' DRUG FEATURES '''
        canonical_smiles = self.data_df.iloc[idx]['Canonical SMILES']
        atom_features_interactions= self.get_atom_graph(smiles=canonical_smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_atom_features = atom_features_interactions.ndata.pop('h')
        num_actual_atoms = actual_atom_features.shape[0]
        num_virtual_atoms = self.max_drug_atoms - num_actual_atoms
        virtual_node_bit = torch.zeros([num_actual_atoms, 1])
        actual_atom_features = torch.cat((actual_atom_features, virtual_node_bit), 1)
        atom_features_interactions.ndata['h'] = actual_atom_features
        virtual_node_features = torch.cat((torch.zeros(num_virtual_atoms, 74), torch.ones(num_virtual_atoms, 1)), 1)
        atom_features_interactions.add_nodes(num_virtual_atoms, {"h": virtual_node_features})
        atom_features_interactions = atom_features_interactions.add_self_loop()
       #Load Morgan FP
        morgan_fp = generate_morgan_fp(canonical_smiles)
        ''' PROTEIN FEATURES '''
        sequence = self.data_df.iloc[idx]['Sequence']
        protein_integer_encodes = integer_encoding_protein(sequence, self.max_seq_length)
        BERT_features = PreparedDataset.BERT_features_dict[sequence]
        return canonical_smiles, sequence, atom_features_interactions, morgan_fp, protein_integer_encodes, BERT_features