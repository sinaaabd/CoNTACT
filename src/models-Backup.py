import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
from mca import MCA_ED


''' Loss function for binary classification '''
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

''' Loss function for multi-class classification '''
def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

''' Our Model Definition '''
class MainModel(nn.Module):
    def __init__(self, device='cuda:0', **config):
        super(MainModel, self).__init__()
        self.device = device
        atom_input_features_dim = config["DRUG"]["ATOM_INPUT_FEATURES_DIM"]
        self.atom_embedding_dim = config["DRUG"]["ATOM_EMBEDDING_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        drug_hidden_layers_dims = config["DRUG"]["HIDDEN_LAYERS_DIMS"]
        
        target_padding =  config["TARGET"]["PADDING"]
        target_embedding_dim = config["TARGET"]["EMBEDDING_DIMS"]
        self.kernel_sizes = config["TARGET"]["CNN_KERNEL_SIZES"]
        cnn_output_channel_dim = config["TARGET"]["CNN_OUTPUT_CHANNEL_DIM"]
        
        self.dot_att_hidden_dim = config["SETTINGS"]["DOTATT_HIDDEN_DIM"]
        self.max_num_atoms = config["DRUG"]["MAX_NUM_ATOMS"]
        
        self.drug_gcn = DrugSubmodel(node_input_features_dim=atom_input_features_dim, embedding_dim=self.atom_embedding_dim,
                                padding=drug_padding,hidden_layers_dims=drug_hidden_layers_dims)
        
        self.target_submodel = TargetSubmodel(target_embedding_dim, self.kernel_sizes, cnn_output_channel_dim, target_padding, device)
        
        
        self.co_attention = MCA_ED(config)
        
        # Define multi-head attention layer
        # ---> self.dot_att = nn.MultiheadAttention(embed_dim=self.atom_embedding_dim, num_heads=4)

        
        #self.dot_att = DoTAttention(self.atom_embedding_dim, hidden_dim=self.dot_att_hidden_dim)
        
        # (64, 1, 4, 37120) ->conv-> (64, 1, 1, 928) ->maxpool-> (64, 1, 1, 116)
        self.final_cnn = nn.Sequential(
                            nn.Conv2d(1, 1, kernel_size=(4,40), stride=40, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(1,8), stride=8))
        
        self.dense_layers = DenseLayers()
        
        
    def forward(self, drug_atom_features, aa_encodes, mode="train"):
        #(batch_size, max num of atoms, atom embedding dim) -> in this case (64, 290, 128)
        d_atom_features = self.drug_gcn(drug_atom_features)        
        #aa_encodes: (batch_size, max num of residues) -> (64, 1400)
        #(batch_size, number of regions, aa embedding dim) -> in this case (64, <One of the 140, 56, 50, 40>, 128)
        #p_aa_features_list[0]: (64, 140, 128)
        #p_aa_features_list[1]: (64, 56, 128)
        #p_aa_features_list[2]: (64, 50, 128)
        #p_aa_features_list[3]: (64, 40, 128)
        p_aa_features_list = self.target_submodel(aa_encodes)
        
        #average_attention: (batch_size, <One of the 140, 56, 50, 40>) specify how important a region of the protein for the drug is
        #att_features: (batch_size, max num of atoms, hidden_dim of attention) -> (64, 290, 32)
        #att_features, average_attention = self.dot_att(d_atom_features, p_aa_features_list[0])
        
        # Inputs: query, key, value
        # ---> fusion_features = [self.dot_att(d_atom_features.transpose(0, 1), p_aa_features_list[i].transpose(0, 1), p_aa_features_list[i].transpose(0, 1)) for i in range(len(p_aa_features_list))]
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)
        lang_feat, img_feat = self.co_attention(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        
        
        concatenated_features = torch.empty(fusion_features[0][0].shape[1], 0, self.atom_embedding_dim*self.max_num_atoms, device=self.device)  # Initialize an empty tensor with the appropriate size
        attention_results = []
        for att_features, average_attention in fusion_features:
            att_features = att_features.transpose(0, 1)
            att_features = att_features.reshape(att_features.shape[0], self.atom_embedding_dim*self.max_num_atoms)
            concatenated_features = torch.cat((concatenated_features, att_features.unsqueeze(1)), dim=1)
            #(batch_size, <One of the 140, 56, 50, 40>)
            average_attention = torch.mean(average_attention, dim=1)
            attention_results.append(average_attention)
        concatenated_features = self.final_cnn(concatenated_features.view(concatenated_features.shape[0], 1, len(self.kernel_sizes), self.atom_embedding_dim*self.max_num_atoms))
        concatenated_features = concatenated_features.view(concatenated_features.shape[0], 116)
        
        result = self.dense_layers(concatenated_features)
        
        if(mode == "train"):
            return result
        elif(mode == "eval"):
            return result, attention_results  
        
        
    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)        

    
        
        
class DrugSubmodel(nn.Module):
    def __init__(self, node_input_features_dim, embedding_dim=128, padding=True, hidden_layers_dims=None, activation=None):
        super(DrugSubmodel, self).__init__()
        self.init_transform = nn.Linear(node_input_features_dim, embedding_dim, bias=False)
        if(padding):
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=embedding_dim, hidden_feats=hidden_layers_dims, activation=activation)
        self.output_features = hidden_layers_dims[-1]

    def forward(self, batch_graph):
        g_node_features = batch_graph.ndata.pop('h')
        g_node_features = self.init_transform(g_node_features)
        g_node_features = self.gnn(batch_graph, g_node_features)
        batch_size = batch_graph.batch_size
        g_node_features = g_node_features.view(batch_size, -1, self.output_features)
        return g_node_features        
    
    
    
class TargetSubmodel(nn.Module):
    def __init__(self, embedding_dim, kernel_sizes=[10, 25, 28, 35], output_channels=128, padding=True, device='cuda:0'):
        super(TargetSubmodel, self).__init__()
        self.output_channels = output_channels
        self.device = device
        if(padding):
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
            
        self.conv1d_layers = [nn.Conv1d(in_channels=embedding_dim, out_channels=output_channels, kernel_size=kernel_size, stride=kernel_size).to(torch.device(self.device)) for kernel_size in kernel_sizes]
            
    def forward(self, aa_encodes):
        #(batch_size, seq_len, embedding_dim)
        aa_embeddings = self.embedding(aa_encodes.long())
        
        #if self.conv1d_layers[0].weight.requires_grad:
        #    print("THIS Embedding layer is trainable")
        
        #(batch_size, embedding_dim, seq_len)
        aa_embeddings = aa_embeddings.permute(0, 2, 1)
        batch_size = aa_embeddings.shape[0]
        
        aa_embeddings = [
            nn.functional.relu(
                nn.BatchNorm1d(self.output_channels).to(torch.device(self.device))(
                    conv(aa_embeddings)
                )).permute(0, 2, 1)
            for conv in self.conv1d_layers
        ]
        return aa_embeddings
    
    
    
class DenseLayers(nn.Module):
    def __init__(self, binary=1):
        super(DenseLayers, self).__init__()
        self.fc1 = nn.Linear(116, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, binary)
        
    def forward(self, input_f):
        input_f = self.bn1(F.relu(self.fc1(input_f)))
        input_f = self.bn2(F.relu(self.fc2(input_f)))
        input_f = self.bn3(F.relu(self.fc3(input_f)))
        input_f = self.fc4(input_f)
        return input_f