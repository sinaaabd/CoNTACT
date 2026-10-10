import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
from mca import MCA_ED
from net_utils import FC, MLP, LayerNorm


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

''' The CoNTACT model (Graph co-Attention-based drug-target InTeraction prediction Using Drug fingerprints and protein Embeddings) '''
class CoNTACT(nn.Module):
    def __init__(self, device='cuda:0', **config):
        super(CoNTACT, self).__init__()
        self.device = device
        atom_input_features_dim = config["DRUG"]["ATOM_INPUT_FEATURES_DIM"]
        self.atom_embedding_dim = config["DRUG"]["ATOM_EMBEDDING_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        drug_hidden_layers_dims = config["DRUG"]["HIDDEN_LAYERS_DIMS"]
        self.batch_size = config["SETTINGS"]["BATCH_SIZE"]
        target_padding =  config["TARGET"]["PADDING"]
        target_embedding_dim = config["TARGET"]["EMBEDDING_DIMS"]
        self.kernel_sizes = config["TARGET"]["CNN_KERNEL_SIZES"]
        cnn_output_channel_dim = config["TARGET"]["CNN_OUTPUT_CHANNEL_DIM"]
        
        self.dot_att_hidden_dim = config["SETTINGS"]["DOTATT_HIDDEN_DIM"]
        self.max_num_atoms = config["DRUG"]["MAX_NUM_ATOMS"]
        
        self.drug_gcn = DrugSubmodel(node_input_features_dim=atom_input_features_dim, embedding_dim=self.atom_embedding_dim,
                                padding=drug_padding,hidden_layers_dims=drug_hidden_layers_dims)
        
        self.target_embedding = nn.Embedding(26, target_embedding_dim, padding_idx=0)
        self.conv1d_target = nn.Conv1d(in_channels=target_embedding_dim, out_channels=128, kernel_size=10, stride=5)
        
        self.co_attention = MCA_ED(**config)
        
        self.attflat_target = AttFlat(**config)
        self.attflat_drug = AttFlat(**config)

        self.bert_fc = nn.Linear(768, 512)
        self.bert_bn = nn.BatchNorm1d(512)
        
        self.fp_fc1 = nn.Linear(2048, 1024)
        self.fp_bn1 = nn.BatchNorm1d(1024)
        self.fp_dropout = nn.Dropout(p=0.2)
        self.fp_fc2 = nn.Linear(1024, 512)
        self.fp_bn2 = nn.BatchNorm1d(512)
        
        self.combined_fc = nn.Linear(2048, 1024)
        self.combined_bn = nn.BatchNorm1d(1024)
        self.combined_dropout = nn.Dropout(p=0.2)
        
        self.proj_norm = LayerNorm(config["CORE"]["FLAT_OUT_SIZE"])

        self.fp_fc_3 = nn.Linear(config["CORE"]["FLAT_OUT_SIZE"], 256)
        self.fp_bn_3 = nn.BatchNorm1d(256)
        self.fp_fc_4 = nn.Linear(256, 64)
        self.fp_bn_4 = nn.BatchNorm1d(64)
        self.before_last = nn.Linear(64, 16)

        self.proj = nn.Linear(16, 1)
        
        
    def forward(self, drug_atom_features, morgan_fp, aa_encodes, bert, mode="train"):
        #(batch_size, max num of atoms, atom embedding dim) -> in this case (64, 290, 128)
        d_atom_features = self.drug_gcn(drug_atom_features)        
        #aa_encodes: (batch_size, max num of residues) -> (64, 1400)
        #aa_embeddings: (batch_size, max num of residues, aa embedding dim) -> (64, 1400, 128)
        aa_embeddings = self.target_embedding(aa_encodes.long())
        aa_embeddings = aa_embeddings.permute(0, 2, 1)
        #If stride=10: (batch_size, number of regions, aa embedding dim) -> in this case (64, 140, 128)
        #If stride=5: (batch_size, number of regions, aa embedding dim) -> in this case (64, 279, 128)
        aa_embeddings = nn.functional.relu(
                nn.BatchNorm1d(128).to(torch.device(self.device))(
                    self.conv1d_target(aa_embeddings)
                )).permute(0, 2, 1)
        
        target_feature_mask = self.make_mask(aa_embeddings)
        drug_feature_mask = self.make_mask(d_atom_features)
        #drug_features: (batch_size, 290, 128); target_features: (batch_size, 140, 128)
        drug_features, target_features = self.co_attention(
            d_atom_features,
            aa_embeddings,
            drug_feature_mask,
            target_feature_mask
        )
        
        #drug_features: (batch_size, 1024)
        drug_features = self.attflat_drug(
            drug_features,
            drug_feature_mask
        )
        #drug_features: (batch_size, 2048)
        #drug_features = torch.cat((drug_features, morgan_fp), dim=1)
        #target_features: (batch_size, 1024)
        target_features = self.attflat_target(
            target_features,
            target_feature_mask
        )
        concat_feat = drug_features + target_features

        morgan_fp = self.fp_dropout(F.relu(self.fp_bn1(self.fp_fc1(morgan_fp))))
        morgan_fp = F.relu(self.fp_bn2(self.fp_fc2(morgan_fp)))
        bert  = F.relu(self.bert_bn(self.bert_fc(bert)))
        morgan_bert = torch.cat((bert, morgan_fp), dim=1)
        concat_feat = torch.cat((morgan_bert, concat_feat), dim=1)
        concat_feat = self.combined_dropout(F.relu(self.combined_bn(self.combined_fc(concat_feat))))
        concat_feat = self.proj_norm(concat_feat)
        concat_feat = F.relu(self.fp_bn_3(self.fp_fc_3(concat_feat)))
        concat_feat = self.before_last(F.relu(self.fp_bn_4(self.fp_fc_4(concat_feat))))
        concat_feat = self.proj(concat_feat)
        #bert: (batch_size, 768)
        #morgan_fp: (batch_size, 2048)
        #joint_representation: (batch_size, 8)
        #joint_representation = self.bp_layer(morgan_fp, esm2)
        #joint_representation = torch.cat((bert, morgan_fp), dim=1)
        return concat_feat
        
    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)        
    
# Flatten the sequence 
class AttFlat(nn.Module):
    def __init__(self, **cfg):
        super(AttFlat, self).__init__()
        self.cfg = cfg
        self.mlp = MLP(
            in_size=cfg["CORE"]["HIDDEN_SIZE"],
            mid_size=cfg["CORE"]["FLAT_MLP_SIZE"],
            out_size=cfg["CORE"]["FLAT_GLIMPSES"],
            dropout_r=cfg["CORE"]["DROPOUT_R"],
            use_relu=True
        )
        self.linear_merge = nn.Linear(
            cfg["CORE"]["HIDDEN_SIZE"] * cfg["CORE"]["FLAT_GLIMPSES"],
            cfg["CORE"]["FLAT_OUT_SIZE"]
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.cfg["CORE"]["FLAT_GLIMPSES"]):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted    
    

        
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


    
''' In the case you might need to do transfer learning for regression (TSA)-based dataset '''    
class RegressionModel(nn.Module):
    def __init__(self, pretrained_model, device='cuda:0', **config):
        super(RegressionModel, self).__init__()
        self.device = device
        self.drug_gcn = pretrained_model.drug_gcn
        self.target_embedding = pretrained_model.target_embedding
        self.conv1d_target = pretrained_model.conv1d_target
        self.co_attention = pretrained_model.co_attention
        self.attflat_target = pretrained_model.attflat_target
        self.attflat_drug = pretrained_model.attflat_drug
        self.bert_fc = pretrained_model.bert_fc
        self.bert_bn = pretrained_model.bert_bn
        self.fp_fc1 = pretrained_model.fp_fc1
        self.fp_bn1 = pretrained_model.fp_bn1
        self.fp_dropout = pretrained_model.fp_dropout
        self.fp_fc2 = pretrained_model.fp_fc2
        self.fp_bn2 = pretrained_model.fp_bn2
        self.combined_fc = pretrained_model.combined_fc
        self.combined_bn = pretrained_model.combined_bn
        self.combined_dropout = pretrained_model.combined_dropout
        self.proj_norm = pretrained_model.proj_norm
        self.fp_fc_3 = nn.Linear(config["CORE"]["FLAT_OUT_SIZE"], 256)
        self.fp_bn_3 = nn.BatchNorm1d(256)
        self.fp_fc_4 = nn.Linear(256, 64)
        self.fp_bn_4 = nn.BatchNorm1d(64)
        self.before_last = nn.Linear(64, 16)
        self.regressor = nn.Linear(16, 1)  # Regression output

        
        
    def forward(self, drug_atom_features, morgan_fp, aa_encodes, bert, mode="train"):
        d_atom_features = self.drug_gcn(drug_atom_features)        
        aa_embeddings = self.target_embedding(aa_encodes.long())
        aa_embeddings = aa_embeddings.permute(0, 2, 1)
        aa_embeddings = nn.functional.relu(
                nn.BatchNorm1d(128).to(torch.device(self.device))(
                    self.conv1d_target(aa_embeddings)
                )).permute(0, 2, 1)
        target_feature_mask = self.make_mask(aa_embeddings)
        drug_feature_mask = self.make_mask(d_atom_features)
        drug_features, target_features = self.co_attention(
            d_atom_features,
            aa_embeddings,
            drug_feature_mask,
            target_feature_mask
        )
        drug_features = self.attflat_drug(
            drug_features,
            drug_feature_mask
        )
        target_features = self.attflat_target(
            target_features,
            target_feature_mask
        )
        concat_feat = drug_features + target_features
        morgan_fp = self.fp_dropout(F.relu(self.fp_bn1(self.fp_fc1(morgan_fp))))
        morgan_fp = F.relu(self.fp_bn2(self.fp_fc2(morgan_fp)))
        bert  = F.relu(self.bert_bn(self.bert_fc(bert)))
        morgan_bert = torch.cat((bert, morgan_fp), dim=1)
        concat_feat = torch.cat((morgan_bert, concat_feat), dim=1)
        concat_feat = self.combined_dropout(F.relu(self.combined_bn(self.combined_fc(concat_feat))))
        embeddings = self.proj_norm(concat_feat)
        embeddings = F.relu(self.fp_bn_3(self.fp_fc_3(embeddings)))
        embeddings = F.relu(self.fp_bn_4(self.fp_fc_4(embeddings)))
        concat_feat = self.before_last(embeddings)
        concat_feat = self.regressor(concat_feat)
        return concat_feat
        #If you are also interested in having embeddings
        #return embeddings, concat_feat
        
        
    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)    