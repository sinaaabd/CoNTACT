from net_utils import FC, MLP, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch, math
import pickle


#Multi-Head Attention
class MHAtt(nn.Module):
    def __init__(self, **cfg):
        super(MHAtt, self).__init__()
        self.cfg = cfg            
        self.linear_v = nn.Linear(cfg["CORE"]["HIDDEN_SIZE"], cfg["CORE"]["HIDDEN_SIZE"])
        self.linear_k = nn.Linear(cfg["CORE"]["HIDDEN_SIZE"], cfg["CORE"]["HIDDEN_SIZE"])
        self.linear_q = nn.Linear(cfg["CORE"]["HIDDEN_SIZE"], cfg["CORE"]["HIDDEN_SIZE"])
        self.linear_merge = nn.Linear(cfg["CORE"]["HIDDEN_SIZE"], cfg["CORE"]["HIDDEN_SIZE"])

        self.dropout = nn.Dropout(cfg["CORE"]["DROPOUT_R"])

    def forward(self, v, k, q, mask, layer_num):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.cfg["CORE"]["MULTI_HEAD"],
            self.cfg["CORE"]["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.cfg["CORE"]["MULTI_HEAD"],
            self.cfg["CORE"]["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.cfg["CORE"]["MULTI_HEAD"],
            self.cfg["CORE"]["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask, layer_num)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.cfg["CORE"]["HIDDEN_SIZE"]
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask, layer_num):    
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

        #######################################################################################################
        ###################Uncomment this part if you wish to extract the order of the atoms###################
        ######################################To obtain Attention Map##########################################

'''
        if(layer_num == 7):
            if(att_map.shape[2] != att_map.shape[3]):
                #print(scores)
                #print(scores.shape)
                print(scores[10][0])
                print(scores[10][0].shape)
                exit(0)
                #Save the attention map of the first couple of (drug-target)
                #pickle.dump(att_map[0].detach().cpu().numpy(), open('Attention_Map_GrAtITUDEv1_1EGE_0.pickle', 'wb'))
                exit(0)
'''
        #PUT BEFORE THE 'return'
        #######################################################################################################
        

# FeedForward Nets
class FFN(nn.Module):
    def __init__(self, **cfg):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=cfg["CORE"]["HIDDEN_SIZE"],
            mid_size=cfg["CORE"]["FF_SIZE"],
            out_size=cfg["CORE"]["HIDDEN_SIZE"],
            dropout_r=cfg["CORE"]["DROPOUT_R"],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


#Self Attention
class SA(nn.Module):
    def __init__(self, layer_num, **cfg):
        super(SA, self).__init__()

        self.layer_num = layer_num
        self.mhatt = MHAtt(**cfg)
        self.ffn = FFN(**cfg)

        self.dropout1 = nn.Dropout(cfg["CORE"]["DROPOUT_R"])
        self.norm1 = LayerNorm(cfg["CORE"]["HIDDEN_SIZE"])

        self.dropout2 = nn.Dropout(cfg["CORE"]["DROPOUT_R"])
        self.norm2 = LayerNorm(cfg["CORE"]["HIDDEN_SIZE"])

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask, self.layer_num)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x



# Self Guided Attention
class SGA(nn.Module):
    def __init__(self, layer_num, **cfg):
        super(SGA, self).__init__()

        self.layer_num = layer_num
        self.mhatt1 = MHAtt(**cfg)
        self.mhatt2 = MHAtt(**cfg)
        self.ffn = FFN(**cfg)

        self.dropout1 = nn.Dropout(cfg["CORE"]["DROPOUT_R"])
        self.norm1 = LayerNorm(cfg["CORE"]["HIDDEN_SIZE"])

        self.dropout2 = nn.Dropout(cfg["CORE"]["DROPOUT_R"])
        self.norm2 = LayerNorm(cfg["CORE"]["HIDDEN_SIZE"])

        self.dropout3 = nn.Dropout(cfg["CORE"]["DROPOUT_R"])
        self.norm3 = LayerNorm(cfg["CORE"]["HIDDEN_SIZE"])

    #x: Protein regions; y: atoms of drugs
    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask, self.layer_num)
        ))
        # Atoms (y): Key and Value
        # Protein Regions (x): Query
        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask, self.layer_num)
        ))
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        
        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------
class MCA_ED(nn.Module):
    def __init__(self, **cfg):
        super(MCA_ED, self).__init__()
        self.enc_list = nn.ModuleList([SA(i, **cfg) for i in range(cfg["CORE"]["LAYER"])])
        self.dec_list = nn.ModuleList([SGA(i, **cfg) for i in range(cfg["CORE"]["LAYER"])])

    #Here=> x: Atoms of drug [64, 290, 128] and y: Protein Regions [64, 140, 128]
    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
            
        return x, y
