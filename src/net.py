from net_utils import FC, MLP, LayerNorm
from mca import MCA_ED
import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------
class AttFlat(nn.Module):
    def __init__(self, __cfg):
        super(AttFlat, self).__init__()
        self.__cfg = __cfg

        self.mlp = MLP(
            in_size=__cfg.CORE.HIDDEN_SIZE,
            mid_size=__cfg.CORE.FLAT_MLP_SIZE,
            out_size=__cfg.CORE.FLAT_GLIMPSES,
            dropout_r=__cfg.CORE.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __cfg.CORE.HIDDEN_SIZE * __cfg.CORE.FLAT_GLIMPSES,
            __cfg.CORE.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__cfg.CORE.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

    
# -------------------------
# ---- Main MCAN Model ----
# -------------------------
class Net(nn.Module):
    def __init__(self, __cfg, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__cfg.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __cfg.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__cfg.WORD_EMBED_SIZE,
            hidden_size=__cfg.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __cfg.IMG_FEAT_SIZE,
            __cfg.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__cfg)

        self.attflat_img = AttFlat(__cfg)
        self.attflat_lang = AttFlat(__cfg)

        self.proj_norm = LayerNorm(__cfg.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__cfg.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        #(batch_size, seq_length, 512)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
