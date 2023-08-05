import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn



class MTRHN(pl.LightningModule):
    def __init__(self, input_dim=35, cur_x_dim = 21, his_x_dim = 119, hidden_dim=64, aux_output_dim = 17, num_layers = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cur_x_dim = cur_x_dim
        self.his_x_dim = his_x_dim
        self.aux_output_dim = aux_output_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(cur_x_dim + his_x_dim, self.hidden_dim, num_layers=self.num_layers)
        self.l_out_aux = torch.nn.Linear(self.hidden_dim, self.aux_output_dim)
        self.embed_his1 = torch.nn.Linear(self.his_x_dim, self.his_x_dim)
        self.embed_his2 = torch.nn.Linear(self.his_x_dim, self.his_x_dim)

        self.embed_cur1 = torch.nn.Linear(self.cur_x_dim, self.cur_x_dim)
        self.embed_cur2 = torch.nn.Linear(self.cur_x_dim, self.cur_x_dim)

        self.out_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, **kwargs):
        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        cur_x = x[:,:,:self.cur_x_dim]
        his_x = x[:,:,self.cur_x_dim:]

        embed_cur = self.embed_cur2(self.embed_cur1(cur_x)) + cur_x
        embed_his = self.embed_his2(self.embed_his1(his_x)) + his_x

        embed_cat = torch.cat((embed_cur, embed_his), -1)
        # x = pack_padded_sequence(embed_cat, lens.cpu(),batch_first=True,enforce_sorted=False)
        x = embed_cat

        _, hidden_t = self.rnn(x)
        hn = hidden_t[-1].squeeze()
        out = self.out_proj(hn)
        # o_aux = self.l_out_aux(hn)
        return out
