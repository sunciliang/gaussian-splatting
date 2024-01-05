import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)

def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

class View_Embedding(nn.Module):
    def __init__(self, num_embed, embed_dim):
        super(View_Embedding, self).__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.view_embed_layer = nn.Embedding(num_embed, embed_dim)

    def forward(self, x):
        return self.view_embed_layer(x)

class debulrnet(nn.Module):
    def __init__(self,num_img):
        super().__init__()
        self.view_embed_layer = View_Embedding(num_embed=num_img,embed_dim=64)
        self.embed_fn, self.input_ch = get_embedder(10,0)
        self.embeddirs_fn, self.input_ch_views = get_embedder(4,0)
        self.mlp = nn.ModuleList(
            [nn.Linear(self.input_ch+64+4+3,64)]+[nn.Linear(64,64)]+
            [nn.Linear(64,7)]
        )
    def forward(self,view,gauss,idx):
        idx_embedded = self.view_embed_layer(idx)
        xyz = gauss.get_xyz
        idx_embedded = idx_embedded.expand(xyz.shape[0],64)
        r = gauss.get_rotation
        s = gauss.get_scaling
        xyz_embed = self.embed_fn(xyz)
        h = view_embed = torch.cat((xyz_embed,idx_embedded,r,s),dim=1)
        for i, _ in enumerate(self.mlp):
            h = self.mlp[i](h)
            if i==2:
                h = F.relu(h)+1
            else:
                h = F.relu(h)
        new_r, new_s = torch.split(h,[4,3],dim=1)
        return new_r, new_s


