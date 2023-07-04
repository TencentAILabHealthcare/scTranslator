import re
import torch
from torch import nn
from performer_pytorch import *
from math import ceil

ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    return enc_kwargs, dec_kwargs, kwargs

#################################################
#-------------------- Model --------------------#
#################################################

class scPerformerLM(nn.Module):
    def __init__(
        self,
        *,
        
        max_seq_len,
        dim,depth,
        heads,
        num_tokens=1,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        shift_tokens = False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.to_vector = nn.Linear(1,dim)
        self.pos_emb = nn.Embedding(85500,dim,padding_idx=0)# There are 75500 NCBI Gene ID obtained on 19th July, 2022 
        self.layer_pos_emb = Always(None)
        self.dropout = nn.Dropout(emb_dropout)
        self.performer = Performer(dim, depth, heads, dim_head)
        # self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, geneID, return_encodings = False, **kwargs):
        b, n = x.shape[0], x.shape[1] 
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        
        # token and positional embeddings
        if len(x.shape)<3:
            x = torch.unsqueeze(x,dim=2)
            x = self.to_vector(x)
        
        x += self.pos_emb(geneID)
        x = self.dropout(x)
        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs)

        if return_encodings:
            return x

        return torch.squeeze(self.to_out(x))



class MLPTranslator(nn.Module):
    """
    Class description: translator from RNA to protein
    fully connected layer with adjustable number of layers and variable dropout for each layer
    
    """
    #----- Define all layers -----#
    def __init__(self, num_fc_input, num_output_nodes, num_fc_layers, initial_dropout, act = nn.ReLU(), **kwargs):
        super(MLPTranslator, self).__init__(**kwargs)
        fc_d = pow(num_fc_input/num_output_nodes,1/num_fc_layers) # reduce factor of fc layer dimension
        #--- Fully connected layers ---#
        self.num_fc_layers = num_fc_layers
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/fc_d)))
            self.dropout0 = nn.Dropout(initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input/fc_d)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1,num_fc_layers-1):
                    tmp_input = int(ceil(num_fc_input/fc_d**i))
                    tmp_output = int(ceil(num_fc_input/fc_d**(i+1)))
                    exec('self.fc{} = nn.Linear(tmp_input, tmp_output)'.format(i))
                    if i < ceil(num_fc_layers/2) and 1.1**(i+1)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)'.format(i))
                    elif i >= ceil(num_fc_layers/2) and 1.1**(num_fc_layers-1-i)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)'.format(i))
                    else:
                        exec('self.dropout{} = nn.Dropout(initial_dropout)'.format(i))
                # the last fc layer
                exec('self.fc{} = nn.Linear(tmp_output, num_output_nodes)'.format(i+1))
            
        #--- Activation function ---#
        self.act = act
    
    #----- Forward -----#
    def forward(self, x):
        # x size:  [batch size, feature_dim] 
        
        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1,self.num_fc_layers-1):
                    outputs = eval('self.act(self.dropout{}(self.fc{}(outputs)))'.format(i,i))
                # the last fc layer
                outputs = eval('self.fc{}(outputs)'.format(i+1))
            
        return outputs

class scPerformerEncDec(nn.Module):
    def __init__(
        self,
        dim,
        translator_depth, 
        initial_dropout,
        ignore_index = 0,
        pad_value = 0,
        tie_token_embeds = False,
        no_projection = False,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = dec_kwargs['no_projection'] = no_projection
        enc = scPerformerLM(**enc_kwargs)
        dec = scPerformerLM(**dec_kwargs)


        self.enc = enc
        self.translator = MLPTranslator(enc_kwargs['max_seq_len'], dec_kwargs['max_seq_len'], translator_depth, initial_dropout)
        self.dec = dec

    def forward(self, seq_in, seq_inID, seq_outID, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        encodings = self.enc(seq_in, seq_inID, return_encodings = True, **enc_kwargs)# batch_size, input_seq_lenth, dim
        seq_out = self.translator(encodings.transpose(1,2).contiguous()).transpose(1,2).contiguous() # batch_size, out_seq_lenth, dim 
        return encodings, self.dec(seq_out, seq_outID, **dec_kwargs)
