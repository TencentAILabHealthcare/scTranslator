import os
import argparse
import torch
import torch.nn as nn
import scanpy as sc
import numpy as np
import pandas as pd

import sys 
sys.path.append('code/model') 
from performer_enc_dec import *
from utils import *

#################################################
#------------ Train & Test Function ------------#
#################################################   

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args
    
class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, output_attentions = False, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if output_attentions:
            attn_weights = []
        for (f, g), (f_args, g_args) in layers_and_args:
            if output_attentions:
                x = x + f(x, output_attentions = output_attentions, **f_args)[0]
                attn_weights.append(f(x, output_attentions = output_attentions, **f_args)[1].unsqueeze(0))
            else:
                x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        if output_attentions:
            attn_weights = torch.transpose(torch.cat(attn_weights, dim=0), 0, 1)    # the final dim is (batch, layer, head, len, len)
            attn_weights = torch.mean(attn_weights, dim=1)                        # the dim is (batch, head, len, len)
            return x, attn_weights
        else:
            return x
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, output_attentions = False):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        if output_attentions:
            v_diag = torch.eye(v.shape[-2]).to(device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0],v.shape[1],1,1)
            attn_weights = torch.zeros(1, 1, q.shape[2], q.shape[2]).to(device)
            for head_dim in range(q.shape[1]):
                attn_weights += attn_fn(q[:,head_dim], k[:,head_dim], v_diag[:,head_dim])
            attn_weights /= q.shape[1]
            return out, attn_weights
        else:
            return out



class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, output_attentions = False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k, = apply_rotary_pos_emb(q, k, pos_emb)

            if output_attentions:
                out, attn_weights = self.fast_attention(q, k, v, output_attentions)
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if output_attentions:
            return self.dropout(out), attn_weights
        else:
            return self.dropout(out)
            
class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,#64,#
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):

            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, output_attentions = True, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, output_attentions = output_attentions, **kwargs)
    
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
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, geneID, return_encodings = False, output_attentions = True,**kwargs):
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
        x, attn_weights = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

        if return_encodings:
            return x, attn_weights

        return torch.squeeze(self.to_out(x)), attn_weights

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
        encodings, enc_weights = self.enc(seq_in, seq_inID, return_encodings = True, **enc_kwargs)# batch_size, input_seq_lenth, dim
        seq_out = self.translator(encodings.transpose(1,2).contiguous()).transpose(1,2).contiguous() # batch_size, out_seq_lenth, dim \
        _, dec_weights = self.dec(seq_out, seq_outID, **dec_kwargs)
        enc2dec_weights = torch.einsum('...ik,...kj->...ij', self.translator(enc_weights.type_as(encodings)), dec_weights.type_as(encodings))
        return torch.squeeze(enc_weights), torch.squeeze(dec_weights), torch.squeeze(enc2dec_weights)


#################################################
#---------------- Main Function ----------------#
#################################################
def main():
    #--- Training Settings ---#
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--dim', type=int, default=128,
                        help='latend dimension of each token')
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder')
    parser.add_argument('--dec_max_seq_len', type=int, default=1000,
                        help='sequence length of decoder')
    parser.add_argument('--translator_depth', type=int, default=2,
                        help='translator depth')
    parser.add_argument('--initial_dropout', type=float, default=0.1,
                        help='sequence length of decoder')
    parser.add_argument('--enc_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--enc_heads', type=int, default=8,
                        help='sequence length of decoder')
    parser.add_argument('--dec_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--dec_heads', type=int, default=8,
                        help='sequence length of decoder')              
    parser.add_argument('--pretrain_checkpoint', default='checkpoint/Dataset1_fine-tuned_scTranslator.pt',
                        help='path for loading the checkpoint')
    parser.add_argument('--RNA_path', default='dataset/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='dataset/test/dataset1/GSM5008738_protein_finetune_withcelltype.h5ad',
                        help='path for loading the protein')
    args = parser.parse_args()


    print('seed', args.seed)

    #--- Prepare The Model ---#
    model = scPerformerEncDec(
        dim=args.dim,
        translator_depth=args.translator_depth,
        initial_dropout=args.initial_dropout,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        enc_max_seq_len=args.enc_max_seq_len,
        dec_depth=args.dec_depth,
        dec_heads=args.dec_heads,
        dec_max_seq_len=args.dec_max_seq_len
        )

    
    model.load_state_dict(torch.load(args.pretrain_checkpoint).cpu().state_dict())
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #-----  Load Single Cell Data  -----#
    scRNA_adata = sc.read_h5ad(args.RNA_path)[:10]
    scP_adata = sc.read_h5ad(args.Pro_path)[:10]
    print('Total number of origin RNA genes: ', scRNA_adata.n_vars)
    print('Total number of origin proteins: ', scP_adata.n_vars)
    print('Total number of origin cells: ', scRNA_adata.n_obs)
    print('# of NAN in X', np.isnan(scRNA_adata.X).sum())
    print('# of NAN in X', np.isnan(scP_adata.X).sum())

    #---  Seperate Training and Testing set ---#
    setup_seed(1105)
    att_index = scRNA_adata.obs.index
    my_testset = fix_SCDataset(scRNA_adata[att_index], scP_adata[att_index], args.enc_max_seq_len, args.dec_max_seq_len)
    test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.test_batch_size)
    print("load data ended!")
   
    enc_weights, dec_weights, enc2dec_weights = np.zeros((args.enc_max_seq_len,args.enc_max_seq_len)), np.zeros((args.dec_max_seq_len,args.dec_max_seq_len)), \
        np.zeros((args.enc_max_seq_len,args.dec_max_seq_len))

    torch.cuda.empty_cache()
    with torch.no_grad():
        i = 0
        for x, y in test_loader:
            #--- Extract Feature ---#
            RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
            pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
            y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

            #--- Prediction ---#
            enc_weight, dec_weight, enc2dec_weight = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)
            enc_weights += enc_weight.detach().cpu().numpy()
            dec_weights += dec_weight.detach().cpu().numpy()
            enc2dec_weights += enc2dec_weight.detach().cpu().numpy()
            i+=1
            print('attention for cell', i)
    args.enc_max_seq_len = min(args.enc_max_seq_len, len(scRNA_adata.var.index))
    args.dec_max_seq_len = min(args.dec_max_seq_len, len(scP_adata.var.index))
    enc_weights =  pd.DataFrame(enc_weights[:args.enc_max_seq_len, :args.enc_max_seq_len], columns=scRNA_adata.var.index[:args.enc_max_seq_len].tolist(),index= scRNA_adata.var.index[:args.enc_max_seq_len].tolist()) 
    dec_weights =  pd.DataFrame(dec_weights[:args.dec_max_seq_len, :args.dec_max_seq_len], columns=scP_adata.var.index[:args.dec_max_seq_len].tolist(), index=scP_adata.var.index[:args.dec_max_seq_len].tolist())
    enc2dec_weights =  pd.DataFrame(enc2dec_weights[:args.enc_max_seq_len, :args.dec_max_seq_len], columns=scP_adata.var.index[:args.dec_max_seq_len].tolist(), index=scRNA_adata.var.index[:args.enc_max_seq_len].tolist())
    enc_weights = attention_normalize(enc_weights)
    dec_weights = attention_normalize(dec_weights)
    enc2dec_weights = attention_normalize(enc2dec_weights)
    file_path = 'result/test/attention_matrix'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    enc_weights.to_csv(file_path + '/encoder_attention_score.csv')
    dec_weights.to_csv(file_path + '/decoder_attention_score.csv')
    enc2dec_weights.to_csv(file_path + '/encoder2decoder_attention_score.csv')

    print('completed')


if __name__ == '__main__':
    main()