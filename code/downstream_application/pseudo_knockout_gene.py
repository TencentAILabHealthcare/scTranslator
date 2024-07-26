import os
import argparse
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from torch.utils.data import Dataset

import sys 
sys.path.append('code/model') 
from performer_enc_dec import *
from utils import *

#################################################
#------------ Train & Test Function ------------#
#################################################   
def test(model, test_loader, device):
    model.eval()
    y_hat_all = []
    with torch.no_grad():
        for x, y in test_loader:
            #--- Extract Feature ---#
            RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
            pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)

            #--- Prediction ---#
            _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)
            y_hat = torch.squeeze(y_hat)
            if device == 'cpu':
                y_hat_all.extend(y_hat.numpy().tolist())
            else:
                y_hat_all.extend(y_hat.detach().cpu().numpy().tolist())
       
    return np.array(y_hat_all)
    
#################################################
#---------- scData Preprocess Function ---------#
#################################################
def pro_fix_sc_truncate_padding(x, length):
    '''
    x = (num_gene,1)

    '''
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    if len_x >= length: # truncate
        x_value = tmp[:length]
        gene = x.var.iloc[:length]['my_Id'].astype(int).values.tolist()
        mask = np.full(length, True).tolist()
    else: # padding
        x_value = tmp.tolist()
        x_value.extend([0 for i in range(length-len_x)])
        gene = x.var['my_Id'].astype(int).values.tolist()
        gene.extend([0 for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return x_value, gene, mask

class fix_SCDataset(Dataset):
    def __init__(self, scRNA_adata, scP_adata, len_rna, len_protein):
        super().__init__()
        self.scRNA_adata = scRNA_adata
        self.scP_adata = scP_adata
        self.len_rna = len_rna
        self.len_protein = len_protein

    def __getitem__(self, index):
        k = self.scRNA_adata.obs.index[index]
        rna_value, rna_gene, rna_mask = fix_sc_normalize_truncate_padding(self.scRNA_adata[k], self.len_rna)
        pro_value, pro_gene, pro_mask = pro_fix_sc_truncate_padding(self.scP_adata[k], self.len_protein)
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs

#################################################
#---------------- Main Function ----------------#
#################################################
def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder')
    parser.add_argument('--dec_max_seq_len', type=int, default=1000,
                        help='sequence length of decoder')
    parser.add_argument('--test_batch_size', type=int, default=2,
                        help='input batch size for testing (default: 2)')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')         
    parser.add_argument('--pretrain_checkpoint', default='checkpoint/Dataset1_fine-tuned_scTranslator.pt',
                        help='path for loading the checkpoint')
    parser.add_argument('--RNA_path', default='dataset/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='dataset/test/query_protein_ID.csv',
                        help='path for loading the protein')
    parser.add_argument('--gene', default='org', help='knock out gene, eg: Predictability(org), TRIM39')
    args = parser.parse_args()

    print('seed', args.seed)
    #-----  Load model  -----#
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device)
    model = torch.load(args.pretrain_checkpoint, map_location=torch.device(device))
    
    #-----  Load single-cell data  -----#
    scRNA_adata = sc.read_h5ad(args.RNA_path)
    obs = pd.DataFrame(scRNA_adata.obs.values.tolist(), index=scRNA_adata.obs.index)
    protein = pd.read_csv(args.Pro_path)
    X = np.zeros((scRNA_adata.n_obs, protein.shape[0]))
    scP_adata = ad.AnnData(X, obs=obs, var=protein)
    print('Total number of origin RNA genes: ', scRNA_adata.n_vars)
    print('Total number of origin proteins: ', scP_adata.n_vars)
    print('Total number of origin cells: ', scRNA_adata.n_obs)
    print('# of NAN in X', np.isnan(scRNA_adata.X).sum())
    print('# of NAN in X', np.isnan(scP_adata.X).sum())
    
    #---  Knock-out ---#
    gene = args.gene
    if gene != 'org':
        scRNA_adata = scRNA_adata[:, scRNA_adata.var.drop(index=gene).index]

    #---  Inference 1.4W protein ---#
    # setup_seed(1105+10)
    test_index = scRNA_adata.obs.index
    for i in range(int(scP_adata.n_vars/args.dec_max_seq_len)):
        my_testset = fix_SCDataset(scRNA_adata[test_index], scP_adata[test_index,1000*i:1000*(i+1)], args.enc_max_seq_len, args.dec_max_seq_len)
        test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.test_batch_size, drop_last=True)
        y_hat = test(model, test_loader, device)
        if i == 0:
            y_all = y_hat #(num_cell, num_protein)
        else:
            y_all = np.concatenate((y_all, y_hat), axis=1)
            print(y_hat.shape)
    #---  Save results ---#
    y_all =  pd.DataFrame(y_all, columns=scP_adata.var['Hugo_Symbol'].tolist())
    file_path = 'result/fig5/e'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    y_all.to_pickle(file_path+'/knock_out_'+gene+'.pkl')
    

    print('completed')


if __name__ == '__main__':
    main()