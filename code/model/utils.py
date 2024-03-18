import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from torch.utils.data import Dataset

#################################################
#------------ Train & Test Function ------------#
#################################################   
def setup_seed(seed):
    #--- Fix random seed ---#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    train_loss = 0
    train_ccc = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        #--- Extract Feature ---#
        RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
        Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
        rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
        pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
        x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
        y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

        #--- Prediction ---#
        optimizer.zero_grad()
        _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

        #--- Compute Performance Metric ---#
        y_hat = torch.squeeze(y_hat)
        y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
        y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)

        loss = F.mse_loss(y_hat[pro_mask], y[pro_mask])
        train_loss += loss.item()
        
        train_ccc += loss2(y_hat[pro_mask], y[pro_mask]).item()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    train_ccc /= len(train_loader)
    print('-'*15)
    print('--- Epoch {} ---'.format(epoch), flush=True)
    print('-'*15)
    print('Training set: Average loss: {:.4f}, Average ccc: {:.4f}'.format(train_loss, train_ccc), flush=True)
    return train_loss, train_ccc
    
def test(model, device, test_loader):
    model.eval()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    test_loss = 0
    test_ccc = 0
    with torch.no_grad():
        for x, y in test_loader:
            #--- Extract Feature ---#
            RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
            pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
            y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

            #--- Prediction ---#
            _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

            #--- Compute Performance Metric ---#
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
            y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)
            test_loss += F.mse_loss(y_hat[pro_mask], y[pro_mask]).item()
            test_ccc += loss2(y_hat[pro_mask], y[pro_mask]).item()

    test_loss /= len(test_loader)
    test_ccc /= len(test_loader)
    return test_loss, test_ccc
    
#################################################
#---------- Dataset Preprocess Function ---------#
#################################################
def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    x = low + (x-MIN)/(MAX-MIN)*(high-low) # zoom to (low, high)
    return x

def fix_sc_normalize_truncate_padding(x, length):
    '''
    x = (num_gene,1)

    '''
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    tmp = normalization(tmp)
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
        pro_value, pro_gene, pro_mask = fix_sc_normalize_truncate_padding(self.scP_adata[k], self.len_protein)
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs

def sc_normalize_truncate_padding(x, length):
    '''
    x = (num_gene,1)

    '''
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    tmp = normalization(tmp)
    if len_x >= length: # truncate
        gene = random.sample(range(len_x), length)
        x_value = [i for i in tmp[gene]] 
        gene = x.var.iloc[gene]['my_Id'].astype(int).values.tolist()
        mask = np.full(length, True).tolist()
    else: # padding
        x_value = tmp.tolist()
        x_value.extend([0 for i in range(length-len_x)])
        gene = x.var['my_Id'].astype(int).values.tolist()
        gene.extend([0 for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return x_value, gene, mask

class SCDataset(Dataset):
    def __init__(self, scRNA_adata, scP_adata, len_rna, len_protein):
        super().__init__()
        self.scRNA_adata = scRNA_adata
        self.scP_adata = scP_adata
        self.len_rna = len_rna
        self.len_protein = len_protein

    def __getitem__(self, index):
        k = self.scRNA_adata.obs.index[index]
        rna_value, rna_gene, rna_mask = sc_normalize_truncate_padding(self.scRNA_adata[k], self.len_rna)
        pro_value, pro_gene, pro_mask = sc_normalize_truncate_padding(self.scP_adata[k], self.len_protein)
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs
    
def attention_normalize(weights):
    for i in weights.columns:
        W_min = weights[i].min()
        W_max = weights[i].max()
        weights[i] = (weights[i]-W_min)/(W_max-W_min)
    for i in range(weights.shape[0]):
        W_min = weights.iloc[i].min()
        W_max = weights.iloc[i].max()
        weights.iloc[i] = (weights.iloc[i]-W_min)/(W_max-W_min)
    return(weights)