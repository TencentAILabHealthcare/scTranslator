import os
import time
import argparse
import warnings

import scanpy as sc
import numpy as np
import pandas as pd


import sys 
sys.path.append('code/model') 
from performer_enc_dec import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--repeat', type=int, default=1,
                        help='for repeating experiments to change seed (default: 1)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder')
    parser.add_argument('--dec_max_seq_len', type=int, default=1000,
                        help='sequence length of decoder')
    parser.add_argument('--fix_set', action='store_false',
                        help='fix (aligned) or disordering (un-aligned) dataset')
    parser.add_argument('--pretrain_checkpoint', default='checkpoint/stage2_single-cell_scTranslator.pt',
                        help='path for loading the pretrain checkpoint')
    parser.add_argument('--RNA_path', default='dataset/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='dataset/test/dataset1/GSM5008738_protein_finetune_withcelltype.h5ad',
                        help='path for loading the protein')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    
    ###########################
    #--- Prepare The Model ---#
    ###########################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device)
    model = torch.load(args.pretrain_checkpoint, map_location=torch.device(device))
    # model = model.to(device)

    ##########################
    #--- Prepare The Data ---#
    ##########################
         
    #---  Load Single Cell Data  ---#
    scRNA_adata = sc.read_h5ad(args.RNA_path)[:100]
    scP_adata = sc.read_h5ad(args.Pro_path)[:100]
    print('Total number of origin RNA genes: ', scRNA_adata.n_vars)
    print('Total number of origin proteins: ', scP_adata.n_vars)
    print('Total number of origin cells: ', scRNA_adata.n_obs)
    print('# of NAN in X', np.isnan(scRNA_adata.X).sum())
    print('# of NAN in X', np.isnan(scP_adata.X).sum())

    #---  Seperate Training and Testing set ---#
    test_rna = scRNA_adata
    # --- Protein ---#
    test_protein = scP_adata[test_rna.obs.index]
    # #---  Construct Dataloader ---#
    if args.fix_set == True:
        my_testset = fix_SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)
    else:
        my_testset = SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)

    test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.test_batch_size, drop_last=True)
    print("load data ended")

    ##################
    #---  Testing ---#
    ##################
    start_time = time.time()
    test_loss, test_ccc, y_hat, y = test(model, device, test_loader)
    y_pred =  pd.DataFrame(y_hat, columns=test_protein.var.index.tolist())
    y_truth = pd.DataFrame(y, columns=test_protein.var.index.tolist())
    ##############################
    #---  Prepare for Storage ---#
    ##############################
    

    if args.RNA_path == 'dataset/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad':
        dataset_flag = '/seuratv4_16W-without_fine-tune'
    else:
        dataset_flag = '/new_data-without_fine-tune'
    file_path = 'result/test'+dataset_flag
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    dict = vars(args)
    filename = open(file_path+'/args'+str(args.repeat)+'.txt','w')
    for k,v in dict.items():
        filename.write(k+':'+str(v))
        filename.write('\n')
    filename.close()
 
    #---  Save the Final Results ---#
    log_path = file_path+'/performance_log.csv'
    log_all = pd.DataFrame(columns=['test_loss', 'test_ccc'])
    log_all.loc[args.repeat] = np.array([test_loss, test_ccc])
    log_all.to_csv(log_path)
    y_pred.to_csv(file_path+'/y_pred.csv')
    y_truth.to_csv(file_path+'/y_truth.csv')
        
    print('-'*40)
    print('single cell '+str(args.enc_max_seq_len)+' RNA To '+str(args.dec_max_seq_len)+' Protein on dataset'+dataset_flag)
    print('Overall performance in repeat_%d costTime: %.4fs' % ( args.repeat, time.time() - start_time))
    print('Test Set: AVG mse %.4f, AVG ccc %.4f' % (np.mean(log_all['test_loss'][:args.repeat]), np.mean(log_all['test_ccc'][:args.repeat])))
if __name__ == '__main__':
    main()