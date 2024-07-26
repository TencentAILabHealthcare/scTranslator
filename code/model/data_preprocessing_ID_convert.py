import argparse
import pickle
import os
import scanpy as sc
import anndata

def EntrezID_to_myID(EntrezID):
    myID = EntrezID_to_myID_dict.get(EntrezID)
    if myID:
        return myID
    else:
        return None

def hugo_symbol_to_myID(hugo_symbol):
    EntrezID = hgs_to_EntrezID_dict.get(hugo_symbol)
    myID = EntrezID_to_myID_dict.get(EntrezID)
    if myID:
        return myID
    else:
        return None

def mouse_id_to_myID(mouse_id):
    hugo_symbol = mouseID_to_hgs_dict.get(mouse_id)
    EntrezID = hgs_to_EntrezID_dict.get(hugo_symbol)
    myID = EntrezID_to_myID_dict.get(EntrezID)
    if myID:
        return myID
    else:
        return None

def mouse_name_to_myID(mouse_name):
    hugo_symbol = mouse_gene_symbol_to_hgs_dict.get(mouse_name)
    EntrezID = hgs_to_EntrezID_dict.get(hugo_symbol)
    myID = EntrezID_to_myID_dict.get(EntrezID)
    if myID:
        return myID
    else:
        return None

#################################################
#---------------- Main Function ----------------#
#################################################
def main():
    global hgs_to_EntrezID_dict
    global EntrezID_to_myID_dict
    global mouseID_to_hgs_dict
    global mouse_gene_symbol_to_hgs_dict

    #--- Settings ---#
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--origin_gene_type', type=str, 
                        choices=['mouse_gene_ID', 'mouse_gene_symbol', 'human_gene_symbol', 'EntrezID'], 
                        default='mouse_gene_symbol',
                        help='original gene type (must be one of: mouse_gene_ID, mouse_gene_symbol, human_gene_symbol, EntrezID)')
    parser.add_argument('--origin_gene_column', type=str, default='index',
                        help='Colum name of origin gene location, eg. index, feature, gene, protein')
    parser.add_argument('--data_path', default='dataset/test/cite-seq_mouse/spleen_lymph_111.h5ad',
                        help='path for loading the anndata')
    args = parser.parse_args()
    feature_column = args.origin_gene_column
    dic_path = 'code/model/ID_dic/'
    file_path = os.path.join(dic_path, 'hgs_to_EntrezID.pkl')
    with open(file_path, 'rb') as f:
        hgs_to_EntrezID_dict = pickle.load(f)

    file_path = os.path.join(dic_path, 'EntrezID_to_myID.pkl')
    with open(file_path, 'rb') as f:
        EntrezID_to_myID_dict = pickle.load(f)

    file_path = os.path.join(dic_path, 'mouse_gene_ID_to_human_gene_symbol.pkl')
    with open(file_path, 'rb') as f:
        mouseID_to_hgs_dict = pickle.load(f)

    file_path = os.path.join(dic_path, 'mouse_gene_symbol_to_human_gene_symbol.pkl')
    with open(file_path, 'rb') as f:
        mouse_gene_symbol_to_hgs_dict = pickle.load(f)

    data_file_path = os.path.join(args.data_path)
    if '.h5ad' in data_file_path:
        data_file_path = data_file_path.replace('.h5ad', '')

    try:
        adata = sc.read_h5ad(os.path.join(data_file_path + '.h5ad'))
    except FileNotFoundError:
        print(f"can not find file: {data_file_path}")
        return
    print('# of genes before mapping:' , adata.n_vars)

    if args.origin_gene_type == 'mouse_gene_ID':
        orgID_to_myID = mouse_id_to_myID
    elif args.origin_gene_type == 'mouse_gene_symbol':
        orgID_to_myID = mouse_name_to_myID
    elif args.origin_gene_type == 'human_gene_symbol':
        orgID_to_myID = hugo_symbol_to_myID
    elif args.origin_gene_type == 'EntrezID':
        orgID_to_myID = EntrezID_to_myID

    if feature_column == 'index':
        feature_column = 'feature'
        adata.var['feature'] = adata.var.index

    adata.var['my_Id'] = adata.var[feature_column].tolist()
    for index, org_id in zip(adata.var.index.tolist(), adata.var[feature_column].tolist()):
        adata.var.loc[index, 'my_Id'] = orgID_to_myID(org_id)

    # delete no mapping gene
    flag = adata.var.index[~adata.var['my_Id'].isna()]
    new_var = adata.var.loc[flag, :]
    # delete the expression value of no mapping gene
    new_X = adata[:, flag].X
    # create new AnnData object
    filtered_adata = anndata.AnnData(X=new_X, var=new_var, obs=adata.obs)

    filtered_adata.write(data_file_path + '_mapped.h5ad')
    print('# of genes after mapping:' , filtered_adata.n_vars)
    print('Gene mapping completed!')

if __name__ == '__main__':
    main()
