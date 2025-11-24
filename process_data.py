import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import pickle

from myutils import GraphDataset, GraphDatasetWithBonds, collate, collate_with_bonds, sim_graph_construction, combine_hypergraphs
from scipy.sparse import coo_matrix


# Obtain the topk neighbors for each cell line





def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list, third_element = drug_feature.iloc[i]
        # If third element is number 1 (CCLE format), compute degree_list
        # If third element is degree_list (GDSC format), use directly
        if isinstance(third_element, int) and third_element == 1:
            degree_list = [len(neighbors) for neighbors in adj_list]
        else:
            degree_list = third_element
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


def FeatureExtractWithBonds(drug_feature_with_bonds):
    drug_data = [[] for item in range(len(drug_feature_with_bonds))]
    for i in range(len(drug_feature_with_bonds)):
        feat_mat, adj_list, _, bond_matrix = drug_feature_with_bonds.iloc[i]
        drug_data[i] = CalculateGraphFeatWithBonds(feat_mat, adj_list, bond_matrix)
    return drug_data


def CalculateGraphFeatWithBonds(feat_mat, adj_list, bond_matrix):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    edge_attr = []
    max_dim = 480  # Determined from statistical results
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for j, each in enumerate(nodes):
            adj_mat[i, int(each)] = 1
            edge_idx = i * len(adj_list) + int(each)
            if edge_idx < len(bond_matrix):
                edge_feat = np.asarray(bond_matrix[edge_idx], dtype='float32').flatten()
                # Pad zeros or truncate to max_dim
                if edge_feat.shape[0] < max_dim:
                    edge_feat = np.pad(edge_feat, (0, max_dim - edge_feat.shape[0]), mode='constant')
                elif edge_feat.shape[0] > max_dim:
                    edge_feat = edge_feat[:max_dim]
                edge_attr.append(edge_feat)
            else:
                edge_attr.append(np.zeros(max_dim, dtype='float32'))
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    if len(edge_attr) == 0:
        edge_attr = np.zeros((0, max_dim), dtype='float32')
    else:
        edge_attr = np.stack(edge_attr, axis=0)  # [E, max_dim]
    return [feat_mat, adj_index, edge_attr]




def cmask(num, train_ratio, valid_ratio, seed):
    mask = np.zeros(num)
    mask[0:int(train_ratio * num)] = 0
    mask[int(train_ratio * num):int((train_ratio+valid_ratio) * num)] = 1
    mask[int((train_ratio + valid_ratio) * num):] = 2
    np.random.seed(seed)
    np.random.shuffle(mask)
    train_mask = (mask == 0)
    valid_mask = (mask == 1)
    test_mask = (mask == 2)
    return train_mask, valid_mask, test_mask


# Split the response into train/valid/test set
def process_label_random(allpairs, nb_celllines, nb_drugs, train_ratio=0.6, valid_ratio=0.2, seed=100):
    # split into positive and negative pairs
    pos_pairs = allpairs[allpairs[:, 2] == 1]
    neg_pairs = allpairs[allpairs[:, 2] == -1]
    pos_num = len(pos_pairs)
    neg_num = len(neg_pairs)

    # random
    train_mask, valid_mask, test_mask = cmask(len(allpairs), train_ratio, valid_ratio, seed)
    train = allpairs[train_mask][:, 0:3]
    valid = allpairs[valid_mask][:, 0:3]
    test = allpairs[test_mask][:, 0:3]

    train_edge = np.vstack((train, train[:, [1, 0, 2]]))
    train[:, 1] -= nb_celllines
    test[:, 1] -= nb_celllines
    valid[:, 1] -= nb_celllines

    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    valid_mask = coo_matrix((np.ones(valid.shape[0], dtype=bool), (valid[:, 0], valid[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()

    pos_pairs[:, 1] -= nb_celllines
    neg_pairs[:, 1] -= nb_celllines
    label_pos = coo_matrix((np.ones(pos_pairs.shape[0]), (pos_pairs[:, 0], pos_pairs[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)

    return train_mask, valid_mask, test_mask, train_edge, label_pos



def build_sen_edge_and_hyper_edge_index(allpairs, nb_celllines, nb_drugs):
    """
    allpairs: [N, 3], each row [cell_idx, drug_idx, label]
    Returns:
        sen_edge: [2, E], Regular sensitive edges
        hyper_edge_index: [2, M], Hypergraph edge index
    """
    # sen_edge
    sen_mask = allpairs[:, 2] == 1
    sen_pairs = allpairs[sen_mask]
    sen_edge = np.stack([sen_pairs[:, 0], sen_pairs[:, 1]], axis=0)  # [2, E]

    # Hypergraph edges
    # For each cell line, find all drugs with sen=1 response, form a hyperedge
    hyper_src = []
    hyper_dst = []
    for cell_idx in range(nb_celllines):
        # Find all drugs with sen=1 response for this cell line
        drugs = sen_pairs[sen_pairs[:, 0] == cell_idx][:, 1]
        if len(drugs) == 0:
            continue  # This cell line has no sensitive drugs
        # Hyperedge: cell line node and all sensitive drug nodes
        # PyG hypergraph format: each hyperedge is assigned a hyperedge number
        # Here each hyperedge number is cell_idx
        
        # Hyperedge node numbers: cell_idx + nb_celllines + nb_drugs
        # First nb_celllines: cell lines
        # Next nb_drugs: drugs
        # After that are hyperedge nodes
        hyper_src.append(cell_idx) # Hyperedge node number array (cell line or drug nodes)
        hyper_dst.append(cell_idx + nb_celllines + nb_drugs) # Hyperedge number array
        for drug_idx in drugs:
            hyper_src.append(drug_idx)
            hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
    # Merge
    hyper_edge_index = np.stack([hyper_src, hyper_dst], axis=0)  # [2, M]
    # Result:
    # [[ 0  1  2  0  3] <--- Hyperedge node numbers
    #  [55 55 55 56 56]] <--- Belonging hyperedge numbers
    return sen_edge, hyper_edge_index

def build_sen_res_hyper_edges(allpairs, nb_celllines, nb_drugs):
    """
    allpairs: [N, 3], each row [cell_idx, drug_idx, label]
    Returns:
        sen_edge: [2, E], Edges with label=1
        res_edge: [2, E], Edges with label=-1
        hyper_edge_index: [2, M], Hypergraph edge index
    """
    # sen_edge
    sen_mask = allpairs[:, 2] == 1
    res_mask = allpairs[:, 2] == -1
    sen_pairs = allpairs[sen_mask]
    res_pairs = allpairs[res_mask]
    # sen_edge = np.stack([sen_pairs[:, 0], sen_pairs[:, 1]], axis=0)  # [2, E_sen]
    # res_edge = np.stack([res_pairs[:, 0], res_pairs[:, 1]], axis=0)  # [2, E_res]
    res_edge = np.stack([res_pairs[:, 0], res_pairs[:, 1]], axis=0)
    sen_edge = np.stack([sen_pairs[:, 0], sen_pairs[:, 1]], axis=0)


    print(f"res_pairs drug idx max: {res_pairs[:,1].max()}, nb_drugs: {nb_drugs}")
    print(f"nb_celllines: {nb_celllines}, offset drug idx max: {(res_pairs[:,1] + nb_celllines).max()}")


    # Hypergraph edges
    hyper_src = []
    hyper_dst = []
    for cell_idx in range(nb_celllines):
        drugs = sen_pairs[sen_pairs[:, 0] == cell_idx][:, 1]
        if len(drugs) == 0:
            continue
        hyper_src.append(cell_idx)
        hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
        for drug_idx in drugs:
            hyper_src.append(drug_idx)
            hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
    hyper_edge_index = np.stack([hyper_src, hyper_dst], axis=0)
    return sen_edge, res_edge, hyper_edge_index


def build_gexp_hyper_edge_index(allpairs, nb_celllines, nb_drugs):
    # Expression-based hypergraph edge construction, can follow build_sen_res_hyper_edges
    # This is just an example, can be adjusted based on expression grouping needs
    sen_mask = allpairs[:, 2] == 1
    sen_pairs = allpairs[sen_mask]
    hyper_src = []
    hyper_dst = []
    for cell_idx in range(nb_celllines):
        drugs = sen_pairs[sen_pairs[:, 0] == cell_idx][:, 1]
        if len(drugs) == 0:
            continue
        hyper_src.append(cell_idx)
        hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
        for drug_idx in drugs:
            hyper_src.append(drug_idx)
            hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
    hyper_edge_index = np.stack([hyper_src, hyper_dst], axis=0)
    return torch.LongTensor(hyper_edge_index)

def build_mut_hyper_edge_index(allpairs, nb_celllines, nb_drugs):
    # Mutation-based hypergraph edge construction, can follow build_sen_res_hyper_edges
    sen_mask = allpairs[:, 2] == 1
    sen_pairs = allpairs[sen_mask]
    hyper_src = []
    hyper_dst = []
    for cell_idx in range(nb_celllines):
        drugs = sen_pairs[sen_pairs[:, 0] == cell_idx][:, 1]
        if len(drugs) == 0:
            continue
        hyper_src.append(cell_idx)
        hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
        for drug_idx in drugs:
            hyper_src.append(drug_idx)
            hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
    hyper_edge_index = np.stack([hyper_src, hyper_dst], axis=0)
    return torch.LongTensor(hyper_edge_index)

def build_methy_hyper_edge_index(allpairs, nb_celllines, nb_drugs):
    # Methylation-based hypergraph edge construction, can follow build_sen_res_hyper_edges
    sen_mask = allpairs[:, 2] == 1
    sen_pairs = allpairs[sen_mask]
    hyper_src = []
    hyper_dst = []
    for cell_idx in range(nb_celllines):
        drugs = sen_pairs[sen_pairs[:, 0] == cell_idx][:, 1]
        if len(drugs) == 0:
            continue
        hyper_src.append(cell_idx)
        hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
        for drug_idx in drugs:
            hyper_src.append(drug_idx)
            hyper_dst.append(cell_idx + nb_celllines + nb_drugs)
    hyper_edge_index = np.stack([hyper_src, hyper_dst], axis=0)
    return torch.LongTensor(hyper_edge_index)


def combine_omics_hypergraphs(gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge, nb_celllines):
    """
    Concatenate three omics hypergraphs into a unified hypergraph
    
    Args:
        gexp_hyper_edge: [2, E_gexp] Gene expression hypergraph edges
        mut_hyper_edge: [2, E_mut] Mutation hypergraph edges  
        methy_hyper_edge: [2, E_methy] Methylation hypergraph edges
        nb_celllines: Number of cell lines
    
    Returns:
        combined_hyper_edge: [2, E_total] Combined unified hypergraph edges
    """
    # Use new utility function to combine hypergraphs
    combined_hyper_edge = combine_hypergraphs(gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge)
    
    # Get edge count for each hypergraph for printing information
    E_gexp = gexp_hyper_edge.shape[1]
    E_mut = mut_hyper_edge.shape[1]
    E_methy = methy_hyper_edge.shape[1]
    
    print(f"Combined hypergraph: {combined_hyper_edge.shape[1]} total edges")
    print(f"  - Gene expression: {E_gexp} edges")
    print(f"  - Mutation: {E_mut} edges") 
    print(f"  - Methylation: {E_methy} edges")
    
    return combined_hyper_edge




def process_data(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs, k=5, device='cpu'):
    """
    Enhanced version of process_v2, integrated with cell line hypergraph construction
    Returns: drug_set, cellline_set, allpairs, atom_shape, gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge
    """
    # construct cell line-drug response pairs
    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    pubmedid = list(set([item[1] for item in data_new]))
    pubmedid.sort()
    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))
    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])
    pubmed_num = np.squeeze([[j[1] for j in pubmedmap if i[1] == j[0]] for i in data_new])
    IC_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((cellline_num, pubmed_num, IC_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    # process drug feature
    pubid = [item[0] for item in pubmedmap]
    drug_feature = pd.DataFrame(drug_feature).T
    drug_feature = drug_feature.loc[pubid]
    atom_shape = drug_feature[0][0].shape[-1]
    drug_data = FeatureExtract(drug_feature)

    #----cell line_feature_input
    cellid = [item[0] for item in cellmap]
    gexpr_feature = gexpr_feature.loc[cellid]
    mutation_feature = mutation_feature.loc[cellid]
    methylation_feature = methylation_feature.loc[cellid]

    mutation = torch.from_numpy(np.array(mutation_feature, dtype='float32'))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature, dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation_feature, dtype='float32'))

    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data), collate_fn=collate, batch_size=nb_drugs, shuffle=False, num_workers=0)
    cellline_set = Data.DataLoader(dataset=Data.TensorDataset(mutation, gexpr, methylation), batch_size=nb_celllines, shuffle=False, num_workers=0)

    # Three omics hypergraph edges: based on feature similarity
    mut_hyper_edge, _ = sim_graph_construction(mutation.squeeze(1).squeeze(1).to(device), k, device)
    gexp_hyper_edge, _ = sim_graph_construction(gexpr.to(device), k, device)
    methy_hyper_edge, _ = sim_graph_construction(methylation.to(device), k, device)
    
    print(f"cellline_num min: {cellline_num.min()}, max: {cellline_num.max()}")
    print(f"pubmed_num min: {pubmed_num.min()}, max: {pubmed_num.max()}")
    return drug_set, cellline_set, allpairs, atom_shape, gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge


def process_v3(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs, k=5, device='cpu'):
    # construct cell line-drug response pairs
    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    pubmedid = list(set([item[1] for item in data_new]))
    pubmedid.sort()
    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))
    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])
    pubmed_num = np.squeeze([[j[1] for j in pubmedmap if i[1] == j[0]] for i in data_new])
    IC_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((cellline_num, pubmed_num, IC_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    # process drug feature
    pubid = [item[0] for item in pubmedmap]
    drug_feature = pd.DataFrame(drug_feature).T
    drug_feature = drug_feature.loc[pubid]
    atom_shape = drug_feature[0][0].shape[-1]
    drug_data = FeatureExtract(drug_feature)

    #----cell line_feature_input
    cellid = [item[0] for item in cellmap]
    gexpr_feature = gexpr_feature.loc[cellid]
    mutation_feature = mutation_feature.loc[cellid]
    methylation_feature = methylation_feature.loc[cellid]

    mutation = torch.from_numpy(np.array(mutation_feature, dtype='float32'))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature, dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation_feature, dtype='float32'))

    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data), collate_fn=collate, batch_size=nb_drugs, shuffle=False, num_workers=0)
    cellline_set = Data.DataLoader(dataset=Data.TensorDataset(mutation, gexpr, methylation), batch_size=nb_celllines, shuffle=False, num_workers=0)

    sen_edge, res_edge, hyper_edge_index = build_sen_res_hyper_edges(allpairs, nb_celllines, nb_drugs)
    sen_edge = torch.LongTensor(sen_edge)
    res_edge = torch.LongTensor(res_edge)
    hyper_edge_index = torch.LongTensor(hyper_edge_index)
    
    # Three omics hypergraph edges: based on feature similarity
    mut_hyper_edge, _ = sim_graph_construction(mutation.squeeze(1).squeeze(1).to(device), k, device)
    gexp_hyper_edge, _ = sim_graph_construction(gexpr.to(device), k, device)
    methy_hyper_edge, _ = sim_graph_construction(methylation.to(device), k, device)
    
    # Concatenate three omics hypergraphs into a unified hypergraph
    combined_hyper_edge = combine_omics_hypergraphs(gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge, nb_celllines)
    
    print(f"cellline_num min: {cellline_num.min()}, max: {cellline_num.max()}")
    print(f"pubmed_num min: {pubmed_num.min()}, max: {pubmed_num.max()}")
    return drug_set, cellline_set, allpairs, atom_shape, sen_edge, res_edge, hyper_edge_index, gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge, combined_hyper_edge





