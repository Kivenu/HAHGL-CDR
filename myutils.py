from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
from sklearn import metrics
import torch.nn as nn




def torch_corr_x_y(tensor1, tensor2, eps=1e-8):
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
    lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))
    lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))
    std_x_y = torch.mm(torch.sqrt(lxx + eps).view([-1, 1]), torch.sqrt(lyy + eps).view([1, -1]))
    corr_x_y = torch.div(lxy, std_x_y + eps)
    return corr_x_y

def scale_sigmoid(tensor, alpha):
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    out = torch.sigmoid(torch.mul(alpha, tensor))
    return out

def build_incidence_matrix_from_edge_index(edge_index, num_nodes):
    """
    Build incidence matrix from edge index
    Args:
        edge_index: [2, num_edges] Edge index
        num_nodes: Total number of nodes
    Returns:
        incidence_matrix: [num_nodes, num_edges] Incidence matrix
    """
    if edge_index.shape[1] == 0:
        return torch.zeros(num_nodes, 0, device=edge_index.device)
    
    # Create incidence matrix
    incidence_matrix = torch.zeros(num_nodes, edge_index.shape[1], device=edge_index.device)
    
    # Fill incidence matrix
    incidence_matrix[edge_index[0], torch.arange(edge_index.shape[1])] = 1
    incidence_matrix[edge_index[1], torch.arange(edge_index.shape[1])] = 1
    
    return incidence_matrix

def compute_edge_features_from_incidence(node_features, incidence_matrix, edge_projection):
    """
    Compute edge features from incidence matrix: average pooling of all node features connected by each edge, then project
    Args:
        node_features: Node features [N, dim_feat]
        incidence_matrix: Incidence matrix [N, num_edges]
        edge_projection: Edge feature projection layer
    Returns:
        edge_features: Edge features [num_edges, dim_feat]
    """
    import torch.nn.functional as F
    
    if incidence_matrix.shape[1] == 0:
        return torch.zeros(0, node_features.shape[1], device=node_features.device)
    
    # Calculate the number of nodes connected by each edge
    edge_node_counts = incidence_matrix.sum(dim=0, keepdim=True)  # [1, num_edges]
    
    # Calculate average node features for each edge
    edge_features = torch.mm(incidence_matrix.T, node_features) / (edge_node_counts.T + 1e-8)
    
    # Project edge features
    edge_features = edge_projection(edge_features)
    edge_features = F.leaky_relu(edge_features)
    
    return edge_features

def convert_train_edge_to_sen_resistant(train_edge, device):
    """
    Convert training edges to sen_edge and resistant_edge format
    Args:
        train_edge: Training edges, format [num_edges, 3], each row is [cell_idx, drug_idx, label]
        device: Target device (cuda or cpu)
    Returns:
        sen_edge: Sensitive edges (positive samples), format [2, num_positive_edges]
        resistant_edge: Resistant edges (negative samples), format [2, num_negative_edges]
    """
    sen_edge = None
    resistant_edge = None
    
    if train_edge is not None and len(train_edge) > 0:
        # train_edge format: [cell_idx, drug_idx, label]
        # Separate positive samples (label=1) and negative samples (label=0)
        positive_mask = train_edge[:, 2] == 1
        negative_mask = train_edge[:, 2] == 0
        
        positive_edges = train_edge[positive_mask]
        negative_edges = train_edge[negative_mask]
        
        # Convert to sen_edge format [2, num_edges] (label=1)
        if len(positive_edges) > 0:
            sen_edge = torch.tensor(positive_edges[:, [0, 1]].T, dtype=torch.long, device=device)
        
        # Convert to resistant_edge format [2, num_edges] (label=0)
        if len(negative_edges) > 0:
            resistant_edge = torch.tensor(negative_edges[:, [0, 1]].T, dtype=torch.long, device=device)
    
    return sen_edge, resistant_edge

class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features, edge_index = data_mol[0], data_mol[1]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GraphDatasetWithBonds(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDatasetWithBonds, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features, edge_index, edge_attr = data_mol[0], data_mol[1], data_mol[2]
            GCNData = DATA.Data(
                x=torch.Tensor(features), 
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.Tensor(edge_attr)
            )
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA


def collate_with_bonds(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA



def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    
    # Move data to CPU to avoid GPU memory issues
    true_data_cpu = true_data.detach().cpu()
    predict_data_cpu = predict_data.detach().cpu()
    
    # Use sklearn's precision_recall_curve to calculate F1 score
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Get unique thresholds
    thresholds = torch.unique(predict_data_cpu)
    
    # If too many thresholds, sample some to reduce computation
    if len(thresholds) > 1000:
        # Uniformly sample 1000 thresholds
        step = len(thresholds) // 1000
        thresholds = thresholds[::step]
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    # Calculate F1 score for each threshold
    for threshold in thresholds:
        # Convert predictions to binary
        pred_binary = (predict_data_cpu >= threshold).float()
        
        # Calculate F1 score
        f1 = f1_score(true_data_cpu.numpy(), pred_binary.numpy(), zero_division='warn')
        
        if isinstance(f1, (int, float)) and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold.item()
    
    return torch.tensor(best_f1, device=true_data.device), torch.tensor(best_threshold, device=true_data.device)


def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    
    # Move data to CPU to avoid GPU memory issues
    true_data_cpu = true_data.detach().cpu()
    predict_data_cpu = predict_data.detach().cpu()
    threshold_cpu = threshold.detach().cpu() if isinstance(threshold, torch.Tensor) else threshold
    
    n = true_data_cpu.size()[0]
    ones = torch.ones(n, dtype=torch.float32)
    zeros = torch.zeros(n, dtype=torch.float32)
    predict_value = torch.where(predict_data_cpu.ge(threshold_cpu), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data_cpu), ones, zeros))
    score = torch.div(tpn, n)
    return score.to(true_data.device)


def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    
    # Move data to CPU to avoid GPU memory issues
    true_data_cpu = true_data.detach().cpu()
    predict_data_cpu = predict_data.detach().cpu()
    threshold_cpu = threshold.detach().cpu() if isinstance(threshold, torch.Tensor) else threshold
    
    ones = torch.ones(true_data_cpu.size()[0], dtype=torch.float32)
    zeros = torch.zeros(true_data_cpu.size()[0], dtype=torch.float32)
    predict_value = torch.where(predict_data_cpu.ge(threshold_cpu), ones, zeros)
    tp = torch.sum(torch.mul(true_data_cpu, predict_value))
    true_neg = torch.sub(ones, true_data_cpu)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score.to(true_data.device)


def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    
    # Move data to CPU to avoid GPU memory issues
    true_data_cpu = true_data.detach().cpu()
    predict_data_cpu = predict_data.detach().cpu()
    threshold_cpu = threshold.detach().cpu() if isinstance(threshold, torch.Tensor) else threshold
    
    ones = torch.ones(true_data_cpu.size()[0], dtype=torch.float32)
    zeros = torch.zeros(true_data_cpu.size()[0], dtype=torch.float32)
    predict_value = torch.where(predict_data_cpu.ge(threshold_cpu), ones, zeros)
    tp = torch.sum(torch.mul(true_data_cpu, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data_cpu))
    score = torch.div(tp, torch.add(tp, fn))
    return score.to(true_data.device)


def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    
    # Move data to CPU to avoid GPU memory issues
    true_data_cpu = true_data.detach().cpu()
    predict_data_cpu = predict_data.detach().cpu()
    threshold_cpu = threshold.detach().cpu() if isinstance(threshold, torch.Tensor) else threshold
    
    ones = torch.ones(true_data_cpu.size()[0], dtype=torch.float32)
    zeros = torch.zeros(true_data_cpu.size()[0], dtype=torch.float32)
    predict_value = torch.where(predict_data_cpu.ge(threshold_cpu), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data_cpu)
    tp = torch.sum(torch.mul(true_data_cpu, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data_cpu, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32)
    score = torch.div((tp * tn - fp * fn), torch.add(delta, torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    return score.to(true_data.device)



def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt.detach().cpu().numpy(), yp.detach().cpu().numpy())
    aupr = metrics.auc(recall, precision)
    # aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy())
    #---f1,acc,recall, specificity, precision
    f1, thresholds = f1_score_binary(yt, yp)
    acc = accuracy_binary(yt, yp, thresholds)
    precision = precision_binary(yt, yp, thresholds)
    recall = recall_binary(yt, yp, thresholds)
    mcc = mcc_binary(yt, yp, thresholds)

    return auc, aupr, f1.detach().cpu().numpy(), acc.detach().cpu().numpy(), precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), mcc.detach().cpu().numpy()


# ============ Similarity computation and hyperedge construction utility functions ============

def compute_cosine_similarity(features1, features2):
    """
    Compute cosine similarity between two sets of features
    
    Args:
        features1: [N1, dim] torch.Tensor First set of features
        features2: [N2, dim] torch.Tensor Second set of features
    
    Returns:
        similarity: [N1, N2] torch.Tensor Similarity matrix
    """
    # Ensure features are on the same device
    features2 = features2.to(features1.device)
    
    # Normalize features
    features1_norm = torch.nn.functional.normalize(features1, p=2, dim=1)
    features2_norm = torch.nn.functional.normalize(features2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(features1_norm, features2_norm.t())
    
    return similarity


def sim_graph_construction(feature, k, device):
    """
    Build k-NN graph based on feature similarity (from sim_graph_construction function in process_data.py)
    
    Args:
        feature: [N, D] torch.Tensor Node features
        k: int Number of top-k neighbors
        device: str Device type
    
    Returns:
        edge_index: [2, N*k] torch.Tensor Edge index
        edge_weights: [N*k] torch.Tensor Edge weights
    """
    sim = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-10)
    sim = torch.mm(sim, sim.T)
    diag = torch.diag(sim)
    sim = sim - torch.diag_embed(diag)
    tmp = torch.topk(sim, dim=1, k=k)
    index = torch.arange(tmp.values.shape[0]).unsqueeze(1).to(device)
    edge = torch.empty((tmp.values.shape[0] * k, 2), dtype=torch.long)
    for i in range(k):
        index_tmp = torch.cat((index, tmp.indices[:, i].unsqueeze(1)), 1)
        edge[tmp.values.shape[0] * i:tmp.values.shape[0] * (i + 1), :] = index_tmp
    edge = edge.T.to(device)
    edge_weights = tmp.values.t().reshape(-1).to(device)
    return edge, edge_weights


def build_knn_hyper_edges(similarity_matrix, k, node_offset=0, device='cpu'):
    """
    Build k-NN hyperedges based on similarity matrix
    
    Args:
        similarity_matrix: [N1, N2] torch.Tensor Similarity matrix
        k: int Number of neighbors each node connects to
        node_offset: int Node index offset
        device: str Device type
    
    Returns:
        hyper_edge_index: [2, E] torch.Tensor Hyperedge index
    """
    hyper_edges = []
    
    for i in range(similarity_matrix.shape[0]):
        # Get top-k similar nodes
        _, top_indices = torch.topk(similarity_matrix[i], min(k, similarity_matrix.shape[1]))
        
        # Build hyperedges for current node
        for j in top_indices:
            hyper_edges.append([node_offset + i, node_offset + j.item()])
    
    if len(hyper_edges) > 0:
        hyper_edge_index = torch.tensor(hyper_edges, dtype=torch.long, device=device).t()
    else:
        hyper_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return hyper_edge_index


def build_pipe_node_hyper_edges(cell_features, drug_features, pipe_node_features, threshold, device='cpu'):
    """
    Build hyperedges based on pipeNode (using threshold method)
    
    Args:
        cell_features: [num_cells, dim] torch.Tensor Cell line features
        drug_features: [num_drugs, dim] torch.Tensor Drug features  
        pipe_node_features: [num_pipe_nodes, dim] torch.Tensor PipeNode features
        threshold: float Similarity threshold
        device: str Device type
    
    Returns:
        hyper_edge_index: [2, E] torch.Tensor Hyperedge index
        cell_pipe_sim: [num_cells, num_pipe_nodes] torch.Tensor Similarity between cell lines and pipeNode
        drug_pipe_sim: [num_drugs, num_pipe_nodes] torch.Tensor Similarity between drugs and pipeNode
    """
    num_cells = cell_features.shape[0]
    num_drugs = drug_features.shape[0]
    num_pipe_nodes = pipe_node_features.shape[0]
    
    # Compute similarity between cell lines and pipeNode
    cell_pipe_sim = compute_cosine_similarity(cell_features, pipe_node_features)
    
    # Compute similarity between drugs and pipeNode
    drug_pipe_sim = compute_cosine_similarity(drug_features, pipe_node_features)
    
    # Build hyperedges
    hyper_edges = []
    
    # Build hyperedges for each pipeNode
    for pipe_idx in range(num_pipe_nodes):
        # PipeNode index after concatenation
        pipe_node_idx = num_cells + num_drugs + pipe_idx
        
        # Find cell lines similar to current pipeNode (using threshold)
        cell_similarities = cell_pipe_sim[:, pipe_idx]
        similar_cells = torch.where(cell_similarities > threshold)[0]
        
        # Find drugs similar to current pipeNode (using threshold)
        drug_similarities = drug_pipe_sim[:, pipe_idx]
        similar_drugs = torch.where(drug_similarities > threshold)[0]
        
        # Build hyperedge: include pipeNode and similar cell lines, drugs
        if len(similar_cells) > 0 or len(similar_drugs) > 0:
            # Add cell lines to hyperedge
            for cell_idx in similar_cells:
                hyper_edges.append([cell_idx.item(), pipe_node_idx])
            
            # Add drugs to hyperedge
            for drug_idx in similar_drugs:
                drug_node_idx = num_cells + drug_idx.item()
                hyper_edges.append([drug_node_idx, pipe_node_idx])
    
    # Convert to torch tensor
    if len(hyper_edges) > 0:
        hyper_edge_index = torch.tensor(hyper_edges, dtype=torch.long, device=device).t()
        # Ensure indices are within valid range
        max_node_idx = num_cells + num_drugs + num_pipe_nodes - 1
        hyper_edge_index = torch.clamp(hyper_edge_index, 0, max_node_idx)
    else:
        # Create empty hyperedge index
        hyper_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return hyper_edge_index, cell_pipe_sim, drug_pipe_sim


def combine_hypergraphs(*hyper_edge_indices):
    """
    Combine multiple hypergraphs
    
    Args:
        *hyper_edge_indices: Multiple hyperedge index tensors [2, E_i]
    
    Returns:
        combined_hyper_edge: [2, E_total] torch.Tensor Combined hyperedge index
    """
    if len(hyper_edge_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Get edge count for each hypergraph
    edge_counts = [hyper_edge.shape[1] for hyper_edge in hyper_edge_indices]
    total_edges = sum(edge_counts)
    
    if total_edges == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Calculate hyperedge number offset and adjust
    adjusted_hyper_edges = []
    current_offset = 0
    
    for i, hyper_edge in enumerate(hyper_edge_indices):
        if hyper_edge.shape[1] > 0:
            adjusted_hyper_edge = hyper_edge.clone()
            # Adjust hyperedge numbers (second row)
            adjusted_hyper_edge[1, :] += current_offset
            adjusted_hyper_edges.append(adjusted_hyper_edge)
        current_offset += edge_counts[i]
    
    # Concatenate all hypergraphs
    if adjusted_hyper_edges:
        combined_hyper_edge = torch.cat(adjusted_hyper_edges, dim=1)
    else:
        combined_hyper_edge = torch.empty((2, 0), dtype=torch.long)
    
    return combined_hyper_edge
