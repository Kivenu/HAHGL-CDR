import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
import numpy as np
from AdaptiveHypergraph import HeteroAdaptiveHypergraph, OmicsAdaptiveHypergraph

from myutils import torch_corr_x_y, scale_sigmoid, build_incidence_matrix_from_edge_index, compute_edge_features_from_incidence, convert_train_edge_to_sen_resistant





# ========== FeatureExtraction ==========
def calculate_mutation_output_dim(dim_mutation):
    """
    Calculate the output dimension after mutation convolution and pooling layers.
    Formula:
    - conv1: (dim_mutation - 700) // 5 + 1
    - pool1: ((dim_mutation - 700) // 5 + 1) // 5
    - conv2: ((((dim_mutation - 700) // 5 + 1) // 5) - 5) // 2 + 1
    - pool2: (((((dim_mutation - 700) // 5 + 1) // 5) - 5) // 2 + 1) // 10
    - flatten: 30 * (((((dim_mutation - 700) // 5 + 1) // 5) - 5) // 2 + 1) // 10
    """
    # conv1 output: (dim_mutation - 700) // 5 + 1
    conv1_out = (dim_mutation - 700) // 5 + 1
    # pool1 output: conv1_out // 5
    pool1_out = conv1_out // 5
    # conv2 output: (pool1_out - 5) // 2 + 1
    conv2_out = (pool1_out - 5) // 2 + 1
    # pool2 output: conv2_out // 10
    pool2_out = conv2_out // 10
    # flatten output: 30 * pool2_out (30 is the number of output channels from conv2)
    return 30 * pool2_out

class Multi_Omics_Ada_Hypergraph_Fusion(nn.Module):
    def __init__(self, dim_drug, drug_layer, dim_gexp, dim_methy, dim_mutation, dim_feat, k, num_layers, dropout, use_bn_at, negative_slope, threshold, device, omics_num_layers=1, tau=10.0):
        super(Multi_Omics_Ada_Hypergraph_Fusion, self).__init__()
        self.device = device
        self.dropout = dropout
        self.leakyRelu = nn.LeakyReLU(negative_slope)
        self.use_bn_at = use_bn_at
        
        # Drug encoder with GCN
        self.drug_layer = drug_layer
        self.drug_conv = GCNConv(dim_drug, drug_layer[0])
        self.drug_graph_bn1 = nn.BatchNorm1d(drug_layer[0])
        self.graph_conv = []
        self.graph_bn = []
        for i in range(len(drug_layer) - 1):
            self.graph_conv.append(GCNConv(drug_layer[i], drug_layer[i + 1]).to(device))
            self.graph_bn.append(nn.BatchNorm1d(drug_layer[i + 1]).to(device))
        self.conv_end = GCNConv(drug_layer[-1], dim_feat)
        self.batch_end = nn.BatchNorm1d(dim_feat)
        
        # Multi-omics encoder
        self.k = k

        self.num_layers = num_layers
        self.gexp_fc1 = nn.Linear(dim_gexp, 256)
        self.gexp_fc2 = nn.Linear(256, dim_feat)
        self.batch_gexp = nn.BatchNorm1d(256)
        self.methy_fc1 = nn.Linear(dim_methy, 256)
        self.methy_fc2 = nn.Linear(256, dim_feat)  
        self.batch_methy = nn.BatchNorm1d(256)
        self.mut_cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.mut_cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.mut_fla = nn.Flatten()
        # Calculate mutation output dimension dynamically
        mut_output_dim = calculate_mutation_output_dim(dim_mutation)
        self.mut_fc = nn.Linear(mut_output_dim, dim_feat)
        self.batch_mut1 = nn.BatchNorm2d(50)
        self.batch_mut2 = nn.BatchNorm2d(30)
        self.batch_mut3 = nn.BatchNorm1d(mut_output_dim)
        
        # Edge feature projection layers (for computing initial edge features)
        self.edge_projection_mut = nn.Linear(dim_feat, dim_feat)
        self.edge_projection_gexp = nn.Linear(dim_feat, dim_feat)
        self.edge_projection_methy = nn.Linear(dim_feat, dim_feat)
        
        # Multi-omics adaptive hypergraph fusion module 
        self.omics_adaptive_hypergraph = OmicsAdaptiveHypergraph(
            dim_feat=dim_feat,
            num_heads=1,  # 1 attention head
            threshold=threshold,  # threshold parameter
            dropout=dropout,
            device=device,
            num_layers=omics_num_layers,  # Number of omics adaptive hypergraph convolution layers
            tau=tau  # tau parameter
        )
        
        # Final output layer
        self.cell_fc = nn.Linear(dim_feat, dim_feat)
        self.batch_all = nn.BatchNorm1d(dim_feat)

    def forward(self, drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, nb_drug=None, 
                gexp_hyper_edge=None, mut_hyper_edge=None, methy_hyper_edge=None, return_participation=False):
        # Drug GCN part
        x_drug = self.drug_conv(drug_feature, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.drug_graph_bn1(x_drug)
        for i in range(len(self.drug_layer) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)
        x_drug_all = global_max_pool(x_drug, drug_batch)
        x_drug_all = x_drug_all[:nb_drug]
        
        # Multi-omics features
        x_mutation = torch.tanh(self.mut_cov1(mutation_data))
        x_mutation = self.batch_mut1(x_mutation)
        x_mutation = F.max_pool2d(x_mutation, (1, 5))
        x_mutation = F.relu(self.mut_cov2(x_mutation))
        x_mutation = self.batch_mut2(x_mutation)
        x_mutation = F.max_pool2d(x_mutation, (1, 10))
        x_mutation = self.mut_fla(x_mutation)
        x_mutation = self.batch_mut3(x_mutation)
        x_mutation = F.relu(self.mut_fc(x_mutation))
        
        x_gexpr = torch.tanh(self.gexp_fc1(gexpr_data))
        x_gexpr = self.batch_gexp(x_gexpr)
        x_gexpr = F.relu(self.gexp_fc2(x_gexpr))
        
        x_methylation = torch.tanh(self.methy_fc1(methylation_data))
        x_methylation = self.batch_methy(x_methylation)
        x_methylation = F.relu(self.methy_fc2(x_methylation))
        
        # original omics features (without GCN processing)
        x_omics_raw = torch.cat((x_mutation, x_gexpr, x_methylation), 1)
        
        # Build incidence matrix and edge features using provided hypergraph edge indices
        # Build mutation omics hypergraph
        if mut_hyper_edge is not None:
            mut_incidence_matrix = build_incidence_matrix_from_edge_index(mut_hyper_edge, x_mutation.shape[0])
            mut_edge_features = compute_edge_features_from_incidence(x_mutation, mut_incidence_matrix, self.edge_projection_mut)
        else:
            mut_incidence_matrix = torch.zeros(x_mutation.shape[0], 0, device=x_mutation.device)
            mut_edge_features = torch.zeros(0, x_mutation.shape[1], device=x_mutation.device)
        
        # Build gene expression omics hypergraph
        if gexp_hyper_edge is not None:
            gexp_incidence_matrix = build_incidence_matrix_from_edge_index(gexp_hyper_edge, x_gexpr.shape[0])
            gexp_edge_features = compute_edge_features_from_incidence(x_gexpr, gexp_incidence_matrix, self.edge_projection_gexp)
        else:
            gexp_incidence_matrix = torch.zeros(x_gexpr.shape[0], 0, device=x_gexpr.device)
            gexp_edge_features = torch.zeros(0, x_gexpr.shape[1], device=x_gexpr.device)
        
        # Build methylation omics hypergraph
        if methy_hyper_edge is not None:
            methy_incidence_matrix = build_incidence_matrix_from_edge_index(methy_hyper_edge, x_methylation.shape[0])
            methy_edge_features = compute_edge_features_from_incidence(x_methylation, methy_incidence_matrix, self.edge_projection_methy)
        else:
            methy_incidence_matrix = torch.zeros(x_methylation.shape[0], 0, device=x_methylation.device)
            methy_edge_features = torch.zeros(0, x_methylation.shape[1], device=x_methylation.device)
        
        # Use multi-omics adaptive hypergraph fusion module
        
        if return_participation:
            x_cell, omics_participation = self.omics_adaptive_hypergraph(
                x_mutation, x_gexpr, x_methylation, 
                mut_edge_features, gexp_edge_features, methy_edge_features,
                return_participation=True
            )
        else:
            x_cell = self.omics_adaptive_hypergraph(
                x_mutation, x_gexpr, x_methylation, 
                mut_edge_features, gexp_edge_features, methy_edge_features
            )
            omics_participation = None
        
        x_all = torch.cat((x_cell, x_drug_all), 0)
        if self.use_bn_at:
            x_all = self.batch_all(x_all)
        
        if return_participation:
            return x_all, x_omics_raw, omics_participation
        return x_all, x_omics_raw


class Hetero_Ada_Hypergraph_Learning(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_bn, negative_slope, device, num_layers=2, dropout=0.4, num_pipe_nodes=10, threshold=0.8, tau=10.0):
        super().__init__()
        self.device = device
        self.use_bn = use_bn
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_pipe_nodes = num_pipe_nodes
        self.threshold = threshold
        
        # Hetero adaptive hypergraph module
        self.sen_adaptive_hypergraph = HeteroAdaptiveHypergraph(
            dim_feat=in_channels,
            threshold=threshold,
            dropout=dropout,
            device=device,
            num_layers=num_layers,
            tau=tau  # tau parameter
        )
        

        self.sen_adaptive_projection = nn.Linear(in_channels, out_channels)
        

        self.cell_fc = nn.Linear(out_channels, out_channels)
        self.drug_fc = nn.Linear(out_channels, out_channels)
    
    def forward(self, feature, sen_edge, resistant_edge, nb_celllines, nb_drugs, return_participation=False):
        # Hetero adaptive hypergraph convolution 
        if return_participation:
            x_sen_adaptive, sen_participation = self.sen_adaptive_hypergraph(feature, sen_edge, resistant_edge, return_participation=True)
        else:
            x_sen_adaptive = self.sen_adaptive_hypergraph(feature, sen_edge, resistant_edge)
            sen_participation = None
            
        x_sen_adaptive_mapped = self.sen_adaptive_projection(x_sen_adaptive)
        
        # Separate cell line and drug features
        adaptive_cell = x_sen_adaptive_mapped[:nb_celllines]
        adaptive_drug = x_sen_adaptive_mapped[nb_celllines:nb_celllines+nb_drugs]
        
        # Process cell line and drug features through MLP
        x_cell = self.cell_fc(adaptive_cell)
        x_drug = self.drug_fc(adaptive_drug)
        x_cell = F.relu(x_cell)
        x_drug = F.relu(x_drug)
        
        # Combine into final output
        x_out = torch.cat([x_cell, x_drug], dim=0)
        
        if return_participation:
            return x_sen_adaptive_mapped, x_out, sen_participation
        return x_sen_adaptive_mapped, x_out

class HAHGL_CDR(nn.Module):
    def __init__(self, dim_drug, drug_layer, dim_gexp, dim_methy, dim_mutation, dim_feat, out_channels, k, num_layers, dropout, use_bn_at, use_bn_as, negative_slope, device, alpha=8, num_pipe_nodes=10, threshold=0.8, omics_num_layers=1, tau=10.0):
        super().__init__()
        self.branch_at = Multi_Omics_Ada_Hypergraph_Fusion(dim_drug, drug_layer, dim_gexp, dim_methy, dim_mutation, dim_feat, k, num_layers, dropout, use_bn_at, negative_slope, threshold, device, omics_num_layers=omics_num_layers, tau=tau)
        self.branch_as = Hetero_Ada_Hypergraph_Learning(dim_feat, out_channels, out_channels, use_bn_as, negative_slope, device, num_layers, dropout, num_pipe_nodes=num_pipe_nodes, threshold=threshold, tau=tau)
        self.alpha = alpha
    
    def forward(self, drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, train_edge, nb_celllines, nb_drugs,
                gexp_hyper_edge=None, mut_hyper_edge=None, methy_hyper_edge=None):
        # Convert train_edge to sen_edge and resistant_edge format
        sen_edge, resistant_edge = convert_train_edge_to_sen_resistant(train_edge, drug_feature.device)
            
        initial_feature, x_omics_raw = self.branch_at(drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, nb_drug=nb_drugs,
                                                      gexp_hyper_edge=gexp_hyper_edge, mut_hyper_edge=mut_hyper_edge, methy_hyper_edge=methy_hyper_edge)
        
        # Hetero adaptive hypergraph processing 
        x_sen_adaptive_mapped, x_out = self.branch_as(initial_feature, sen_edge, resistant_edge, nb_celllines, nb_drugs)
        
        feat_cell = x_out[:nb_celllines, :]
        feat_drug = x_out[nb_celllines:nb_celllines+nb_drugs, :]
        corr = torch_corr_x_y(feat_cell, feat_drug)
        final_prob = scale_sigmoid(corr, alpha=self.alpha)
        
        return final_prob.view(-1)
    
    def get_participation_matrices(self, drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, train_edge, nb_celllines, nb_drugs,
                                   gexp_hyper_edge=None, mut_hyper_edge=None, methy_hyper_edge=None):
        """
        Get incidence matrices of two adaptive hypergraphs
        Args:
            Same input parameters as forward
        Returns:
            omics_participation: incidence matrix of Omics adaptive hypergraph [N, num_omics_edges]
            sen_participation: incidence matrix of Hetero adaptive hypergraph [N, num_sen_edges]
        """
        # Convert train_edge to sen_edge and resistant_edge format
        sen_edge, resistant_edge = convert_train_edge_to_sen_resistant(train_edge, drug_feature.device)
        
        # Multi-omics adaptive hypergraph fusion 
        initial_feature, x_omics_raw, omics_participation = self.branch_at(
            drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, 
            nb_drug=nb_drugs, gexp_hyper_edge=gexp_hyper_edge, mut_hyper_edge=mut_hyper_edge, 
            methy_hyper_edge=methy_hyper_edge, return_participation=True
        )
        
        # Hetero adaptive hypergraph learning 
        x_sen_adaptive_mapped, x_out, sen_participation = self.branch_as(
            initial_feature, sen_edge, resistant_edge, nb_celllines, nb_drugs, return_participation=True
        )
        
        return omics_participation, sen_participation
    
    def compute_total_loss(self, predictions, targets, supervision_weight=1.0):
        """
        Compute total loss: supervision loss only
        Args:
            predictions: Model prediction results
            targets: Ground truth labels
            supervision_weight: Supervision loss weight
        Returns:
            total_loss: Total loss
            supervision_loss: Supervision loss
        """
        # Supervision loss (binary cross entropy)
        supervision_loss = F.binary_cross_entropy(predictions, targets)
        
        # Total loss 
        total_loss = supervision_weight * supervision_loss
        
        return total_loss, supervision_loss 