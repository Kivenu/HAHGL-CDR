import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdActivation(nn.Module):
    """
    Fixed threshold activation function, used as a replacement for ReLU
    Activates similarities > threshold to near 1, and < threshold to near 0
    f(x)=σ(τ⋅(x−θ))
    """
    def __init__(self, tau=10.0, threshold=0.8):
        super().__init__()
        self.tau = tau  # Fixed tau parameter
        self.threshold = threshold  # Fixed threshold

    def forward(self, similarity):
        """
        Args:
            similarity: Similarity matrix
        Returns:
            Activated matrix, range [0, 1]
        """
        return torch.sigmoid(self.tau * (similarity - self.threshold))


class AdaptiveHypergraph(nn.Module):
    def __init__(self, dim_feat, threshold=0.8, dropout=0.1, device='cuda', num_layers=1, tau=10.0):
        super(AdaptiveHypergraph, self).__init__()
        self.dim_feat = dim_feat
        self.threshold = threshold
        self.dropout = dropout
        self.device = device
        self.num_layers = num_layers
        self.tau = tau
        
        # Fixed threshold activation function
        self.threshold_activation = ThresholdActivation(tau=tau, threshold=threshold)
        
        # Edge feature projection layer
        self.edge_projection = nn.Linear(dim_feat, dim_feat)
        
        # Multi-layer adaptive hypergraph message passing projection layers
        self.edge_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_feat, dim_feat),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_layers)
        ])
        
        self.node_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_feat, dim_feat),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_layers)
        ])
        
        # BatchNorm layers
        self.node_bns = nn.ModuleList([
            nn.BatchNorm1d(dim_feat) for _ in range(num_layers)
        ])
    
    def compute_edge_features_from_edges(self, node_features, edge_index, edge_projection=None):
        if edge_index.shape[1] == 0:
            return torch.zeros(0, node_features.shape[1], device=node_features.device)
        
        if edge_projection is None:
            edge_projection = self.edge_projection
        
        # Get features of two nodes connected by each edge
        src_features = node_features[edge_index[0]]  # [num_edges, dim_feat]
        dst_features = node_features[edge_index[1]]  # [num_edges, dim_feat]
        
        # Average pooling
        edge_features = (src_features + dst_features) / 2.0
        
        # Project edge features
        edge_features = edge_projection(edge_features)
        edge_features = F.leaky_relu(edge_features)
        
        return edge_features
    
    def compute_edge_features_from_incidence(self, node_features, incidence_matrix, edge_projection=None):
        if incidence_matrix.shape[1] == 0:
            return torch.zeros(0, node_features.shape[1], device=node_features.device)
        
        if edge_projection is None:
            edge_projection = self.edge_projection
        
        # Calculate the number of nodes connected by each edge
        edge_node_counts = incidence_matrix.sum(dim=0, keepdim=True)  # [1, num_edges]
        
        # Calculate average node features for each edge
        edge_features = torch.mm(incidence_matrix.T, node_features) / (edge_node_counts.T + 1e-8)
        
        # Project edge features
        edge_features = edge_projection(edge_features)
        edge_features = F.leaky_relu(edge_features)
        
        return edge_features
    
    def compute_adaptive_incidence(self, node_features, edge_features):
        if edge_features.shape[0] == 0:
            return torch.zeros(node_features.shape[0], 0, device=node_features.device)
        
        # Compute cosine similarity using original features directly
        node_norm = F.normalize(node_features, dim=1)  # [N, dim_feat]
        edge_norm = F.normalize(edge_features, dim=1)  # [num_edges, dim_feat]
        
        # Compute cosine similarity, range [-1, 1]
        similarity = torch.mm(node_norm, edge_norm.T)  # [N, num_edges]
        
        # Use threshold_activation
        adaptive_incidence = self.threshold_activation(similarity)
        
        return adaptive_incidence
    
    def adaptive_hypergraph_convolution(self, node_features, adaptive_incidence_matrix):
        if adaptive_incidence_matrix.shape[1] == 0:
            return node_features, torch.zeros(0, node_features.shape[1], device=node_features.device)
            
        N, E = adaptive_incidence_matrix.shape
        
        # Save original node features for residual connection
        residual = node_features
        
        # Calculate degree vectors (avoid constructing diagonal matrix)
        eps = 1e-8
        D_v = adaptive_incidence_matrix.sum(dim=1)  # Node degree vector [N]
        D_e = adaptive_incidence_matrix.sum(dim=0)  # Edge degree vector [E]
        
        # Calculate normalization factors
        D_v_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(D_v, min=eps))  # [N]
        D_e_inv = 1.0 / torch.clamp(D_e, min=eps)  # [E]
        
        current_node_features = node_features
        final_edge_features = None
        
        # Perform multi-layer adaptive hypergraph convolution
        for layer_idx in range(self.num_layers):
            # Step 1: Node to edge features
            # Compute edge features: He = D_e^(-1) * H^T * D_v^(-1/2) * X
            # Use broadcasting to avoid matrix multiplication
            normalized_node_features = current_node_features * D_v_inv_sqrt.unsqueeze(1)  # [N, dim_feat]
            edge_features = torch.mm(adaptive_incidence_matrix.T, normalized_node_features)  # [E, dim_feat]
            edge_features = edge_features * D_e_inv.unsqueeze(1)  # [E, dim_feat]
            
            # Project and activate edge features
            edge_features = self.edge_projs[layer_idx](edge_features)
            
            # Step 2: Edge features to nodes
            # Update node features: X_new = D_v^(-1/2) * H * He
            updated_node_features = torch.mm(adaptive_incidence_matrix, edge_features)  # [N, dim_feat]
            updated_node_features = updated_node_features * D_v_inv_sqrt.unsqueeze(1)  # [N, dim_feat]
            
            # Project and activate node features
            updated_node_features = self.node_projs[layer_idx](updated_node_features)
            
            # BatchNorm → Dropout → Residual connection
            updated_node_features = self.node_bns[layer_idx](updated_node_features)
            updated_node_features = F.dropout(updated_node_features, self.dropout, training=self.training)
            
            # Residual connection (each layer connects to original input except first layer)
            if layer_idx == 0:
                # First layer connects to original input
                if residual.shape[1] == updated_node_features.shape[1]:
                    current_node_features = updated_node_features + residual
                else:
                    current_node_features = updated_node_features
            else:
                # Subsequent layers connect to previous layer output
                if current_node_features.shape[1] == updated_node_features.shape[1]:
                    current_node_features = updated_node_features + current_node_features
                else:
                    current_node_features = updated_node_features
            
            # Save edge features from the last layer
            final_edge_features = edge_features
        
        return current_node_features, final_edge_features
    
    def build_incidence_matrix_from_edge_index(self, edge_index, num_nodes):
        if edge_index.shape[1] == 0:
            return torch.zeros(num_nodes, 0, device=edge_index.device)
        
        # Create incidence matrix
        incidence_matrix = torch.zeros(num_nodes, edge_index.shape[1], device=edge_index.device)
        
        # Fill incidence matrix
        incidence_matrix[edge_index[0], torch.arange(edge_index.shape[1])] = 1
        incidence_matrix[edge_index[1], torch.arange(edge_index.shape[1])] = 1
        
        return incidence_matrix
    
    def incidence_to_edge_index(self, incidence_matrix):
        if incidence_matrix.numel() == 0 or incidence_matrix.shape[1] == 0:
            return torch.empty(2, 0, device=incidence_matrix.device, dtype=torch.long)
        
        # Find positions of non-zero elements
        node_indices, edge_indices = torch.nonzero(incidence_matrix, as_tuple=True)
        
        # Build edge index
        edge_index = torch.stack([node_indices, edge_indices], dim=0)
        
        return edge_index
    
    def forward(self, node_features, edge_index):
        # 1. Compute edge features
        edge_features = self.compute_edge_features_from_edges(node_features, edge_index)
        
        # 2. Compute adaptive incidence matrix
        adaptive_incidence = self.compute_adaptive_incidence(node_features, edge_features)
        
        # 3. Adaptive hypergraph convolution
        updated_node_features, _ = self.adaptive_hypergraph_convolution(node_features, adaptive_incidence)
        
        return updated_node_features


class HeteroAdaptiveHypergraph(AdaptiveHypergraph):
    def __init__(self, dim_feat, threshold=0.8, dropout=0.1, device='cuda', num_layers=1, tau=10.0):
        super(HeteroAdaptiveHypergraph, self).__init__(
            dim_feat=dim_feat,
            threshold=threshold,
            dropout=dropout,
            device=device,
            num_layers=num_layers,
            tau=tau
        )
        
        # Debug counter
        self._debug_counter = 0
    
    def compute_adaptive_incidence_with_mask(self, node_features, edge_features, sen_edge, resistant_edge=None):
        """
        Compute adaptive hyperedge incidence matrix (with enhanced mask processing)
        Process: cosine similarity → threshold_activation → enhanced mask processing (Plan A)
        
        Enhanced mask (Plan A - edge-specific exclusion):
        1) Set participation of sen=1 to 1
        2) Exclude resistant nodes only in sen edges related to resistant
        
        Args:
            node_features: Node features [N, dim_feat]
            edge_features: Edge features [num_edges, dim_feat]
            sen_edge: Sen graph edge index [2, num_sen_edges]
            resistant_edge: Resistant graph edge index [2, num_resistant_edges]
        Returns:
            adaptive_incidence: Adaptive incidence matrix [N, num_edges]
        """
        if edge_features.shape[0] == 0:
            return torch.zeros(node_features.shape[0], 0, device=node_features.device)
        
        # Compute cosine similarity using original features directly
        node_norm = F.normalize(node_features, dim=1)  # [N, dim_feat]
        edge_norm = F.normalize(edge_features, dim=1)  # [num_edges, dim_feat]
        
        # Compute cosine similarity, range [-1, 1]
        similarity = torch.mm(node_norm, edge_norm.T)  # [N, num_edges]
        
        # Use threshold_activation
        adaptive_incidence = self.threshold_activation(similarity)
        
        # Enhanced mask processing (Plan A)
        if sen_edge is not None and sen_edge.shape[1] > 0:
            # Create sen mask: set participation of sen=1 cell-drug pairs to 1
            sen_mask = torch.zeros_like(adaptive_incidence)
            for i in range(sen_edge.shape[1]):
                cell_idx = sen_edge[0, i].item()
                drug_idx = sen_edge[1, i].item()
                # In adaptive_incidence matrix, rows are nodes, columns are edges
                # Need to find corresponding edge index, assume edge order matches sen_edge
                if i < adaptive_incidence.shape[1]:
                    sen_mask[cell_idx, i] = 1.0
                    sen_mask[drug_idx, i] = 1.0
            
            # Create resistant mask (Plan A: edge-specific exclusion)
            resistant_mask = torch.ones_like(adaptive_incidence)  # Default: keep all participation
            if resistant_edge is not None and resistant_edge.shape[1] > 0:
                # Iterate over each resistant pair
                for res_idx in range(resistant_edge.shape[1]):
                    res_cell_idx = resistant_edge[0, res_idx].item()
                    res_drug_idx = resistant_edge[1, res_idx].item()
                    
                    # Iterate over all sen edges to find edges related to this resistant pair
                    for sen_idx in range(sen_edge.shape[1]):
                        sen_cell_idx = sen_edge[0, sen_idx].item()
                        sen_drug_idx = sen_edge[1, sen_idx].item()
                        
                        # If sen edge contains resistant cell or drug
                        if res_cell_idx == sen_cell_idx or res_drug_idx == sen_drug_idx:
                            # Exclude resistant cell and drug only in this edge
                            resistant_mask[res_cell_idx, sen_idx] = 0.0
                            resistant_mask[res_drug_idx, sen_idx] = 0.0
            
            # Apply enhanced mask: first apply resistant mask (edge-specific zeroing), then apply sen mask (set to one)
            adaptive_incidence = adaptive_incidence * resistant_mask  # Zero resistant participation in related edges
            adaptive_incidence = torch.where(sen_mask > 0, torch.ones_like(adaptive_incidence), adaptive_incidence)  # Set sen participation to one
            
            # Debug info: output number of non-zero participation nodes in each adaptive edge
            self._debug_counter += 1
            
            # Output debug info only during training and every 10 forward passes
            if self.training and self._debug_counter % 10 == 1:
                num_edges = adaptive_incidence.shape[1]
                non_zero_counts = []
                for edge_idx in range(num_edges):
                    # Calculate number of nodes with non-zero participation in this edge
                    non_zero_count = (adaptive_incidence[:, edge_idx] > 1e-6).sum().item()
                    non_zero_counts.append(non_zero_count)
                
                
        
        return adaptive_incidence
    
    def forward(self, node_features, sen_edge, resistant_edge=None, return_participation=False):
        """
        Forward propagation
        Args:
            node_features: Node features [N, dim_feat]
            sen_edge: Sen graph edge index [2, num_sen_edges]
            resistant_edge: Resistant graph edge index [2, num_resistant_edges]
            return_participation: Whether to return participation matrix
        Returns:
            adaptive_conv_features: Node features after adaptive hypergraph convolution [N, dim_feat]
            adaptive_incidence (optional): Adaptive incidence matrix [N, num_edges]
        """
        # 1. Compute edge features of sen graph
        sen_edge_features = self.compute_edge_features_from_edges(
            node_features, sen_edge, self.edge_projection
        ) if sen_edge is not None and sen_edge.shape[1] > 0 else torch.zeros(0, node_features.shape[1], device=node_features.device)
        
        # 2. Compute adaptive incidence matrix (with enhanced mask processing)
        adaptive_incidence = self.compute_adaptive_incidence_with_mask(node_features, sen_edge_features, sen_edge, resistant_edge)
        
        # 3. Adaptive hypergraph convolution
        adaptive_conv_features, _ = self.adaptive_hypergraph_convolution(node_features, adaptive_incidence)
        
        if return_participation:
            return adaptive_conv_features, adaptive_incidence
        return adaptive_conv_features


class OmicsAdaptiveHypergraph(AdaptiveHypergraph):
    def __init__(self, dim_feat, num_heads=2, threshold=0.8, dropout=0.1, device='cuda', num_layers=1, tau=10.0):
        super(OmicsAdaptiveHypergraph, self).__init__(
            dim_feat=dim_feat,
            threshold=threshold,
            dropout=dropout,
            device=device,
            num_layers=num_layers,
            tau=tau
        )
        
        self.num_heads = num_heads
        
        # Projection layers for three omics features
        self.projection_mut = nn.Linear(dim_feat, dim_feat)
        self.projection_gexp = nn.Linear(dim_feat, dim_feat)
        self.projection_methy = nn.Linear(dim_feat, dim_feat)
        
        # Attention mechanism for omics fusion
        self.omics_fusion_attention = nn.MultiheadAttention(
            embed_dim=dim_feat,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(dim_feat, dim_feat)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def fuse_omics_features(self, x_mutation, x_gexpr, x_methylation):
        """
        Fuse three omics features through attention mechanism
        Args:
            x_mutation: Mutation features [N, dim_feat]
            x_gexpr: Gene expression features [N, dim_feat]
            x_methylation: Methylation features [N, dim_feat]
        Returns:
            fused_features: Fused features [N, dim_feat]
        """
        # Project three omics features
        x_mut_proj = self.projection_mut(x_mutation)
        x_gexp_proj = self.projection_gexp(x_gexpr)
        x_methy_proj = self.projection_methy(x_methylation)
        
        # Stack three omics features as sequence [N, 3, dim_feat]
        omics_sequence = torch.stack([x_mut_proj, x_gexp_proj, x_methy_proj], dim=1)
        
        # Fuse using attention mechanism
        attn_output, attn_weights = self.omics_fusion_attention(
            omics_sequence, omics_sequence, omics_sequence
        )
        
        # Get attention weights for weighted fusion
        if attn_weights.dim() == 4:
            avg_attn_weights = attn_weights.mean(dim=1)  # [N, 3, 3]
        else:
            avg_attn_weights = attn_weights
        
        # Use first row as weights for three omics
        omics_weights = avg_attn_weights[:, 0, :]  # [N, 3]
        
        # Weighted fusion
        fused_features = (x_mut_proj * omics_weights[:, 0:1] + 
                         x_gexp_proj * omics_weights[:, 1:2] + 
                         x_methy_proj * omics_weights[:, 2:3])
        
        return fused_features
    
    def forward(self, x_mutation, x_gexpr, x_methylation,
                mut_edge_features, gexp_edge_features, methy_edge_features, return_participation=False):
        """
        Forward propagation: omics fusion + adaptive hypergraph convolution
        Args:
            x_mutation: Mutation features [N, dim_feat]
            x_gexpr: Gene expression features [N, dim_feat]
            x_methylation: Methylation features [N, dim_feat]
            mut_edge_features: Mutation edge features [num_edges_mut, dim_feat]
            gexp_edge_features: Gene expression edge features [num_edges_gexp, dim_feat]
            methy_edge_features: Methylation edge features [num_edges_methy, dim_feat]
            return_participation: Whether to return participation matrix
        Returns:
            output: Output features [N, dim_feat]
            combined_adaptive_incidence (optional): Combined adaptive incidence matrix [N, total_edges]
        """
        N = x_mutation.shape[0]
        
        # 1. Fuse three omics node features into one through attention-weighted fusion
        fused_node_features = self.fuse_omics_features(x_mutation, x_gexpr, x_methylation)
        
        # 2. Compute adaptive incidence between fused node features and three omics edge features separately
        mut_adaptive_incidence = self.compute_adaptive_incidence(fused_node_features, mut_edge_features)
        gexp_adaptive_incidence = self.compute_adaptive_incidence(fused_node_features, gexp_edge_features)
        methy_adaptive_incidence = self.compute_adaptive_incidence(fused_node_features, methy_edge_features)
        
        # 3. Concatenate adaptive incidence matrices of three omics
        total_edges = mut_adaptive_incidence.shape[1] + gexp_adaptive_incidence.shape[1] + methy_adaptive_incidence.shape[1]
        combined_adaptive_incidence = torch.zeros(N, total_edges, device=fused_node_features.device)
        
        start_idx = 0
        combined_adaptive_incidence[:, start_idx:start_idx + mut_adaptive_incidence.shape[1]] = mut_adaptive_incidence
        start_idx += mut_adaptive_incidence.shape[1]
        combined_adaptive_incidence[:, start_idx:start_idx + gexp_adaptive_incidence.shape[1]] = gexp_adaptive_incidence
        start_idx += gexp_adaptive_incidence.shape[1]
        combined_adaptive_incidence[:, start_idx:start_idx + methy_adaptive_incidence.shape[1]] = methy_adaptive_incidence
        
        # 4. Adaptive hypergraph convolution
        adaptive_conv_output, _ = self.adaptive_hypergraph_convolution(fused_node_features, combined_adaptive_incidence)
        
        # 5. Output projection
        output = self.output_projection(adaptive_conv_output)
        output = self.leaky_relu(output)
        
        if return_participation:
            return output, combined_adaptive_incidence
        return output

