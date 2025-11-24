import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
import random


class DrugResponseSampler:
    """
    Drug response data sampler
    Supports random splitting and K-fold cross-validation
    """
    
    def __init__(self, seed=2023):
        """
        Initialize sampler
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.set_global_seed(seed)
    
    def set_global_seed(self, seed):
        """Set global random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_masks(self, num_samples, train_ratio=0.8, test_ratio=0.2):
        """
        Create train/test set masks
        
        Args:
            num_samples: Total number of samples
            train_ratio: Training set ratio
            test_ratio: Test set ratio
            
        Returns:
            train_mask, test_mask: Boolean masks
        """
        assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Create mask
        mask = np.zeros(num_samples)
        train_end = int(train_ratio * num_samples)
        
        mask[0:train_end] = 0  # Training set
        mask[train_end:] = 1   # Test set
        
        # Random shuffle
        np.random.shuffle(mask)
        
        train_mask = (mask == 0)
        test_mask = (mask == 1)
        
        return train_mask, test_mask
    
    def process_label_random(self, allpairs, nb_celllines, nb_drugs, train_ratio=0.8, test_ratio=0.2):
        """
        Randomly split drug response data
        
        Args:
            allpairs: All drug-cell line pairs (cellline_id, drug_id, response)
            nb_celllines: Number of cell lines
            nb_drugs: Number of drugs
            train_ratio: Training set ratio
            test_ratio: Test set ratio
            
        Returns:
            train_mask, test_mask: Sparse matrix masks
            train_edge: Training edges
            label_pos: Positive sample labels
        """
        # Separate positive and negative samples
        pos_pairs = allpairs[allpairs[:, 2] == 1]
        neg_pairs = allpairs[allpairs[:, 2] == -1]
        
        # Create masks
        train_mask, test_mask = self.create_masks(
            len(allpairs), train_ratio, test_ratio
        )
        
        # Split data
        train = allpairs[train_mask][:, 0:3]
        test = allpairs[test_mask][:, 0:3]
        
        # Create training edges (bidirectional)
        train_edge = np.vstack((train, train[:, [1, 0, 2]]))
        
        # Adjust drug IDs (subtract number of cell lines)
        train[:, 1] -= nb_celllines
        test[:, 1] -= nb_celllines
        
        # Create sparse matrix masks - use unique pairs to avoid duplicates
        def create_unique_mask(pairs, shape):
            """Create unique sparse matrix mask"""
            if len(pairs) == 0:
                return np.zeros(shape, dtype=bool)
            
            # Get unique drug-cell line pairs
            unique_pairs = set((int(p[0]), int(p[1])) for p in pairs)
            rows, cols = zip(*unique_pairs)
            
            mask = np.zeros(shape, dtype=bool)
            mask[rows, cols] = True
            return mask
        
        train_mask_sparse = create_unique_mask(train, (nb_celllines, nb_drugs))
        test_mask_sparse = create_unique_mask(test, (nb_celllines, nb_drugs))
        
        # Create positive sample labels
        pos_pairs[:, 1] -= nb_celllines
        neg_pairs[:, 1] -= nb_celllines
        
        # Create positive sample labels using unique pairs
        unique_pos_pairs = set((int(p[0]), int(p[1])) for p in pos_pairs)
        pos_rows, pos_cols = zip(*unique_pos_pairs)
        
        label_pos = np.zeros((nb_celllines, nb_drugs), dtype=np.float32)
        label_pos[pos_rows, pos_cols] = 1.0
        label_pos = torch.from_numpy(label_pos).view(-1)
        
        return train_mask_sparse, test_mask_sparse, train_edge, label_pos
    
    def create_kfold_splits(self, allpairs, nb_celllines, nb_drugs, n_splits=5):
        """
        Create K-fold cross-validation data splits
        
        Args:
            allpairs: All drug-cell line pairs
            nb_celllines: Number of cell lines
            nb_drugs: Number of drugs
            n_splits: Number of folds
            
        Returns:
            splits: List containing data for each fold
        """
        splits = []
        
        # Use KFold for splitting
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(allpairs)):
            # Get training and test sets
            train_pairs = allpairs[train_idx]
            test_pairs = allpairs[test_idx]
            
            # Process data
            train = train_pairs[:, 0:3]
            test = test_pairs[:, 0:3]
            
            # Create training edges (bidirectional)
            train_edge = np.vstack((train, train[:, [1, 0, 2]]))
            
            # Adjust drug IDs
            train[:, 1] -= nb_celllines
            test[:, 1] -= nb_celllines
            
            # Create sparse matrix masks - use unique pairs to avoid duplicates
            def create_unique_mask(pairs, shape):
                """Create unique sparse matrix mask"""
                if len(pairs) == 0:
                    return np.zeros(shape, dtype=bool)
                
                # Get unique drug-cell line pairs
                unique_pairs = set((int(p[0]), int(p[1])) for p in pairs)
                rows, cols = zip(*unique_pairs)
                
                mask = np.zeros(shape, dtype=bool)
                mask[rows, cols] = True
                return mask
            
            train_mask_sparse = create_unique_mask(train, (nb_celllines, nb_drugs))
            test_mask_sparse = create_unique_mask(test, (nb_celllines, nb_drugs))
            
            # Create positive sample labels
            pos_pairs = allpairs[allpairs[:, 2] == 1]
            pos_pairs_adjusted = pos_pairs.copy()
            pos_pairs_adjusted[:, 1] -= nb_celllines
            
            # Create positive sample labels using unique pairs
            unique_pos_pairs = set((int(p[0]), int(p[1])) for p in pos_pairs_adjusted)
            pos_rows, pos_cols = zip(*unique_pos_pairs)
            
            label_pos = np.zeros((nb_celllines, nb_drugs), dtype=np.float32)
            label_pos[pos_rows, pos_cols] = 1.0
            label_pos = torch.from_numpy(label_pos).view(-1)
            
            # Convert to torch tensors
            train_mask_tensor = torch.from_numpy(train_mask_sparse).view(-1)
            test_mask_tensor = torch.from_numpy(test_mask_sparse).view(-1)
            
            splits.append({
                'fold': fold,
                'train_mask': train_mask_tensor, # Training set mask
                'test_mask': test_mask_tensor, # Test set mask
                'train_edge': train_edge, # Training edges, processed bidirectionally
                'label_pos': label_pos, # Positive sample labels
                'train_pairs': train_pairs, # Training pairs, drugs processed, subtracted cell line count
                'test_pairs': test_pairs # Test pairs, drugs processed, subtracted cell line count
            })
        
        return splits
    
    def run_multiple_cv(self, allpairs, nb_celllines, nb_drugs, n_splits=5, n_repeats=5):
        """
        Run multiple K-fold cross-validation
        
        Args:
            allpairs: All drug-cell line pairs
            nb_celllines: Number of cell lines
            nb_drugs: Number of drugs
            n_splits: Number of folds
            n_repeats: Number of repeats
            
        Returns:
            all_splits: Data splits for all repeated experiments
        """
        all_splits = []
        
        for repeat in range(n_repeats):
            # Set different seed for each repeat
            current_seed = self.seed + repeat
            self.set_global_seed(current_seed)
            
            print(f"Running CV repeat {repeat + 1}/{n_repeats}")
            splits = self.create_kfold_splits(allpairs, nb_celllines, nb_drugs, n_splits)
            
            # Add repeat information to each split
            for split in splits:
                split['repeat'] = repeat
            
            all_splits.extend(splits)
        
        return all_splits


    def run_tuning_cv(self, allpairs, nb_celllines, nb_drugs, n_splits=5):
        """
        Run CV split for hyperparameter tuning, fixed to first fold
        
        Args:
            allpairs: All drug-cell line pairs
            nb_celllines: Number of cell lines
            nb_drugs: Number of drugs
            n_splits: Number of folds (used to generate first fold)
            
        Returns:
            tuning_split: Data split for first fold
        """
        # Generate K-fold splits
        splits = self.create_kfold_splits(allpairs, nb_celllines, nb_drugs, n_splits)
        
        # Return only first fold
        tuning_split = splits[0]
        tuning_split['repeat'] = 0  # Fixed to 0
        
        print(f"Tuning mode: using first fold data (fold {tuning_split['fold'] + 1})")
        print(f"Number of training samples: {tuning_split['train_mask'].sum().item()}")
        print(f"Number of test samples: {tuning_split['test_mask'].sum().item()}")
        
        return [tuning_split]  # Return list to maintain interface consistency with run_multiple_cv


def get_sampler(seed=2023):
    """
    Get sampler instance
    
    Args:
        seed: Random seed
        
    Returns:
        DrugResponseSampler instance
    """
    return DrugResponseSampler(seed=seed) 