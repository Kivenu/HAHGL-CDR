import torch
import numpy as np
import argparse
import os
import pandas as pd

from load_data import LoadData_gdsc, LoadData_ccle
from process_data import process_data
from sampler import get_sampler
from myutils import metrics_graph
from HAHGL_CDR import HAHGL_CDR


def evaluate_model(args):
    """
    Evaluate model performance on test set
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda:"+args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Evaluation Configuration:")
    print("="*60)
    print(f"Model file: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Random seed: {args.seed}")
    print(f"Repeat: {args.repeat} (0-based)")
    print(f"Fold: {args.fold} (0-based)")
    print(f"K: {args.k}")
    print(f"Dropout: {args.dropout}")
    print(f"Feature dimension: {args.dim_feat}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Alpha: {args.alpha}")
    print(f"Threshold: {args.threshold}")
    print(f"Omics num layers: {args.omics_num_layers}")
    print(f"Tau: {args.tau}")
    print("="*60)
    
    # ===== Data paths =====
    if args.dataset.lower() == 'gdsc':
        Drug_info_file = 'data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
        IC50_threds_file = 'data/Drug/drug_threshold.txt'
        Drug_feature_file = 'data/Drug/new_edge'
        Cell_line_info_file = 'data/Celline/Cell_lines_annotations.txt'
        Genomic_mutation_file = 'data/Celline/genomic_mutation_34673_demap_features.csv'
        Gene_expression_file = 'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
        Methylation_file = 'data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
        Cancer_response_exp_file = 'data/Celline/GDSC_IC50.csv'
    elif args.dataset.lower() == 'ccle':
        Drug_feature_file = 'data/CCLE/CCLE_smiles.csv'
        Genomic_mutation_file = 'data/Celline/genomic_mutation_34673_demap_features.csv'
        Gene_expression_file = 'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
        Methylation_file = 'data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
        Cancer_response_exp_file = 'data/CCLE/CCLE_response.csv'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Must be 'gdsc' or 'ccle'")
    
    # ===== Load data =====
    print("\nLoading data...")
    if args.dataset.lower() == 'gdsc':
        drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = LoadData_gdsc(
            Drug_info_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file, 
            Gene_expression_file, Methylation_file, Cancer_response_exp_file, IC50_threds_file)
    else:  # CCLE
        drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = LoadData_ccle(
            Drug_feature_file, Genomic_mutation_file, Gene_expression_file, 
            Methylation_file, Cancer_response_exp_file)
    
    drug_set, cellline_set, allpairs, atom_shape, gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge = process_data(
        drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs, device=device)
    
    # ===== Generate data splits =====
    print(f"\nGenerating data splits (Seed={args.seed}, Repeat={args.repeat}, Fold={args.fold})...")
    sampler = get_sampler(seed=args.seed)
    all_splits = sampler.run_multiple_cv(allpairs, nb_celllines, nb_drugs, 
                                         n_splits=args.n_splits, n_repeats=args.n_repeats)
    
    # Find the specified repeat and fold (0-based)
    target_split = None
    for split in all_splits:
        if split['repeat'] == args.repeat and split['fold'] == args.fold:
            target_split = split
            break
    
    if target_split is None:
        raise ValueError(f"Split not found for Repeat {args.repeat}, Fold {args.fold}")
    
    train_mask = target_split['train_mask'].to(device)
    test_mask = target_split['test_mask'].to(device)
    label_pos = target_split['label_pos'].to(device)
    train_edge = target_split['train_edge']
    
    print(f"Number of training samples: {train_mask.sum().item()}")
    print(f"Number of test samples: {test_mask.sum().item()}")
    
    # ===== Build model =====
    print("\nBuilding model...")
    model = HAHGL_CDR(
        dim_drug=atom_shape, drug_layer=[256,256,256],
        dim_gexp=gexpr_feature.shape[1], dim_methy=methylation_feature.shape[1],
        dim_mutation=mutation_feature.shape[1],
        dim_feat=args.dim_feat, out_channels=100, k=args.k, 
        num_layers=args.num_layers,
        dropout=args.dropout, use_bn_at=True, use_bn_as=True,
        negative_slope=0.2, device=device, alpha=args.alpha,
        threshold=args.threshold,
        omics_num_layers=args.omics_num_layers,
        tau=args.tau
    ).to(device)
    
    # ===== Load model weights =====
    print(f"\nLoading model weights: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully!")
    
    # ===== Evaluate model =====
    print("\nEvaluating model...")
    
    # Get data (DataLoader should have only one batch)
    for drug, cell in zip(drug_set, cellline_set):
        drug.x, drug.edge_index, drug.batch = drug.x.to(device), drug.edge_index.to(device), drug.batch.to(device)
        mutation_data, gexpr_data, methylation_data = cell[0].to(device), cell[1].to(device), cell[2].to(device)
    
    # Forward propagation
    with torch.no_grad():
        output = model(
            drug.x, drug.edge_index, drug.batch,
            mutation_data, gexpr_data, methylation_data,
            train_edge,
            nb_celllines, nb_drugs,
            gexp_hyper_edge=gexp_hyper_edge.to(device),
            mut_hyper_edge=mut_hyper_edge.to(device),
            methy_hyper_edge=methy_hyper_edge.to(device)
        )
        
        # Calculate test set metrics
        ytest_p = output[test_mask]
        ytest_t = label_pos[test_mask]
        test_auc, test_aupr, test_f1, test_acc, test_prec, test_rec, test_mcc = metrics_graph(ytest_t, ytest_p)
        
        # Calculate training set metrics
        ytrain_p = output[train_mask]
        ytrain_t = label_pos[train_mask]
        train_auc, train_aupr, train_f1, train_acc, train_prec, train_rec, train_mcc = metrics_graph(ytrain_t, ytrain_p)
    
    # ===== Output results =====
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nTraining Set Metrics:")
    print(f"  AUC:       {train_auc:.4f}")
    print(f"  AUPR:      {train_aupr:.4f}")
    print(f"  F1:        {train_f1:.4f}")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  MCC:       {train_mcc:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  AUPR:      {test_aupr:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  MCC:       {test_mcc:.4f}")
    
    print("="*60)
    
    # ===== Save results =====
    if args.save_results:
        results = {
            'model_path': [args.model_path],
            'dataset': [args.dataset],
            'seed': [args.seed],
            'repeat': [args.repeat],
            'fold': [args.fold],
            'train_auc': [train_auc],
            'train_aupr': [train_aupr],
            'train_f1': [train_f1],
            'train_acc': [train_acc],
            'train_prec': [train_prec],
            'train_rec': [train_rec],
            'train_mcc': [train_mcc],
            'test_auc': [test_auc],
            'test_aupr': [test_aupr],
            'test_f1': [test_f1],
            'test_acc': [test_acc],
            'test_prec': [test_prec],
            'test_rec': [test_rec],
            'test_mcc': [test_mcc]
        }
        
        df = pd.DataFrame(results)
        
        # Generate output filename
        if args.output_file is None:
            model_name = os.path.splitext(os.path.basename(args.model_path))[0]
            output_file = f"eval_results_{model_name}.csv"
        else:
            output_file = args.output_file
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(output_file, index=False)
        print(f"\nEvaluation results saved to: {output_file}")
    
    return {
        'train': {
            'auc': train_auc, 'aupr': train_aupr, 'f1': train_f1,
            'acc': train_acc, 'prec': train_prec, 'rec': train_rec, 'mcc': train_mcc
        },
        'test': {
            'auc': test_auc, 'aupr': test_aupr, 'f1': test_f1,
            'acc': test_acc, 'prec': test_prec, 'rec': test_rec, 'mcc': test_mcc
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model performance on test set")
    
    # Model and dataset arguments
    parser.add_argument('--model_path', type=str, default='save/HAHGL_CDR_GDSC_results_seed2024_repeat1_fold1.pth', help='Path to model weight file')
    parser.add_argument('--dataset', type=str, default='gdsc', choices=['gdsc', 'ccle'], help='Dataset name (gdsc or ccle)')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--repeat', type=int, default=0, help='Repeat number (0-based, same as target_repeat in training)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-based, same as target_fold in training)')
    
    # Model configuration arguments
    parser.add_argument('--k', type=int, default=5, help='K_NN parameter for omics-specific hypergraph')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--dim_feat', type=int, default=100, help='Feature dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hetero adaptive hypergraph convolution layers')
    parser.add_argument('--alpha', type=float, default=8.0, help='Correlation scaling parameter')
    parser.add_argument('--threshold', type=float, default=0.8, help='Adaptive hypergraph sparsification parameter')
    parser.add_argument('--omics_num_layers', type=int, default=1, help='Number of Multi-omics adaptive hypergraph convolution layers')
    parser.add_argument('--tau', type=float, default=10.0, help='Adaptive hypergraph sparsification parameter')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of cross-validation repeats')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default="0", help='CUDA device number or cpu')
    
    # Output arguments
    parser.add_argument('--save_results', action='store_true', help='Whether to save evaluation results')
    parser.add_argument('--output_file', type=str, default=None, help='Path to output result file')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.model_path is None:
        raise ValueError("--model_path is required. Please specify the path to the model weight file.")
    
    # Run evaluation
    evaluate_model(args)
