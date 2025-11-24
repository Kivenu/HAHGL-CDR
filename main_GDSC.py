import torch
import numpy as np
import random
import os
import pickle
import argparse
import time
from tqdm import tqdm
import pandas as pd
import json

from load_data import LoadData_gdsc
from process_data import process_data
from sampler import get_sampler
from myutils import metrics_graph
from HAHGL_CDR import HAHGL_CDR


parser = argparse.ArgumentParser(description="HAHGL_CDR GDSC")
parser.add_argument('--device', type=str, default="0", help='cuda:number or cpu')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--k', type=int, default=5, help='K_NN parameter for omics-specific hypergraph')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--dim_feat', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2, help='Number of hetero adaptive hypergraph convolution layers')
parser.add_argument('--alpha', type=float, default=8)
parser.add_argument('--threshold', type=float, default=0.8, help='Adaptive hypergraph sparsification parameter')
parser.add_argument('--omics_num_layers', type=int, default=1, help='Number of Multi-omics adaptive hypergraph convolution layers')
parser.add_argument('--tau', type=float, default=10.0, help='Adaptive hypergraph sparsification parameter')
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--n_repeats', type=int, default=1, help='Number of cross-validation repeats')
parser.add_argument('--target_repeat', type=int, default=0, help='Specify the repeat to run (starting from 0)')
parser.add_argument('--target_fold', type=int, default=0, help='Specify the fold to run (starting from 0)')
parser.add_argument('--outfile', type=str, default="HAHGL_CDR_GDSC_results")
parser.add_argument('--save_model', type=bool, default=True)
args = parser.parse_args()

start_time = time.time()

device = torch.device("cuda:"+args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Data paths =====
Drug_info_file = 'data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = 'data/Drug/drug_threshold.txt'
Drug_feature_file = 'data/Drug/new_edge'
Cell_line_info_file = 'data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = 'data/Celline/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = 'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = 'data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
Cancer_response_exp_file = 'data/Celline/GDSC_IC50.csv'

# ===== Load data =====
drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = LoadData_gdsc(
    Drug_info_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file, 
    Gene_expression_file, Methylation_file, Cancer_response_exp_file, IC50_threds_file)

drug_set, cellline_set, allpairs, atom_shape, gexp_hyper_edge, mut_hyper_edge, methy_hyper_edge = process_data(
    drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs, device=device)


sampler = get_sampler(seed=args.seed)

# Check if specific repeat and fold are specified
if args.target_repeat is not None and args.target_fold is not None:
    # Specific repeat and fold mode
    print("=" * 50)
    print(f"Specific mode: Seed {args.seed}, Repeat {args.target_repeat}, Fold {args.target_fold}")
    print("=" * 50)
    
    # Validate parameter ranges
    if args.target_repeat < 0 or args.target_repeat >= args.n_repeats:
        raise ValueError(f"target_repeat must be between 0 and {args.n_repeats-1}")
    if args.target_fold < 0 or args.target_fold >= args.n_splits:
        raise ValueError(f"target_fold must be between 0 and {args.n_splits-1}")
    
    # Generate data for specified repeat and fold
    all_splits = sampler.run_multiple_cv(allpairs, nb_celllines, nb_drugs, 
                                         n_splits=args.n_splits, n_repeats=args.n_repeats)
    
    # Filter out specified repeat and fold
    target_splits = []
    for split in all_splits:
        if split['repeat'] == args.target_repeat and split['fold'] == args.target_fold:
            target_splits.append(split)
            break
    
    if not target_splits:
        raise ValueError(f"Data not found for Repeat {args.target_repeat}, Fold {args.target_fold}")
    
    all_splits = target_splits
else:
    # Default mode: run all repeats and folds
    print("=" * 50)
    print(f"Full cross-validation mode: {args.n_repeats} repeats × {args.n_splits} folds")
    print("=" * 50)
    all_splits = sampler.run_multiple_cv(allpairs, nb_celllines, nb_drugs, 
                                         n_splits=args.n_splits, n_repeats=args.n_repeats)

res = []
patience = 140

# Create directories for loss, para, save directories and output directories
if not os.path.exists("loss"):
    os.mkdir("loss")
if not os.path.exists("para"):
    os.mkdir("para")
if not os.path.exists("save"):
    os.mkdir("save")
if not os.path.exists("out"):
    os.mkdir("out")

for split_idx, split_data in enumerate(all_splits):
    repeat = split_data['repeat']
    fold = split_data['fold']
    
    if args.target_repeat is not None and args.target_fold is not None:
        print(f"\nSpecific mode - Seed {args.seed}, Repeat {repeat + 1}, Fold {fold + 1}")
    else:
        print(f"\nFull CV - Seed {args.seed}, Repeat {repeat + 1}, Fold {fold + 1}")

    train_mask = split_data['train_mask'].to(device)
    test_mask = split_data['test_mask'].to(device)
    label_pos = split_data['label_pos'].to(device)
    train_edge = split_data['train_edge']  # Training set specific edges


    print(f"  Num Layers (Association): {args.num_layers}")
    print(f"  Omics Num Layers: {args.omics_num_layers}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Tau: {args.tau}")
    
    model = HAHGL_CDR(
        dim_drug=atom_shape, drug_layer=[256,256,256],
        dim_gexp=gexpr_feature.shape[1], dim_methy=methylation_feature.shape[1],
        dim_feat=args.dim_feat, out_channels=100, k=args.k, 
        num_layers=args.num_layers,
        dropout=args.dropout, use_bn_at=True, use_bn_as=True,
        negative_slope=0.2, device=device, alpha=args.alpha,
        threshold=args.threshold,
        omics_num_layers=args.omics_num_layers,
        tau=args.tau
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bceloss = torch.nn.BCELoss().to(device)

    best_auc = 0
    early_stop_counter = 0
    best_result = None
    
    # Initialize loss recording list
    epoch_losses = []
    
    # Initialize time and memory statistics
    epoch_train_times = []
    epoch_inference_times = []

    desc = f"R{repeat+1}-F{fold+1}"
    
    for epoch in tqdm(range(args.epochs), desc=desc):
        epoch_start_time = time.time()  # Start timing epoch
        model.train()
        epoch_loss = 0  # Accumulate loss

        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            drug.x, drug.edge_index, drug.batch = drug.x.to(device), drug.edge_index.to(device), drug.batch.to(device)
            mutation_data, gexpr_data, methylation_data = cell[0].to(device), cell[1].to(device), cell[2].to(device)
            
            output = model(
                drug.x, drug.edge_index, drug.batch,
                mutation_data, gexpr_data, methylation_data,
                train_edge,  # Use training set specific edges
                nb_celllines, nb_drugs,
                gexp_hyper_edge=gexp_hyper_edge.to(device),
                mut_hyper_edge=mut_hyper_edge.to(device),
                methy_hyper_edge=methy_hyper_edge.to(device)
            )

            optimizer.zero_grad()
            # Use model's compute_total_loss method (supervision loss only)
            total_loss, supervision_loss = model.compute_total_loss(
                predictions=output[train_mask], 
                targets=label_pos[train_mask]
            )
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()  # Accumulate loss for each batch
        
        # Calculate training time (training phase of entire epoch)
        epoch_train_time = (time.time() - epoch_start_time) * 1000  # Convert to milliseconds
        epoch_train_times.append(epoch_train_time)

        # Calculate average
        num_batches = len(drug_set)
        avg_epoch_loss = epoch_loss / num_batches
        
        # Evaluation phase (inference) - only measure model forward propagation time
        with torch.no_grad():
            model.eval()
            inference_start_time = time.time()
            output = model(
                drug.x, drug.edge_index, drug.batch,
                mutation_data, gexpr_data, methylation_data,
                train_edge,  # Use training set specific edges
                nb_celllines, nb_drugs,
                gexp_hyper_edge=gexp_hyper_edge.to(device),
                mut_hyper_edge=mut_hyper_edge.to(device),
                methy_hyper_edge=methy_hyper_edge.to(device)
            )
            # Calculate inference time (only includes model forward propagation)
            epoch_inference_time = (time.time() - inference_start_time) * 1000  # Convert to milliseconds
            epoch_inference_times.append(epoch_inference_time)
            
            # Calculate metrics (not included in inference time)
            ytest_p = output[test_mask]
            ytest_t = label_pos[test_mask]
            test_auc, test_aupr, test_f1, test_acc, test_prec, test_rec, test_mcc = metrics_graph(ytest_t, ytest_p)
            
            # Calculate training set metrics
            ytrain_p = output[train_mask]
            ytrain_t = label_pos[train_mask]
            train_auc, train_aupr, train_f1, train_acc, train_prec, train_rec, train_mcc = metrics_graph(ytrain_t, ytrain_p)
        
        # Print information (does not affect time statistics)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"=====================LOSS=========================")
        print(f"  Total Loss: {avg_epoch_loss:.4f}")
        print(f"=====================LOSS=========================")
        
        # Record loss and metrics for current epoch
        epoch_losses.append({
            'epoch': epoch + 1,
            'total_loss': avg_epoch_loss,
            'train_auc': train_auc,
            'train_aupr': train_aupr,
            'train_f1': train_f1,
            'train_acc': train_acc,
            'train_prec': train_prec,
            'train_rec': train_rec,
            'train_mcc': train_mcc,
            'test_auc': test_auc,
            'test_aupr': test_aupr,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_mcc': test_mcc,
            'train_time_ms': epoch_train_time,
            'inference_time_ms': epoch_inference_time
        })
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_result = (test_auc, test_aupr, test_f1, test_acc, test_prec, test_rec, test_mcc)
            early_stop_counter = 0
            if args.save_model:
                try:
                    torch.save(model.state_dict(), f'save/{args.outfile}_seed{args.seed}_repeat{repeat}_fold{fold}.pth')
                except OSError as e:
                    print(f"Warning: Could not save model due to file system error: {e}")
                    print("Continuing training without saving...")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stop at epoch {epoch}")
                break
        if epoch % 2 == 0:
            print(f"========================AUC=========================")
            print(f"[Epoch {epoch}] train_auc={train_auc:.4f}, test_auc={test_auc:.4f}, patience={early_stop_counter}")
            print(f"========================AUC=========================")

    # Calculate performance statistics
    avg_train_time = np.mean(epoch_train_times) if epoch_train_times else 0.0
    avg_inference_time = np.mean(epoch_inference_times) if epoch_inference_times else 0.0
    
    # Get and save participation matrices
    with torch.no_grad():
        model.eval()
        omics_participation, sen_participation = model.get_participation_matrices(
            drug.x, drug.edge_index, drug.batch,
            mutation_data, gexpr_data, methylation_data,
            train_edge, nb_celllines, nb_drugs,
            gexp_hyper_edge=gexp_hyper_edge.to(device),
            mut_hyper_edge=mut_hyper_edge.to(device),
            methy_hyper_edge=methy_hyper_edge.to(device)
        )
        
        # Save participation matrices as numpy arrays
        omics_participation_np = omics_participation.cpu().numpy()
        sen_participation_np = sen_participation.cpu().numpy()
        
        # Save to para directory
        para_filename_prefix = f"para/{args.outfile}_seed{args.seed}_repeat{repeat}_fold{fold}"
        np.save(f"{para_filename_prefix}_omics_participation.npy", omics_participation_np)
        np.save(f"{para_filename_prefix}_sen_participation.npy", sen_participation_np)
        
        print(f"Participation matrices saved:")
        print(f"  Omics participation matrix shape: {omics_participation_np.shape}")
        print(f"  Sen participation matrix shape: {sen_participation_np.shape}")
    
    # Save results and performance metrics together
    res.append([repeat+1, fold+1] + list(best_result) + [avg_train_time, avg_inference_time])
    
    print(f"Best result - Seed {args.seed}, Repeat {repeat}, Fold {fold}: {best_result}")
    
    print(f"Performance statistics:")
    print(f"  Average training time/epoch: {avg_train_time:.2f} ms")
    print(f"  Average inference time: {avg_inference_time:.2f} ms")
    
    # Save loss data to CSV
    loss_df = pd.DataFrame(epoch_losses)
    loss_filename = f"loss/{args.outfile}_seed{args.seed}_repeat{repeat}_fold{fold}_loss.csv"
    
    # Create parameter DataFrame
    args_dict = vars(args)
    args_df = pd.DataFrame(list(args_dict.items()), columns=['Parameter', 'Value'])
    
    # Save parameters and loss data to the same CSV file
    with open(loss_filename, 'a', newline='') as f:
        # Write parameter section
        f.write("# ===== EXPERIMENT PARAMETERS =====\n")
        args_df.to_csv(f, index=False)
        f.write("\n# ===== EPOCH LOSS DATA =====\n")
        # Write loss data section
        loss_df.to_csv(f, index=False)
    
    print(f"Loss data saved to {loss_filename}")

# ===== Result statistics and saving =====
res = np.array(res)

results_df = pd.DataFrame(res, columns=['repeat', 'fold', 'auc', 'aupr', 'f1', 'acc', 'prec', 'recall', 'mcc', 
                                                'avg_train_time_ms', 'avg_inference_time_ms'])

# Display results
if args.target_repeat is not None and args.target_fold is not None:
    print(f"\nSpecific mode results - Seed {args.seed}, Repeat {args.target_repeat}, Fold {args.target_fold}:")
else:
    print(f"\nFull cross-validation results - Seed {args.seed}:")
    print(f"Total runs: {len(res)} folds")
    print(f"\nMean ± Std for each metric:")
    for col in ['auc', 'aupr', 'f1', 'acc', 'prec', 'recall', 'mcc']:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        print(f"  {col.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"  Average training time/epoch: {results_df['avg_train_time_ms'].mean():.2f} ms")
    print(f"  Average inference time: {results_df['avg_inference_time_ms'].mean():.2f} ms")

# Create parameter DataFrame
args_dict = vars(args)
args_df = pd.DataFrame(list(args_dict.items()), columns=['Parameter', 'Value'])


# Set output filename based on mode
if args.target_repeat is not None and args.target_fold is not None:
    output_file = f"out/{args.outfile}_seed{args.seed}_repeat{args.target_repeat}_fold{args.target_fold}.csv"
else:
    output_file = f"out/{args.outfile}_seed{args.seed}.csv"

# Save parameters and results to the same CSV file
with open(output_file, 'a', newline='') as f:
    # Write parameter section
    f.write("# ===== EXPERIMENT PARAMETERS =====\n")
    args_df.to_csv(f, index=False)
    f.write("\n# ===== EXPERIMENT RESULTS =====\n")
    # Write results section
    results_df.to_csv(f, index=False)

if args.target_repeat is not None and args.target_fold is not None:
    print(f"\nSpecific mode detailed results:")
    print(f"AUC: {res[0][2]:.4f}")
    print(f"AUPR: {res[0][3]:.4f}")
    print(f"F1: {res[0][4]:.4f}")
    print(f"ACC: {res[0][5]:.4f}")
    print(f"Precision: {res[0][6]:.4f}")
    print(f"Recall: {res[0][7]:.4f}")
    print(f"MCC: {res[0][8]:.4f}")
    print(f"Average training time/epoch: {res[0][9]:.2f} ms")
    print(f"Average inference time: {res[0][10]:.2f} ms")

print(f"Results saved to {output_file}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
