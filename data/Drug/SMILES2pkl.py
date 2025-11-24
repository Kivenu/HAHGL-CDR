import numpy as np
from rdkit import Chem
import pickle
import os

def create_bond_feature_matrix(smiles):
    """
    从SMILES创建包含化学键特征的分子图
    返回: (feat_mat, adj_list, degree_list, bond_matrix)
    """
    # 1. 从SMILES创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析SMILES: {smiles}")
    
    # 2. 获取原子信息（节点特征）
    atom_features = []
    atom_indices = {}
    
    for i, atom in enumerate(mol.GetAtoms()):
        # 原子特征向量
        features = [
            atom.GetAtomicNum(),      # 原子序数
            atom.GetDegree(),         # 连接度
            atom.GetImplicitValence(), # 隐式价
            atom.GetFormalCharge(),   # 形式电荷
            atom.GetIsAromatic(),     # 是否芳香
            atom.GetHybridization(),  # 杂化类型
            atom.GetNumRadicalElectrons(), # 自由基电子数
        ]
        atom_features.append(features)
        atom_indices[atom.GetIdx()] = i
    
    # 3. 创建邻接表和边特征矩阵
    adj_list = [[] for _ in range(len(atom_features))]
    num_atoms = len(atom_features)
    feature_dim = 5  # 化学键特征维度
    bond_matrix = np.zeros((num_atoms, num_atoms, feature_dim))
    
    # 4. 填充化学键信息
    for bond in mol.GetBonds():
        begin_idx = atom_indices[bond.GetBeginAtomIdx()]
        end_idx = atom_indices[bond.GetEndAtomIdx()]
        
        # 化学键特征
        bond_features = [
            bond.GetBondType(),       # 键类型 (SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12)
            bond.GetIsAromatic(),     # 是否芳香键
            bond.GetIsConjugated(),   # 是否共轭
            bond.GetBondDir(),        # 键方向
            bond.GetStereo(),         # 立体化学
        ]
        
        # 添加到邻接表
        adj_list[begin_idx].append(end_idx)
        adj_list[end_idx].append(begin_idx)
        
        # 填充边特征矩阵（无向图）
        bond_matrix[begin_idx, end_idx] = bond_features
        bond_matrix[end_idx, begin_idx] = bond_features
    
    # 5. 计算度数列表
    degree_list = [len(neighbors) for neighbors in adj_list]
    
    return np.array(atom_features), adj_list, degree_list, bond_matrix

def process_smiles_file(smiles_file_path, output_dir="./new_edge"):
    """
    处理SMILES文件，生成包含化学键特征的pkl文件
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    # 读取SMILES文件
    with open(smiles_file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"开始处理 {len(lines)} 个SMILES...")
    
    success_count = 0
    error_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            # 解析行数据 (pubchem_id \t smiles)
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"跳过格式错误的行: {line}")
                continue
                
            pubchem_id = parts[0].strip()
            smiles = parts[1].strip()
            
            # 生成分子图特征
            feat_mat, adj_list, degree_list, bond_matrix = create_bond_feature_matrix(smiles)
            
            # 保存为pkl文件
            output_file = os.path.join(output_dir, f"{pubchem_id}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump((feat_mat, adj_list, degree_list, bond_matrix), f)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"已处理 {success_count} 个分子...")
                
        except Exception as e:
            error_count += 1
            print(f"处理失败 {line}: {e}")
    
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {error_count} 个")
    print(f"输出目录: {output_dir}")

def verify_generated_files(output_dir="./new_edge"):
    """
    验证生成的pkl文件
    """
    if not os.path.exists(output_dir):
        print(f"目录不存在: {output_dir}")
        return
    
    pkl_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
    print(f"\n验证 {len(pkl_files)} 个pkl文件...")
    
    valid_count = 0
    for pkl_file in pkl_files[:5]:  # 只验证前5个文件
        try:
            file_path = os.path.join(output_dir, pkl_file)
            with open(file_path, 'rb') as f:
                feat_mat, adj_list, degree_list, bond_matrix = pickle.load(f)
            
            pubchem_id = pkl_file.split('.')[0]
            print(f"{pubchem_id}: 节点特征{feat_mat.shape}, 边特征{bond_matrix.shape}")
            valid_count += 1
            
        except Exception as e:
            print(f"验证失败 {pkl_file}: {e}")
    
    print(f"验证完成: {valid_count}/{min(5, len(pkl_files))} 个文件有效")

if __name__ == "__main__":
    # 设置文件路径
    smiles_file = "222drugs_pubchem_smiles.txt"
    output_directory = "./new_edge"
    
    print("=== SMILES转pkl工具 ===")
    print(f"输入文件: {smiles_file}")
    print(f"输出目录: {output_directory}")
    
    # 处理SMILES文件
    process_smiles_file(smiles_file, output_directory)
    
    # 验证生成的文件
    verify_generated_files(output_directory)
