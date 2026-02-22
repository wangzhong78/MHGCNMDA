#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MHGCNMDA: Multi-source Heterogeneous Graph Convolutional Network
for Microbe-Drug Association Prediction

Complete implementation with 5-fold cross-validation × 10 repeats evaluation
Modified to load real data from specified paths
Includes microbe functional similarity and drug structural similarity fusion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, precision_recall_curve)
import warnings
import os

warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING - MODIFIED FOR REAL DATA
# =============================================================================

class MDADDataset:
    """
    Load and preprocess MDAD dataset from specified paths
    Includes microbe functional similarity and drug structural similarity
    """

    def __init__(self,
                 a1_path=r'C:\\新建文件夹 (2)\\MKGCN-main - 副本\\data\\MicrobeDrugA\\MDAD\\a\\drug_microbe_matrix.txt',
                 a2_path=r'C:\\新建文件夹 (2)\\MKGCN-main - 副本\\data\\MicrobeDrugA\\MDAD\\a\\drug_dmicrobe_matrix.txt',
                 microbe_func_sim_path=r'C:\\新建文件夹 (2)\\MKGCN-main - 副本\\data\\MicrobeDrugA\\MDAD\\a\\microbe_function_sim.txt',
                 drug_struct_sim_path=r'C:\\新建文件夹 (2)\\MKGCN-main - 副本\\data\\MicrobeDrugA\\MDAD\\a\\drug_structure_sim.txt',
                 use_real_data=True):
        """
        Parameters:
        -----------
        a1_path : str
            Path to A1 matrix file (MDAD core associations)
        a2_path : str
            Path to A2 matrix file (extended associations)
        microbe_func_sim_path : str
            Path to microbe functional similarity matrix
        drug_struct_sim_path : str
            Path to drug structural similarity matrix
        use_real_data : bool
            If True, load from files; if False, generate synthetic data
        """
        self.a1_path = a1_path
        self.a2_path = a2_path
        self.microbe_func_sim_path = microbe_func_sim_path
        self.drug_struct_sim_path = drug_struct_sim_path
        self.use_real_data = use_real_data

        # These will be determined from data
        self.n_drugs = None
        self.n_microbes = None

        # Cache for similarity matrices (shared between A1 and A2)
        self.S_mf = None  # Microbe functional similarity
        self.S_rf = None  # Drug structural similarity

    def load_matrix_from_file(self, filepath, name="matrix"):
        """
        Load matrix from text file with multiple delimiter attempts

        Expected format: space-separated or tab-separated values

        Returns:
            A: np.array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"  Loading {name} from: {filepath}")

        # Try different delimiters
        delimiters = [' ', '\t', ',', ';']
        A = None

        for delim in delimiters:
            try:
                A = np.loadtxt(filepath, delimiter=delim)
                print(f"    Successfully loaded with delimiter: '{delim}'")
                break
            except:
                continue

        if A is None:
            try:
                # Auto-detect
                A = np.loadtxt(filepath)
                print(f"    Successfully loaded with auto-detection")
            except Exception as e:
                raise ValueError(f"Failed to load {filepath}: {e}")

        print(f"    Matrix shape: {A.shape}")
        if np.sum(A) > 0:
            print(f"    Non-zero elements: {int(np.sum(A != 0))}")
            print(f"    Sparsity: {100 * np.sum(A != 0) / A.size:.2f}%")

        return A

    def load_association_matrix(self, matrix_type='A1'):
        """
        Load association matrix A1 or A2

        Parameters:
            matrix_type: 'A1' or 'A2'

        Returns:
            A: np.array, shape (n_drugs, n_microbes)
        """
        if self.use_real_data:
            if matrix_type == 'A1':
                A = self.load_matrix_from_file(self.a1_path, f"A1 (core associations)")
            else:  # A2
                A = self.load_matrix_from_file(self.a2_path, f"A2 (extended associations)")

            # Set dimensions
            if self.n_drugs is None:
                self.n_drugs, self.n_microbes = A.shape
                print(f"  Dataset dimensions: {self.n_drugs} drugs × {self.n_microbes} microbes")

            return A

        else:
            # Synthetic data for testing
            if self.n_drugs is None:
                self.n_drugs = 1373
                self.n_microbes = 173

            A = np.zeros((self.n_drugs, self.n_microbes))

            if matrix_type == 'A1':
                n_pos = 2470
            else:
                n_pos = 5251

            np.random.seed(42)
            pos_indices = np.random.choice(self.n_drugs * self.n_microbes,
                                           n_pos, replace=False)
            A.flat[pos_indices] = 1

            return A

    def load_microbe_functional_similarity(self):
        """
        Load microbe functional similarity matrix from file
        Based on STRING database and Kamneva method

        Returns:
            S_mf: np.array, shape (n_microbes, n_microbes)
        """
        if self.S_mf is not None:
            return self.S_mf

        try:
            S_mf = self.load_matrix_from_file(self.microbe_func_sim_path,
                                              "Microbe Functional Similarity")

            # Verify dimensions
            if self.n_microbes is not None and S_mf.shape != (self.n_microbes, self.n_microbes):
                print(f"    Warning: Expected shape ({self.n_microbes}, {self.n_microbes}), "
                      f"got {S_mf.shape}")

            self.S_mf = S_mf
            return S_mf

        except FileNotFoundError:
            print(f"    Warning: Microbe functional similarity file not found.")
            print(f"    Using Gaussian kernel similarity only.")
            return None

    def load_drug_structural_similarity(self):
        """
        Load drug structural similarity matrix from file
        Based on SIMCOMP2 tool

        Returns:
            S_rf: np.array, shape (n_drugs, n_drugs)
        """
        if self.S_rf is not None:
            return self.S_rf

        try:
            S_rf = self.load_matrix_from_file(self.drug_struct_sim_path,
                                              "Drug Structural Similarity")

            # Verify dimensions
            if self.n_drugs is not None and S_rf.shape != (self.n_drugs, self.n_drugs):
                print(f"    Warning: Expected shape ({self.n_drugs}, {self.n_drugs}), "
                      f"got {S_rf.shape}")

            self.S_rf = S_rf
            return S_rf

        except FileNotFoundError:
            print(f"    Warning: Drug structural similarity file not found.")
            print(f"    Using Gaussian kernel similarity only.")
            return None

    def compute_gaussian_kernel_similarity(self, A, axis=0):
        """
        Compute Gaussian kernel similarity

        Parameters:
            A: association matrix (n_drugs, n_microbes)
            axis: 0 for microbe similarity, 1 for drug similarity

        Returns:
            S: similarity matrix
        """
        if axis == 0:  # Microbe similarity: compare columns
            features = A.T  # (n_microbes, n_drugs)
            n = features.shape[0]
        else:  # Drug similarity: compare rows
            features = A  # (n_drugs, n_microbes)
            n = features.shape[0]

        # Compute bandwidth parameter
        gamma = n / (np.sum(np.linalg.norm(features, axis=1) ** 2) + 1e-10)

        # Compute pairwise squared Euclidean distances
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(features, 'euclidean'))

        # Gaussian kernel
        S = np.exp(-gamma * dists ** 2)

        return S

    def fuse_similarities(self, S_g, S_f, name="similarity"):
        """
        Fuse Gaussian kernel similarity with functional/structural similarity
        According to paper formulas (2) and (4):

        S(i,j) = (S_g(i,j) + S_f(i,j)) / 2,  if S_f(i,j) != 0
        S(i,j) = S_g(i,j),                     otherwise

        Parameters:
            S_g: Gaussian kernel similarity matrix
            S_f: Functional/structural similarity matrix (can be None)
            name: name for logging

        Returns:
            S: fused similarity matrix
        """
        if S_f is None:
            print(f"    Using Gaussian kernel {name} only (no functional/structural data)")
            return S_g

        # Ensure same shape
        if S_g.shape != S_f.shape:
            print(f"    Warning: Shape mismatch S_g{S_g.shape} vs S_f{S_f.shape}")
            print(f"    Using Gaussian kernel {name} only")
            return S_g

        # Create mask where functional/structural similarity is non-zero
        mask = (S_f != 0)

        # Initialize with Gaussian kernel similarity
        S = S_g.copy()

        # Where functional/structural similarity exists, use average
        S[mask] = (S_g[mask] + S_f[mask]) / 2.0

        n_fused = np.sum(mask)
        n_total = mask.size
        print(f"    Fused {n_fused}/{n_total} elements ({100*n_fused/n_total:.1f}%) "
              f"with functional/structural {name}")

        return S

    def load_similarity_matrices(self, A, matrix_type='A1'):
        """
        Compute and fuse similarity matrices

        According to paper:
        - S_m = fuse(S_mg, S_mf)  [Formula 2]
        - S_r = fuse(S_rg, S_rf)  [Formula 4]

        Parameters:
            A: association matrix
            matrix_type: 'A1' or 'A2' (for logging)

        Returns:
            S_d: drug similarity (n_drugs, n_drugs)
            S_m: microbe similarity (n_microbes, n_microbes)
        """
        print(f"  Computing similarities for {matrix_type}:")

        # Compute Gaussian kernel similarities
        print("    Computing Gaussian kernel similarities...")
        S_d_gaussian = self.compute_gaussian_kernel_similarity(A, axis=1)
        S_m_gaussian = self.compute_gaussian_kernel_similarity(A, axis=0)

        print(f"      Drug Gaussian kernel: {S_d_gaussian.shape}")
        print(f"      Microbe Gaussian kernel: {S_m_gaussian.shape}")

        # Load functional/structural similarities (only once, shared between A1 and A2)
        if matrix_type == 'A1':
            print("    Loading functional/structural similarities...")
            S_mf = self.load_microbe_functional_similarity()
            S_rf = self.load_drug_structural_similarity()
        else:
            # Reuse already loaded matrices
            S_mf = self.S_mf
            S_rf = self.S_rf
            if S_mf is not None:
                print("    Reusing loaded microbe functional similarity")
            if S_rf is not None:
                print("    Reusing loaded drug structural similarity")

        # Fuse similarities according to paper formulas (2) and (4)
        S_d = self.fuse_similarities(S_d_gaussian, S_rf, "drug similarity")
        S_m = self.fuse_similarities(S_m_gaussian, S_mf, "microbe similarity")

        print(f"    Final drug similarity matrix: {S_d.shape}")
        print(f"    Final microbe similarity matrix: {S_m.shape}")

        return S_d, S_m

    def construct_heterogeneous_network(self, A, S_d, S_m):
        """
        Construct heterogeneous network H

        H = [S_d   A  ]
            [A^T  S_m]

        Returns:
            H: adjacency matrix (n_drugs+n_microbes, n_drugs+n_microbes)
            X: feature matrix
        """
        n_d, n_m = A.shape

        # Adjacency matrix
        H = np.zeros((n_d + n_m, n_d + n_m))
        H[:n_d, :n_d] = S_d  # Drug-drug similarity
        H[:n_d, n_d:] = A  # Drug-microbe association
        H[n_d:, :n_d] = A.T  # Microbe-drug association
        H[n_d:, n_d:] = S_m  # Microbe-microbe similarity

        # Feature matrix (block diagonal)
        X = np.zeros((n_d + n_m, n_d + n_m))
        X[:n_d, :n_d] = S_d
        X[n_d:, n_d:] = S_m

        print(f"  Heterogeneous network: {H.shape}")

        return H, X

    def prepare_data(self):
        """
        Prepare all data for MHGCNMDA

        Returns:
            data_dict: dictionary containing all matrices
        """
        print("\n[Data Loading]")
        print("-" * 50)

        # Load association matrices
        print("\nLoading A1 (MDAD core dataset):")
        A1 = self.load_association_matrix('A1')

        print("\nLoading A2 (Extended dataset):")
        A2 = self.load_association_matrix('A2')

        # Verify dimensions match
        assert A1.shape == A2.shape, "A1 and A2 must have same dimensions"
        self.n_drugs, self.n_microbes = A1.shape

        # Compute and fuse similarities for H1
        print("\nComputing similarities for H1:")
        S_d1, S_m1 = self.load_similarity_matrices(A1, 'A1')

        # Compute and fuse similarities for H2 (reuse functional/structural matrices)
        print("\nComputing similarities for H2:")
        S_d2, S_m2 = self.load_similarity_matrices(A2, 'A2')

        # Construct heterogeneous networks
        print("\nConstructing heterogeneous networks:")
        H1, X1 = self.construct_heterogeneous_network(A1, S_d1, S_m1)
        H2, X2 = self.construct_heterogeneous_network(A2, S_d2, S_m2)

        data_dict = {
            'A1': A1, 'A2': A2,
            'H1': H1, 'H2': H2,
            'X1': X1, 'X2': X2,
            'S_d1': S_d1, 'S_m1': S_m1,
            'S_d2': S_d2, 'S_m2': S_m2,
            'S_rf': self.S_rf,  # Drug structural similarity
            'S_mf': self.S_mf,  # Microbe functional similarity
            'n_drugs': self.n_drugs,
            'n_microbes': self.n_microbes
        }

        print("\n" + "=" * 50)
        print("Data loading completed successfully!")
        print(f"Drugs: {self.n_drugs}, Microbes: {self.n_microbes}")
        print(f"A1 associations: {int(np.sum(A1))}")
        print(f"A2 associations: {int(np.sum(A2))}")
        print(f"A2 - A1 additional: {int(np.sum(A2)) - int(np.sum(A1))}")
        if self.S_rf is not None:
            print(f"Drug structural similarity: Loaded and fused")
        else:
            print(f"Drug structural similarity: Not available")
        if self.S_mf is not None:
            print(f"Microbe functional similarity: Loaded and fused")
        else:
            print(f"Microbe functional similarity: Not available")
        print("=" * 50)

        return data_dict


# =============================================================================
# 2. MKGCN MODEL (SIMPLIFIED VERSION)
# =============================================================================

class MKGCN:
    """
    Simplified MKGCN for demonstration
    In full implementation, this would be a PyTorch nn.Module
    """

    def __init__(self, n_nodes, hidden_dims=[128, 64, 32]):
        self.n_nodes = n_nodes
        self.hidden_dims = hidden_dims

    def get_embeddings(self, X, H, seed=42):
        """
        Get node embeddings using simplified GCN propagation

        Returns:
            embeddings: list of embeddings for each layer
        """
        np.random.seed(seed)

        # Normalize adjacency
        D = np.diag(np.sum(H, axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        A_norm = D_inv_sqrt @ H @ D_inv_sqrt

        embeddings = []
        H_current = X.copy()

        for dim in self.hidden_dims:
            # Simplified GCN: H^(l) = ReLU(A_norm * H^(l-1) * W)
            W = np.random.randn(H_current.shape[1], dim) * 0.01
            H_next = A_norm @ H_current @ W
            H_next = np.maximum(H_next, 0)  # ReLU
            embeddings.append(H_next)
            H_current = H_next

        return embeddings


# =============================================================================
# 3. MULTI-KERNEL FUSION AND DLapRLS
# =============================================================================

class MultiKernelFusion:
    """
    Multi-kernel fusion and DLapRLS prediction
    """

    def __init__(self, gamma_hl=2 ** -6, lambda_d=2 ** -4, lambda_m=2 ** -4,
                 n_iterations=10):
        self.gamma_hl = gamma_hl
        self.lambda_d = lambda_d
        self.lambda_m = lambda_m
        self.n_iterations = n_iterations

    def compute_gip_kernel(self, H_embed):
        """
        Compute Gaussian Interaction Profile (GIP) kernel
        """
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(H_embed, 'euclidean'))
        K = np.exp(-self.gamma_hl * dists ** 2)
        return K

    def combine_kernels(self, K_prior, K_h1, K_h2, K_h3):
        """
        Combine multiple kernels with average weighting
        """
        return (K_prior + K_h1 + K_h2 + K_h3) / 4.0

    def compute_laplacian(self, K):
        """
        Compute normalized Laplacian matrix
        """
        D = np.diag(np.sum(K, axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L = D_inv_sqrt @ (D - K) @ D_inv_sqrt
        return L

    def dlaprls_predict(self, K_d, K_m, Y_train, n_drugs, n_microbes):
        """
        Dual Laplacian Regularized Least Squares

        Corrected version with proper matrix dimensions
        """
        # Initialize alpha matrices
        # alpha_d: (n_drugs, n_microbes) - represents drug latent factors
        # alpha_m: (n_microbes, n_drugs) - represents microbe latent factors
        alpha_d = np.zeros((n_drugs, n_microbes))
        alpha_m = np.zeros((n_microbes, n_drugs))

        # Laplacians
        L_d = self.compute_laplacian(K_d)
        L_m = self.compute_laplacian(K_m)

        # Iterative update
        for iteration in range(self.n_iterations):
            # Update alpha_d
            # Formula: alpha_d = (K_d^T K_d + lambda_d L_d)^{-1} K_d^T (2Y - (K_m @ alpha_m).T)
            left_d = K_d.T @ K_d + self.lambda_d * L_d + 1e-6 * np.eye(n_drugs)

            # Corrected: (K_m @ alpha_m).T has shape (n_drugs, n_microbes), matching Y_train
            reconstruction_m = (K_m @ alpha_m).T  # (n_drugs, n_microbes)
            right_d = K_d.T @ (2 * Y_train - reconstruction_m)

            try:
                alpha_d = np.linalg.solve(left_d, right_d)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                alpha_d = np.linalg.lstsq(left_d, right_d, rcond=None)[0]

            # Update alpha_m
            # Formula: alpha_m = (K_m^T K_m + lambda_m L_m)^{-1} K_m^T (2Y^T - (K_d @ alpha_d).T)
            left_m = K_m.T @ K_m + self.lambda_m * L_m + 1e-6 * np.eye(n_microbes)

            # Corrected: (K_d @ alpha_d).T has shape (n_microbes, n_drugs), matching Y_train.T
            reconstruction_d = (K_d @ alpha_d).T  # (n_microbes, n_drugs)
            right_m = K_m.T @ (2 * Y_train.T - reconstruction_d)

            try:
                alpha_m = np.linalg.solve(left_m, right_m)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                alpha_m = np.linalg.lstsq(left_m, right_m, rcond=None)[0]

        # Final prediction: Score = K_d @ alpha_d + (K_m @ alpha_m).T
        # Both terms have shape (n_drugs, n_microbes)
        Score = K_d @ alpha_d + (K_m @ alpha_m).T

        return Score


# =============================================================================
# 4. MHGCNMDA COMPLETE MODEL
# =============================================================================

class MHGCNMDA:
    """
    Complete MHGCNMDA model
    """

    def __init__(self, n_drugs, n_microbes, hidden_dims=[128, 64, 32],
                 w=0.77, gamma_hl=2 ** -6, lambda_d=2 ** -4, lambda_m=2 ** -4):
        self.n_drugs = n_drugs
        self.n_microbes = n_microbes
        self.w = w

        self.mkgcn = MKGCN(n_drugs + n_microbes, hidden_dims)
        self.mkf = MultiKernelFusion(gamma_hl, lambda_d, lambda_m)

    def predict_single_network(self, H, X, S_d, S_m, Y_train):
        """
        Predict using single network (H1 or H2)
        """
        # Get embeddings
        embeddings = self.mkgcn.get_embeddings(X, H)

        # Split drug and microbe embeddings
        drug_embeds = [e[:self.n_drugs, :] for e in embeddings]
        microbe_embeds = [e[self.n_drugs:, :] for e in embeddings]

        # Compute GIP kernels
        K_d_gcn = [self.mkf.compute_gip_kernel(e) for e in drug_embeds]
        K_m_gcn = [self.mkf.compute_gip_kernel(e) for e in microbe_embeds]

        # Combine kernels
        K_d_combined = self.mkf.combine_kernels(S_d, K_d_gcn[0],
                                                K_d_gcn[1], K_d_gcn[2])
        K_m_combined = self.mkf.combine_kernels(S_m, K_m_gcn[0],
                                                K_m_gcn[1], K_m_gcn[2])

        # DLapRLS prediction
        Score = self.mkf.dlaprls_predict(K_d_combined, K_m_combined,
                                         Y_train, self.n_drugs, self.n_microbes)

        return Score

    def predict(self, data_dict, train_mask, network='both'):
        """
        Predict with optional fusion

        network: 'H1', 'H2', or 'both'
        """
        A1 = data_dict['A1']

        # Create training label matrix
        Y_train = A1.copy()
        if train_mask is not None:
            # Mask out test samples
            test_mask_2d = train_mask.reshape(A1.shape) == False
            Y_train[test_mask_2d] = 0

        scores = {}

        if network in ['H1', 'both']:
            scores['H1'] = self.predict_single_network(
                data_dict['H1'], data_dict['X1'],
                data_dict['S_d1'], data_dict['S_m1'], Y_train
            )

        if network in ['H2', 'both']:
            scores['H2'] = self.predict_single_network(
                data_dict['H2'], data_dict['X2'],
                data_dict['S_d2'], data_dict['S_m2'], Y_train
            )

        if network == 'both':
            # Weighted fusion
            scores['fusion'] = self.w * scores['H1'] + (1 - self.w) * scores['H2']

        return scores


# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================

class MetricsCalculator:
    @staticmethod
    def calculate_all(y_true, y_score, threshold=0.5):
        """
        Calculate all metrics
        """
        y_pred = (y_score >= threshold).astype(int)

        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            return {'AUROC': 0.5, 'AUPR': 0, 'Accuracy': 0,
                    'F1-Score': 0, 'F1-max': 0}

        metrics = {
            'AUROC': roc_auc_score(y_true, y_score),
            'AUPR': average_precision_score(y_true, y_score),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0)
        }

        # F1-max
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        metrics['F1-max'] = np.max(f1_scores)

        return metrics


# =============================================================================
# 6. CROSS-VALIDATION
# =============================================================================

class CrossValidator:
    def __init__(self, n_splits=5, n_repeats=10, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        # 存储所有fold的真实标签和预测分数
        self.all_y_true = []
        self.all_y_score = []

    def create_train_test_split(self, A, fold_idx, repeat_idx):
        """
        Create stratified train/test split
        """
        np.random.seed(self.random_state + repeat_idx * 100 + fold_idx)

        n_drugs, n_microbes = A.shape

        # Get positive samples
        pos_indices = np.argwhere(A == 1)  # (n_pos, 2)
        n_pos = len(pos_indices)

        # Sample negative samples (same number as positive for balance)
        neg_indices = np.argwhere(A == 0)
        n_neg_samples = min(n_pos * 5, len(neg_indices))  # 5:1 ratio
        neg_sample_idx = np.random.choice(len(neg_indices), n_neg_samples, replace=False)
        neg_indices = neg_indices[neg_sample_idx]

        # Combine
        all_indices = np.vstack([pos_indices, neg_indices])
        all_labels = np.concatenate([np.ones(len(pos_indices)),
                                     np.zeros(len(neg_indices))])

        # Shuffle
        shuffle_idx = np.random.permutation(len(all_indices))
        all_indices = all_indices[shuffle_idx]
        all_labels = all_labels[shuffle_idx]

        # K-fold split
        fold_size = len(all_indices) // self.n_splits
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < self.n_splits - 1 else len(all_indices)

        test_idx = np.arange(start_idx, end_idx)
        train_idx = np.concatenate([np.arange(0, start_idx),
                                    np.arange(end_idx, len(all_indices))])

        train_pairs = all_indices[train_idx]
        test_pairs = all_indices[test_idx]
        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]

        return train_pairs, test_pairs, train_labels, test_labels

    def run_single_fold(self, model, data_dict, train_pairs, test_pairs,
                        test_labels, fold_idx):
        """
        Run single fold
        """
        print(f"    Fold {fold_idx + 1}/{self.n_splits}")

        # Create training mask
        A1 = data_dict['A1']
        train_mask = np.zeros_like(A1, dtype=bool)
        for i, j in train_pairs:
            train_mask[i, j] = True

        # Predict
        scores = model.predict(data_dict, train_mask, network='both')
        Score_fused = scores['fusion']

        # Extract test scores
        test_scores = np.array([Score_fused[i, j] for i, j in test_pairs])

        # 收集数据用于可视化
        self.all_y_true.extend(test_labels)
        self.all_y_score.extend(test_scores)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all(test_labels, test_scores)

        print(f"      AUROC: {metrics['AUROC']:.4f}, "
              f"AUPR: {metrics['AUPR']:.4f}, "
              f"F1-max: {metrics['F1-max']:.4f}")

        return metrics

    def get_roc_pr_data(self):
        """
        获取用于绘制ROC和PR曲线的数据
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

        y_true = np.array(self.all_y_true)
        y_score = np.array(self.all_y_score)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'auc': auc(fpr, tpr),
            'ap': average_precision_score(y_true, y_score),
            'y_true': y_true,
            'y_score': y_score
        }


    def run_single_repeat(self, model, data_dict, repeat_idx):
        """
        Run one repeat of 5-fold CV
        """
        print(f"\n  Repeat {repeat_idx + 1}/{self.n_repeats}")
        print("  " + "-" * 40)

        A1 = data_dict['A1']
        fold_metrics = []

        for fold_idx in range(self.n_splits):
            train_pairs, test_pairs, train_labels, test_labels = \
                self.create_train_test_split(A1, fold_idx, repeat_idx)

            metrics = self.run_single_fold(model, data_dict, train_pairs,
                                           test_pairs, test_labels, fold_idx)
            fold_metrics.append(metrics)

        # Average across folds
        mean_metrics = {}
        for key in ['AUROC', 'AUPR', 'Accuracy', 'F1-Score', 'F1-max']:
            values = [f[key] for f in fold_metrics]
            mean_metrics[key] = np.mean(values)
            mean_metrics[key + '_std'] = np.std(values)

        print(f"\n  Repeat {repeat_idx + 1} Average:")
        print(f"    AUROC: {mean_metrics['AUROC']:.4f} ± {mean_metrics['AUROC_std']:.4f}")

        return fold_metrics, mean_metrics

    def run_full_experiment(self, model, data_dict):
        """
        Run complete experiment
        """
        print(f"\n{'=' * 70}")
        print(f"Starting {self.n_splits}-fold CV × {self.n_repeats} repeats")
        print(f"{'=' * 70}")

        all_repeat_metrics = []

        for repeat_idx in range(self.n_repeats):
            _, mean_metrics = self.run_single_repeat(model, data_dict, repeat_idx)
            all_repeat_metrics.append(mean_metrics)

        # Final statistics
        final_results = {}
        for key in ['AUROC', 'AUPR', 'Accuracy', 'F1-Score', 'F1-max']:
            values = [m[key] for m in all_repeat_metrics]
            final_results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

        return final_results, all_repeat_metrics


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution with real data loading
    Includes microbe functional similarity and drug structural similarity
    """
    print("=" * 70)
    print("MHGCNMDA: Multi-source Heterogeneous Graph Convolutional Network")
    print("for Microbe-Drug Association Prediction")
    print("With Microbe Functional Similarity and Drug Structural Similarity")
    print("=" * 70)

    # Configuration
    USE_REAL_DATA = True  # Set to False for synthetic data testing

    # Paths to your data files
    base_path = r'C:\\新建文件夹 (2)\\MKGCN-main - 副本\\data\\MicrobeDrugA\\MDAD\\a'

    A1_PATH = os.path.join(base_path, 'drug_microbe_matrix.txt')
    A2_PATH = os.path.join(base_path, 'drug_dmicrobe_matrix.txt')
    MICROBE_FUNC_SIM_PATH = os.path.join(base_path, 'microbe_function_sim.txt')
    DRUG_STRUCT_SIM_PATH = os.path.join(base_path, 'drug_structure_sim.txt')

    print("\n[Configuration]")
    print(f"  A1 (core): {A1_PATH}")
    print(f"  A2 (extended): {A2_PATH}")
    print(f"  Microbe functional similarity: {MICROBE_FUNC_SIM_PATH}")
    print(f"  Drug structural similarity: {DRUG_STRUCT_SIM_PATH}")

    # 1. Load data
    print("\n[Step 1] Loading data...")
    dataset = MDADDataset(
        a1_path=A1_PATH,
        a2_path=A2_PATH,
        microbe_func_sim_path=MICROBE_FUNC_SIM_PATH,
        drug_struct_sim_path=DRUG_STRUCT_SIM_PATH,
        use_real_data=USE_REAL_DATA
    )

    try:
        data_dict = dataset.prepare_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nFalling back to synthetic data for demonstration...")
        dataset.use_real_data = False
        data_dict = dataset.prepare_data()

    # 2. Initialize model
    print("\n[Step 2] Initializing model...")
    model = MHGCNMDA(
        n_drugs=data_dict['n_drugs'],
        n_microbes=data_dict['n_microbes'],
        hidden_dims=[128, 64, 32],
        w=0.77,
        gamma_hl=2 ** -6,
        lambda_d=2 ** -4,
        lambda_m=2 ** -4
    )

    # 3. Run cross-validation
    print("\n[Step 3] Running cross-validation...")
    cv = CrossValidator(n_splits=5, n_repeats=10, random_state=42)

    final_results, all_repeats = cv.run_full_experiment(model, data_dict)

    # 收集ROC/PR数据
    roc_pr_data = cv.get_roc_pr_data()

    # 保存实验结果供可视化使用
    experiment_data = {
        'mhgcnda': {
            'fpr': roc_pr_data['fpr'],
            'tpr': roc_pr_data['tpr'],
            'precision': roc_pr_data['precision'],
            'recall': roc_pr_data['recall'],
            'auc': roc_pr_data['auc'],
            'ap': roc_pr_data['ap'],
            'color': '#E94F37',
            'linewidth': 3
        },
        'final_metrics': final_results
    }

    # 保存到文件
    import pickle
    with open('experiment_results.pkl', 'wb') as f:
        pickle.dump(experiment_data, f)
    print("\n实验结果已保存到 experiment_results.pkl")

    # 4. Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Table 3 format)")
    print("Mean ± Std over 10 repeats of 5-fold CV")
    print("=" * 70)

    for metric in ['AUROC', 'AUPR', 'Accuracy', 'F1-Score']:
        mean = final_results[metric]['mean']
        std = final_results[metric]['std']
        print(f"{metric:12s}: {mean:.4f} ± {std:.4f}")

    # F1-max (recommended)
    print(f"{'F1-max':12s}: {final_results['F1-max']['mean']:.4f} ± "
          f"{final_results['F1-max']['std']:.4f} (recommended)")

    # 5. LaTeX format - 使用简单的print避免转义问题
    print("\n" + "=" * 70)
    print("LaTeX Table Row:")
    print("=" * 70)

    auroc_mean = final_results['AUROC']['mean']
    auroc_std = final_results['AUROC']['std']
    aupr_mean = final_results['AUPR']['mean']
    aupr_std = final_results['AUPR']['std']
    acc_mean = final_results['Accuracy']['mean']
    f1_mean = final_results['F1-Score']['mean']

    # 使用多个print避免复杂的f-string转义
    print("MHGCNMDA & ", end="")
    print(f"{auroc_mean:.4f} $\\pm$ {auroc_std:.4f} & ", end="")
    print(f"{aupr_mean:.4f} $\\pm$ {aupr_std:.4f} & ", end="")
    print(f"{acc_mean:.4f} & ", end="")
    print(f"{f1_mean:.4f} \\")

    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)

    return final_results



if __name__ == "__main__":
    results = main()