MHGCNMDA: Multi-source Heterogeneous Graph Convolutional Network for Microbe-Drug Association Prediction

MHGCNMDA is a novel computational framework for predicting potential microbe-drug associations by integrating multi-source heterogeneous biological information. The model addresses three key limitations of existing methods:
1.Single data source: Integrates drug-drug, microbe-microbe, microbe-drug, microbe-disease, and drug-disease associations
2.Static feature fusion: Employs multi-layer GCN with dynamic multi-kernel fusion
3.Low optimization efficiency: Utilizes Dual Laplacian Regularized Least Squares (DLapRLS)


Requirements
Python 3.8 or higher
NumPy >= 1.21.0
PyTorch >= 1.9.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
scipy >= 1.7.0

Data Preparation
Download the MDAD dataset from Figshare
Place data files in the data/MicrobeDrugA/MDAD/a/ directory:
drug_microbe_matrix.txt - Core association matrix A¹
drug_dmicrobe_matrix.txt - Extended association matrix A²
microbe_function_sim.txt - Microbe functional similarity
drug_structure_sim.txt - Drug structural similarity
Training
Run the main experiment with 5-fold cross-validation (10 repeats):
