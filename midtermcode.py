from pathlib import Path
from warnings import filterwarnings
!pip install rdkit
# Silence some expected warnings
filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# Neural network specific libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

%matplotlib inline

def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.
    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.
    method : str
        The type of fingerprint to use. Default is MACCS keys.
    n_bits : int
        The length of the fingerprint.
    Returns
    -------
    array
        The fingerprint array.
    """

    # Convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        return np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    if method == "morgan3":
        return np.array(GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
    else:
        print(f"Warning: Wrong method specified: {method}." " Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))
        
def convert_target_id(chembl_code, input_length=25):
    """Converts a ChEMBL target id to a numpy array of fixed length that can be used as input the neural network"""
    
    # Remove the 'CHEMBL' prefix from the ChEMBL code
    code_integer = int(chembl_code.replace('CHEMBL', ''))
    
    # Convert the integer to a numpy array of fixed length
    binary_str = '{0:b}'.format(code_integer).zfill(input_length)
    input_array = np.array([float(bit) for bit in binary_str], dtype=np.float64)
    
    return input_array
    
    def neural_network_model(hidden1, hidden2):
    """
    Creating a neural network from two hidden layers
    using ReLU as activation function in the two hidden layers
    and a linear activation in the output layer.
    Parameters
    ----------
    hidden1 : int
        Number of neurons in first hidden layer.
    hidden2: int
        Number of neurons in second hidden layer.
    Returns
    -------
    model
        Fully connected neural network model with two hidden layers.
    """

    model = Sequential()
    # First hidden layer
    model.add(Dense(hidden1, activation="relu", name="layer1"))
    # Second hidden layer
    model.add(Dense(hidden2, activation="relu", name="layer2"))
    # Output layer
    model.add(Dense(1, activation="linear", name="layer4"))
    
    # Compile model
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=["mse", "mae"])
    return model
    
# Mount Google Drive to access files
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/kinase.csv', index_col=0)
df = df.reset_index(drop=True)

# Keep necessary columns
chembl_df = df[["smiles", "target_chembl_id","standard_value"]]
# Convert IC50 to pIC50 (standart_value to the pIC50)
chembl_df['pIC50'] = -np.log10(df['standard_value'])

chembl_df.replace([np.inf, -np.inf], np.nan, inplace=True)
chembl_df.dropna(inplace=True)

chembl_df["fingerprints_df"] = chembl_df["smiles"].apply(smiles_to_fp)

#binary representation of the chembl ID
chembl_df["binary_id"] = chembl_df["target_chembl_id"].apply(convert_target_id)

#combiens information from traget and chemical structure
chembl_df['combined_input'] = chembl_df.apply(lambda row: np.concatenate([row['fingerprints_df'], row['binary_id']]), axis=1)

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(
    chembl_df["combined_input"], chembl_df[["pIC50"]], test_size=0.3, random_state=42
)

# Neural network parameters
batch_sizes = [16, 32, 64]
nb_epoch = 50
layer1_size = 64
layer2_size = 32

model = neural_network_model(layer1_size, layer2_size)

# Save the trained model
weights_folder = "/content/drive/MyDrive/Colab Notebooks/"

# Save the trained model
filepath = os.path.join(weights_folder, "best_weights.hdf5")
checkpoint = ModelCheckpoint(
    str(filepath),
    monitor="loss",
    verbose=0,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)
callbacks_list = [checkpoint]

# Fit the model
model.fit(
    np.array(list((x_train))).astype(float),
    y_train.values,
    epochs=nb_epoch,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=0,
)
