import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

DATASET_PATH = "data"
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

def extract_features(file_path):
    # Load audio
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Extract MFCC (Mel-Frequency Cepstral Coefficients)
    # We take 40 coefficients to capture rich vocal detail
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Calculate Delta (velocity) and Delta-Delta (acceleration) for temporal patterns
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack features to create a comprehensive vector
    comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    
    # Transpose to (Time, Features) for GMM
    return comprehensive_mfccs.T

def train_gmm(label):
    features = []
    folder = os.path.join(DATASET_PATH, label)
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    
    print(f"[INFO] Extracting features for '{label}' from {len(files)} files...")
    
    for f in files:
        path = os.path.join(folder, f)
        try:
            vector = extract_features(path)
            if vector.shape[0] > 0:
                features.append(vector)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Stack all time frames from all files vertically
    features_stacked = np.vstack(features)
    
    print(f"[INFO] Training GMM for '{label}' with {features_stacked.shape[0]} feature vectors...")
    
    # GMM Configuration: 
    # 16 components is standard for voice; adjust to 32 if you have lots of data
    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, n_init=3)
    gmm.fit(features_stacked)
    
    # Save Model
    joblib.dump(gmm, os.path.join(MODEL_DIR, f"{label}.gmm"))
    print(f"[SUCCESS] Saved model: {label}.gmm")

if __name__ == "__main__":
    train_gmm("Soumar")
    train_gmm("Others")