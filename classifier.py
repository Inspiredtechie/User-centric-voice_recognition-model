import joblib
import librosa
import numpy as np
import os

class VoiceClassifier:
    def __init__(self, model_dir="models"):
        self.models = {}
        # Load all .gmm models
        for file in os.listdir(model_dir):
            if file.endswith(".gmm"):
                name = file.split(".")[0]
                self.models[name] = joblib.load(os.path.join(model_dir, file))
        
        print(f"[INIT] Loaded models: {list(self.models.keys())}")

    def extract_features_realtime(self, audio_buffer, sample_rate):
        # Convert buffer to float32
        audio_float = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
        
        # Normalize
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))
            
        mfccs = librosa.feature.mfcc(y=audio_float, sr=sample_rate, n_mfcc=40)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        combined = np.concatenate((mfccs, delta, delta2))
        return combined.T

    def predict(self, audio_buffer, sample_rate=16000):
        features = self.extract_features_realtime(audio_buffer, sample_rate)
        if features.shape[0] == 0:
            return "Silence"

        scores = {}
        for name, model in self.models.items():
            # Score_samples returns log-likelihood for each frame
            # We sum them to get the total likelihood for the audio clip
            scores[name] = np.sum(model.score_samples(features))
        
        # Determine winner
        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]
        
        # Calculate a confidence margin (Owner score vs Unknown score)
        # If the difference is too small, it's ambiguous
        
        return best_match, scores