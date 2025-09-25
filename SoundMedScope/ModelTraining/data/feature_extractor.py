import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, n_mfcc=40):
        self.n_mfcc = n_mfcc # Broj MFCC koeficijenata
        # Mel-Frequency Cepstral Coefficients — karakteristike temeljene na ljudskom sluhu

    def extract(self, all_audio_data):
        feature_list, labels_list = [], []
        for y, sr, category in all_audio_data:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

            features = np.hstack([
                np.mean(mfcc, axis=1),
                np.mean(chroma, axis=1),
                np.mean(spec_centroid, axis=1),
                np.mean(spec_bw, axis=1)
            ])
            feature_list.append(features)
            labels_list.append(category)

        return np.array(feature_list), np.array(labels_list)
