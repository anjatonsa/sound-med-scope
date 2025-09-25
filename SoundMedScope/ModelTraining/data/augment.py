import numpy as np
import random
import librosa

class Augmenter:
    # Dodavanje belog šuma
    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    # Promena brzine reprodukcije
    def time_stretch(self, audio, rate=1.0):
        return librosa.effects.time_stretch(y=audio, rate=rate)

    # Promena visine tona bez promene trajanja
    def pitch_shift(self, audio, sr, n_steps=0):
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    def augment(self, y, sr):
        choice = random.choice(['noise', 'stretch', 'pitch', 'none'])
        if choice == 'noise':
            return self.add_noise(y)
        elif choice == 'stretch':
            return self.time_stretch(y, rate=random.uniform(0.8, 1.2))
        elif choice == 'pitch':
            return self.pitch_shift(y, sr, n_steps=random.uniform(-2, 2))
        return y
