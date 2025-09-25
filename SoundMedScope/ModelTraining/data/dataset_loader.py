import os, random, librosa
from config import dataset_path, categories
from utils.helpers import AudioUtils
from .augment import Augmenter

class DatasetLoader:
    def __init__(self, target_len=2048):
        self.target_len = target_len
        self.augmenter = Augmenter()

    def load_dataset(self, max_count=None):
        all_audio_data = []
        category_counts = {
            category: len([f for f in os.listdir(os.path.join(dataset_path, category)) if f.endswith('.wav')])
            for category in categories
        }
        if max_count is None:
            max_count = max(category_counts.values())

        for category in categories:
            category_path = os.path.join(dataset_path, category)
            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            current_count = len(audio_files)

            loaded_signals = []
            for audio_file in audio_files:
                y, sr = librosa.load(os.path.join(category_path, audio_file), sr=None)
                y = AudioUtils.pad_audio(y, self.target_len)
                loaded_signals.append((y, sr))
                all_audio_data.append((y, sr, category))

            if current_count < max_count:
                needed = max_count - current_count
                for _ in range(needed):
                    orig_y, orig_sr = random.choice(loaded_signals)
                    y_aug = self.augmenter.augment(orig_y, orig_sr)
                    y_aug = AudioUtils.pad_audio(y_aug, self.target_len)
                    all_audio_data.append((y_aug, orig_sr, category))

        return all_audio_data
