import numpy as np

class AudioUtils:
    @staticmethod
    def pad_audio(audio, target_len=2048):
        if len(audio) < target_len:
            padding = target_len - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
