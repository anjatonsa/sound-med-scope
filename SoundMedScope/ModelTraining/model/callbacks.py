import numpy as np
from tensorflow.keras.callbacks import Callback # pyright: ignore[reportMissingImports]

class AccuracyLossCheckpoint(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_val_acc = -np.inf
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch+1}: val_accuracy improved, saving model...")
        elif np.isclose(val_acc, self.best_val_acc):
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(self.filepath)
                print(f"\nEpoch {epoch+1}: val_loss improved, saving model...")
