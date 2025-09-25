import tensorflow as tf
from tensorflow.keras import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import ( # pyright: ignore[reportMissingImports]
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input
)
from config import learning_rate

class ModelBuilder:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Prvi konvolucioni blok: Conv1D -> BatchNorm -> MaxPooling
        model.add(Conv1D(64, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        # Drugi konvolucioni blok
        model.add(Conv1D(128, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        # LSTM slojevi za obradu vremenskih/sekvencijalnih podataka
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
