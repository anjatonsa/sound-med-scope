import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import test_size, random_state, batch_size, epochs
from data.dataset_loader import DatasetLoader
from data.feature_extractor import FeatureExtractor
from model.model_builder import ModelBuilder
from model.callbacks import AccuracyLossCheckpoint


def main():
    # 1. Učitavanje podataka
    loader = DatasetLoader()
    all_audio_data = loader.load_dataset()

    # 2. Ekstrakcija osobina
    extractor = FeatureExtractor()
    X, y = extractor.extract(all_audio_data)

    # 3. Kodiranje labela
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 4. Podela na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # 5. Reshape podataka
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"Input feature length: {X_train.shape[1]}")

    # 6. Kreiranje modela
    builder = ModelBuilder((X_train.shape[1], 1), len(np.unique(y_encoded)))
    model = builder.build()

    # 7. Callback funkcije
    callbacks = [
        AccuracyLossCheckpoint('best_model.keras'),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 8. Trening modela
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # 9. Evaluacija najboljeg modela
    best_model = tf.keras.models.load_model('best_model.keras')
    _, test_acc = best_model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # 10. Predikcije
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # 11. Konfuziona matrica
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
