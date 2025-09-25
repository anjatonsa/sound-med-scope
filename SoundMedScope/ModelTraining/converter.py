import tensorflow as tf
import numpy as np
import tf2onnx


def convert_to_onnx(model_path="best_model.keras", output_path="model.onnx", input_shape=(54, 1)):
    # 1. Učitavanje Keras modela
    model = tf.keras.models.load_model(model_path)

    # 2. Dummy input da "probudi" model
    dummy_input = tf.random.normal([1, *input_shape])
    model(dummy_input)

    # 3. Konverzija u ONNX
    full_model = tf.function(model)
    onnx_model, _ = tf2onnx.convert.from_function(
        full_model,
        input_signature=[tf.TensorSpec([None, *input_shape], tf.float32)],
        opset=12
    )

    # 4. Snimanje ONNX fajla
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ ONNX model uspešno sačuvan u: {output_path}")


def main():
    convert_to_onnx(
        model_path="best_model.keras",
        output_path="model.onnx",
        input_shape=(54, 1)
    )


if __name__ == "__main__":
    main()
