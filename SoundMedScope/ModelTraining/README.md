# Model Training and Conversion

## Running Training
To train the model, run the following command in the console:

```bash
python main.py
```

After execution, the trained model will be saved as **`best_model.keras`**.

## Conversion to ONNX Format
If you want to convert the model to **ONNX** format, run:

```bash
python converter.py
```

This script will take the model from **`best_model.keras`** and save it as **`model.onnx`**.
