# Trening i konverzija modela

## Pokretanje treninga
Za treniranje modela pokrenite sledeću komandu u konzoli:

```bash
python main.py
```

Nakon izvršavanja, istrenirani model će biti sačuvan u fajlu **`best_model.keras`**.

## Konverzija u ONNX format
Ako želite da konvertujete model u **ONNX** format, pokrenite:

```bash
python converter.py
```

Ovaj skript će uzeti model iz fajla **`best_model.keras`** i sačuvati ga u **`model.onnx`**.
