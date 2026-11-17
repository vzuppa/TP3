# ¿Nuevo o usado? — Parcial 3, Análisis Predictivo (ITBA)

Valentín Zuppa · Competencia Kaggle AP 2026 Q2 · métrica: accuracy.

## El modelo final

1. **Tres modelos de lenguaje** leen la publicación "escrita" (título, garantía, precio, atributos…) y dan una probabilidad de "usado": DistilBERT, BETO y XLM-RoBERTa large (5 folds cada uno).
2. Esas probabilidades entran como **variables** junto con las de la publicación, el vendedor y la categoría (target encodings dentro de cada fold), el texto (TF-IDF + regresión logística), las fotos, los tiempos exactos y los vecinos de título: 133 variables.
3. **LightGBM (3 semillas), XGBoost (2) y CatBoost (1)** se entrenan con esas variables y se promedian.
4. En las **publicaciones dudosas** la probabilidad se combina, con una regresión logística, con la de un **LLM** que leyó la ficha completa (`llm_prompt.md`).
5. Si el resultado es ≥ 50%, la predicción es `used`.

| | Accuracy |
|---|---|
| CV de los 3 boosting promediados | 0.9465 |
| CV con el LLM en las dudosas | ≈ 0.948 |

## Orden de ejecución

| # | Notebook | Qué hace | Dónde | Salida |
|---|---|---|---|---|
| 1a | `1a_bert_distilbert.ipynb` | Ajusta DistilBERT multilingüe | GPU | `preds/oof_bert.npy`, `preds/test_bert.npy` |
| 1b | `1b_beto_kaggle.ipynb` | Ajusta BETO | Kaggle, GPU | `preds/oof_beto.npy`, `preds/test_beto.npy` |
| 1c | `1c_xlmr_large_kaggle.ipynb` | Ajusta XLM-RoBERTa large | Kaggle, GPU (~8 h) | `preds/oof_xlmr.npy`, `preds/test_xlmr.npy` |
| 2 | **`2_entrenar_modelo_final.ipynb`** | Calcula las variables, entrena y guarda los 3 boosting y la ponderación con el LLM | CPU, ~1 h | `modelos/` |
| 3 | **`3_predecir_test.ipynb`** | Aplica el modelo guardado al test | CPU, minutos | `submission.csv` |

Las predicciones de los modelos de lenguaje ya están en `preds/`, así que se puede arrancar directamente desde la notebook 2.

## Datos de entrada

- `train_data.jsonlines`, `test_data.jsonlines`: datos de la competencia. **No están en el repositorio** porque superan el límite de 100 MB de GitHub: hay que descargarlos de la pestaña *Data* de la competencia de Kaggle (AP 2026 Q2) y ponerlos en esta carpeta o en la de arriba.
- `preds/`: probabilidades de los tres modelos de lenguaje (salida de las notebooks 1a–1c).
- `llm_dudosas_scores_v2.csv`: puntajes del LLM en las publicaciones dudosas. Ver `llm_prompt.md`.
- `tp3_utils.py`: funciones de variables.

## Notas

- Las variables que usan la etiqueta se calculan dentro de cada fold: para las filas de validación solo se usan etiquetas de las filas de entrenamiento de ese fold.
- La notebook 2 entrena con 1 hilo para que el resultado sea idéntico entre corridas. Las variables de texto y de vecinos pueden variar levemente entre computadoras.
- Versiones: ver `requirements.txt`.
