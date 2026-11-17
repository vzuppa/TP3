# Puntajes del LLM (`llm_dudosas_scores_v2.csv`)

Dato de entrada del modelo final. El uso de una columna puntuada por un LLM fue consultado y aprobado por el profesor.

## Qué contiene

| Columna | Significado |
|---|---|
| `base` | `train` o `test` |
| `fila` | número de fila (desde 0) dentro de `train_data.jsonlines` / `test_data.jsonlines` |
| `p_modelo` | probabilidad de "usado" del ensamble sin XLM-R en el momento de puntuar (sólo referencia; el modelo final no la usa) |
| `p_sonnet` | probabilidad de "usado" según Claude Sonnet (0 a 1) |
| `p_fuerte` | segunda lectura con un modelo de Claude más grande; sólo en el núcleo más dudoso (vacío en el resto) |

- **Qué filas se puntuaron:** las publicaciones donde el ensamble sin XLM-R daba una probabilidad entre 20% y 80%: 5.522 de las 6.456 de train que caían en esa zona y las 2.759 de test. La segunda lectura se hizo donde daba entre 35% y 65%: 1.284 de train y 1.157 de test.
- **Para qué se usan las de train:** para ajustar los pesos de la ponderación (regresión logística) en `2_entrenar_modelo_final.ipynb`.
- **Sin fuga de target:** en la ficha de una publicación de train, el historial del vendedor y de la categoría cuenta sólo las OTRAS publicaciones etiquetadas (nunca la propia). En las de test se usa todo train.

## Qué leyó el LLM (ficha de cada publicación)

Título, precio y moneda, tipo de publicación, garantía, stock inicial y vendidos, cantidad de fotos, atributos y variaciones, envío, medios de pago, fechas, ubicación, categoría, y dos líneas de historial:
- `VENDEDOR`: cuántas publicaciones tiene en la base y cuántas de sus otras publicaciones etiquetadas son usadas o nuevas.
- `CATEGORÍA`: cuántas otras publicaciones de la misma categoría son usadas o nuevas.

Las publicaciones se enviaron en tandas de 125. Los archivos con las fichas no se conservaron; sí los puntajes.

## Instrucciones que recibió el LLM (prompt)

```
NO escribas código, scripts ni reglas automáticas: leé cada publicación y decidí con tu propio juicio.

Vas a leer un archivo con publicaciones de MercadoLibre Argentina de 2015. Cada una empieza con una línea "### <k>" y sigue con su ficha completa: título, precio, tipo de publicación (free = gratuita, típica de particulares; bronze/silver/gold = pagas), garantía, stock, fotos, atributos, envío, medios de pago, fechas, ubicación, categoría, y dos líneas de historial:
- VENDEDOR: cuántas publicaciones tiene en la base y cuántas de sus OTRAS publicaciones ya etiquetadas son USADAS o NUEVAS (si tiene historial, es la pista más fuerte: los vendedores suelen publicar siempre lo mismo).
- CATEGORÍA: cuántas otras publicaciones de la misma categoría exacta son USADAS o NUEVAS.

Son casos DIFÍCILES: un modelo estadístico ya dudó en todos. Para cada publicación estimá la probabilidad (entero 0 a 100) de que el vendedor la haya publicado como USADO. Combiná todo:
- Tu conocimiento del mundo sobre el producto del título (¿se vende normalmente de segunda mano? ¿es un modelo/edición vieja? ¿es coleccionable, antigüedad, vehículo de año anterior?).
- Señales del texto: "usado", "impecable", "como nuevo", "poco uso", "funcionando", "antiguo", "permuto" → usado; "nuevo", "sin uso", "sellado", "a estrenar", "0km", "por mayor", "fabricantes" → nuevo.
- Perfil del vendedor: stock 1 + publicación free + pocas publicaciones + sin garantía sugiere particular (más usado); stock alto, garantía de fábrica, muchas publicaciones, envío gratis sugiere comercio (más nuevo).
- Convenciones del dataset: muchas librerías y vendedores de coleccionables etiquetan como NUEVO cosas viejas; por eso el historial del vendedor y de la categoría pesa más que el sentido común cuando existe.
Usá toda la escala: 50 solo si de verdad no hay pistas.

Salida: escribí el archivo de salida con exactamente una línea por publicación, en el mismo orden, con formato "<k> <probabilidad>" (ejemplo: "1532 85"). Sin encabezado ni nada más.
```
