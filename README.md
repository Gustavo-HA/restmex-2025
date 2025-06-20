# REST-MEX 2025

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >3.12](https://img.shields.io/badge/python->3.12-blue.svg)](https://www.python.org/downloads/)

Este proyecto es un pipeline de extremo a extremo para el procesamiento y modelado de datos de reseñas de restaurantes mexicanos. El objetivo principal es realizar un análisis de sentimientos y una clasificación de texto para poder predecir la polaridad, la ciudad y el tipo de reseña.

## Datos

> El conjunto de datos se encuentra íntegramente en el fichero `data/raw/restmex-corpus.csv`.
> El conjunto de datos consta de 208.051 filas (el 70% del conjunto de datos original. El 30% restante se utilizará como conjunto de prueba), una por cada opinión. Cada fila contiene 6 columnas:
>
> 1.  **Title**: El título que el turista asignó a su opinión. Tipo de datos: Texto.
> 2.  **Review**: La opinión emitida por el turista. Tipo de datos: Texto.
> 3.  **Polarity**: La etiqueta que representa la polaridad del sentimiento de la opinión. Tipo de datos: [1, 2, 3, 4, 5].
> 4.  **Town**: El pueblo en el que se centra la reseña. Tipo de datos: Texto.
> 5.  **Region**: La región (estado de México) donde se encuentra el pueblo. Tipo de datos: Texto. Esta característica no está pensada para ser clasificada, pero se proporciona como información adicional que podría ser aprovechada en los modelos de clasificación.
> 6.  **Type**: El tipo de lugar al que se refiere la reseña. Tipo de datos: [Hotel, Restaurante, Atractivo].

## Características

- **Preprocesamiento de datos**: Limpieza y preparación de los datos de texto para el modelado.
- **Análisis exploratorio de datos (EDA)**: Visualizaciones y estadísticas para entender los datos.
- **Modelado**: Implementación de varios modelos de clasificación, incluyendo regresión logística, SVM, MLP y XGBoost.
- **Evaluación de modelos**: Informes de clasificación y métricas para evaluar el rendimiento de los modelos.
- **Despliegue**: Scripts para preparar el despliegue de los modelos.

## Estructura del Proyecto

```
.
├── data
│   ├── interim
│   ├── preprocessed
│   └── raw
├── models
├── notebooks
│   ├── eda
│   └── test_models
├── report
│   ├── figures
│   └── other stuff
├── src
│   ├── polarity
│   ├── town
│   └── type
├── LICENSE
├── README.md
└── requirements.txt
```

## Requisitos

- Python >3.12

## Instalación

Para instalar las dependencias del proyecto, ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar el pipeline completo, puedes ejecutar los scripts en el directorio `src/` en el siguiente orden:

1.  Limpiar los datos:
    ```bash
    python -m src.cleaning
    ```
2.  Preprocesar los datos para cada una de las tareas:
    ```bash
    python -m src.polarity.preprocess
    python -m src.town.preprocess
    python -m src.type.preprocess
    ```
3.  Entrenar los modelos:
    ```bash
    python -m src.polarity.train
    python -m src.town.train
    python -m src.type.lr_training
    python -m src.type.mlp_training
    python -m src.type.svm_training
    python -m src.type.xgb_training
    ```

## Contribuciones

Las contribuciones son bienvenidas. Si quieres contribuir al proyecto, por favor, abre un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Para más información, consulta el archivo [LICENSE](LICENSE).
