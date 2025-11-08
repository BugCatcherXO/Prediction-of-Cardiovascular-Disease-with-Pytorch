Predicción de Enfermedad Cardíaca con PyTorch (MLP)

Proyecto de clasificación binaria que, a partir de variables clínicas básicas, predice la presencia de HeartDisease. Está implementado en un notebook con pandas / scikit-learn / PyTorch e incluye EDA, split estratificado, preprocessing sin fugas de información, entrenamiento de una MLP y evaluación con ROC y Precision-Recall, además de búsqueda de umbral por F1.

🗂 Estructura del proyecto
cardiovascular_illness/
├─ 01_proyecto.ipynb
└─ dataset/
   └─ heart.csv


🚀 Requisitos e instalación

Python 3.9+

Jupyter / IPython

Paquetes principales: pandas, numpy, matplotlib, scikit-learn, torch (CUDA opcional)

Instalación rápida (Windows / macOS / Linux):

# (opcional) entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install pandas numpy matplotlib scikit-learn torch jupyter


Abrir el notebook:

jupyter notebook 01_proyecto.ipynb

📦 Datos

Archivo: dataset/heart.csv
Target: HeartDisease ∈ {0,1}
Features usadas: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.

El notebook arranca con una sanity check: shape, tipos, valores perdidos y distribución del objetivo.

🔎 EDA (Exploratory Data Analysis)

Incluye:

Histograma de la variable objetivo.

Histogramas y boxplots de variables numéricas.

Matriz de correlaciones (Pearson) para numéricas + target.

En EDA se excluye FastingBS de algunos gráficos por ser binaria, pero sí se utiliza como feature en el modelo.

✂️ Split y Preprocessing

Split estratificado y reproducible (seed=42):

Train 70%, Valid 15%, Test 15%.

Sin fugas: estadísticas de imputación/estandarización se calculan solo en train.

Numéricas → imputación por mediana + z-score (media/STD de train; STD=1 si es 0).

Categóricas (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) → one-hot con columnas fijadas por train (reindex en val/test).

Tratamiento específico: Cholesterol == 0 se considera missing y se imputa con la mediana.

🧠 Modelo

MLP (PyTorch)

Capas: in_features → 128 → 32 → 1

Activación: ReLU

Dropout p=0.325

Pérdida: BCEWithLogitsLoss

Optimizador: Adam lr=1e-3, weight_decay=1e-4

Batch size: 64

Épocas: 250

Device: cuda si hay GPU, si no cpu

Semillas fijadas: numpy/torch (42)

Durante el entrenamiento se reportan por época: train_loss, val_loss, val_acc, val_auroc, val_f1.

📊 Evaluación y selección de umbral

ROC en validación
AUROC ≈ 0.929 (según la figura incluida).

Precision-Recall en validación
AUPRC ≈ 0.942.

Umbral óptimo por F1 (validación)
Se barre el umbral sobre las scores de validación y se selecciona el que maximiza F1.

Reporte en test
Se imprimen métricas a:

Umbral 0.5

Mejor umbral (F1-val)

Para cada caso: accuracy, precision, recall, F1, AUROC y matriz de confusión [[TN, FP],[FN, TP]]