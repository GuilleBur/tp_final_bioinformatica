# tp_final_bioinformatica
## Trabajo Practico Final de la Diplomatura en Deep Learning del ITBA
### Instalación:
- De manera local: puede utilizar el contenedor docker que ya tiene todas las lib. necesarias --> **docker pull gburgener/deep_learning_itba**
- Con Colab: La mayoria de las notebooks son compatibles para ejecutarlas desde Colab, solo se debera cambiar el path donde se encuentra los dataset y demas cosas.

### Guias de Notebooks extras:

- **007_Data_preprocessing.ipynb:** Se agrega al dataset columnas nuevas:
  - Lipinski descriptors
  - Clasificación de componentes Activos,Componente Activo, Intermedio y Inactivo
  - Fingerprint Descriptors
- **008_Baseline:** Se busca obtener un baseline utilizando Random Forest y otros Modelos de Machine Learning utilizando como inputs los descriptores fingerprints. Se obtiene un Val_R2 entre 0.26 y 0.32 con alto overfitting 
- **009__Gradient_Boosting:** Se aplica el modelo de LightGBM utilizando como inputs los descriptores fingerprints (modelo que mejor métrica dio en prueba LazyRegressor realizada en notebook 008_Baseline). Se realiza una búsqueda de hiper-parámetros, para encontrar los más adecuados al problema. Se obtiene una métrica val_R2=0.35 - 0.36
- **10_deepLearning:** Se aplican técnicas diversas de deep learning utilizados distintos inputs:
    - Inputs= Smiles Vectorizados y con data augmentation (usando lib. Molvecgen)+ Modelo= LSTM → val_R2=0.68  en 200 Epochs
    - Inputs= Smiles Vectorizados y con data augmentation (usando lib. Molvecgen)+ Modelo= Bidirectional-LSTM → val_R2=0.32  en 17 Epochs
    - Inputs= Smiles Vectorizados y con data augmentation (usando lib. Molvecgen)+ Modelo= CNN-Inception → val_R2=0.65  en 16 Epochs
    - Inputs= Finger Prints+ Modelo= MLP → val_R2=0.27  en 17 Epochs
    - Inputs= Finger Prints+ Modelo= MLP → val_R2=0.27  en 17 Epochs
    - Inputs= Smiles tokenizados con data augmentation + Modelo = TextCNN →  val_R2=0.69  en 57 Epochs
    - Inputs= Smiles tokenizados con data augmentation + Modelo = MLP- EMBEDDINGS-ATTENTION→  val_R2=0.25  en 17 Epochs
    - Inputs= Smiles tokenizados con data augmentation + Modelo = MLP- EMBEDDINGS-ATTENTION-CONTEXTCNN→  val_R2=0  en 29 Epochs 
- **11_SmilesTokenizer:** utilización de técnicas modernas de tokenización utilizando [SmilesTokenizer](https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html) de la lib deepchem:
Inputs= Smiles tokenizados con data augmentation + Modelo = TextCNN → val_R2=0.7 en 148 Epochs
- **012-Transfer-Learning:** utilización de modelos tipo BERT pre-entrenados con lenguaje= SMILES. 
    - Inputs= Smiles Canonicos + Modelo = [PubChem10M_SMILES_BPE_450k](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k) val_R2= 0.7 en 10 Epochs
    - Inputs= Smiles Canonicos + Modelo = [ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) val_R2=0.674 en 20 Epochs
