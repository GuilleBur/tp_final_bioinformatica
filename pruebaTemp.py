

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

"""# Modelo CNN con generador, embedings de smiles y data-augmentation"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datagen import smiles_dict, smiles_to_seq
from dataaug import SmilesEnumerator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datagen import smiles_dict, smiles_to_seq
from dataaug import SmilesEnumerator, SmilesIterator
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential,regularizers
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, Activation, BatchNormalization, Conv1D, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K


sme= SmilesEnumerator()
print(smiles_dict)


df = pd.read_csv('/notebooks/TP-FINAL/bioinformatics_final_project/data/acetylcholinesterase_02_bioactivity_data_preprocessed.csv')

max_len_idx = df['canonical_smiles'].apply(len).argmax()
min_len_idx = df['canonical_smiles'].apply(len).argmin()
max_sequence_len = len(df['canonical_smiles'].iloc[max_len_idx]) + 20

df.head()

X = df['canonical_smiles'].values
y = df['pIC50'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sme = SmilesEnumerator()
sme.fit(df['canonical_smiles'])
sme.leftpad= True
generator = SmilesIterator(X_train, y_train, sme, batch_size=200, shuffle=True, dtype=K.floatx())
generatorTest = SmilesIterator(X_test, y_test, sme, batch_size=200, dtype=K.floatx())

#for i, (X_b, y_b) in enumerate(generator):
#    print(f'{i}\r', end='')



#Xshape = X.reshape(X.shape[0], 1)
#np.apply_along_axis(lambda t: sme.randomize_smiles(t[0]),1,Xshape)

from datagen import DataGenerator

dgen = DataGenerator(X, y, max_sequence_len, batch_size=16, shuffle=False, data_augmentation=True, smilesTokenizer=True, vocab_path='/notebooks/TP-FINAL/bioinformatics_final_project/data' )

len(dgen) * dgen.batch_size

for i, (X_b, y_b) in enumerate(dgen):
    print(f'{i}\r', end='')

