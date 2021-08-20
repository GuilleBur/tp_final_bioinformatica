import numpy as np
import tensorflow.keras as keras
from dataaug import SmilesEnumerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from smiles_tokenizer import SmilesTokenizer
import os


smiles_dict = {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33, '@': 34, '.': 35, 'a': 36, 'B': 37, 'e': 38, 'i': 39, '9': 40, '10': 41, '11': 42, '+':43}


# Pasa 
def smiles_to_seq(smiles, seq_length, char_dict=smiles_dict):
    """ Tokenize characters in smiles to integers
    """
    smiles_len = len(smiles)
    seq = []
    keys = char_dict.keys()
    i = 0
    while i < smiles_len:
        # Skip all spaces
        if smiles[i:i + 1] == ' ':
            i = i + 1
        # For 'Cl', 'Br', etc.
        elif smiles[i:i + 2] in keys:
            seq.append(char_dict[smiles[i:i + 2]])
            i = i + 2
        elif smiles[i:i + 1] in keys:
            seq.append(char_dict[smiles[i:i + 1]])
            i = i + 1
        else:
            print(smiles)
            print(smiles[i:i + 1], i)
            raise ValueError('character not found in dict')
    for i in range(seq_length - len(seq)):
      # Padding with '_'
      seq.append(0)
    return seq


from dataaug import SmilesEnumerator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, seq_length, batch_size=128, data_augmentation=True, shuffle=True, smilesTokenizer=False, vocab_path = 'data', vocab_file='vocab.txt'):
        # Agregar aca todas las propiedades necesarias para resolver el problema
        # No olvidar la aumentación de datos
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.on_epoch_end()
        self.sme = SmilesEnumerator()

        self.token = Tokenizer(num_words=len(smiles_dict), char_level=True)
        self.token.fit_on_texts(self.X)
        self.smilesTokenizer = smilesTokenizer
        if self.smilesTokenizer:
            current_dir = os.getcwd()
            vocab_path = os.path.join(current_dir, vocab_path, vocab_file)
            self.token = SmilesTokenizer(vocab_path)            

        
    def get_token(self):
        return self.token

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Implementar
        
        smilesBatch = self.X[self.indexes]
        # Generate data
        if self.data_augmentation:
            smilesBatch = np.array([self.sme.randomize_smiles_aletorio(smile) for smile in smilesBatch])                        
                    
        # Opcion 1. es muy lenta
        #X = np.array([smiles_to_seq(smile, self.seq_length) for smile in smilesBatch])
        
        #Opcion 2. es un poco mas rapida (casi 3 veces mas rapida que Opcion 1)
        if self.smilesTokenizer:
            X_token = np.array([self.token.encode(smile) for smile in smilesBatch])
        else:
            X_token = self.token.texts_to_sequences(smilesBatch)
        X = pad_sequences(X_token, maxlen=self.seq_length, padding='pre', value=0)            
        y = self.y[self.indexes]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)