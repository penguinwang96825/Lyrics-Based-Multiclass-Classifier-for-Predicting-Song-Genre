# Lyrics-Based Music Genre Classifier Using a CNN-LSTM Network
Classify the genre (Rock, Pop, Hip-Hop, Not Available, Metal, Other, Country, Jazz, Electronic, R&B, Indie, Folk) of the song by its lyrics. Music genre classification, especially using lyrics alone, remains a challenging topic in Music Information Retrieval. The accuracy about the model is not good enough considering there is too much factors that effect song's genre instead of using song's lyrics only. 
In particular, we will go through some baselines and full Deep Learning pipeline, from:

* Exploring and Processing the Data
* Building and Training our Model
  1. Baselines: Naive Bayes Classifier, Linear Support Vector Machine, and Logistic Regression
  2. Neural Network: CNN-LSTM
* Define a Plot Function to Visualize Loss and Accuracy

## Pre-requisites
### Create a conda environment
1. Install python version 3.7.3: https://www.python.org/downloads/release/python-373/
2. Install Anaconda 3 for win10: https://www.anaconda.com/distribution/#download-section
3. Create a virtual environment and change the PYTHONPATH of an ipython kernel: 
* `conda update conda`
* `conda create --name my_env python=3.7.3`
* `conda activate my_env`
* `conda install ipykernel -y`
* `python -m ipykernel install --user --name my_env --display-name "My Env"`
4. GPU support software requirements:
* NVIDIA® GPU drivers: https://www.nvidia.com/Download/index.aspx?lang=en-us
* CUDA® Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
* cuDNN SDK: https://developer.nvidia.com/cudnn
* (Optional) TensorRT 5.0: https://developer.nvidia.com/tensorrt
5. Windows setup
* Add the CUDA, CUPTI, and cuDNN installation directories to the `%PATH%` environmental variable. For example, if the CUDA Toolkit is installed to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0` and cuDNN to `C:\tools\cuda`, update your `%PATH%` to match:
```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```
* Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH

## Check whether GPU is working
1. Choose which GPU you want to use
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
2. Check what all devices are used by tensorflow
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
3. Check using Keras backend function
```python
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```
4. How to get the nvidia driver version from the command line?
* Open the terminal and type in `cd C:\Program Files\NVIDIA Corporation\NVSMI` and input `nvidia-smi`
```console
C:\Users\YangWang>cd C:\Program Files\NVIDIA Corporation\NVSMI

C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi
Sun Jun 16 03:32:53 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.86       Driver Version: 430.86       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060   WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P8     6W /  N/A |     85MiB /  3072MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Exploring and Processing the Data
### Import the data
First install and library our needed package.
```python
import pandas as pd
import codecs
import re
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
%matplotlib inline
```
The method to disable the python warning.
```python
import sys
import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning) 
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
```
Secondly, open our file and read into a pandas dataframe.
```python
with codecs.open("lyrics.csv", "r", "Shift-JIS", "ignore") as file:
    df = pd.read_csv(file, delimiter=",")
    
print(len(df))
df.head()
```
### Pre-processing
```python
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r'(', '') 
    df[text_field] = df[text_field].str.replace(r')', '')
    df[text_field] = df[text_field].str.replace(r',', '')
    df[text_field] = df[text_field].str.replace(r'_', '')
    df[text_field] = df[text_field].str.replace(r"'", "")
    df[text_field] = df[text_field].str.replace('\n','')
    df[text_field] = df[text_field].str.replace('\t','')
    df[text_field] = df[text_field].str.replace(r"^[a-z]+\[0-9]\d+", "")
    df[text_field] = df[text_field].str.replace(r"^[0-9]{1,2,3,4,5}?$", "")
    return df
    
df = standardize_text(df, 'lyrics')
df = df[pd.notnull(df['lyrics'])]
```
Split the data into train set and test set (80% for traing and 20% for testing)
```python
lyrics = df['lyrics'].apply(str).values
genre = df['genre'].apply(str).values

lyrics_train, lyrics_test, genre_train, genre_test = train_test_split(lyrics, genre, test_size = 0.20, random_state = 1000)
```
Finally, visualizing the data distribution
```python
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline = False, world_readable = True)

plt.figure(figsize = (10,4))
# print(df.genre.value_counts())
df.genre.value_counts().sort_values(ascending = False).iplot(kind = 'bar', title = 'Number of songs in each genre');
```
## Baselines
### Naive Bayes Classifier
In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

my_tags = ["Rock", "Pop", "Hip-Hop", "Not Available", "Metal", "Other", 
           "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk"]
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(lyrics_train, genre_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(lyrics_test)

print('accuracy %s' % accuracy_score(y_pred, genre_test))
print(classification_report(genre_test, y_pred,target_names = my_tags))
```
```shell
accuracy 0.42821503601440575
               precision    recall  f1-score   support

         Rock       0.00      0.00      0.00      2852
          Pop       0.00      0.00      0.00      1597
      Hip-Hop       0.00      0.00      0.00       456
Not Available       0.93      0.07      0.13      4945
        Metal       0.00      0.00      0.00       627
        Other       0.00      0.00      0.00      1582
      Country       1.00      0.00      0.01      4780
         Jazz       0.54      0.04      0.08      4768
   Electronic       0.00      0.00      0.00       975
          R&B       0.39      0.10      0.16      8168
        Indie       0.00      0.00      0.00       651
         Folk       0.43      0.98      0.59     21911

     accuracy                           0.43     53312
    macro avg       0.27      0.10      0.08     53312
 weighted avg       0.46      0.43      0.29     53312
 ```
 
### Linear Support Vector Machine
In machine learning, support-vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
```python
from sklearn.linear_model import SGDClassifier

my_tags = ["Rock", "Pop", "Hip-Hop", "Not Available", "Metal", "Other", 
           "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk"]
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(lyrics_train, genre_train)

y_pred = sgd.predict(lyrics_test)

print('accuracy %s' % accuracy_score(y_pred, genre_test))
print(classification_report(genre_test, y_pred,target_names=my_tags))
```
```shell
accuracy 0.5024572328931572
               precision    recall  f1-score   support

         Rock       0.40      0.11      0.18      2852
          Pop       0.27      0.03      0.05      1597
      Hip-Hop       0.35      0.04      0.06       456
Not Available       0.54      0.81      0.65      4945
        Metal       0.03      0.00      0.01       627
        Other       0.40      0.11      0.17      1582
      Country       0.63      0.35      0.45      4780
         Jazz       0.45      0.10      0.17      4768
   Electronic       0.25      0.05      0.08       975
          R&B       0.43      0.15      0.23      8168
        Indie       0.03      0.04      0.03       651
         Folk       0.51      0.85      0.64     21911

     accuracy                           0.50     53312
    macro avg       0.36      0.22      0.23     53312
 weighted avg       0.47      0.50      0.43     53312
 ```
 
 ### Logistic Regression
In statistics, the logistic model (or logit model) is a widely used statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist.
```python
from sklearn.linear_model import LogisticRegression

my_tags = ["Rock", "Pop", "Hip-Hop", "Not Available", "Metal", "Other", 
           "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk"]
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(lyrics_train, genre_train)

y_pred = logreg.predict(lyrics_test)

print('accuracy %s' % accuracy_score(y_pred, genre_test))
print(classification_report(genre_test, y_pred,target_names = my_tags))
```
```shell
accuracy 0.5567226890756303
               precision    recall  f1-score   support

         Rock       0.48      0.38      0.43      2852
          Pop       0.49      0.17      0.25      1597
      Hip-Hop       0.49      0.14      0.21       456
Not Available       0.80      0.76      0.78      4945
        Metal       0.32      0.06      0.10       627
        Other       0.45      0.29      0.35      1582
      Country       0.64      0.49      0.56      4780
         Jazz       0.34      0.23      0.27      4768
   Electronic       0.30      0.12      0.17       975
          R&B       0.44      0.41      0.42      8168
        Indie       0.44      0.16      0.24       651
         Folk       0.58      0.78      0.66     21911

     accuracy                           0.56     53312
    macro avg       0.48      0.33      0.37     53312
 weighted avg       0.54      0.56      0.53     53312
 ```
 
 ### Define a heat map for confusion matrix
 ```python
 def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize = (10,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
 ```
 
 ```python
 plot_confusion_matrix(cm           = confusion_matrix(genre_test, y_pred), 
                      normalize    = True,
                      target_names = my_tags,
                      title        = "Confusion Matrix")
 ```
 
## Neural Networks
1. Tokenize the lyrics
```python
from keras.preprocessing.text import Tokenizer

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words = MAX_NB_WORDS, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(lyrics)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
```
2. Sentence padding
```python
from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(df['lyrics'].apply(str).values)
X = pad_sequences(X, maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
```
```python
Y = pd.get_dummies(df['genre']).values
print('Shape of label tensor:', Y.shape)
```
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
```
3. Import the packages from Keras API for neural networks.
```python
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout1D
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.engine import Input
from keras.optimizers import SGD
from keras.preprocessing import text,sequence
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
```
4. Build the model and train the model.
```python
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Convolution1D(filters = 150,
                        kernel_size = 3,
                        border_mode = 'same',
                        activation = 'relu',
                        subsample_length = 1))
model.add(MaxPooling1D(pool_size = 2))
model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(12, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    validation_split = 0.1, 
                    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3, min_delta = 0.0001)])
```
5. Evaluate the accuracy on test set.
```python
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
```
6. Plot the history of accuracy and loss.
```python
plt.style.use('ggplot')

def plot_history_ggplot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
plot_history_ggplot(history)
```
![History](https://github.com/penguinwang96825/Lyrics-based-multi-classifier-for-predicting-song-genre/blob/master/history.png)

## Predict a New Song's Genre Based on its Lyrics
I put in "Jonas Blue – What I Like About You" into this CNN-LSTM neural network.
```python
import numpy as np

new_lyrics = [   
    "All my life I've been a good girl\
    Tryna do what's right, never told no lies\
    Then you came around and suddenly my world\
    Turned upside down, now there's no way out\
    I tried to fight against it, shut out what all my friends said\
    Can't get you out of my head, oh-woah-woah-woah-woah\
    I keep letting you in, though I know it's not a good thing\
    I got you under my skin, oh-woah-woah-woah\
    You're so outta line\
    You make me bad and I don't know why\
    But that's what I like about ya\
    Yeah, that's what I like about ya\
    I'm out of my mind\
    You got me runnin' all the red lights\
    But that's what I like about ya\
    Yeah, that's what I like about ya\
    That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya\
    (Ooh, make me bad and I don't know)\
    (Ooh, make me bad and I don't know)\
    (Ooh) That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya\
    I don't care if my daddy don't think we're the perfect pair\
    It's my affair\
    Yeah, I don't know why my momma worries when I'm out all night\
    She thinks I'm nine, uh-uh\
    I tried to fight against it, shut out what all my friends said\
    Can't get you out of my head, oh-woah-woah-woah-woah\
    I keep letting you in, though I know it's not a good thing\
    I got you under my skin, oh-woah-woah-woah\
    You're so outta line\
    You make me bad and I don't know why\
    But that's what I like about ya\
    Yeah, that's what I like about ya\
    I'm out of my mind\
    You got me runnin' all the red lights\
    But that's what I like about ya\
    Yeah, that's what I like about ya\
    That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya\
    (Ooh, make me bad and I don't know)\
    (Ooh, make me bad and I don't know)\
    (Ooh) That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya\
    I'm a fool for you\
    I tried to fight against it, shut out what all my friends said\
    Can't get you out of my head (Out of my head, out of my head)\
    I keep letting you in, though I know it's not a good thing\
    I got you under my skin, oh-woah-woah-woah\
    You're so out of line (You're so outta line)\
    You make me bad and I don't know why (I don't know why)\
    But that's what I like about ya\
    Yeah, that's what I like about ya (Yeah, that's what I like)\
    I'm out of my mind\
    You got me runnin' all the red lights (You got me runnin')\
    But that's what I like about ya\
    Yeah, that's what I like about ya\
    That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya\
    (Ooh, make me bad and I don't know)\
    (Ooh, make me bad and I don't know)\
    (Ooh) That's what I like about ya\
    (Ooh) Yeah, that's what I like about ya"]
seq = tokenizer.texts_to_sequences(new_lyrics)
padded = pad_sequences(seq, maxlen = MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
my_tags = ["Rock", "Pop", "Hip-Hop", "Not Available", "Metal", "Other", 
           "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk"]

print(pred, my_tags[np.argmax(pred)])
```
```shell
[[0.00443605 0.11274784 0.00268884 0.01278187 0.02782017 0.00551375
  0.01185072 0.12756461 0.01382886 0.29636037 0.0140664  0.3703405 ]] Folk
```
  
```python
from cleantext import clean
import unidecode

def plotSoftmaxProb(new_lyrics, title):
    clean("new_lyrics",
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    lang="en")
    # Convert texts into sequences and pad sentences 0
    seq = tokenizer.texts_to_sequences(new_lyrics)
    padded = pad_sequences(seq, maxlen = MAX_SEQUENCE_LENGTH)
    # Predict the genre of the song
    pred = model.predict(padded)
    # Make a flat list out of list of lists
    l = pred.tolist()
    flat_list = [item for sublist in l for item in sublist]
    # Put the probabilties from softmax layer into dataframe
    my_tags = ["Rock", "Pop", "Hip-Hop", "Not Available", "Metal", "Other", 
               "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk"]
    data = {'genre' : pd.Series(flat_list, index = my_tags)}
    pred_df = pd.DataFrame(data)
    # Illustrate the probabilities
    pred_df.iplot(kind = 'bar', title = title);
```
Plot "Jonas Blue – What I Like About You" probabilities through Softmax activation function.
```python
plotSoftmaxProb(new_lyrics, title = 'Probability of genres for the testing lyric')
```

## Word Cloud
WordCloud for Rock genre song.
```python
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

df_Rock = df[genre == "Rock"]

standardize_text(df_Rock, "lyrics")

df_Rock = df_Rock[pd.notnull(df_Rock['lyrics'])]
df_Rock.head()
```
```python
text = df_Rock.lyrics.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'white',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(figsize = (40, 30), facecolor = 'white', edgecolor = 'white')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
```
![wordcloud](https://github.com/penguinwang96825/Lyrics-based-multi-classifier-for-predicting-song-genre/blob/master/RockWordCloud.png)
