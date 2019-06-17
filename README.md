# Lyrics-Based Music Genre Classification Using a CNN-LSTM Network
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
![History](data:image/png;base64,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23ncs74Pj7+f1vcTeO7/g4xkTO//wPM+V6tTTiOPH7NnnVyf/Hn49ttveffdd+nbty979uzht99+Y+LEiU8dv+dt39vbm06dOrFixQri4uJYv359pm99CpET5Pwv5/+Myqnzf0a29fj+arVafvrpJ/bv30/NmjXZtGkTZcuW5cCBAwDUqVOHv//+m7lz56JWqxkxYgQ+Pj7ExMRkKkN+I8V0Ljt58iRJSUksWrSIhg0bUr58ee7evWuRLOXKlUOr1XL8+PFU7584ceK5y2XXPlSuXJnz58+b+9BByhWF+Pj4dJft168fZ86c4fz58wQEBKQaAHHhwgXCwsKYO3cuTZo0oUKFCpmez7Ny5cqYTKZUxyI+Pj7V4JJ//vmHP//8k6lTp/Laa69RqVIl1Gp1qj5/Go0GjUZDcnJyuts7evRoqj59p06dIj4+nsqVK2cq++MykrF27drcuXOHP/74I8111K5dm99///2ZV8G8vLwAuHPnjvm9Jwc4Psvhw4epV68eo0aNonbt2pQtW5YbN26YP1epVNSsWZN9+/Y9dz3Dhg1jy5YtLFu2DJPJRPfu3TO0fSFyk5z//yPn/9Tby4nz/5O8vLwoUKAAP//8c6r3Dx8+TLly5dDpdEDKebd+/fpMnTqVo0ePUrdu3VTjUpydnenSpQtffvklx44d4/z58+Y/WF5WUkznsnLlymEymfjss8+4ceMG33//PR9//LFFsri7uzNw4EAmTpzInj17uHLlChMmTODGjRvPvVqRXfvQv39/dDod/fr1448//uDo0aMMHz48QwPb6tSpQ6VKlejfvz/R0dGpiqdSpUqh0+lYvHgxf/31F/v372fChAmZylalShVatmzJsGHDOHz4MBcvXmTAgAGpTvReXl64ubmxbNkyrl27xtGjR+nbt695xDaknJRKlCjBTz/9REhIyDNvx40ePZp79+4xePBgLl68yM8//8zAgQPx9fWlTp06mcr+uIxkbN26NXXr1qVLly7s3LmTGzdu8Msvv7B69WoA8yweHTp04KeffuLGjRscOHDA/MCDihUrUrhwYaZPn86VK1f4+eefee+99zKUr3z58pw9e5Zdu3Zx/fp1FixYwM6dO1O1mT59Olu2bGHChAn88ccfXL58mVWrVqUand+iRQuKFSvGxIkT6dWrl8y3K6ySnP//I+f//+TU+T8tkyZN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%0A)

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
[[0.00443605 0.11274784 0.00268884 0.01278187 0.02782017 0.00551375
  0.01185072 0.12756461 0.01382886 0.29636037 0.0140664  0.3703405 ]] Folk
  
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
![wordcloud]()
