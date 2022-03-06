import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

data = pd.read_csv('restaurants reviews.csv')
df = pd.DataFrame(data)

from decision_tree import do_text_cleaning 

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df['cleaned_text']= df['Review'].apply(lambda x: do_text_cleaning(x, stop_words, lem))

max_vocab = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(df['cleaned_text'].values)

X = tokenizer.texts_to_sequences(df['cleaned_text'].values)
X = tf.keras.preprocessing.sequence.pad_sequences(X)
#print(X[22])


from tensorflow import keras as k 

model = k.Sequential()
model.add(k.layers.Embedding(20000, 256, input_length=X.shape[1]))
model.add(k.layers.Dropout(0.3))
model.add(k.layers.LSTM(256, return_sequences= True, dropout=0.3, recurrent_dropout=0.2))
model.add(k.layers.LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(k.layers.Dense(1, activation='sigmoid'))
model.add(k.layers.Flatten())


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#@print(model.summary())

#y = pd.get_dummies(df['Review']).values
y = df['Liked']
#print([print(df['Review'][i], y[i]) for i in range(0,5)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 32
epochs = 8

model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)


#results of rnn model ...increasing the number of epochs will increase the percent of accuracy
'''Before text cleaning 
Epoch 1/8
25/25 - 20s - loss: 0.6901 - accuracy: 0.5312 - 20s/epoch - 804ms/step
Epoch 2/8
25/25 - 16s - loss: 0.5187 - accuracy: 0.7550 - 16s/epoch - 643ms/step
Epoch 3/8
25/25 - 16s - loss: 0.3217 - accuracy: 0.8675 - 16s/epoch - 634ms/step
Epoch 4/8
25/25 - 15s - loss: 0.1560 - accuracy: 0.9350 - 15s/epoch - 618ms/step
Epoch 5/8
25/25 - 16s - loss: 0.0751 - accuracy: 0.9775 - 16s/epoch - 641ms/step
Epoch 6/8
25/25 - 16s - loss: 0.0596 - accuracy: 0.9787 - 16s/epoch - 631ms/step
Epoch 7/8
25/25 - 16s - loss: 0.0299 - accuracy: 0.9925 - 16s/epoch - 639ms/step
Epoch 8/8
25/25 - 16s - loss: 0.0203 - accuracy: 0.9912 - 16s/epoch - 643ms/step'''




'''after text cleaning 
Epoch 1/8
25/25 - 18s - loss: 0.6929 - accuracy: 0.5175 - 18s/epoch - 721ms/step
Epoch 2/8
25/25 - 11s - loss: 0.6154 - accuracy: 0.6938 - 11s/epoch - 460ms/step
Epoch 3/8
25/25 - 11s - loss: 0.3647 - accuracy: 0.8388 - 11s/epoch - 442ms/step
Epoch 4/8
25/25 - 11s - loss: 0.2068 - accuracy: 0.9237 - 11s/epoch - 444ms/step
Epoch 5/8
25/25 - 12s - loss: 0.1397 - accuracy: 0.9513 - 12s/epoch - 463ms/step
Epoch 6/8
25/25 - 14s - loss: 0.0768 - accuracy: 0.9737 - 14s/epoch - 576ms/step
Epoch 7/8
25/25 - 15s - loss: 0.0517 - accuracy: 0.9850 - 15s/epoch - 601ms/step
Epoch 8/8
25/25 - 14s - loss: 0.0418 - accuracy: 0.9875 - 14s/epoch - 564ms/step'''