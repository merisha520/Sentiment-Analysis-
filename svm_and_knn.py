import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag 
from nltk.corpus import stopwords, wordnet
import string
import nltk
from decision_tree import do_text_cleaning
from nltk.stem import WordNetLemmatizer


data = pd.read_csv('restaurants reviews.csv')
df = pd.DataFrame(data)

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df['cleaned_text']= df['Review'].apply(lambda x: do_text_cleaning(x, stop_words, lem))

# '''using sklearn to split train and test data '''
training, testing = train_test_split(df, test_size = 0.33, random_state=42)

train_x = training['cleaned_text']
train_y = training['Liked']

test_x = testing['cleaned_text']
test_y = testing['Liked']


# '''bag of words method for vectorization'''
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

'''classification models'''

#svm classifier
from sklearn.svm import SVC
clf_svm = SVC()
clf_svm.fit(train_x_vectors, train_y)

from sklearn.metrics import roc_auc_score
test_y_pred = clf_svm.predict(test_x_vectors)
y_accuracy = roc_auc_score(test_y, test_y_pred)
print("Accuracy of svm classifier is " + str(y_accuracy))

#accuracy score before cleaning --> 82%
#accuracy score after cleaning --> 77.263%

#KNearestNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier as KNN
clf_svm = KNN()
clf_svm.fit(train_x_vectors, train_y)

from sklearn.metrics import roc_auc_score
test_y_pred = clf_svm.predict(test_x_vectors)
y_accuracy = roc_auc_score(test_y, test_y_pred)
print("Accuracy of KNN classifier is " + str(y_accuracy))

#accuracy score before cleaning--> 74.52%
#accuracy score after cleaning --> 73.48%

