import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

data = pd.read_csv('restaurants reviews.csv')
df = pd.DataFrame(data)

from decision_tree import do_text_cleaning 

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

#required for GaussianNB to convert sparse matrix into a dense array
train_x_mat = train_x_vectors.todense()
test_x_mat = test_x_vectors.todense()

#GaussianNB classifier
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_mat, train_y)
print("The mean accuracy of GaussianNB model is "+ str((clf_gnb.score(test_x_mat, test_y))))

#accuracy score before text cleaning --> 69.39%  
#accuracy score after text cleaning --> 70.3%
