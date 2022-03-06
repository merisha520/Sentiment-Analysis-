import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


data = pd.read_csv('restaurants reviews.csv')
df = pd.DataFrame(data)

from decision_tree import do_text_cleaning 

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df['cleaned_text']= df['Review'].apply(lambda x: do_text_cleaning(x, stop_words, lem))

vectorizer = TfidfVectorizer(use_idf = True, lowercase = True)

#dependent variable
y = df.Liked 

#converting our reviews from text into a matrix of features 
X = vectorizer.fit_transform(df.cleaned_text)

#splitting the data into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

#using naive bayes multinomialNB classifier 
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

#get the accuracy of the model
print(roc_auc_score(y_test, clf.predict_proba(X_test) [:, 1]))

#accuracy score before cleaning --> 86.64% 
#accuracy score after cleaning --> 85.95% 