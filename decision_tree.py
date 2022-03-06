import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords, wordnet
#from data_cleaning import pos_tagger, pos_tagging, word_tag, wordnet_tagging, lemmatization, remove_stop_words
from nltk import pos_tag 

data = pd.read_csv('restaurants reviews.csv')
df = pd.DataFrame(data)

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def do_text_cleaning(headline, stop_words, lem):
    cleaned_text = []
    headline = headline.lower().strip().replace("'", "")
    #tokenized_sent = sent_tokenize(headline)

    #tokenize the sentences into individual words
    tokenized_word = word_tokenize(headline)

    for i in tokenized_word:
        if i not in stop_words:
            cleaned_text.append(i)
#print(cleaned_text)
    lemmatized_sent = []
    for word, tag in pos_tag(cleaned_text):#using pos_tag is important to give the context to the lemmatizer
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        if word not in string.punctuation:#remove the punctuations
            lemmatized_sent.append(lem.lemmatize(word, pos))
    return(" ".join(lemmatized_sent))



#add cleaned text as a column in the df
df['cleaned_text']= df['Review'].apply(lambda x: do_text_cleaning(x, stop_words, lem))



#using sklearn to split train and test data
training, testing = train_test_split(df, test_size = 0.33, random_state=42)

train_x = training['cleaned_text']
train_y = training['Liked']

test_x = testing['cleaned_text']
test_y = testing['Liked']


#bag of words method for vectorization
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

#classification models

#decisiontree classifier 
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

from sklearn.metrics import roc_auc_score
test_y_pred = clf_dec.predict(test_x_vectors)
y_accuracy = roc_auc_score(test_y, test_y_pred)
print(y_accuracy)

#accuracy score before cleaning --> 71.91% 
#accuracy score after cleaning --> 71.51%