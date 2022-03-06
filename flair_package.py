#import necessary modules for flair package 
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords, wordnet


#load the text classifier
sia = TextClassifier.load('en-sentiment')

#method to predict and return the sentiment 
def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"

#read the data and put it in a dataframe
data = pd.read_csv("restaurants reviews.csv")
df = pd.DataFrame(data)

'''before cleaning the dataset'''
#add a sentiment column in the dataframe after applying the above method 
df["sentiment"] = df['Review'].apply(flair_prediction)

#since we know that the dataset contains 500 positive and 500 negative reviews, we calculate the accuracy by dividing the predcitions by 500 each  
negatives = df[(df['Liked']==0) & (df['sentiment']=='neg')].count()
true_negatives = negatives[1] #number of negative scores correctly predicted 
positives = df[(df['Liked']==1) & (df['sentiment']=='pos')].count()
true_positives = positives[1] #number of positive scores correctly predicted

negative_accuracy = (true_negatives/500)*100 
positive_accuracy = (true_positives/500)*100

print(f'The accuracy of negative reviews is {negative_accuracy} % and positive reviews is {positive_accuracy:.2f} %')



'''After Cleaning the dataset'''
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#custom pos tagging function
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

#function to pos tag each sentence
def pos_tagging(text):
    return [nltk.pos_tag(nltk.word_tokenize(text))]

pos_tagged_review = df['Review'].apply(pos_tagging)

#extra function to get reviews in desired state
def word_tag(text):
    for i in text: 
        return i
word_tagged = pos_tagged_review.apply(word_tag)

#tagging using custom function
def wordnet_tagging(pos_tagged):
    return list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

wordnet_tagged = word_tagged.apply(wordnet_tagging)

#lemmatizing using tags
def lemmatization(word_tagged1):
    lemmatized_sentence = []
    for word, tag in word_tagged1:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    return lemmatized_sentence

lemmatized_sentence = wordnet_tagged.apply(lemmatization)

def remove_stop_words(text):
    cleaned_sentence = []
    tokens = word_tokenize(text)
    for word in tokens:
        if word not in stop_words:
            cleaned_sentence.append(word)
    return " ".join(cleaned_sentence)

cleaned_review = lemmatized_sentence.apply(remove_stop_words)

#add a sentiment column in the dataframe after applying the above method 
df["sentiment"] = cleaned_review.apply(flair_prediction)

#since we know that the dataset contains 500 positive and 500 negative reviews, we calculate the accuracy by dividing the predcitions by 500 each  
negatives = df[(df['Liked']==0) & (df['sentiment']=='neg')].count()
true_negatives = negatives[1] #number of negative scores correctly predicted 
positives = df[(df['Liked']==1) & (df['sentiment']=='pos')].count()
true_positives = positives[1] #number of positive scores correctly predicted

negative_accuracy = (true_negatives/500)*100 
positive_accuracy = (true_positives/500)*100

print(f'The accuracy of negative reviews is {negative_accuracy} % and positive reviews is {positive_accuracy:.2f} %')

'''
 Before cleaning --> The accuracy of negative reviews is 92.2 % and positive reviews is 93.60 %
After Cleaning --> The accuracy of negative reviews is 80.0 % and positive reviews is 91.60 %'''