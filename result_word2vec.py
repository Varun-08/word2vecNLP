import numpy as np
import pandas as pd

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0,\
                    delimiter="\t", quoting=3)

test = pd.read_csv("testData.tsv",header=0,\
                    delimiter="\t", quoting=3)

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,\
                    delimiter="\t", quoting=3)

# Verifying the number of reviews
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews \n"%(train["review"].size, test["review"].size, unlabeled_train["review"].size))

# Removing stop words and numbers might not be useful. 
# Therefore  it is optional function

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words.
    # 1. Remove html tags
    review_text = BeautifulSoup(review).get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
       stops=set(stopwords.words("english"))     
       words = [w for w in words if not w in stops]
    # 5. Returning a list of words
    return(words)

# word2vec expects a list of lists.
# Using punkt tokenizer for beter splitting of a paragraph into sentences.

import nltk.data
#nltk.download('popular')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_sentences(review, tokenizer, remove_stopwords= False):
    # Function to split a review into sentences
    # 1. Using nltk tokenizer
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence, remove_stopwords))

    # This returns the list of lists
    return sentences

# Applying this function to prepare our data for word2vec

sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_sentences(review, tokenizer)

# Importing the built in logging module
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Setting values for the various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(sentences,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

# doesnt_match function will get which word is most dissimilar from the others
#checking few tests
model.wv.doesnt_match("man woman dog child kitchen".split())

model.wv.doesnt_match("france england germany berlin".split())

model.wv.most_similar("man")

model.wv.most_similar("queen")

model.wv.most_similar("awful")


# Loading the model
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

model.wv.syn0.shape # Gives the total number of words in vocabulary

def featureVecMethod(words, model, num_features):
    # Function to average all word vectors in a paragraph
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word a list to a set for speed containing the names of words in model's vocabulary
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Calculating the average feature vector
    counter = 0
    
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs
    
# Calculating average feature vector for training and test sets

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))
    
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    
print("Creating average feature vecs for test reviews ")
    
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_wordlist(review,remove_stopwords=True))
    
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    
# fitting a random forest to a training data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
    
print("Fitting random forest to training data....")    
forest = forest.fit(trainDataVecs, train["sentiment"])
    
result = forest.predict(testDataVecs)
    
    
    



