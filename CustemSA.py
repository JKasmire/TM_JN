# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:45:27 2020

Creating a custom sentiment analyser/classifier with textblob

@author: mzyssjkc
"""


import os                         # os is a module for navigating your machine (e.g., file directories).
import nltk                       # nltk stands for natural language tool kit and is useful for text-mining. 
import csv                        # csv is for importing and working with csv files
import statistics
import textblob
from textblob import TextBlob 
from textblob.classifiers import NaiveBayesClassifier


#### First steps with sample docs
Doc1 = TextBlob("Textblob is just super. I love it!")   # A few simple documents to analyse in string format
Doc2 = TextBlob("Cabbages are the worst. Say no to cabbages!")  
Doc3 = TextBlob("Paris is the capital of France. ")   
print("...")
print(Doc1.sentiment)
print(Doc2.sentiment)
print(Doc3.sentiment)
print(Doc1.sentiment.polarity)
print(Doc1.sentiment.subjectivity)

#### Import and format more interesting set of docs, use basic sentiment analyser
with open('./Sentiment_Analysis/test_min.csv', newline='') as f:              # 2, a csv of scored product reviews
    reader = csv.reader(f)
    Doc_set = list(reader)

Doc_set_corrected = []
for pair in Doc_set:
    x = []
    x.append(pair[0])
    if (pair[0][1] >= '4'):
        x.append(1)
    elif (pair[0][1] == '2'):
        x.append(0)
    else:
        x.append(-1)
    Doc_set_corrected.append(x)

Doc_set_scored = []
for pair in Doc_set_corrected:
    x=[]
    Doc_set_scored.append([pair[0], pair[1], 
                           TextBlob(pair[0]).sentiment.polarity])
    Doc_set_scored.append

print(Doc_set_scored[:100])

#### Calculate accuracy score
Doc_set_accuracy =[]

for item in Doc_set_scored:
    x = item[1]
    y = item[2]
    Doc_set_accuracy.append(abs (x-y))

print(statistics.mean(Doc_set_accuracy)) ##not a great result

#### Now, create and apply custom analyser trained and tested on full data set
install autocorrect
from autocorrect import Speller
check = Speller(lang='en')
import re
from nltk import word_tokenize
English_punctuation = "!\"#$%&()*+,./:;<=>?@[\]--'^_`{|}~“”"      
table_punctuation = str.maketrans('','', English_punctuation)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



with open('./Sentiment_Analysis/training_min.csv', newline='') as f:              # 2, a csv of scored product reviews
    reader = csv.reader(f)
    train_1 = list(reader)
    
with open('./Sentiment_Analysis/test_min.csv', newline='') as f:              # 2, a csv of scored product reviews
    reader = csv.reader(f)
    test_1 = list(reader)
    
quick_test = train_1[:10]
            
## WORKING SO FAR            
train_2 = []
for thingy in train_1:
    no_url = re.sub(r'^https?:\/\/.*[\r\n]*', '', thingy[0], flags=re.MULTILINE)
    x0 = word_tokenize(no_url)
    x1 = [w.translate(table_punctuation) for w in x0]  
    x2 = [word.lower() for word in x1]
    x3 = [check(word) for word in x2]
    x4 = list(filter(None, x3))  
    x5 = ' '.join(x4)  
    train_2.append([x5, thingy[1]])

    
test_2 = []
for thingy in test_1:
    no_url = re.sub(r'^https?:\/\/.*[\r\n]*', '', thingy[0], flags=re.MULTILINE)
    x0 = word_tokenize(no_url)
    x1 = [w.translate(table_punctuation) for w in x0]  
    x2 = [word.lower() for word in x1]
    x3 = [check(word) for word in x2]
    x4 = list(filter(None, x3))  
    x5 = ' '.join(x4)  
    test_2.append([x5, thingy[1]])

cl_custom = NaiveBayesClassifier(train_1)

cl_2 = NaiveBayesClassifier(train_2)

Doc_set_train_1 = []
for thingy in train_1[:10]:
    no_url = re.sub(r'^https?:\/\/.*[\r\n]*', '', thingy[0][0], flags=re.MULTILINE)
    x0 = word_tokenize(no_url)
    x1 = [w.translate(table_punctuation) for w in x0]  
    x2 = [word.lower() for word in x1]
    x3 = [check(word) for word in x2]
    x4 = list(filter(None, x3))  
    x5 = ' '.join(x4)  
    Doc_set_train_1.append([x5, thingy[0][1]])

cl_Doc_set = NaiveBayesClassifier(Doc_set_train_1)

Doc_set_train_1

cl_Doc_set.classify("This is an amazing library!")
prob_dist = cl.prob_classify("This one's a doozy.")


