import string
import re
import os

import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold




def tokenizeTweets(tweet):
	#lower tweets and remove punctuation

	lowerTweet = tweet.lower()
	punctTweet = lowerTweet.translate(str.maketrans('','',string.punctuation))

	# Tokenize tweets

	tknzr = TweetTokenizer()
	tokens = tknzr.tokenize(punctTweet)
	
	#set stopwordlist dutch

	stop = set(stopwords.words('dutch'))
	filtered_sentence = []


	# remove links, mentions, stopwords and the hastagsign. 
	for w in tokens:
		if w[:4] != "http":
			if w[0] != "@":
				if w not in stop:
					if w[0] == "#":
						filtered_sentence.append(w[1:])
					else:
						filtered_sentence.append(w)

	return(filtered_sentence)

# function that makes character ngrams (3-, 4-,5- and 6-gram)

def ngramschar(tweet):
	t = tweet.split(' ')
	ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1)
	counts = ngram_vectorizer.fit_transform(t)
	ngramschar = ngram_vectorizer.get_feature_names()

	return(ngramschar)

# function that makes word ngrams (1-, 2-, and 3-gram)

def ngrams(tweet):
	ngram_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                    token_pattern=r'\b\w+\b', min_df=1)
	analyze = ngram_vectorizer.build_analyzer()
	ngramstweet = analyze(tweet)

	return ngramstweet


def readTweets(tweetfile):
	documents =[]
	labels =[]

	# Open the twitter file en split.
	# Tokenize the tweet, then join the tweet for making character ngrams and word ngram. 
	# Append all ngrams to documents list
	# Append positive score en negative score to labels list

	with open (tweetfile, encoding='utf-8') as tweets:
		for line in tweets:
			line = line.rstrip()

			#Check if data is annoteted
			annotetedTest = line.split("\t")
			if len(annotetedTest) == 9:
				(id, user, text, words, hastags, date, place, posScore, negScore) = line.split("\t")
				score = posScore + negScore
				labels.append(score)
			else:
				(id, user, text, words, hastags, date, place) = line.split("\t")


			tokens = tokenizeTweets(words)
			tokendtweet = " ".join(tokens)
			ngramlist = ngrams(tokendtweet) + ngramschar(tokendtweet)
			documents.append(ngramlist)
			
	

	# retrun list with tokenized ngram words and characters 
	# return list with labels
	return documents, labels

# a dummy function that just returns its input
def identity(x):
	return x

def getgoal():
	print("Train = Train develepment set of annoteted 850 tweets with 5 fold cross validation and returns the evaluation report, and  5 fold average precision, recall and f1-score.")
	print("Test = Test the classifier with testset of 150 tweets return evaluation report")
	print("Gemeenteresults = Returns the results of the research for every gemeente.")

	g = input("What would you like to do? Typ in : train, test, gemeenteresults -->   ")
	goal = g.lower()

	return goal

def crosstrain(X,Y):
	psum = 0
	fsum = 0
	rsum = 0

	kf = KFold(n_splits=5)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Yguess = classifier(X_train,X_test,Y_train)

		p,r,f,s = precision_recall_fscore_support(Y_test,Yguess, average= 'weighted')

		evaluate(Y_test,Yguess)

		psum = psum + p
		fsum = fsum + f
		rsum = rsum + r

	# Calculate average over 5 folds
		
	favg = fsum / 5
	pavg = psum / 5
	ravg = rsum / 5

	print("====== Average over 5 folds scores ======")

	# print average scores over 5 folds

	print("Favg: ", favg)
	print("Pavg: ", pavg)
	print("Ravg: ", ravg)



def evaluate(Y_test,Yguess):

	#print report and accuracy score

	print(metrics.classification_report(Y_test, Yguess))
	
	print("accurancy: ",accuracy_score(Y_test, Yguess))

	

def classifier(X_train,X_test,Y_train):
	# Use the TF-IDF vectorizer
	tfidf = True

	# Dummy function as tokenizer and preprocessor
	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
	else:
		vec = CountVectorizer(preprocessor = identity, tokenizer = identity)


	# combine the vectorizer with linear suport vector machines
	classifier = Pipeline([('vec', vec),("clf", svm.LinearSVC())])

	# combine the vectorizer with Naive Bayes classifier (Remove the # and place # in front of SVM classfier)
	#classifier = Pipeline( [('vec', vec), ('cls', MultinomialNB())] )

	# train the classifier
	classifier.fit(X_train, Y_train)

	# test
	Yguess = classifier.predict(X_test)

	return (Yguess)


def main():

	goal = getgoal()

	if goal == "train":
		X,Y = np.array(readTweets("trainset.txt"))
		crosstrain(X,Y)
	elif goal == "test":
		X_train,Y_train = readTweets("trainset.txt")
		X_test,Y_test = readTweets("testset.txt")
		Yguess = classifier(X_train,X_test,Y_train)
		evaluate(Y_test,Yguess)
	elif goal == "gemeenteresults":
		gemeentefile = open('gemeenteresults.txt', 'w')
		gemeentefile.write('{0:45} {1:10} {2:10} {3:10} {4:10} {5}'.format('XX','neutral','positive','negative','both', '\n'))
		path = 'gemeenten/'
		X_train,Y_train = readTweets("trainset.txt")
		for filename in os.listdir(path):
			X_test, _ = readTweets('gemeenten/' + filename) 
			Yguess = classifier(X_train,X_test,Y_train)
			
			neu = 0
			pos = 0
			neg = 0
			teg = 0

			for score in Yguess:
				if score == '00':
					neu = neu + 1
				elif score == '10':
					pos = pos + 1
				elif score == '01':
					neg = neg + 1
				elif score == '11':
					teg = teg + 1
			
			gemeentefile.write('{0:40} {1:10} {2:10} {3:10} {4:10} {5}'.format(filename,neu,pos,neg,teg, '\n'))
		gemeentefile.close()
		



main()
