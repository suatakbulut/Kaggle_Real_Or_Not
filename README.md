# Kaggle_Real_Or_Not
Natural Language Processing

Here I try to build a Machine Learning algorithm both as a solution to Kaggle's Real or Not Competition question and as a way to learn about NLP.

The data is downloaded from the competition's webpage.

The data consists of tweets about which we have 5 set of information, namely 
  - id - a unique for each tweet
  - text - the text of the tweet
  - location - the location the tweet was sent from (may be empty)
  - keyword - a keyword (may be empty)
  - target - whether a tweet is about a real disaster (1) or not (0)

I will have multiple models where 

I will use :
  - one of the following approcahes during the celaning process:
      1) Stemming (Porter stemming)
      2) Lemmatizing (WordNet Lemmatizer)
    - one of the following three vecorization method:
      1) Count Vectorization
      2) N-gram
      3) Tfdif Vectorization
  - build and compare a random forest classifier and a gradient boost classifier with default parameters
  - optimize the classifiers' hyper-parameters and then compare them. 

As a result, I will have 12 folders. For instance in the first one I will be using stemming, count vectorization and un-optimazed random forest model. 
