import pandas as pd
import re
import string
import nltk

# upload the data from data_folder
train = pd.read_csv("../data_folder/train.csv")
test  = pd.read_csv("../data_folder/test.csv")

# for the data cleaning, define thes top words and the stemming
stopwords = nltk.corpus.stopwords.words("english")
ps = nltk.stem.PorterStemmer()


# Clean and Stem
def clean_text(text):
    """
    for a given tweet text, this function
        - gets rid of punctuations, non-word characters
        - tokenize the tweet
        - removes the stopwords
        - returns a string of stemmed tokens with no stopwords
    """
    text = text.replace("\n", "|").lower()
    text_noPunct = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split("\W+", text_noPunct)
    tokens_NoStop = [token for token in tokens if token not in stopwords]
    # At this stage we convert the stemmed tokens into a string again
    # the difference between this cleaning function and the one we use 
    # when we only stem is this one returns a string while the former one 
    # returns a list of words
    stemmed_tokens_str = " ".join([ps.stem(token) for token in tokens_NoStop])
        
    return stemmed_tokens_str

# Now we will create a text_Ngram column in our data set
train["text_Ngram"] = train.text.apply(lambda x: clean_text(x))

# Let us split our data into training and validation sets before we start vectorizing
# For simplicity, I will only use the tweets' text bodies for the classification    
X = train.text_Ngram 
y = train.target
    
from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# Vectorize    
from sklearn.feature_extraction.text import CountVectorizer
# initiate the vectorizer with our clean_text function as its analyzer
# and fit it to the train data
cv = CountVectorizer(ngram_range=(1,2)).fit(X_train)

# now vectorize both train and validation data
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_valid)

# Create the sparse matrices for Classification stage
X_train_df = pd.DataFrame(X_train_cv.toarray(), columns= cv.get_feature_names())
X_valid_df = pd.DataFrame(X_test_cv.toarray(), columns= cv.get_feature_names())

# Import Random Forest Classifier and fit it to the train data
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rf_model = rf.fit(X_train_df, y_train)

# use the fitted model for prediction on our validation data
y_pred = rf.predict(X_valid_df)

# calculate the accuracy
accuracy = (y_pred == y_valid).sum()/len(y_valid)

"""
Accuracy may not be a complete measure for our analysis. 
Similar to spam filterings, we may be interested in the 
recall and the precision of our algorithm, too. 

"""

from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(y_valid, y_pred, pos_label = 1, average="binary")

# Now, let's save the precision, recall, and the accuracy of our model
f = open("STEM Ngram RandomForest.txt", "w")

f.write("\t\t\t\tResults of the non-optimized model\n")
f.write("\n---------------------------------------------------------------------------------\n")
f.write("Precision: {} \t Recall: {} \t Accuracy: {}\n".format( round( precision ,3), round(recall ,3), round(accuracy ,3) ))

# ---------------------------------------------------------
#       Optimization of the Hyper Parameters:
# ---------------------------------------------------------
f.write("\n---------------------------------------------------------------------------------\n")
f.write("\n\t\t\t\tOptimization Results\n")
f.write("\n---------------------------------------------------------------------------------\n")
# define a function that takes the n_estimator parameter of the Random Forest Classifier as
# an arguement and returns the precision, recall, fscore, support, and the accuracy of the 
# Random Forest Classifier with that arguement being its parameter.

def rf_optimizer(n_estimators): 
    interim_model = RandomForestClassifier(n_estimators = n_estimators, n_jobs=-1) 
    interim_model.fit(X_train_df, y_train) 
    y_pred = interim_model.predict(X_valid_df) 
    precision, recall, fscore, support = precision_recall_fscore_support(y_valid, y_pred, pos_label = 1, average="binary")
    accuracy = (y_pred == y_valid).sum()/len(y_valid)
    
    result = "n_estimators: {} \t Precision: {} \t Recall: {} \t Accuracy: {}\n".format(n_estimators, round( precision ,3), round(recall ,3), round(accuracy ,3) )
    return result 

for n_estimators in range(20,70,20):
    res = rf_optimizer(n_estimators)
    f.write(res)

f.close()