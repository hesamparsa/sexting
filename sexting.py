import pandas as pd 
import glob
import re 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# read file into Pandas using a relative path

#path1 = 'C:/Users/Hesam/Dropbox/job application/Data_Science/Insight/sexting/SMSSpamCollection.txt'
#path2 = 'C:/Users/Hesam/Dropbox/job application/Data_Science/Insight/sexting/sextlist.txt'
path1='/Users/hesamparsa/Dropbox/job application/Data_Science/Insight/sexting/SMSSpamCollection.txt'
path2='/Users/hesamparsa/Dropbox/job application/Data_Science/Insight/sexting/sextlist.txt'


sms1 = pd.read_table(path1, header=None, names=['label', 'message'])  # UCI machine learning Repository
sms2 = pd.read_table(path2, header=None, names=['label', 'message'])  # My online search for sexts
sms = sms1.append(sms2,  ignore_index=True)
sms.message.replace(to_replace='[\d\.\?]', value='',inplace= True, regex=True)
sms=sms[(sms.label =='ham')  | (sms.label =='spam') | (sms.label =='sexpam')] 

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1, 'sexpam':2})
#sms.fillna(2, inplace=True)

# examine the class distribution
sms.label_num.value_counts()

# required way to define X and y for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
#print(X.shape)
#print(y.shape)

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#print(X_train.shape)
#print(X_test.shape)


# ## Part 4: Vectorizing the SMS data

def tokenize_test(vect):
    
    # create document-term matrices using the vectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    # print the number of features that were generated
    print('Features: ', X_train_dtm.shape[1])
    
    # use Multinomial Naive Bayes to predict the star rating
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    
    # print the accuracy of its predictions
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
    print ('Confusion matrix:', metrics.confusion_matrix(y_test, y_pred_class))

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), min_df=1, max_df=0.5)  # include 1-grams and 2-grams
#print(sorted(vect.get_stop_words()))
#print(vect.stop_words_)
tokenize_test(vect)

# alternative: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)

# examine the last 50 features
#print(vect.get_feature_names())

#print('Features: ', X_train_dtm.shape[1])

# calculate null accuracy
print('null accuracy:', y_test.value_counts().head(1) / y_test.shape)

# ## Part 2: Term Frequency-Inverse Document Frequency (TF-IDF)

# TfidfVectorizer (CountVectorizer + TfidfTransformer)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()  # this parameter will increase the norm=None
tokenize_test(vect)


# TF-IDF (scikit-learn's default implementation)
pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names())

# ## Part 4: Sentiment analysis using TextBlob

# Sentiment is the "attitude" of the speaker:
# 
# - **polarity** ranges from -1 (most negative) to 1 (most positive)
# - **subjectivity** ranges from 0 (very objective) to 1 (very subjective)
# save it as a TextBlob object
from textblob import TextBlob




# define a function that accepts text and returns the polarity

def detect_sentiment_polarity(text):   
    blob = TextBlob(text)   
    return blob.sentiment.polarity 
    
def detect_sentiment_subjectivity(text):   
    blob = TextBlob(text)   
    return blob.sentiment.subjectivity 
# create a new DataFrame column for sentiment (WARNING: SLOW!)

#def make_features(df):
#    df['sentiment_polarity'] = df.message.apply(detect_sentiment_polarity)
#    return df
#    
#make_features(sms)
sms['length'] = sms.message.apply(len)
sms['sentiment_polarity'] = sms.message.apply(detect_sentiment_polarity)
sms['sentiment_subjectivity'] = sms.message.apply(detect_sentiment_subjectivity)

sms.boxplot('sentiment_polarity', by='label')    
sms.boxplot('sentiment_subjectivity', by='label')   
sms.boxplot('length', by='label').set_ylim([0,300])


sms.groupby('label').length.describe().unstack()

# ## Part 3: Model evaluation using `train_test_split` and `cross_val_score`
# 
# - The motivation for model evaluation is that you need a way to **choose between models** (different model types, tuning parameters, and features).
# - You use a model evaluation procedure to estimate how well a model will **generalize** to out-of-sample data.
# - This requires a model evaluation metric to **quantify** a model's performance.

# define X and y
feature_cols = ['length', 'sentiment_polarity','sentiment_subjectivity' ]
Xn = sms[feature_cols]
yn = sms.label_num

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)

from sklearn.cross_validation import train_test_split
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, random_state=1)

knn.fit(Xn_train, yn_train)
yn_pred_class = knn.predict(Xn_test)

metrics.accuracy_score(yn_test, yn_pred_class)

from sklearn.cross_validation import cross_val_score
cross_val_score(knn, Xn, yn, cv=5, scoring='accuracy').mean()

yn_test.value_counts().head(1) / yn_test.shape

print ('Confusion matrix:', metrics.confusion_matrix(yn_test, yn_pred_class))
print ('Confusion matrix:', metrics.confusion_matrix(y_test, y_pred_class))



