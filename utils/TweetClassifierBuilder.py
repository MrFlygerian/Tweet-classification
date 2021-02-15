# Data Manipulation
import pandas as pd
pd.set_option('use_inf_as_na', True)
import numpy as np
from sklearn.model_selection import train_test_split

# tweets processing
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

# Visual modules
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Converting words to numbers (bags of words)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

#Save model
import pickle


# CONSTANTS
TWITTER_LOGO = 'Twitter logo 2012.png'
EXTRA_WORDS = 'extra_words.txt'
TRAIN_FILE = 'train.csv'
SUB_FILE = 'test.csv'

MODEL_FILE = 'some-model'



#------------------------------------------------------------Functions----------------------------------------------
def plot_calibration_curve(est, name, fig_index):

    print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

def make_cloud(word_tokens, image_file_path):

    word_str = ' '.join(word_tokens)

    mask = Image.open(image_file_path)
    img_mask = Image.new(mode='RGB', size=mask.size, color = (255,255,255))
    img_mask.paste(mask, box = mask)

    rgb_array = np.array(img_mask)


    cloud = WordCloud(font_path=r'C:\Users\bless\Python Scripts\Twitter Projects\WordClouds\GatsbyFLF-BoldItalic.ttf',
                  mask=rgb_array,background_color='black',
                  max_words=600, colormap = 'Set3')

    cloud.generate(word_str.upper())
    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.imshow(cloud, interpolation='bilinear')

def ProcessTweet(tweet):
    extra_words = (pd.read_table(EXTRA_WORDS, header = None))[0].values.tolist()
    _stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'] + extra_words)

    tweet = tweet.lower() # convert tweets to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet =  re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet =  word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)

    return [word for word in tweet if word not in _stopwords]

def test_models(x_train, y_train, x_test, y_test):
    fitted_models = []

    models = {'NB': GaussianNB(),
              'SVC': LinearSVC(max_iter = 1000),
              'LR': LogisticRegression(C = 0.6),
              'RF': RandomForestClassifier(n_estimators = 200),
              'SVC calibrated':CalibratedClassifierCV((LinearSVC(max_iter = 10000)), cv = 2, method = 'sigmoid'),
              'LR calibrated': CalibratedClassifierCV(LogisticRegression(), method = 'sigmoid',cv = 2)}

    for name, model in models.items():
        classifier = model.fit(X_train, y_train)
        y_pred = classifier.predict(x_test)
        fitted_models.append(classifier)
        print(name)
        print('Accuracy score:', round(accuracy_score(y_test, y_pred), 2))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    return fitted_models
#------------------------------------------------------------------------------------------------------------------



#Load data and extract inputs and outputs
data_df = pd.read_csv(r'..\data\train.csv')
data_df = data_df.fillna(0)

texts = data_df.drop(['target', 'id'], axis = 1)
tweets = texts.text
y = data_df.target

#Clean tweets and extract words
clean_tweets_tokened = tweets.apply(ProcessTweet)
words = [word for item in clean_tweets_tokened for word in item]
clean_tweets = clean_tweets_tokened.apply(' '.join)

#Make wordcloud
make_cloud(words, TWITTER_LOGO)


#Create bag of words and turn into numpy arrays
vectorizer = CountVectorizer(max_features = 3000, min_df = 5, max_df = 0.7)
X = vectorizer.fit_transform(clean_tweets).toarray()

converter = TfidfTransformer()
X_1 = converter.fit_transform(X).toarray()

#split arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Instantiate, fit and evaluate classifier
fitted_models = test_models(X_train, y_train, X_test, y_test)

# Plot calibration curves
#plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

plot_calibration_curve(LinearSVC(max_iter=10000), "SVC", 2)

plot_calibration_curve(RandomForestClassifier(), "Random Forest", 3)



#with open(MODEL_FILE, 'wb') as some_model:
#    pickle.dump(fitted_models[4], some_model)
