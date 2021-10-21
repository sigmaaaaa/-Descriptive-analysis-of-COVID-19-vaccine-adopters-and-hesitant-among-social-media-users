# _*_ coding: utf-8 _*_
# Created on Sat Jun 10 08:50:16 2021
# @Author : Sigma.G
# @Function: Model-Final Version(Twitter)

#%%
import numpy as np
import pandas as pd
import string
import re
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from spacy.util import compounding
from spacy.util import minibatch
#%%
# Step1 Train set preparation

# Read the Whole Dataset=======================================================
Data = pd.read_csv('Tweets_ALL_afterscoring.csv')
covidvaccine = pd.read_csv('covidvaccine.csv')

# Data Cleaning================================================================
Data = Data.drop_duplicates(subset='m_content').reset_index(drop=True)
covidvaccine = covidvaccine.drop_duplicates(subset='text').reset_index(drop=True)
del Data['neu']
del Data['neg']
del Data['pos']
del Data['Unnamed: 0']
del Data['Unnamed: 0.1']
del Data['Unnamed: 0.1.1']

# Deal with attached websites and other signals impacting the textblob scores==
def clean(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    return text
Data['TEXT'] = Data['m_content'].apply(lambda x:clean(x))
covidvaccine['TEXT'] = covidvaccine['text'].apply(lambda x:str_trans(x))
covidvaccine['TEXT'] = covidvaccine['TEXT'].apply(lambda x:clean(x))

# Append text VADER score======================================================
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
Data['compound'] = Data['TEXT'].apply(lambda x:sia.polarity_scores(x)["compound"])
Data['pos'] = Data['TEXT'].apply(lambda x:sia.polarity_scores(x)["pos"])
Data['neg'] = Data['TEXT'].apply(lambda x:sia.polarity_scores(x)["neg"])
Data['neu'] = Data['TEXT'].apply(lambda x:sia.polarity_scores(x)["neu"])

covidvaccine['compound'] = covidvaccine['TEXT'].apply(lambda x:sia.polarity_scores(x)["compound"])
covidvaccine['pos'] = covidvaccine['TEXT'].apply(lambda x:sia.polarity_scores(x)["pos"])
covidvaccine['neg'] = covidvaccine['TEXT'].apply(lambda x:sia.polarity_scores(x)["neg"])
covidvaccine['neu'] = covidvaccine['TEXT'].apply(lambda x:sia.polarity_scores(x)["neu"])

# Append textblob polarity score===============================================
import textblob
Data['textblob'] = Data['TEXT'].apply(lambda x:textblob.TextBlob(x).sentiment[0])
covidvaccine['textblob'] = covidvaccine['TEXT'].apply(lambda x:textblob.TextBlob(x).sentiment[0])

# Tweets_split=================================================================
# 1. Remove punctuations + lowercase every word
def RemovePunctuations(string):
    Punctuations = '''!()-[]{};:'"\,<>./?|@#$%^&*_~+…'''
    for i in string:  
        if i in Punctuations:  
            string = string.replace(i,"") 
    return string

Data['Tweets_split'] = 0
covidvaccine['Tweets_split'] = 0

for i in range(len(Data)):
    Data.loc[i,'Tweets_split'] = RemovePunctuations(Data.loc[i,'TEXT'])
    Data.loc[i,'Tweets_split'] = Data.loc[i,'Tweets_split'].lower()
    
for i in range(len(covidvaccine)):
    covidvaccine.loc[i,'Tweets_split'] = RemovePunctuations(covidvaccine.loc[i,'TEXT'])
    covidvaccine.loc[i,'Tweets_split'] = covidvaccine.loc[i,'Tweets_split'].lower()

# 2. Split
Data['Tweets_split'] = Data.Tweets_split.str.split()
covidvaccine['Tweets_split'] = covidvaccine.Tweets_split.str.split()

# 3. Remove stopwords
def remove_stopword(x):
    stopwords_list = stopwords.words('english')
    stopwords_list = set(stopwords_list)
    more_stopwords = {'u', "im","covid","covid19","vaccines","vaccine","us","get","one","like",
                      "vaccinated","gotvaccinated","vaxxed","vaccine death","covidvaccinedeaths",
         "vaccine side effects","covid","covid-19","covid19","coronavirus",
         "vaccine","vaccines","vaccination","vaccinations","covidcaccine","covidvaccines","covidvaccination",
         "vaccinated","gotvaccinated","vaccinesWork","vaxxed","fullyvaccinated",
         "coronavaccineishope","getvaccinated","vaccinesideeffects","deathbyvaccines",
         "vaccinedeaths","vaccineDeath","vaccinevictims","vaccinevictim","vaccinedamage",
         "vaccineinjury","vaccineinjury","killervaccines","vaccineconcerns","novccinepassports",
         "vaccinefraud","vaccinefraud","vaccinefraud","vaccineviolence","stopmandatoryvaccines",
         "novaccinepassports","novaccinepassportsanywhere","novaccine","novaccinenovacancy",
         "nocovidvaccine","novaxxed","nocovidvaccine","antivaxxers","coronavirus","covid19",
         "covidvaccine","covidvaccines","covid19vaccines","covidvaccintion",
         "vaccinationcovid","vaccineforall","vaccinationdrive","vaccines","covid19vaccine",
         "vaccinated","vaccination","covidvaccine","vaccinations","covidvaccines","covid19vaccines",
         "side","effects","novaccine","vaccineswork","getvaccinated","covid-19vaccine","amp","people","first"}
    stopwords_list = stopwords_list.union(more_stopwords)
    return [y for y in x if y not in stopwords_list]

Data['Tweets_split'] = Data['Tweets_split'].apply(lambda x:remove_stopword(x))
covidvaccine['Tweets_split'] = covidvaccine['Tweets_split'].apply(lambda x:remove_stopword(x))

# 4. Using stemming
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return text
Data['Tweets_split_sl']= Data['Tweets_split'].apply(lambda x: stemming_on_text(x))
covidvaccine['Tweets_split_sl']= covidvaccine['Tweets_split'].apply(lambda x: stemming_on_text(x))

# 5. Using Lemmatizer 
lm = nltk.WordNetLemmatizer()
nltk.download('wordnet')
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text
Data['Tweets_split_sl']= Data['Tweets_split_sl'].apply(lambda x: lemmatizer_on_text(x))
covidvaccine['Tweets_split_sl']= covidvaccine['Tweets_split_sl'].apply(lambda x: lemmatizer_on_text(x))

# 6. Tweets_split_str
def str_trans(text):
    text_str = str(text)
    return text_str

Data['Tweets_split_sl_str'] = Data['Tweets_split_sl'].apply(lambda x:str_trans(x))
covidvaccine['Tweets_split_sl_str'] = covidvaccine['Tweets_split_sl'].apply(lambda x:str_trans(x))

Data['Tweets_split_str'] = Data['Tweets_split'].apply(lambda x:str_trans(x))
covidvaccine['Tweets_split_str'] = covidvaccine['Tweets_split'].apply(lambda x:str_trans(x))

Data.to_csv('Data_0812.csv')
covidvaccine.to_csv('Data_covidvaccine_0812.csv')

# Extract tweets from extremely positive and negative hashtags=================
neg_key = ["vaccine death","NoVaccine","vaccinedeaths","VaccineInjury","VaccineDeath",
           "nocovidvaccine","VaccineFraud","VaccineDamage","vaccinevictims","covidvaccinedeaths",
           "KillerVaccines","VaccineVictim","deathbyvaccines"]

pos_key = ["Vaccinated","GetVaccinated","GotVaccinated","FullyVaccinated"]

neutral_key = ["coronavirus","Covid19","CovidVaccine","Vaccines","Covid","covid19vaccine","Covid-19",
               "Vaccination","Vaccinations","COVID19vaccines","COVIDVaccination","VaccinationDrive",
               "VaccinationCovid","Vaccine","CoronaVirus","COVID19","CovidVaccines","NoVaccineNoVacancy",
               "StopMandatoryVaccines","vaccine side effects","vaccineSideEffects","NoVaccinePassports",
               "NoVaccinePassportsAnywhere","FullyVaccinated","antivaxxers","vaccineconcerns","VAXXED",
               "vaxxed","VaccinesWork"]

def rules_def(dataset,rules_key):
    rules = (dataset["keyword"] == rules_key[0])
    i = 1
    for i in range(len(rules_key)):
        rules = rules|(dataset["keyword"] == rules_key[i])
    return rules

Rules_neg = rules_def(Data,neg_key)
Rules_pos = rules_def(Data,pos_key)
Rules_neutral = rules_def(Data,neutral_key)

data_content_neg = Data[Rules_neg]
data_content_neg = pd.DataFrame(data_content_neg).reset_index(drop=False)
data_content_neg['sentiment'] = 0
#data_content_neg.to_csv('data_content_neg_0729.csv')

data_content_pos = Data[Rules_pos]
data_content_pos = pd.DataFrame(data_content_pos).reset_index(drop=False)
data_content_pos['sentiment'] = 1
data_content_pos_1 = data_content_pos.sample(n=22047)
#data_content_pos.to_csv('data_content_pos_0729.csv')

data_content_neutral = Data[Rules_neutral]
data_content_neutral = pd.DataFrame(data_content_neutral).reset_index(drop=False)
#data_content_neutral_2['sentiment'] = data_content_neutral['sentiment']

train_set = data_content_neg.append(data_content_pos_1)
index = train_set.loc[(train_set['g_publish_time']=='g_publish_time')].index
train_set = train_set.drop(index)
train_set = train_set.reset_index(drop=True)

#7. TF-IDF Vectorizer
#Define Spacy Tokenizer
def spacy_tokenizer(document):
    tokens = nlp(document)
    tokens = [token for token in tokens if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.lemma_.strip()!= '')]
    tokens = [token.lemma_ for token in tokens]
    return tokens

tfidf_vector = TfidfVectorizer(input = 'content', max_features = 3000,tokenizer = spacy_tokenizer)
corpus = train_set['Tweets_split_sl_str']
result = tfidf_vector.fit_transform(corpus)
dense = result.todense()
feature = pd.DataFrame(train_set.loc[:,('neu','neg','pos','compound','textblob')])
dense = pd.DataFrame(dense)
feature = feature.join(dense)

corpus_neu = data_content_neutral['Tweets_split_sl_str']
result_neu = tfidf_vector.fit_transform(corpus_neu)
dense_neu = result_neu.todense()
feature_neu = pd.DataFrame(data_content_neutral.loc[:,('neu','neg','pos','compound','textblob')])
dense_neu = pd.DataFrame(dense_neu)
feature_neu = feature_neu.join(dense_neu)

corpus_data3 = covidvaccine['Tweets_split_sl_str']
result_data3 = tfidf_vector.fit_transform(corpus_data3)
dense_data3 = result_data3.todense()
feature_data3 = pd.DataFrame(covidvaccine.loc[:,('neu','neg','pos','compound','textblob')])
dense_data3 = pd.DataFrame(dense_data3)
feature_data3 = feature_data3.join(dense_data3)

#%%
pd.DataFrame(data_content_neg).to_csv('data_content_neg_0826')
pd.DataFrame(data_content_neutral).to_csv('data_content_neutral_0826')
pd.DataFrame(data_content_pos).to_csv('data_content_pos_0826')
pd.DataFrame(data_content_pos_1).to_csv('data_content_pos_1_0826')

pd.DataFrame(dense).to_csv('dense_0826')
pd.DataFrame(dense_data3).to_csv('dense_data3_0826')
pd.DataFrame(dense_neu).to_csv('dense_neu_0826')

pd.DataFrame(feature).to_csv('feature_0826')
pd.DataFrame(feature_data3).to_csv('feature_data3_0826')
pd.DataFrame(feature_neu).to_csv('feature_neu_0826')

pd.DataFrame(result).to_csv('result_0826')
pd.DataFrame(result_data3).to_csv('result_data3_0826')
pd.DataFrame(result_neu).to_csv('result_neu_0826')

#%%
# Step2 Split the dataset used for model training
# Split the train set into train/set = 0.85:0.15===============================
from sklearn.model_selection import train_test_split
xtrain, xtest,ytrain, ytest = train_test_split(feature, train_set['sentiment'], random_state=42, test_size=0.15)
xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)
xtrain = xtrain.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)

#%%
# Step3 Parameter selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,cross_val_score

kf =KFold(n_splits=5, shuffle=True, random_state=42)

#LR-MODEL-Optimization=========================================================
algorithms = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
for algo in algorithms:
    score = cross_val_score(LogisticRegression(max_iter= 4000, solver= algo, random_state= 42), xtrain, ytrain, cv= kf, scoring="accuracy")
    print(f'Average score({algo}): {"{:.3f}".format(score.mean())}')
#==============================================================================
# sag 0.865
    
#Decision Tree Classifier-MODEL-Optimization===================================
max_depth = [1,2,3,4,5,6,7,8,9,10]
for val in max_depth:
    score = cross_val_score(DecisionTreeClassifier(max_depth= val, random_state= 42), xtrain, ytrain, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
#==============================================================================
# 10 0.804

#Random Forest Classifier-MODEL-Optimization===================================
n_estimators = [50, 100, 150, 200, 250, 300, 350]
for val in n_estimators:
    score = cross_val_score(RandomForestClassifier(n_estimators= val, random_state= 42), xtrain, ytrain, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
#==============================================================================
# 250 0.853

#SVM Classifier-MODEL-Optimization=============================================
n_C = [1,2,3,4,5,6,7,8,9,10]
for val in n_C:
    score = cross_val_score(svm.SVC(kernel='rbf',C=val,gamma=0.1), xtrain, ytrain, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
#==============================================================================
#

#KNN Classifier-MODEL-Optimization=============================================
n_K = [1,2,3,4,5,6,7,8,9,10]
for val in n_K:
    score = cross_val_score(KNeighborsClassifier(n_neighbors=val), xtrain, ytrain, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
#==============================================================================
#

#XGBoost=======================================================================
score = cross_val_score(XGBClassifier(max_depth=4,
 min_child_weight=6), xtrain, ytrain, cv= kf, scoring="accuracy")
score.mean()
#==============================================================================
#

#%%
# Step4 Model selection
# Sub Functions================================================================
def SecondRule_ParaSelection(result):
    Best_Index = list(result.loc[(result['Acc_select'] == result['Acc_select'].max())].index)
    if len(Best_Index) == 1:
        bindex = Best_Index[0]
    else:
        percent_2 = result.loc[(result['Acc_select'] == result['Acc_select'].max())]
        bindex = percent_2['P2'].argmin()
    return bindex

def second_rule(df,prob0,prob1):
    pred_result = []
    for i in range(len(df)):
        if df.iloc[i,0] == df.iloc[i,1]:
            pred_result.append(2)
        elif df.iloc[i,0] >= prob0:
            pred_result.append(0)
        elif df.iloc[i,1] >= prob1:
            pred_result.append(1)
        else:
            pred_result.append(2)
    return pred_result

def acc_calculation(original_result,pred_result):
    counter = 0
    count_2 = 0
    for j in range(len(pred_result)):
       if original_result[j] == pred_result[j]:
            counter = counter + 1
       elif pred_result[j] == 2:
           count_2 = count_2 + 1 
       else:
               counter = counter
    acc = counter/(len(pred_result)-count_2)
    return acc

import matplotlib.pyplot as plt
import seaborn as sns
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
     # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

# Model Appending==============================================================
models = []
models.append(('LR',LogisticRegression(random_state= 42,solver = 'sag')))
#models.append(('SGD',SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)))
models.append(('DecisionTree',DecisionTreeClassifier(max_depth= 10, random_state= 42)))
models.append(('RandomForest',RandomForestClassifier(n_estimators= 250, random_state= 42)))
#models.append(('svm',svm.SVC(kernel='linear',C=9,gamma=0.1)))
models.append(('XGBoost',XGBClassifier(max_depth=4,min_child_weight=6)))
models.append(('KNN',KNeighborsClassifier(n_neighbors=9)))
#models.append(('NaiveByes',MultinomialNB()))

# Main Model Part==============================================================
from sklearn.model_selection import StratifiedKFold
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


prob0_selected = []
prob1_selected = []

#LOOP1: For different models
for name,model in models:
    Acc_815 = pd.DataFrame([])
    Acc_815_select = pd.DataFrame([])
    P0_815 = pd.DataFrame([])
    P1_815 = pd.DataFrame([])
    P2_815 = pd.DataFrame([])
    Prob0_815 = pd.DataFrame([])
    Prob1_815 = pd.DataFrame([])
    skf = StratifiedKFold(n_splits=5)
    #LOOP2: For K-Fold CV train/valid selection
    k = 0
    for train_index, test_index in skf.split(xtrain,ytrain):
        x_train, x_valid = xtrain.loc[train_index,:], xtrain.loc[test_index,:]
        y_train, y_valid = ytrain[train_index], ytrain[test_index]
        x_train = x_train.reset_index(drop=True)
        x_valid = x_valid.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)
        model.fit(x_train,y_train)
        # Confusion Matrix Heatmap=============================================
        y_pred = model.predict(x_train)
        cf_matrix = confusion_matrix(y_train, y_pred)
        title = name
        gorup_names = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['0', '1']
        make_confusion_matrix(cf_matrix, 
                      group_names=gorup_names,
                      categories=categories, 
                      cmap='Blues',title = title)
        # Confusion Matrix Heatmap Codes End===================================
        model_pred = model.predict_proba(x_valid)
        acc_5 = []
        acc_5_select = []
        p0_5 = []
        p1_5 = []
        p2_5 = []
        Prob0 = []
        Prob1 = []
        #LOOP3: For Prob0 of the second rule
        for i in range(55,100,5):
            i=i/100
            #LOOP4: For Prob1 of the second rule
            for j in range(55,100,5):
                j=j/100
                Prob0.append(i)
                Prob1.append(j)
                model_pred_sentiment = second_rule(pd.DataFrame(model_pred),i,j)
                acc = acc_calculation(y_valid,model_pred_sentiment)
                acc_5.append(acc)
                c = collections.Counter(model_pred_sentiment)
                p0 = c[0]/len(model_pred_sentiment)
                p1 = c[1]/len(model_pred_sentiment)
                p2 = c[2]/len(model_pred_sentiment)
                acc_select = 2*(acc-0.5) - p2
                p0_5.append(p0)
                p1_5.append(p1)
                p2_5.append(p2)
                acc_5_select.append(acc_select)
                j=j*100
            i=i*100
        Acc_815['k'] = acc_5
        P0_815['k'] = p0_5
        P1_815['k'] = p1_5
        P2_815['k'] = p2_5
        Acc_815_select['k'] = acc_5_select
        k = k+1
    Acc = Acc_815.mean(axis=1)
    P0 = P0_815.mean(axis=1)
    P1 = P1_815.mean(axis=1)
    P2 = P2_815.mean(axis=1)
    Acc_select = Acc_815_select.mean(axis=1)
    Result = pd.DataFrame({'Prob0':Prob0,'Prob1':Prob1,'Acc':Acc,'P0':P0,'P1':P1,'P2':P2,'Acc_select':Acc_select})
    # Plot Codes===============================================================
    fig= plt.figure()
    ax = fig.gca(projection='3d')
    X= Result['Prob0']
    Y= Result['Prob1']
    Z = Result['Acc_select']
    surf=ax.plot_trisurf(Y, X, Z, cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)
    plt.show()
    # Plot Codes End===========================================================
    bindex = SecondRule_ParaSelection(Result)
    prob0 = Result.loc[bindex,'Prob0']
    prob1 = Result.loc[bindex,'Prob1']
    prob0_selected.append(prob0)
    prob1_selected.append(prob1)
    Test_pred_prob = model.predict_proba(xtest)
    Test_pred = second_rule(pd.DataFrame(Test_pred_prob), prob0, prob1)
    Accuracy_F = acc_calculation(ytest,Test_pred)
    Ct = collections.Counter(Test_pred)
    P0_t = Ct[0]/len(Test_pred)
    P1_t = Ct[1]/len(Test_pred)
    P2_t = Ct[2]/len(Test_pred)
    msg = "%s: Acc(%f),P0(%f),P1(%f),P2(%f)"%(name,Accuracy_F,P0_t,P1_t,P2_t)
    print(msg)

prob_selected = pd.DataFrame({'prob0':prob0_selected,'prob1':prob1_selected})

# Use the model on neutral tweets==============================================
model_F = LogisticRegression(random_state= 42,solver = 'sag')
prob0 = 0.55
prob1 = 0.55

model_F.fit(xtrain, ytrain)

model_pred_F_prob = model_F.predict_proba(feature_neu)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
data_content_neutral['sentiment'] = model_pred_F

model_pred_F_prob = model_F.predict_proba(feature_data3)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
covidvaccine['sentiment'] = model_pred_F
#==============================================================================

model_F = DecisionTreeClassifier(max_depth= 10, random_state= 42)
prob0 = 0.65
prob1 = 0.70

model_F.fit(xtrain, ytrain)
model_pred_F_prob = model_F.predict_proba(feature_data3)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
covidvaccine['sentiment'] = model_pred_F
#==============================================================================

model_F = RandomForestClassifier(n_estimators= 250, random_state= 42)
prob0 = 0.55
prob1 = 0.55
model_F.fit(xtrain, ytrain)

model_pred_F_prob = model_F.predict_proba(feature_neu)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
data_content_neutral['sentiment_RF'] = model_pred_F

model_pred_F_prob = model_F.predict_proba(feature_data3)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
covidvaccine['sentiment_RF'] = model_pred_F
#==============================================================================

model_F = XGBClassifier(max_depth=4,min_child_weight=6)
prob0 = 0.80
prob1 = 0.75

model_F.fit(xtrain, ytrain)
model_pred_F_prob = model_F.predict_proba(feature_data3)
model_pred_F = second_rule(pd.DataFrame(model_pred_F_prob), prob0, prob1)
covidvaccine['sentiment_XGB'] = model_pred_F
#==============================================================================
covidvaccine.to_csv('F_covidvaccine.csv')

#%%
#0,1,2 Percentage
#LR
data_analysis = data_content_neutral
Ct = collections.Counter(data_analysis['sentiment'])
P0_LR = Ct[0]/len(data_analysis)
P1_LR = Ct[1]/len(data_analysis)
P2_LR = Ct[2]/len(data_analysis)
msg = "%s: P0(%f),P1(%f),P2(%f)"%('LR',P0_LR,P1_LR,P2_LR)
print(msg)

#DT
Ct = collections.Counter(data_analysis['sentiment'])
P0_DT = Ct[0]/len(data_analysis)
P1_DT = Ct[1]/len(data_analysis)
P2_DT = Ct[2]/len(data_analysis)
msg = "%s: P0(%f),P1(%f),P2(%f)"%('DT',P0_DT,P1_DT,P2_DT)
print(msg)

#RF
Ct = collections.Counter(data_analysis['sentiment_RF'])
P0_RF = Ct[0]/len(data_analysis)
P1_RF = Ct[1]/len(data_analysis)
P2_RF = Ct[2]/len(data_analysis)
msg = "%s: P0(%f),P1(%f),P2(%f)"%('RF',P0_RF,P1_RF,P2_RF)
print(msg)

#XGB
Ct = collections.Counter(covidvaccine['sentiment_XGB'])
P0_XGB = Ct[0]/len(covidvaccine)
P1_XGB = Ct[1]/len(covidvaccine)
P2_XGB = Ct[2]/len(covidvaccine)
msg = "%s: P0(%f),P1(%f),P2(%f)"%('XGB',P0_XGB,P1_XGB,P2_XGB)
print(msg)

#%%
# Data Reading
data_content_neutral = pd.read_csv('F_data_content_neutral.csv')
covidvaccine = pd.read_csv('F_covidvaccine.csv')
# Data Grouping
# Dataset1=====================================================================
# LR
NEU_group = data_content_neutral.groupby('sentiment')
NEU_LR_0 = NEU_group.get_group(0)
NEU_LR_1 = NEU_group.get_group(1)
NEU_LR_2 = NEU_group.get_group(2)

# DT
NEU_group = data_content_neutral.groupby('sentiment')
NEU_DT_0 = NEU_group.get_group(0)
NEU_DT_1 = NEU_group.get_group(1)
NEU_DT_2 = NEU_group.get_group(2)

# RF
NEU_group = data_content_neutral.groupby('sentiment_RF')
NEU_RF_0 = NEU_group.get_group(0)
NEU_RF_1 = NEU_group.get_group(1)
NEU_RF_2 = NEU_group.get_group(2)

# XGB
NEU_group = data_content_neutral.groupby('sentiment_XGB')
NEU_XGB_0 = NEU_group.get_group(0)
NEU_XGB_1 = NEU_group.get_group(1)
NEU_XGB_2 = NEU_group.get_group(2)

# Dataset2=====================================================================
# DT
data3_group = covidvaccine.groupby('sentiment')
data3_DT_0 = data3_group.get_group(0)
data3_DT_1 = data3_group.get_group(1)
data3_DT_2 = data3_group.get_group(2)

# RF
data3_group = covidvaccine.groupby('sentiment_RF')
data3_RF_0 = data3_group.get_group(0)
data3_RF_1 = data3_group.get_group(1)
data3_RF_2 = data3_group.get_group(2)

#%%
# Data Analysis

#1. Top n words
def topnwords(dataset,n):
    top = Counter([item for sublist in dataset['Tweets_split'] for item in sublist])
    temp = pd.DataFrame(top.most_common(n))
    temp = temp.iloc[1:,:]
    temp.columns = ['Common_words','count']
    return temp

# Dataset1=====================================================================
top40_LR_0 = topnwords(NEU_LR_0,40)
top40_LR_1 = topnwords(NEU_LR_1,40)
top40_LR_2 = topnwords(NEU_LR_2,40)

top40_DT_0 = topnwords(NEU_DT_0,40)
top40_DT_1 = topnwords(NEU_DT_1,40)
top40_DT_2 = topnwords(NEU_DT_2,40)

top40_RF_0 = topnwords(NEU_RF_0,40)
top40_RF_1 = topnwords(NEU_RF_1,40)
top40_RF_2 = topnwords(NEU_RF_2,40)

top40_XGB_0 = topnwords(NEU_XGB_0,40)
top40_XGB_1 = topnwords(NEU_XGB_1,40)
top40_XGB_2 = topnwords(NEU_XGB_2,40)

# Dataset2=====================================================================
data3_top40_DT_0 = topnwords(data3_DT_0,40)
data3_top40_DT_1 = topnwords(data3_DT_1,40)
data3_top40_DT_2 = topnwords(data3_DT_2,40)

data3_top40_RF_0 = topnwords(data3_RF_0,40)
data3_top40_RF_1 = topnwords(data3_RF_1,40)
data3_top40_RF_2 = topnwords(data3_RF_2,40)

#2. WordCloud
def count_word(id_list):
    top = Counter([item for sublist in id_list for item in sublist])
    return top

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords =  {'u', "im","covid","covid19","vaccines","vaccine","us","get","one","like",
                      "Vaccinated","GotVaccinated","vaxxed","vaccine death","covidvaccinedeaths",
         "vaccine side effects","Covid","Covid-19","Covid19","coronavirus","CoronaVirus",
         "Vaccine","Vaccines","Vaccination","Vaccinations","CovidVaccine","CovidVaccines","COVIDVaccination",
         "Vaccinated","GotVaccinated","VaccinesWork","VAXXED","FullyVaccinated",
         "CoronaVaccineIsHope","GetVaccinated","vaccineSideEffects","deathbyvaccines",
         "vaccinedeaths","VaccineDeath","vaccinevictims","VaccineVictim","VaccineDamage",
         "VaccineInjury","vaccineinjury","KillerVaccines","vaccineconcerns","NoVaccinePassports",
         "VaccineFraud","vaccinefraud","VaccineFRAUD","VaccineViolence","StopMandatoryVaccines",
         "NoVaccinePassports","NoVaccinePassportsAnywhere","NoVaccine","NoVaccineNoVacancy",
         "nocovidvaccine","novaxxed","NoCovidVaccine","antivaxxers","coronavirus","Covid19",
         "COVID19","CovidVaccine","CovidVaccines","COVID19vaccines","COVIDVaccintion",
         "VaccinationCovid","VaccineForAll","VaccinationDrive","Vaccines","covid19vaccine",
          "vaccinated","vaccination","covidvaccine","vaccinations","covidvaccines","covid19vaccines",
          "side","effects","novaccine","vaccineswork","getvaccinated"}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=400, 
                    height=200,
                    mask = mask)
    #wordcloud.generate(str(text))
    wordcloud.generate_from_frequencies(count_word(text))
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

#pos_mask = np.array(Image.open(d+ 'twitter_mask.png'))
plot_wordcloud(NEU_LR_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 LR")
plot_wordcloud(NEU_LR_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 LR")
plot_wordcloud(NEU_LR_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 LR")

plot_wordcloud(NEU_DT_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 DT")
plot_wordcloud(NEU_DT_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 DT")
plot_wordcloud(NEU_DT_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 DT")

plot_wordcloud(NEU_RF_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 RF")
plot_wordcloud(NEU_RF_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 RF")
plot_wordcloud(NEU_RF_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 RF")

plot_wordcloud(NEU_XGB_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 XGB")
plot_wordcloud(NEU_XGB_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 XGB")
plot_wordcloud(NEU_XGB_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 XGB")

# Dataset2=====================================================================
plot_wordcloud(data3_DT_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 DT covidvaccine")
plot_wordcloud(data3_DT_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 DT covidvaccine")
plot_wordcloud(data3_DT_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 DT covidvaccine")

plot_wordcloud(data3_RF_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 0 RF covidvaccine")
plot_wordcloud(data3_RF_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 1 RF covidvaccine")
plot_wordcloud(data3_RF_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="WordCloud of 2 RF covidvaccine")


#%%
# Dataset Combination


    











