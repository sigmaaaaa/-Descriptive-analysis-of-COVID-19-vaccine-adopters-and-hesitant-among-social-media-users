# LDA + HDP + Clustering
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np

# Plotting tools
#import pyLDAvis
#from  pyLDAvis import gensim
#import pyLDAvis.gensim  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

np.random.seed(2222)

data_analysis = Neu_Tweets

processed_docs = data_analysis['Tweets_split_sl']
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from gensim.models import HdpModel
hdp = HdpModel(corpus_tfidf, dictionary)   
hdp.print_topics(num_words=10) 

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

train_vecs = []
for i in range(len(data_analysis)):
    top_topics = lda_model_tfidf.get_document_topics(bow_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    train_vecs.append(topic_vec)


def spacy_tokenizer(document):
    tokens = nlp(document)
    tokens = [token for token in tokens if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.lemma_.strip()!= '')]
    tokens = [token.lemma_ for token in tokens]
    return tokens

tfidf_vector = TfidfVectorizer(input = 'content', max_features = 3000,tokenizer = spacy_tokenizer)
corpus = data_analysis['Tweets_split_sl_str']
result = tfidf_vector.fit_transform(corpus)
dense = result.todense()
train_vecs = np.array(train_vecs)
train = np.concatenate((dense,train_vecs),axis=1)


#Neg_Tweets = Neg_Tweets.drop(columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
#Neu_Tweets = Neu_Tweets.drop(columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
#Pos_Tweets = Pos_Tweets.drop(columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

from sklearn.cluster import KMeans
i = 5
current_kmean = KMeans(n_clusters=i).fit(train)
col_name = str(i) +'means_label'
Neu_Tweets[col_name] = current_kmean.labels_




kmeans_models = {}
for i in range(2,13+1):
    current_kmean = KMeans(n_clusters=i).fit(train)
    kmeans_models[i] = current_kmean
    
cluster_df = pd.DataFrame()
cluster_df['Review Texts'] = data_analysis['text']
for i in range(2, 13+1):
    col_name = str(i) +'means_label'
    data_analysis[col_name] = kmeans_models[i].labels_

#Elbow Method to determine the best K
Sum_of_squared_distances = []
K = range(1,18)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(result)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#Choose K = 9 to do the experiment
cluster9 = data_analysis.iloc[:,[0,8]]
cluster9_0 = cluster9.loc[cluster9["9means_label"] == 0]
cluster9_1 = cluster9.loc[cluster9["9means_label"] == 1]
cluster9_2 = cluster9.loc[cluster9["9means_label"] == 2]
cluster9_3 = cluster9.loc[cluster9["9means_label"] == 3]
cluster9_4 = cluster9.loc[cluster9["9means_label"] == 4]
cluster9_5 = cluster9.loc[cluster9["9means_label"] == 5]
cluster9_6 = cluster9.loc[cluster9["9means_label"] == 6]
cluster9_7 = cluster9.loc[cluster9["9means_label"] == 7]
cluster9_8 = cluster9.loc[cluster9["9means_label"] == 8]
cluster13_9 = cluster13.loc[cluster13["13means_label"] == 9]
cluster13_10 = cluster13.loc[cluster13["13means_label"] == 10]
cluster13_11 = cluster13.loc[cluster13["13means_label"] == 11]
cluster13_12 = cluster13.loc[cluster13["13means_label"] == 12]

cluster9_0.to_csv('DATA3_POS_cluster9_0_0906.csv')
cluster9_1.to_csv('DATA3_POS_cluster9_1_0906.csv')
cluster9_2.to_csv('DATA3_POS_cluster9_2_0906.csv')
cluster9_3.to_csv('DATA3_POS_cluster9_3_0906.csv')
cluster9_4.to_csv('DATA3_POS_cluster9_4_0906.csv')
cluster9_5.to_csv('DATA3_POS_cluster9_5_0906.csv')
cluster9_6.to_csv('DATA3_POS_cluster9_6_0906.csv')
cluster9_7.to_csv('DATA3_POS_cluster9_7_0906.csv')
cluster9_8.to_csv('DATA3_POS_cluster9_8_0906.csv')
cluster13_9.to_csv('DATA3_NEG_cluster13_9_0906.csv')
cluster13_10.to_csv('DATA3_NEG_cluster13_10_0906.csv')
cluster13_11.to_csv('DATA3_NEG_cluster13_11_0906.csv')
cluster13_12.to_csv('DATA3_NEG_cluster13_12_0906.csv')

#%%
NEG_group = Neg_Tweets.groupby('11means_label')
NegTweets_0 = NEG_group.get_group(0)
NegTweets_1 = NEG_group.get_group(1)
NegTweets_2 = NEG_group.get_group(2)
NegTweets_3 = NEG_group.get_group(3)
NegTweets_4 = NEG_group.get_group(4)
NegTweets_5 = NEG_group.get_group(5)
NegTweets_6 = NEG_group.get_group(6)
NegTweets_7 = NEG_group.get_group(7)
NegTweets_8 = NEG_group.get_group(8)
NegTweets_9 = NEG_group.get_group(9)
NegTweets_10 = NEG_group.get_group(10)

POS_group = Pos_Tweets.groupby('6means_label')
PosTweets_0 = POS_group.get_group(0)
PosTweets_1 = POS_group.get_group(1)
PosTweets_2 = POS_group.get_group(2)
PosTweets_3 = POS_group.get_group(3)
PosTweets_4 = POS_group.get_group(4)
PosTweets_5 = POS_group.get_group(5)

NEU_group = Neu_Tweets.groupby('5means_label')
NeuTweets_0 = NEU_group.get_group(0)
NeuTweets_1 = NEU_group.get_group(1)
NeuTweets_2 = NEU_group.get_group(2)
NeuTweets_3 = NEU_group.get_group(3)
NeuTweets_4 = NEU_group.get_group(4)

DATA3_NEG_group = data3_DT_0.groupby('6means_label')
kaggledata_neg0 = DATA3_NEG_group.get_group(0)
kaggledata_neg1 = DATA3_NEG_group.get_group(1)
kaggledata_neg2 = DATA3_NEG_group.get_group(2)
kaggledata_neg3 = DATA3_NEG_group.get_group(3)
kaggledata_neg4 = DATA3_NEG_group.get_group(4)
kaggledata_neg5 = DATA3_NEG_group.get_group(5)

DATA3_POS_group = data3_DT_1.groupby('5means_label')
kaggledata_pos0 = DATA3_POS_group.get_group(0)
kaggledata_pos1 = DATA3_POS_group.get_group(1)
kaggledata_pos2 = DATA3_POS_group.get_group(2)
kaggledata_pos3 = DATA3_POS_group.get_group(3)
kaggledata_pos4 = DATA3_POS_group.get_group(4)

DATA3_NEU_group = data3_DT_2.groupby('11means_label')
kaggledata_neu0 = DATA3_NEU_group.get_group(0)
kaggledata_neu1 = DATA3_NEU_group.get_group(1)
kaggledata_neu2 = DATA3_NEU_group.get_group(2)
kaggledata_neu3 = DATA3_NEU_group.get_group(3)
kaggledata_neu4 = DATA3_NEU_group.get_group(4)
kaggledata_neu5 = DATA3_NEU_group.get_group(5)
kaggledata_neu6 = DATA3_NEU_group.get_group(6)
kaggledata_neu7 = DATA3_NEU_group.get_group(7)
kaggledata_neu8 = DATA3_NEU_group.get_group(8)
kaggledata_neu9 = DATA3_NEU_group.get_group(9)
kaggledata_neu10 = DATA3_NEU_group.get_group(10)



