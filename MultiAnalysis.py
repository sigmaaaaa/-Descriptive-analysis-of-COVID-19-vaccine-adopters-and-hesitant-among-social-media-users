#Multi Analysis
#%%
#Trend=========================================================================
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("whitegrid")

data_analysis = pd.read_csv('NEU_cluster5_0_0906.csv')
data_analysis = data_analysis.reset_index()
del data_analysis['level_0']

#def str_time_trans(strtime):
#    strtime = datetime.strptime(strtime, '%Y-%m-%d')
for i in range(len(data_analysis)):
    data_analysis.loc[i,'date-only'] = datetime.strptime(data_analysis.loc[i,'date-only'],'%Y-%m-%d')


start_remove = pd.to_datetime('2020-07-01')
end_remove = pd.to_datetime('2021-05-31')
data_analysis = data_analysis.loc[~(data_analysis['date-only'] < start_remove)]
data_analysis = data_analysis.loc[~(data_analysis['date-only'] > end_remove)]
data_analysis = data_analysis.reset_index()


for i in range(len(data_analysis)):
   data_analysis.loc[i,'Year_Month'] = data_analysis.loc[i,'date-only'].strftime('%Y-%m')

data_tweets_date_group = data_analysis.groupby('Year_Month')
data_y1 = pd.DataFrame(data_tweets_date_group.size())
data_tweets_date_group = data_analysis.groupby('date-only')
data_y2 = pd.DataFrame(data_tweets_date_group.size())
data_y2 = data_y2.reset_index()

for i in range(len(data_y2)):
   data_y2.loc[i,'Year_Month'] = data_y2.loc[i,'date-only'].strftime('%Y-%m')
   
data_y2 = data_y2.groupby('Year_Month').mean()

x = data_y1.index.values
y1 = data_y1[0].values
y2 = data_y2[0].values

plt.rcParams['figure.figsize'] = (12.0,5.0)
fig = plt.figure()
#bar=========================================
ax1 = fig.add_subplot(111)
ax1.bar(x, y1,width = 0.8,alpha=.7,color='g')
ax1.set_ylabel('Monthly Number of Tweets',fontsize='15')
#ax1.set_title("data analysis",fontsize='20')
#line========================================
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r',ms=10)
ax2.set_ylabel('Average Daily Number of Tweets',fontsize='15')
ax2.get_xaxis().set_visible(False)
plt.title('Trend_POS5')
#plt.hist(fig_new['Year_Month'],fig_new['num'])
fig.tight_layout()
plt.show()

#%%
#Location======================================================================
from geotext import GeoText

data_analysis = data_4

locations = data_analysis['u_area'].value_counts()
locations = pd.DataFrame(locations)
locations.reset_index(inplace=True)
locations = locations.rename(columns={'index':'area','u_area':'num'})
locations['area']=locations['area'].apply(lambda x:x.replace(',',' '))
locations['country']=locations['area'].apply(lambda x:(GeoText(x).country_mentions))
locations.drop(locations[locations['country']=='[]'].index,inplace=True)
locations['country']=locations['country'].apply(lambda x:(x.keys()))
locations['country']=locations['country'].apply(lambda x:list(x))
locations.drop(locations.index[locations.country.map(len)==0],inplace=True)
locations['country']=locations['country'].apply(lambda x:str(x[0]))
agg_func={'num':'sum'}
locations=locations.groupby(['country']).aggregate(agg_func)
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#num_minmax = []
#num_minmax = np.array(locations['num'])
#minmax = MinMaxScaler().fit_transform(num_minmax)
#zscore = preprocessing.scale(locations['num'])
#locations['zscore'] = zscore
#locations['minmax'] = minmax
#locations.sort_values(by=['num'],ascending=False,inplace=True)
locations.reset_index(inplace=True)
locations.to_csv('locations_neu5.csv')





# Bar plot of locaitons
import plotly.express as px
Count_graph=px.bar(x='Location',y='count',data_frame=Location_country[:15],color='Location')
Count_graph.show()
# Map plot of locations
import country_converter as coco
import json
import kaleido
with open('countries.geo.json') as response:
    counties = json.load(response)
    
cc = coco.CountryConverter()
locations['country']=locations['country'].apply(lambda x:cc.convert(names=x,to='ISO3'))
locations.to_csv('locations.csv')

india_states = json.load(open("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"))

fig = px.choropleth(
    locations,
    locations="country",
    color="zscore",
    hover_name="country",
    #hover_data=["num"],
    title="number of tweets from each country",
    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_geos(fitbounds="country", visible=False)
fig.show()
fig.write_image('map_test.png')

df = px.data.gapminder().query("year==2007")
fig = px.choropleth(df, locations="iso_alpha",
                    color="lifeExp", # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
plt.show()

#%%
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
plot_wordcloud(NegTweets_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="neg0")

#1. Top n words

# Dataset1=====================================================================
top40_neg0 = topnwords(NegTweets_0,40)
top40_neg1 = topnwords(NegTweets_1,40)
top40_neg2 = topnwords(NegTweets_2,40)
top40_neg3 = topnwords(NegTweets_3,40)
top40_neg4 = topnwords(NegTweets_4,40)
top40_neg5 = topnwords(NegTweets_5,40)
top40_neg6 = topnwords(NegTweets_6,40)
top40_neg7 = topnwords(NegTweets_7,40)
top40_neg8 = topnwords(NegTweets_8,40)
top40_neg9 = topnwords(NegTweets_9,40)
top40_neg10 = topnwords(NegTweets_10,40)

top40 = topnwords(data_analysis,40)

top40_pos0 = topnwords(PosTweets_0,40)
top40_pos1 = topnwords(PosTweets_1,40)
top40_pos2 = topnwords(PosTweets_2,40)
top40_pos3 = topnwords(PosTweets_3,40)
top40_pos4 = topnwords(PosTweets_4,40)
top40_pos5 = topnwords(PosTweets_5,40)

top40_neu0 = topnwords(NeuTweets_0,40)
top40_neu1 = topnwords(NeuTweets_1,40)
top40_neu2 = topnwords(NeuTweets_2,40)
top40_neu3 = topnwords(NeuTweets_3,40)
top40_neu4 = topnwords(NeuTweets_4,40)


NegTweets_0.to_csv('NegTweets_0_F.csv')
NegTweets_1.to_csv('NegTweets_1_F.csv')
NegTweets_2.to_csv('NegTweets_2_F.csv')
NegTweets_3.to_csv('NegTweets_3_F.csv')
NegTweets_4.to_csv('NegTweets_4_F.csv')
NegTweets_5.to_csv('NegTweets_5_F.csv')
NegTweets_6.to_csv('NegTweets_6_F.csv')
NegTweets_7.to_csv('NegTweets_7_F.csv')
NegTweets_8.to_csv('NegTweets_8_F.csv')
NegTweets_9.to_csv('NegTweets_9_F.csv')
NegTweets_10.to_csv('NegTweets_10_F.csv')

PosTweets_0.to_csv('PosTweets_0_F.csv')
PosTweets_1.to_csv('PosTweets_1_F.csv')
PosTweets_2.to_csv('PosTweets_2_F.csv')
PosTweets_3.to_csv('PosTweets_3_F.csv')
PosTweets_4.to_csv('PosTweets_4_F.csv')
PosTweets_5.to_csv('PosTweets_5_F.csv')






