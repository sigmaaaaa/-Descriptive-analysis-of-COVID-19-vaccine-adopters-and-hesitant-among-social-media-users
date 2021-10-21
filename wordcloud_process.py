#Wordcloud
def count_word(id_list):
    top = Counter([item for sublist in id_list for item in sublist])
    return top

def plot_wordcloud(text, mask=pos_mask, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
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
                    colormap = 'cool',
                    mask = pos_mask)
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

pos_mask = np.array(Image.open('wrong.png'))
plot_wordcloud(NegTweets_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg1")
plot_wordcloud(NegTweets_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg2")
plot_wordcloud(NegTweets_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg3")
plot_wordcloud(NegTweets_3['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg4")
plot_wordcloud(NegTweets_4['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg5")
plot_wordcloud(NegTweets_5['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg6")
plot_wordcloud(NegTweets_6['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg7")
plot_wordcloud(NegTweets_7['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg8")
plot_wordcloud(NegTweets_8['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg9")
plot_wordcloud(NegTweets_9['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg10")
plot_wordcloud(NegTweets_10['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neg11")

pos_mask = np.array(Image.open('Neutral.png'))
plot_wordcloud(PosTweets_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos1")
plot_wordcloud(PosTweets_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos2")
plot_wordcloud(PosTweets_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos3")
plot_wordcloud(PosTweets_3['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos4")
plot_wordcloud(PosTweets_4['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos5")
plot_wordcloud(PosTweets_5['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Pos6")

pos_mask = np.array(Image.open('Neutral.png'))
plot_wordcloud(NeuTweets_0['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neu1")
plot_wordcloud(NeuTweets_1['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neu2")
plot_wordcloud(NeuTweets_2['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neu3")
plot_wordcloud(NeuTweets_3['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neu4")
plot_wordcloud(NeuTweets_4['Tweets_split'],color='white',max_font_size=100,title_size=30,title="Neu5")
