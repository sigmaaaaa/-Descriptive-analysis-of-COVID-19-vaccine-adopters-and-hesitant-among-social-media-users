import pandas as pd
import tweepy
import datetime
  
  
# function to display data of each tweet
def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Description:{ith_tweet[1]}")
    print(f"Location:{ith_tweet[2]}")
    print(f"Following Count:{ith_tweet[3]}")
    print(f"Follower Count:{ith_tweet[4]}")
    print(f"Total Tweets:{ith_tweet[5]}")
    print(f"Retweet Count:{ith_tweet[6]}")
    print(f"Tweet Text:{ith_tweet[7]}")
    print(f"Hashtags Used:{ith_tweet[8]}")
  
  
def scrape(words, date_since, date_to,numtweet):
      
    db = pd.DataFrame(columns=['username', 'description', 'location', 'following',
                               'followers', 'totaltweets', 'retweetcount', 'publish_time','text', 'hashtags'])
    
    #places = api.search_geo(query="USA",granularity="country")
    #place_id = places[0].id
    #place_id = api.geo_id('6416b8512febefc9')

    #tweets = tweepy.Cursor(api.search_tweets, q=words and ("place:%s" % place_id), lang="en",
    #                       since=date_since, tweet_mode='extended').items(numtweet)
    #search_string = '%20OR%20'.join(words) 
    #search_place_string = words + '%20place:' + place_id.id
    tweets = tweepy.Cursor(api.search_full_archive, label='CORMSIS42', query='#CovidVaccine lang:en place_country:GB',
                       fromDate=date_since,toDate=date_to).items(numtweet)
     

    list_tweets = [tweet for tweet in tweets]
      
    # Counter to maintain Tweet Count
    i = 1  
      
    for tweet in list_tweets:
        username = tweet.user.screen_name
        description = tweet.user.description
        location = tweet.user.location
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        publish_time = tweet.created_at
        hashtags = tweet.entities['hashtags']
          
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
          
        # Appending all the extracted information in the DataFrame
        ith_tweet = [username, description, location, following,
                     followers, totaltweets, retweetcount,publish_time, text, hashtext]
        db.loc[len(db)] = ith_tweet
          
        # Function call to print tweet data on screen
        printtweetdata(i, ith_tweet)
        i = i+1
    return db
  
  
if __name__ == '__main__':
      
    consumer_key= 'prxz9V3yjut2ylOg0hEZFMnS4'
    consumer_secret= 'n0IGu2fRd1R4WjQa0Opw6oloRMjtQjZcfvybjpBh3wC6kE6sAl'
    access_key = "1424765228768714755-52LiswrEsvgP6tZug3RdCum5wznydf"
    access_secret = "4nDDDEjKz3pRJAu8cI3ScoNUX6JgfoOHaIhJDiCX9BdFf"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
      
    words = 'CovidVaccine'
    date_since = '202012010000'
    date_to = '202012020000'      
    # number of tweets you want to extract in one run
    numtweet = 10 
    dbTTL = scrape(words, date_since, date_to,numtweet)
    #dbTTL = dbTTL.append(db_2)
    print('Scraping has completed!')
    
    
filename = 'scraped_tweets_UK_hashtag1_test2.csv'
db.to_csv(filename)