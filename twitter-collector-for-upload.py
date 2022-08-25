# Importing modules
import pandas as pd
import textacy
from textacy import preprocessing
import tweepy as tw
import os
import emoji
import datetime as dt

def create_and_save_twitter_dataframe(bearer_token:str,
                                      query:str,
                                      end_time:str,
                                      start_time:str,
                                      expansions:list,
                                      tweet_fields:list,
                                      user_fields:list):
    """
    This function uses Tweepy to return a dataframe of tweets matching
    the user's search query. It then exports the dataframe to an Excel
    file in a subdirectory of the location in which this script is stored.
    Given that the data-gathering functionality of the function is
    provided by Tweepy, users might find the Tweepy documentation helpful,
    e.g., for any queries regarding the search parameters of the function.
    Likewise, further information on expansions (i.e. the fields returned
    by the Twitter API in addition to the standard fields) can be found
    on the Twitter developer portal.

    Parameters:

    1. bearer_token (str): the user's bearer token for v2 of the Twitter API.
    2. query (str): the query fed to Tweepy's search_recent_tweets function.
    3. end_time (str): end time for search in YYYY-MM-DDTHH:MM:SSZ format.
    4. start_time (str): start time for the search in the same format.
    5. expansions (list of str): expansions to request from Twitter API.
    6. tweet_fields (list of str): the tweet-related fields to request.
    7. user_fields (list of str): the user-related fields to request.

    Returns:
    
    Nothing.
    """

    # STAGE 1. GATHERING THE DATA

    # Begin by creating directory for dataframe export.
    os.makedirs('folder/subfolder', exist_ok=True)
    # Authenticating with Twitter.
    bearer_token = bearer_token
    client = tw.Client(bearer_token,wait_on_rate_limit=True)
    # Gathering tweets.
    responses = tw.Paginator(client.search_recent_tweets,
        # Excluding retweets; looking only at English language tweets.   
        query=query,
        end_time=end_time,
        # Getting user-related fields for each tweet.
        expansions=expansions,
        max_results=100,
        media_fields=None,
        next_token=None,
        place_fields=None,
        poll_fields=None,
        since_id=None,
        sort_order=None,
        # Setting start date.
        start_time=start_time,
        # Stipulating requested.
        tweet_fields=tweet_fields,
        until_id=None,
        user_fields=user_fields,
        user_auth=False,
        limit=500)
    
    # STAGE 2: CONVERTING TO DATAFRAME
    
    # We will convert our data into a form that is easier to manipulate.
    # First, create two lists: one to store tweet data; another for user data.
    tweetData = []
    userData = []
    # Then create dicts for each set of user / tweet fields and append each dict
    # to their respective list.
    for page in responses:
        for tweet in page.data:
            result = dict(tweet)
            tweetData.append(result)
        for user in page.includes['users']:
            result = dict(user)
            userData.append(result)
    # Combine the user fields with the tweet fields.
    for tweet in tweetData:
        for user in userData:
            if tweet['author_id'] == user['id']:
                for key, val in user.items():
                    newKey = "user"+key
                    tweet[newKey] = val
                break
    # Unpack any features that are themselves dictionaries.
    for tweet in tweetData:
        additionalValues = {}
        for key, val in tweet.items():
            if type(val) == dict:
                for subkey, subval in val.items():
                    additionalValues[subkey] = subval
        # And create new fields for each sub-field.
        tweet.update(additionalValues)
    # Creating a Pandas dataframe to store the data.
    df = pd.DataFrame(tweetData)

    # STAGE 3. FORMATTING DATAFRAME

    # Dropping redundant columns.
    df = df.drop(labels=['public_metrics', 'userpublic_metrics'], axis=1)
    # Stripping timezone info for export to Excel.
    df['created_at'] = df['created_at'].dt.tz_localize(None)
    df['usercreated_at'] = df['usercreated_at'].dt.tz_localize(None)
    # Replacing empty cells, URLs, emojis, line-breaks, etc.
    # Normalizing white-space, bullet-points, quotation marks, etc.
    for feature in ['text','userdescription','userlocation','userurl','username']:
        df[feature] = df[feature].fillna('None').apply(str)
        df[feature] = df[feature].apply(lambda x: textacy.preprocessing.replace.urls(text= x, repl= '_URL_'))
        df[feature] = df[feature].apply(lambda x: emoji.demojize(x))
        df[feature] = df[feature].apply(lambda x: textacy.preprocessing.normalize.bullet_points(text=x))
        df[feature] = df[feature].apply(lambda x: textacy.preprocessing.normalize.quotation_marks(text=x))
        df[feature] = df[feature].apply(lambda x: textacy.preprocessing.normalize.whitespace(text=x))
        df[feature] = df[feature].replace('\n', ' ', regex=True).replace('\r', '', regex=True)
    # Renaming columns that we may feed as text to transformer model
    df.rename(columns={"userverified": "user is verified",
                        "userurl": "user has url",
                        "userdescription": "user description",
                        "usercreated_at": "user created at",
                        "followers_count": "followers count",
                        "following_count": "following count",
                        "tweet_count": "tweet count",
                        "userlocation": "user location"},
                        inplace=True)
    # Converting URL column to Boolean values.
    df['user has url'].replace({'_URL_': 'TRUE', '': 'FALSE'}, inplace=True)
    
    # STAGE 4. Exporting DataFrame to Excel

    df.to_excel('folder/subfolder/monkeypox-delete2.xlsx')



if __name__ == "__main__":
    create_and_save_twitter_dataframe(bearer_token= "",
                                      query = "(monkeypox OR \"monkey pox\" OR \"moneypox\") -is:retweet lang:en",
                                      end_time='2022-08-15T02:00:00Z',
                                      start_time='2022-08-15T00:00:00Z',
                                      expansions = ['author_id'],
                                      tweet_fields = ['author_id','created_at','public_metrics','source'],
                                      user_fields = ['created_at','description','location','public_metrics','url','verified'])










