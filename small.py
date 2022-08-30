# Importing necessary libraries.
import streamlit as st
st.set_page_config(page_title="Monkeypox misinformation detector",
                   page_icon=":lion:",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)
import tweepy as tw
import textacy
from textacy import preprocessing
import emoji
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import datetime as dt
import time
import copy
import altair as alt


@st.experimental_singleton(show_spinner=False)
def load_model():
    """
    This function loads the fine-tuned HuggingFace model and caches
    it (using the experimental_singleton decorator) to improve
    computation times.

    Parameters: none.
    Returns: HuggingFace transformer model.
    """

    model = TFAutoModelForSequenceClassification.from_pretrained("smcrone/mpox-misinformation-detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        model.get_layer('dropout_73').rate = 0.2
    return model


@st.experimental_singleton(show_spinner=False)
def load_tokenizer():
    """
    This function loads a tokenizer for the transformer model and caches
    it (using the experimental_singleton decorator) to improve
    computation times.

    Parameters: none.
    Returns: tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained("smcrone/mpox-misinformation-detector",use_fast=False)
    return tokenizer


@st.experimental_singleton(show_spinner=False)
def load_client():
    """
    This function authenticates the Tweepy client and caches
    the object (using the experimental_singleton decorator) to 
    improve computation times.

    Parameters: none.
    Returns: Tweepy client.
    """

    bearer_token = st.secrets["bearer_token"]
    client = tw.Client(bearer_token,wait_on_rate_limit=True)
    return client


def dataframe_preprocessing(df_to_preprocess:pd.DataFrame):
    """
    The program overall collects tweet data at two junctures: firstly
    on provision of the initial tweet, and secondly if the classification
    of the initial tweet prompts a review of the user's other recent tweets.
    At both of these junctures certain preprocessing steps -- designed to
    increase the intelligibility of text inputs to the model -- are identical,
    so this function is designed to avoid the unnecessary repetition of this
    code. The function takes a Pandas DataFrame for preprocessing and returns
    the DataFrame, having executed certain preprocessing steps (e.g. removal
    of emojis, normalization of whitespace, removal of columns, etc.)

    Parameters: df_to_preprocess (DataFrame)
    Returns: df_to_preprocess (DataFrame)
    """

    # userlocation will not be in dataframe is user not supplied field. So, for time being, fill with none if it does not exist.
    # !!! note: we will likely NOT use userlocation, so can remove this bit of code in later versions!!!
    if 'userlocation' not in df_to_preprocess.columns:
        df_to_preprocess['userlocation'] = 'None'
    # Dropping redundant columns.
    df_to_preprocess = df_to_preprocess.drop(labels=['public_metrics', 'userpublic_metrics'], axis=1)
    # Stripping timezone info for export to Excel.
    df_to_preprocess['created_at'] = df_to_preprocess['created_at'].dt.tz_localize(None)
    df_to_preprocess['usercreated_at'] = df_to_preprocess['usercreated_at'].dt.tz_localize(None)
    # Replacing URLs and emojis; normalizing bullet points, whitespace, etc.
    for feature in ['text','userdescription','userlocation','userurl','username']:
        df_to_preprocess[feature] = df_to_preprocess[feature].fillna('None').apply(str)
        df_to_preprocess[feature] = df_to_preprocess[feature].apply(lambda x: textacy.preprocessing.replace.urls(text= x, repl= '_URL_'))
        df_to_preprocess[feature] = df_to_preprocess[feature].apply(lambda x: emoji.demojize(x))
        df_to_preprocess[feature] = df_to_preprocess[feature].apply(lambda x: textacy.preprocessing.normalize.bullet_points(text=x))
        df_to_preprocess[feature] = df_to_preprocess[feature].apply(lambda x: textacy.preprocessing.normalize.quotation_marks(text=x))
        df_to_preprocess[feature] = df_to_preprocess[feature].apply(lambda x: textacy.preprocessing.normalize.whitespace(text=x))
        df_to_preprocess[feature] = df_to_preprocess[feature].replace('\n', ' ', regex=True).replace('\r', '', regex=True)
    # Renaming columns (for greater model intelligibility).
    df_to_preprocess.rename(columns={"userverified": "user is verified",
                        "userurl": "user has url",
                        "userdescription": "user description",
                        "usercreated_at": "user created at",
                        "followers_count": "followers count",
                        "following_count": "following count",
                        "tweet_count": "tweet count",
                        "userlocation": "user location"},
                        inplace=True)
    # Making URL column binary.
    df_to_preprocess['user has url'].replace({'_URL_': 'True', "": 'False'}, inplace=True)
    # Adding some extra features.
    df_to_preprocess['years since account created'] = df_to_preprocess['created_at'].dt.year.astype('Int64') - df_to_preprocess['user created at'].dt.year.astype('Int64')
    df_to_preprocess['tweets per day'] = df_to_preprocess['tweet count']/((df_to_preprocess['created_at'] - df_to_preprocess['user created at']).dt.days)
    df_to_preprocess['follower to following ratio'] = df_to_preprocess['followers count']/(df_to_preprocess['following count']+1)
    # Returning processed DataFrame.
    return df_to_preprocess


def feature_concatenation(dataframe_to_concatenate:pd.DataFrame,features:list):
    """
    Our transformer model was fine-tuned on text input that combines
    a number of fields in a single string. This function performs
    the concatenation of these features, which in addition to dataframe
    preprocessing, is a necessary preprocessing step. The final dataframe
    consists of just two columns: one containing the concatenated text and
    the other containing the number of retweets that the tweet received
    (for use later on).

    Parameters:

    1. dataframe_to_concatenate (DataFrame): the df from which to take the features.
    2. features (list of str): the features to concatenate.

    Returns:

    1. finalDataFrame (DataFrame): the dataframe to be passed to the model.
    """

    # Make copy of dataframe consisting only of specified features.
    concatenated_dataframe = dataframe_to_concatenate[features].copy()
    # Concatenate chosen features.
    for i in features:
        concatenated_dataframe[i] = concatenated_dataframe[i].name + ": " + concatenated_dataframe[i].astype(str)
        concatenated_dataframe['combined'] = concatenated_dataframe[features].apply(lambda row: ' [SEP] '.join(row.values.astype(str)), axis=1)
    final_concatenated_dataframe = pd.DataFrame({"combined":concatenated_dataframe['combined'],"retweets":dataframe_to_concatenate['retweet_count']})
    # Return the final DataFrame.
    return final_concatenated_dataframe


def classify_tweets(dataframe_to_classify:pd.DataFrame):
    """
    This function takes a DataFrame of tweets which, having gone through
    the necessary preprocessing steps, is ready to classify. The function
    is called both for the initial classification of a single tweet and,
    where necessary, the superspreader analysis of the user's recent tweets.
    The function iterates through the DataFrame provided, tokenizing and
    classifying each tweet, and assigning it to one of two lists within a
    dictionary: 'goodPosts' (i.e. non-misleading posts) and 'badPosts (i.e.
    misleading posts). The function then returns the dictionary, which for
    each post includes the tweet itself, the predicted class, the confidence
    of the prediction, and the number of retweets received by the post.

    Parameters: dataframe_to_classify (DataFrame) -- the preprocessed
    DataFrame of tweet(s).

    Returns: tweet_dict (dict): a dictionary of classification results.
    """

    # Storing classification results in a dictionary with two keys.
    tweet_dict ={}
    tweet_dict['goodPosts'] = []
    tweet_dict['badPosts'] = []
    # Iterate through each tweet string in the DataFrame provided.
    for i in range(len(dataframe_to_classify['combined'])):
        # First, tokenize the tweet.
        tokenized_tweet = tokenizer(dataframe_to_classify['combined'].iloc[i],padding="max_length",truncation=True)
        # Next, convert tweet to a format that TensorFlow will accept.
        predict_dict = {}
        for x,y in tokenized_tweet.items():
            a = tf.convert_to_tensor(y, dtype=None, dtype_hint=None, name=None)
            b = tf.reshape(a,[1,512])
            predict_dict[x] = b
        # Call model to predict tweet.
        prediction = model(predict_dict,training=False)
        # Take pred. class and confidence in pred. class
        pred_class = np.argmax(np.array(tf.nn.softmax(prediction.logits)))
        pred_conf = np.max(np.array(tf.nn.softmax(prediction.logits)))
        # Construct a list of variables that we wish to store.
        seq_to_append = [dataframe_to_classify['combined'].iloc[i],pred_class,pred_conf,dataframe_to_classify['retweets'].iloc[i]]
        # Add list under appropriate dictionary key.
        if pred_class == 1:
            tweet_dict['badPosts'].append(seq_to_append)
        elif pred_class == 0:
            tweet_dict['goodPosts'].append(seq_to_append)
        else:
            print("Something went wrong.")
            return
    # Return the dictionary of results.
    return tweet_dict


def get_user_tweets(user_id:str, days_to_go_back:int, client:tw.Client):
    """
    If the initial tweet provided to the web app is classified as
    misleading, then relevant tweets from the user must be gathered
    in order to perform the superspreader calculation. This function
    supports this process by collecting relevant user tweets, undertaking
    the necessary preprocessing steps (with support from other functions),
    and classifying the tweets using the classify_tweets function. It
    then returns the dictionary of results produced by classify_tweets.

    Parameters:

    1. user_id (int|str): the user_id to be fed to Tweepy.
    2. days_to_go_back (int): how many days' tweets to investigate.
    3. client: the Tweepy client instantiated by load_client.

    Returns:

    1. user_tweets_classified (dict): model outputs for user tweets. 
    """

    # STAGE 1. FETCH USER TWEETS

    # Converting days_to_go_back into variables that can be fed to Tweepy.
    d = dt.datetime.today() - dt.timedelta(days=days_to_go_back)
    year = str(d.year)
    month = str(d.month)
    if len(month) == 1:
        month = '0'+month
    day = str(d.day)
    if len(day) == 1:
        day = '0'+day
    hour = str(d.hour)
    if len(hour) == 1:
        hour = '0'+hour
    # Gathering tweets from user.
    try:
        tweets_we_want_to_check = tw.Paginator(client.get_users_tweets,
                                               id = user_id,
                                               end_time=None,
                                               exclude=None,
                                               expansions=['author_id'],
                                               max_results=100,
                                               media_fields=None,
                                               pagination_token=None,
                                               place_fields=None,
                                               poll_fields=None,
                                               since_id=None,
                                               start_time='{}-{}-{}T{}:00:00Z'.format(year,month,day,hour),
                                               tweet_fields=['author_id','created_at','public_metrics','source'],
                                               until_id=None,
                                               user_fields=['created_at','description','location','public_metrics','url','verified'],
                                               user_auth=False,
                                               limit=500)
    except:
        return "Something went wrong whilst performing superspreader analysis."
    
    # STAGE 2. PREPROCESSING TWEET DATA

    # Parsing response data into an intermediate form.
    tweet_data_for_user = []
    user_data_for_user = []
    for page in tweets_we_want_to_check:
        # Converting each set of tweet fields into a dict and appending to list.
        for tweet in page.data:
            result = dict(tweet)
            tweet_data_for_user.append(result)
        # Converting each set of user fields into a dict and appending to list.
        for user in page.includes['users']:
            result = dict(user)
            user_data_for_user.append(result)
    # Adding user fields to tweet fields.
    for tweet in tweet_data_for_user:
        for user in user_data_for_user:
            for key, val in user.items():
                newKey = "user"+key
                tweet[newKey] = val
            break
    # Unpack and append any values that are dictionaries.
    for tweet in tweet_data_for_user:
        additional_values = {}
        for key, val in tweet.items():
            if type(val) == dict:
                for subkey, subval in val.items():
                    additional_values[subkey] = subval
        tweet.update(additional_values)
    # Create a Pandas DataFrame to store the data.
    user_df = pd.DataFrame(tweet_data_for_user)
    # Perform additional preprocessing using dedicated function.
    user_df = dataframe_preprocessing(user_df)
    # Drop non-monkeypox related rows.
    user_df['monkeypox'] = user_df['text'].str.contains('monkeypox|monkey pox|money pox', case=False, regex=True)
    user_df.drop(user_df[user_df.monkeypox == False].index, inplace=True)
    # Concatenating chosen features.
    concatenated_df = feature_concatenation(user_df,['text','user is verified'])
    
    # STAGE 3. CALLING CLASSIFIER AND RETURNING RESULTS

    # Calling classifier.
    classified_tweets = classify_tweets(concatenated_df)
    # Returning dictionary of classified tweets.
    return classified_tweets


def on_receipt_of_tweet_query(request:str,client:tw.Client):
    """
    This function defines what the app should do on receipt of a tweet
    URL / ID from the end-user. It performs the following steps:
    (i) formats the string submitted by the userinto a parsable form;
    (ii) fetches data for the tweet using Tweepy; (iii) performs some
    basic preprocessing on the data; (iv) calls dedicated preprocessing
    functions to finish preprocessing the data; (v) calls the classifier
    on the tweet; (vi) determines whether superspreader analysis is
    needed (i.e. if tweet is classed as misleading); (vii) if so,
    calls get_user_tweet function and calculates a superspreader score;
    (viii) returns a tuple of data for the application to display.

    Parameters:

    1. request (str): the URL or ID provided by the end-user.
    2. client: the Tweepy client instantiated by load_client.

    Returns:

    1. classified_tweet (dict): the metrics returned for the tweet by classify_tweets.
    2. spreader_score (float): where applicable, a metric representing the
    3. extent to which the user can be regarded as a superspreader of misinformation.
    4. tweet_text (str): the text of the tweet queried by the end-user.
    5. followers_count (int): the number of followers that the user has.
    6. classified_user_tweets (dict): where applicable, the metrics returned by
    7. get_user_tweets. 
    """

    # STAGE 1. FETCH DATA FOR REQUESTED TWEET

    # If URL is provided by the end-user, strip out the tweet ID.
    if '/' in request:
        request = request.split('/')[-1]
    # Collect tweet data -- interrupt if invalid input provided.
    tweet = client.get_tweets(ids=request,
                              expansions=['author_id'],
                              media_fields=None, 
                              place_fields=None,
                              poll_fields=None,
                              tweet_fields=['author_id','created_at','public_metrics','source'],
                              user_fields=['created_at','description','location','public_metrics','url','verified'],
                              user_auth=False)
    
    # STAGE 2. PREPROCESSING OF TWEET DATA

    # Create dictionaries out of the tweet and user data.
    for i in tweet.data:
        tweet_fields = dict(i)
    for i in tweet.includes['users']:
        user_fields = dict(i)
    # Add the data from the user dict to the tweet dict.
    for key, val in user_fields.items():
        newKey = "user"+key
        tweet_fields[newKey] = val
    # Unpack any values which are themselves dictionaries.
    additional_values = {}
    for key, val in tweet_fields.items():
        if type(val) == dict:
            for subkey, subval in val.items():
                additional_values[subkey] = subval
    tweet_fields.update(additional_values)
    # Convert everything to a DataFrame.
    tweet_df = pd.DataFrame(tweet_fields,index=[0])
    # Store the raw tweet text itself for later use.
    tweet_text = tweet_df['text'][0]
    # Store the followers count for later use.
    followers_count = tweet_df['followers_count'][0]
    # Preprocess the data using dedicated functions.
    tweet_df = dataframe_preprocessing(tweet_df)
    concatenated_tweet_df = feature_concatenation(tweet_df,['text','user is verified'])

    # STAGE 3. CALLING CLASSIFIER AND DETERMINING NEXT STEPS

    # Call the classifier on the tweet.
    classified_tweet = classify_tweets(concatenated_tweet_df)
    # If the tweet is misleading, call get_user_tweets and calculate
    # the user's superspreader score.
    if len(classified_tweet['badPosts']) == 1:
        # Fetch a dictionary of classified user tweets
        classified_user_tweets = get_user_tweets(tweet_df['userid'][0],14,client=client)
        # Calculate the total number of retweets for all misleading posts.
        retweets_total = 0
        for tweet in classified_user_tweets['badPosts']:
            retweets_total += tweet[-1]
        # Assign the p (post) value.
        p = (0.21 * len(classified_user_tweets['badPosts'])) ** 1.13
        # Assign the f (follower) value
        f = (0.25 * (np.log10(followers_count+1))) ** 4.73
        # Assign the r (retweet) value
        r = (1.04 * (np.log10(retweets_total+1))) ** 0.96
        # Calculate spreader_score and return a tuple of info.
        spreader_score = max(((1 - (1/(max(1,p+f+r))))*100),1)
        return classified_tweet, tweet_text, followers_count, classified_user_tweets, retweets_total, spreader_score,
    # Otherwise, if tweet is not misleading, return the same info
    # (excluding any superspreader related variables).
    elif len(classified_tweet['goodPosts']) == 1:
        return classified_tweet, tweet_text, followers_count, 0, 0, 0
    # Contingency in case an error should unexpectedly occur.
    else:
        raise Exception("Something went wrong whilst processing tweet data.")


def webpage():
    """
    This function structures the main page of the web app using the
    conventions of Streamlit. It begins by loading the model, the tokenizer
    and the Tweepy client using the functions dedicated to those tasks.
    Each of these elements is then cached. The remaining content that the
    function generates then depends mostly on the inputs provided by the
    end-user.

    Parameters: none.
    Returns: nothing.
    """

    # Create a container for displaying loading messages which will clear
    # once the tokenizer, Tweepy client and transformer model have loaded.
    loading_container = st.empty()
    with loading_container.container():
        global model
        model = load_model()
        global client
        client = load_client()
        global tokenizer
        tokenizer = load_tokenizer()
    loading_container.empty()

    # Write header content (e.g. banner image, title, description).
    st.image("monkeypox-small.jpg")
    st.title("Monkeypox misinformation detector")
    st.write("Use this tool to detect whether a tweet contains\
            monkeypox misinformation and assess the extent to which its\
            poster can be considered a misinformation superspreader.")

    st.sidebar.subheader("About")
    st.sidebar.write("This app has been developed using a\
                     [COVID-Twitter-BERT](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2)\
                     model fine-tuned on a monkeypox misinformation\
                     dataset. Users can learn more about the\
                     [model](https://www.bbc.co.uk/sport) on the\
                     HuggingFace model repository and can explore on\
                     Kaggle the [dataset](https://www.kaggle.com/datasets/stephencrone/monkeypox)\
                     on which the model was trained. Further\
                     [documentation](https://www.kaggle.com/datasets/stephencrone/monkeypox),\
                     as well as the source code for the app, can be\
                     found in the project's GitHub repository.")

    st.sidebar.subheader("Contact")
    st.sidebar.write("If you have any questions, comments or feedback\
                    regarding this app that are not answered by the\
                    supporting documentation for the underpinning\
                    dataset or transformer model, please feel free\
                    to contact the author at sgscrone@liverpool.ac.uk.")

    # Provide a text box for user to enter tweet ID / URL.
    tweet_to_check = st.text_input("Please provide a tweet URL or ID", key="name")
    # If the string provided by the user is empty, do nothing.
    if tweet_to_check != "":
        # Otherwise, if string is not empty, try fetching tweet using function.
        try:
            classified_tweet, tweet_text, followers_count, classified_user_tweets, retweets_total, spreader_score = on_receipt_of_tweet_query(tweet_to_check,client)
            st.markdown("""<hr style="height:1px;border:none;background-color:#a6a6a6; margin-top:16px; margin-bottom:20px;" /> """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            # In left column, present tweet text.
            col1.subheader("Tweet")
            tweet_text = textacy.preprocessing.normalize.whitespace(tweet_text)
            col1.markdown('<p style="background-color: #F0F2F6; padding: 8px 8px 8px 8px;">{}{}</p>'.format(tweet_text,type(tweet_text)),unsafe_allow_html=True)
            # In right column, present tweet classification.
            col2.subheader("Rating for this tweet")
            if len(classified_tweet['goodPosts']) != 0:
                # Format blue for not misinformation.
                col2.markdown('<p style="color:White; background-color: #1661AD; text-align: center; font-size: 20px;">Not misinformation</p>',unsafe_allow_html=True)
                col2.markdown('<p style="font-size: 40px; text-align: center;">{}</p>'.format(format(classified_tweet['goodPosts'][0][2],'.0%')), unsafe_allow_html=True)
                col2.markdown('<p style="text-align: center;">confidence level</p>', unsafe_allow_html=True)
            else:
                # Format red for misinformation.
                col2.markdown('<p style="color:White; background-color: #701B20; text-align: center; font-size: 20px;">Misinformation</p>',unsafe_allow_html=True)
                col2.markdown('<p style="font-size: 40px; text-align: center;">{}</p>'.format(format(classified_tweet['badPosts'][0][2],'.0%')), unsafe_allow_html=True)
                col2.markdown('<p style="text-align: center;">confidence level</p>', unsafe_allow_html=True)
                # Add additional container to display superspreader analysis.
                superspreader_container = st.container()
                superspreader_container.subheader("Superspreader rating for this user")
                # Plot the superspreader score as a bar chart.
                score_to_plot = pd.DataFrame({"classified_tweet":["score"],"spreader_score":[spreader_score]})
                bar = alt.Chart(score_to_plot).mark_bar().encode(alt.X('spreader_score:Q',scale=alt.Scale(domain=(0, 100)), axis=None), alt.Y('classified_tweet',axis=None)).properties(height=60)
                if spreader_score > 10:
                    label = bar.mark_text(align='right',baseline='middle', dx=-10, color='white', fontSize=20).encode(text=alt.Text("spreader_score:Q", format=",.0f"))
                else:
                    label = bar.mark_text(align='right',baseline='middle', dx=25, color='black', fontSize=20).encode(text=alt.Text("spreader_score:Q", format=",.0f"))
                x = bar+label
                x = x.configure_mark(color='#701B20')
                superspreader_container.altair_chart(x, use_container_width=True)
                # Display stats on which calculation was based.
                superspreader_container.write("Based on the user's **{:,} followers** and the following **{} tweet(s)** published over the last two weeks, which together received **{:,} retweet(s)**.".format(followers_count,len(classified_user_tweets['badPosts']),retweets_total))
                # And print offending tweets from user's recent history.
                for i in range(len(classified_user_tweets['badPosts'])):
                    recent_tweet = classified_user_tweets['badPosts'][i][0]
                    recent_tweet = recent_tweet.split('text:')[-1]
                    recent_tweet = recent_tweet.split('[SEP]')[0]
                    superspreader_container.markdown('<p style="background-color: #F0F2F6; padding: 8px 8px 8px 8px;">{}</p>'.format(recent_tweet),unsafe_allow_html=True)
        except:
            st.error("Could not retrieve information for tweet. Please ensure you are supplying a valid tweet ID or URL.")
    st.markdown("""<hr style="height:1px;border:none;background-color:#a6a6a6; margin-top:16px; margin-bottom:20px;" /> """, unsafe_allow_html=True)

if __name__ == "__main__":
    webpage()

