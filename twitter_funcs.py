import numpy as np
import pandas as pd

import pickle
import tweepy

consumer_key = 'zI3IWu3om7eMpqLNyPtTGGDYS'
consumer_secret = 'jSNks0sW6miXnXRXivwULkT4H254tQHuhuN9TMgaLdsAXhJC8p'
access_token_key = '725285298922356736-r2w8F3HEU1DgzHjF44N59rSXcBiMfzp'
access_token_secret = 'dKZIuPtUeASVlXp8nPNR5NGVwYaErWJobwZIgCpPB0FFa'

# Get fully-trained XGBoostClassifier model
with open('model_pkl', 'rb') as read_file:
    model = pickle.load(read_file)

# Set up connection to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)


def get_user_features(screen_name):

    try:

        # Get user information from screen name
        user = api.get_user(screen_name)
        print("get user", screen_name)

        # account features to return for prediction

        verified = user.verified
        default_profile = user.default_profile
        default_profile_image = user.default_profile_image
        favourites_count = user.favourites_count
        followers_count = user.followers_count
        friends_count = user.friends_count
        statuses_count = user.statuses_count

        # manufactured features

        network = np.round(np.log(1 + friends_count)
                           * np.log(1 + followers_count), 3)
        tweet_to_followers = np.round(
            np.log(1 + statuses_count) * np.log(1 + followers_count), 3)

        # organizing list to be returned
        account_features = [verified, default_profile, default_profile_image,
                            favourites_count, followers_count, friends_count, statuses_count,
                            network, tweet_to_followers]

    except:
        return 'User not found'
    return account_features if len(account_features) == 14 else f'User not found'


def bot_or_not(twitter_handle):

    user_features = get_user_features(twitter_handle)
    print("bot or not:", twitter_handle)

    if user_features == 'User not found':
        return 'User not found'

    else:
        # features for model
        features = ['verified', 'default_profile', 'default_profile_image',
                    'favourites_count', 'followers_count', 'friends_count', 'statuses_count',
                    'network', 'tweet_to_followers']

        # creates df for model.predict() format
        user_df = pd.DataFrame(np.matrix(user_features), columns=features)

        prediction = model.predict(user_df)[0]

        return "Bot" if prediction == 1 else "Not a bot"


def bot_proba(twitter_handle):

    user_features = get_user_features(twitter_handle)
    print("bot probability:", user_features)

    if user_features == 'User not found':
        return 'User not found'
    else:
        user = np.matrix(user_features)
        proba = np.round(model.predict_proba(user)[:, 1][0] * 100, 2)
        return proba
