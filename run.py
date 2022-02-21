import pickle

from flask import Flask, render_template, request
from twitter_funcs import bot_proba, get_user_features

app = Flask(__name__, static_url_path='/static')


def bot_likelihood(prob):
    if prob < 20:
        return '<span class="has-text-info">Not a bot</span>'
    elif prob < 35:
        return '<span class="has-text-info">Likely not a bot</span>'
    elif prob < 50:
        return '<span class="has-text-info">Probably not a bot</span>'
    elif prob < 60:
        return '<span class="has-text-warning">Maybe a bot</span>'
    elif prob < 80:
        return '<span class="has-text-warning">Likely a bot</span>'
    else:
        return '<span class="has-text-danger">Bot</span>'


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/', methods=['GET'])
def home():
    return render_template('Auth.py')


@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    handle = request.form['handle']
    print("make prediction:", handle)

    # make predictions with model from twitter_funcs
    user_lookup_message = f'Prediction for @{handle}'

    result = get_user_features(handle)
    print("result:", result)

    if result == 'User not found':
        prediction = [f'User @{handle} not found', '']

    else:
        prediction = [bot_likelihood(bot_proba(handle)),
                      f'Probability of being a bot: {bot_proba(handle)}%']

    return render_template('index.html', prediction=prediction[0], probability=prediction[1],
                           user_lookup_message=user_lookup_message)


if __name__ == '__main__':
    app.run()
