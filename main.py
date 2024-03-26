from flask import Flask, render_template
import csv
import os

#Machine Learning Imports
import pandas as pd
import numpy as np
import datetime

# required machine learning packages
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

app = Flask(__name__)


@app.route('/login')
def new_tab():
    return render_template('login.html')

@app.route('/home')
def index():
    return render_template('index.html')


# Machine Learning Model Predictions
df = pd.read_csv("Data/data_preprocessed (1).csv")

# training and testing data
train = df.copy()
test = df.copy()
train = train.loc[train['schedule_season'] < 2016]
test = test.loc[test['schedule_season'] > 2015]

X_train = train[['schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite','team_away_current_win_pct']]
y_train = train['result']

X_test = test[['schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite','team_away_current_win_pct']]
y_test = test['result']

# Train model on data
boost = xgb.XGBClassifier()
dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy')
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='soft')

vote = VotingClassifier(estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='soft')
model = vote.fit(X_train, y_train)


# Make Predictions on our database
this_week = pd.read_csv("Data/new.csv")
this_week.info()

# predict probabilities
week_probabilities = model.predict_proba(this_week)[:,1]

# predict home team win or lose (0/1)
week_winner = model.predict(this_week)

print(week_probabilities)
print(week_winner)

#Compute odds
odds = []
for ele in week_probabilities:
    odds.append(1/ele - 1)

# Calculate the daily returns based on the model's predictions
returns = (week_probabilities - 0.5) * 2  # Scale the predicted probabilities to [-1, 1]

# Calculate the daily log returns
daily_log_returns = np.log(1 + returns)

# Calculate the Sharpe Ratio based on your historical data
historical_sharpe_ratio = np.mean(daily_log_returns) / np.std(daily_log_returns)

# Define your risk-free rate
risk_free_rate = 0.02

# Calculate the Kelly Fraction
kelly_fractions = [(p * (b - 1) - (1 - p)) / (b - 1) for p, b in zip(week_probabilities, odds)]

# Adjust the Kelly Fractions to a reasonable range (e.g., 0 to 1)
kelly_fractions = [max(0, min(1, kf)) for kf in kelly_fractions]

# Your current bankroll
bankroll = 10000  # hypothetical

# Calculate the bet sizes based on the Kelly Fractions and your bankroll
bet_sizes = [risk_free_rate * kf * bankroll for kf in kelly_fractions]
print(bet_sizes)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
