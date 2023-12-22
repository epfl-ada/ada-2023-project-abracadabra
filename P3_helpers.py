import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
from tqdm import tqdm

import ast
import json
import os
import re

import spacy
import pyLDAvis.gensim_models
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

from sklearn.neighbors import KernelDensity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold

import mwparserfromhell
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
STOPWORDS = list(STOPWORDS)
from string import punctuation as PUNCTUATION
FINE_TUNE_STOPWORDS = ["--", "n\'t", "\'s", "\'\'", "font", "``", "color=", "style=", "span", "s", "\'m"]

import networkx as nx
import community

import warnings
warnings.filterwarnings('ignore')

########## DATA HANDLING FUNCTIONS ##########

def data_parsing(path = 'wiki-RfA.txt'):
    """ Parse the data from the file at the given path and store it in a DataFrame.

    Args:
        path (str): Path to the file containing the data
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    # Read the file into a list of lines
    with open(path, 'r', encoding = 'utf8') as file:
        lines = file.readlines()

    # Create a list of dictionaries, where each dictionary represents a record
    df = []
    current_entry = {}

    # Iterate through each line, current_entry = one log entry with all columns, df = list of all votee/voter pairs
    for line in lines:
        line = line.strip()
        if line:
            key, value = line.split(':', 1)
            current_entry[key] = value
        else:
            df.append(current_entry)
            current_entry = {}

    # Append  last record
    if current_entry:
        df.append(current_entry)

    # Convert into DataFrame and store in csv
    df = pd.DataFrame(df)
    df.columns = ['Source', 'Target', 'Vote', 'Results', 'Year', 'Date', 'Comment']
    
    return df

def handle_inconsistencies(df):
    """ Handle inconsistencies in the data (NaN values, mispelled dates, missing values, duplicates, etc.)

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes

    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the inconsistencies handled
    """
    # replace field that's entirely space (or empty) with NaN (the case for some Source, Date and Comment)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    #handle NaN values in Comment for vectorization
    df.Comment = df.Comment.replace(np.nan, None)

    # replace inconsistent date
    df['Date'] = df['Date'].str.replace('Julu ', 'July ')
    df['Date'] = df['Date'].str.replace('Janry ', 'January ')
    df['Date'] = df['Date'].str.replace('Mya ', 'May ')
    df['Date'] = df['Date'].str.replace('Jan ', 'January ')
    df['Date'] = df['Date'].str.replace('Feb ', 'February ')
    df['Date'] = df['Date'].str.replace('Mar ', 'March ')
    df['Date'] = df['Date'].str.replace('Apr ', 'April ')
    df['Date'] = df['Date'].str.replace('Jun ', 'June ')
    df['Date'] = df['Date'].str.replace('Jul ', 'July ')
    df['Date'] = df['Date'].str.replace('Aug ', 'August ')
    df['Date'] = df['Date'].str.replace('Sep ', 'September ')
    df['Date'] = df['Date'].str.replace('Oct ', 'October ')
    df['Date'] = df['Date'].str.replace('Nov ', 'November ')
    df['Date'] = df['Date'].str.replace('Dec ', 'December ')

    # Convert into the right type and format
    df['Date'] = pd.to_datetime(df['Date'], format='%H:%M, %d %B %Y', errors='coerce')
    df['Vote'] = df['Vote'].astype(int)
    df['Results'] = df['Results'].astype(int)
    df['Year'] = df['Year'].astype(int)

    # Drop rows with missing values in Source column
    df.dropna(subset=['Source'], inplace=True)  


    ### Remove duplicates ###
    # Select target with a significant number of duplicates (by manually checking the data we found that 6 discriminate perfectly between users with actual duplicates and users with only missing data or basic comments)
    target_with_duplicates = df[df.duplicated(['Source', 'Target', 'Comment', 'Date'], keep=False)].groupby('Target').size() >= 6
    target_with_duplicates = target_with_duplicates[target_with_duplicates].index
    duplicates = df[df.Target.isin(target_with_duplicates) & df.duplicated(['Source', 'Target', 'Comment', 'Date'], keep=False)].sort_values(by=['Target', 'Source', 'Date'])
    #Remove all duplicates from the dataframe df
    df = df.drop(duplicates.index)

    # Deal with duplicates that have different results 
    perc_vote = (duplicates.groupby(['Target', 'Year']).Vote.value_counts(normalize=True) * 100).unstack(level='Vote')
    perc_vote['Result'] = perc_vote.apply(lambda x: 1 if x[1] >= 70 else -1, axis=1, result_type='reduce')
    # Replace results in duplicates with results in perc_vote
    duplicates['Results'] = duplicates.apply(lambda x: perc_vote.loc[(x['Target'], x['Year'])]['Result'], axis=1)

    # Deal with duplicates that have different years
    correct_year = pd.DataFrame({'Year': duplicates.Date.dt.year, 'Target': duplicates.Target})
    # Replace nan values in Year with most common year for each Target (some Dates are missing)
    correct_year.Year = correct_year.groupby('Target').Year.transform(lambda x: x.fillna(x.mode()[0]))
    # Replace years in duplicates with years in correct_year
    duplicates = duplicates.drop(columns='Year').join(correct_year.Year)

    # Drop the duplicate rows 
    duplicates.drop_duplicates(keep='first', inplace=True)
    # Add the duplicates to df
    df = pd.concat([df, duplicates]).sort_index()

    # Deal with duplicates that have different Vote
    double_vote = df[df.duplicated(['Source', 'Target', 'Comment', 'Date'], keep=False) & df.Date.notnull() & (df.Vote == 0)]
    # Drop the double_vote rows
    df.drop(double_vote.index, inplace=True)

    return df

def get_dataframe(path = 'wiki-RfA.txt'):
    """ Parse the data from the file at the given path, handle inconsistencies and return the resulting DataFrame.

    Args:
        path (str): Path to the file containing the data
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    df = data_parsing(path)
    df = handle_inconsistencies(df)
    return df
    



########## TIME SERIES GENERATION FUNCTIONS ##########

def get_timeserie_df(df):
    """ Compute the voting time of each vote and the corresponding election round.

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        
    Returns:
        df_timeserie (pd.DataFrame): Dataframe containing the data of the votes with the voting time and the election round
    """
    # Remove the 2003 election round that represents only 0.1% of the data and which voting time behavior is different from the other rounds
    df = df[df['Year'] != 2003] # or df = df[df['Year'] != '2003']

    # Remove NaN values in the date column
    df = df[~df['Date'].isna()]

    # define voting time as the difference between the date of the vote and the date of the first vote of for each target
    voting_time = (df.groupby('Target').Date.apply(lambda x: x - x.min()).dt.total_seconds()/3600).rename('Voting_time')
    
    # Find the local minima of the kernel density estimation of the voting time
    lower_bound = 100
    upper_bound = 1000
    log_voting_time = np.log10(voting_time[(voting_time > lower_bound) & (voting_time < upper_bound)])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(log_voting_time.values.reshape(-1, 1))
    kde_x = np.linspace(np.min(log_voting_time), np.max(log_voting_time), 1000)
    kde_y = np.exp(kde.score_samples(kde_x.reshape(-1, 1)))
    deriv_kde_sign = np.sign(np.diff(kde_y))
    local_mins = kde_x[np.append((np.roll(deriv_kde_sign, 1) - deriv_kde_sign) != 0, False)]
    y_mins = kde_y[np.append((np.roll(deriv_kde_sign, 1) - deriv_kde_sign) != 0, False)]
    round_threshold = local_mins[(y_mins < 0.1)][0]

    # Create a new dataframe completing the original dataframe with the voting time
    df_timeserie = df.join(voting_time.droplevel(0))

    # Compute the round number of each vote
    rounds = df_timeserie.groupby('Target').apply(lambda x: compute_rounds(x, round_threshold)).rename('Round')
    df_timeserie = df_timeserie.join(rounds.droplevel(0))

    # Use the round number to compute the voting time and the vote number in each round (i.e. the time between the current vote and the first vote of the round)
    Voting_time_round = df_timeserie.groupby(['Target', 'Round']).Voting_time.apply(lambda x: x - x.min())
    df_timeserie = df_timeserie.drop(columns='Voting_time').join(Voting_time_round.droplevel([0,1]))
    Vote_number_round = (df_timeserie.sort_values('Voting_time').groupby(['Target', 'Round']).cumcount() + 1).rename('Vote_number')
    df_timeserie = df_timeserie.join(Vote_number_round)

    return df_timeserie

# Start a new round each time the interval between the current voting and the first voting time of the round is greater than the round threshold
def compute_rounds(data, round_threshold):
    """ Separate the votes of a target into rounds of elections based on threshold defined by data exploration.

    Args:
        data (pd.DataFrame): Dataframe containing the data of the votes for one target
        round_threshold (int): Threshold to define the end of a round of election (in hours). Found using kernel density estimation of the voting time

    Returns:
        rounds (pd.Series): Series containing the round number of each vote
    """
    data = data.sort_values('Voting_time')
    voting_time = data.Voting_time.values
    rounds = np.array([1]) # first vote on a target is always part of the first round
    for i in range(1,len(voting_time)):

        # If the candidate is elected, we stop the round but check if there is no other vote after more than 4 month
        # (what we consider as a new election for a candidate that may have lost its admin rights)
        if data.Results.values[i] == 1 and voting_time.max()-voting_time[i] < 24*30*4:
            round_number = rounds[-1] + (voting_time[i] > round_threshold)
            rounds = np.append(rounds, round_number*np.ones(len(voting_time)-i).astype(int))
            break

        # If the voting time is greater than the round threshold, the candidate is not elected 
        # and more than 1 day has passed since the last vote, we start a new round
        elif (voting_time[i] > round_threshold and data.Results.values[i] != 1 and voting_time[i]-voting_time[i-1]>=24) or \
            (data.Results.values[i] == 1 and (voting_time[i] > 24*30*4 or data.Results.values[i-1] != 1)):
            rounds = np.append(rounds, rounds[-1] + 1)
            voting_time = voting_time - voting_time[i]

        # In some cases, the first vote of a round is far earlier than the other votes of the round
        elif len(rounds) > 1 and rounds[-2] != rounds[-1]:
            rounds = np.append(rounds, rounds[-1])
            voting_time = voting_time - voting_time[i]
        
        else:
            rounds = np.append(rounds, rounds[-1])
            
    rounds = pd.Series(rounds, index=data.index).astype(int)
    return rounds


########## VOTE EVOLUTION FUNCTIONS ##########
def pdf_voting_time(df, plot=None):
    """ Plot the probability density function of the voting time

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    data = df[(df['Voting_time'] != 0) & (df['Voting_time'] < 24*8)]
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(data=data, x='Voting_time', ax=ax, bins=100, stat='percent', log_scale=(False, False), hue='Year', palette='CMRmap', multiple='stack')
    ax.set_title('Histogram of voting time')
    ax.set_xlabel('Voting time in the round (hours)')
    ax.set_ylabel('Percentage of votes')
    ax.set_xlim(0, 24*8)
    if plot is None:
        plt.savefig('Figures/pdf_voting_time.png', dpi=300)
        plt.show()
    else:
        plt.savefig('Figures/pdf_voting_time_' + plot + '.png', dpi=300)
        plt.close()

def cdf_voting_time(df, plot=None):
    """ Plot the cumulative distribution function of the voting time

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    data = df[(df['Voting_time'] != 0) & (df['Voting_time'] < 24*8)]
    fig, ax = plt.subplots(figsize=(10,4))
    sns.ecdfplot(data=data, x='Voting_time', ax=ax, stat='percent', log_scale=(True, True), hue='Year', palette='CMRmap', complementary=True)
    ax.set_title('CCDF of voting time')
    ax.set_xlabel('Voting time in the round (hours)')
    ax.set_ylabel('Percentage of votes')
    ax.set_xlim(np.min(data.Voting_time), np.min(data.groupby('Year').Voting_time.max()))
    if plot is None:
        plt.savefig('Figures/cdf_voting_time.png', dpi=300)
        plt.show()
    else:
        plt.savefig('Figures/cdf_voting_time_' + plot + '.png', dpi=300)
        plt.close()

def get_progressive_mean(df, sent=False):
    """ Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the progressive mean of the votes in each round
    """
    # Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)
    progressive_mean = df.groupby(['Target', 'Round']).apply(lambda x: x.sort_values('Voting_time').Vote.cumsum() / np.arange(1, len(x)+1)).rename('progressive_mean')
    df = df.join(progressive_mean.droplevel([0,1]))
    if sent:
        progressive_sentiment = df.groupby(['Target', 'Round']).apply(lambda x: x.sort_values('Voting_time').sent_score.cumsum() / np.arange(1, len(x)+1)).rename('progressive_sentiment')
        df = df.join(progressive_sentiment.droplevel([0,1]))

    #Convert time in timedelta
    df.Voting_time = pd.to_timedelta(df.Voting_time, unit='h')
    df.sort_values('Voting_time', inplace=True)

    return df

def plot_vote_evolution(data, x, mean_col = 'center', var_cols = ['lower', 'upper']):
    """ Plot the evolution of the votes for a given target and a given round of election

    Args:
        data (pd.DataFrame): Dataframe containing the data of the votes
        x (str): Name of the column to use as x-axis
        mean_col (str): Name of the column to use as mean
        var_cols (list): List of the names of the columns to use as variance
        
    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot): Plot of the evolution of the votes for a given target and a given round of election
    """
    # Plot the evolution of the votes
    data = data[data.Voting_time < 24*8]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=x, y=mean_col, data=data, hue='Results', palette='tab10', ax=ax)
    ax.fill_between(data[data.Results == -1][x], data[data.Results == -1][var_cols[0]], data[data.Results == -1][var_cols[1]], alpha=0.2, color='tab:blue')
    ax.fill_between(data[data.Results == 1][x], data[data.Results == 1][var_cols[0]], data[data.Results == 1][var_cols[1]], alpha=0.2, color='tab:orange')
    ax.set_ylim(-1, 1.01)
    ax.set_xlim(data[x].min(), data[x].max())
    ax.set_ylabel('Progressive mean of the votes')
    if x == 'Voting_time': 
        ax.set_xlabel('Time (hours)')
    elif x == 'Vote_number': 
        ax.set_xlabel('Number of votes')
    ax.legend(handles=[plt.Line2D([0], [0], color='tab:blue', lw=4, label='Rejected'),
                        plt.Line2D([0], [0], color='tab:orange', lw=4, label='Elected')])
    return ax
    
def rolling_average(data, window_size='1h', on='Voting_time'):
    """ Compute the rolling average of the votes in each round (i.e. the mean of the votes in a given time window)

    Args:
        data (pd.DataFrame): Dataframe containing the data of the votes
        window_size (str): Size of the time window
        
    Returns:
        data (pd.DataFrame): Dataframe containing the data of the votes with the rolling average of the votes in each round
    """
    # Sort and compute moving average of each column 
    if on is None:
        data = data.sort_index().reset_index()
    else:
        data = data.sort_values(on).reset_index()
    data = data.groupby('Results').rolling(window_size, on=on, min_periods=1).mean().reset_index(level='Results')
    data.Voting_time = time_to_float(data.Voting_time)
    data.columns = ['Results', 'Voting_time', 'lower', 'center', 'upper']
    return data

def time_to_float(voting_time):
    """ Convert a time in the format hh:mm:ss to a float number of hours

    Args:
        voting_time (pd.Series): Series of 'Voting_time' in timedelta format

    Returns:
        voting_time (pd.Series): Series containing the column 'Voting_time' in float format
    """
    voting_time = voting_time.dt.total_seconds() / 3600
    return voting_time

def add_voting_time(grouper, stats, on):
    """ Add the voting time to the stats dataframe in the right format

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes
        stats (pd.DataFrame): Dataframe containing the stats of the votes in each round
        on (str): Name of the column to use as x-axis

    Returns:
        stats (pd.DataFrame): Dataframe containing the stats of the votes in each round with the voting time in the right format
    """
    if on == 'Vote_number':
        stats = stats.join(grouper.Voting_time.min(), on=['Results', on])

    stats.Voting_time = time_to_float(stats.Voting_time)
    return stats

def get_quartiles(grouper, on, feature='progressive_mean'):
    """ Compute the median, first and last quartile of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes 
        
    Returns:
        quartiles (pd.DataFrame): Dataframe containing the median, first and last quartile of the votes in each round
    """
    # Compute the median, first and last quartile of the votes in each round
    quartiles = grouper[feature].quantile([0.25, 0.5, 0.75]).unstack(level=2).reset_index()
    quartiles.rename(columns={0.25: 'lower', 0.5: 'center', 0.75: 'upper'}, inplace=True)
    quartiles = add_voting_time(grouper, quartiles, on)
    return quartiles

def get_confidence_interval(grouper, on, feature='progressive_mean'):
    """ Compute the mean and 95% confidence interval of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes        
    
    Returns:
        ci (pd.DataFrame): Dataframe containing the mean and 95% confidence interval of the votes in each round
    """
    # Compute the mean and 95% confidence interval of the votes in each round
    ci = grouper[feature].agg(['mean', stats.sem]).reset_index()
    ci['lower'] = ci['mean'] - 1.96 * ci['sem']
    ci['upper'] = ci['mean'] + 1.96 * ci['sem']
    ci.rename(columns={'mean': 'center'}, inplace=True)
    ci = add_voting_time(grouper, ci, on)
    return ci

# Scatter plot of the progressive mean by voting time and vote number
def plot_time_distribution(df, x, feature = 'progressive_mean', plot=None):
    """ Plot the distribution of the progressive mean over time

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        x (str): Name of the column to use as x-axis
    """
    data = df[df.Voting_time < pd.Timedelta('8 days')]
    data.Voting_time = time_to_float(data.Voting_time)
    fig, axes = plt.subplots(1,2,figsize=(20, 6))
    # Plot the histogram in 2D with log scale colorbar
    sns.histplot(data=data[data.Results==-1], x=x, y=feature, ax=axes[0], color='tab:blue', cbar=True, norm=LogNorm(), vmin=None, vmax=None, bins=100)
    sns.histplot(data=data[data.Results==1], x=x, y=feature, ax=axes[1], color='tab:orange', cbar=True, norm=LogNorm(), vmin=None, vmax=None, bins=100)
    axes[0].set_ylim(-1.01, 1.01)
    axes[1].set_ylim(-1.01, 1.01)
    axes[0].set_xlim(data[x].min(), data[x].max())
    axes[1].set_xlim(data[x].min(), data[x].max())
    axes[0].set_ylabel('Progressive mean')
    axes[1].set_ylabel('Progressive mean')
    axes[0].set_title('Rejected targets')
    axes[1].set_title('Accepted targets')
    if x == 'Voting_time': 
        axes[0].set_xlabel('Voting time')
        axes[1].set_xlabel('Voting time')
        if feature == 'progressive_mean':
            fig.suptitle('Histogram of the progressive mean over time')
        elif feature == 'progressive_sentiment':
            fig.suptitle('Histogram of the progressive sentiment over time')
    elif x == 'Vote_number': 
        axes[0].set_xlabel('Number of votes casted')
        axes[1].set_xlabel('Number of votes casted')
        if feature == 'progressive_mean':
            fig.suptitle('Histogram of the progressive mean over the number of votes casted')
        elif feature == 'progressive_sentiment':
            fig.suptitle('Histogram of the progressive sentiment over the number of votes casted')
    
    if plot is None:
        plt.savefig('Figures/hist_progressive_mean_over_' + x.lower() + '.png', dpi=300)
        plt.show()
    else:
        plt.savefig('Figures/hist_progressive_mean_over_' + x.lower() + '_' + plot + '.png', dpi=300)
        plt.close()

def predict_results(df, n_first_votes, n_folds, feature='progressive_mean'):
    """ Predict the results of the votes using the progressive mean of the votes in each round

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        n_first_votes (np.array): Array containing the number of first votes to use for the prediction
        n_folds (int): Number of folds to use for the cross validation

    Returns:
        (pd.DataFrame): Dataframe containing the results of the prediction
    """
    X = df[df.Vote_number <= n_first_votes][['Vote_number', feature, 'Target']]
    X.Target = X.Target.astype('category').cat.codes
    y = df[df.Vote_number <= n_first_votes]['Results']
    clf = GradientBoostingClassifier(random_state=0)
    # Perform cross validation by taking batches of 10 targets (and their votes) 
    scores = cross_validate(clf, X, y, cv=GroupKFold(n_splits=n_folds), groups=X.Target, scoring=('accuracy', 'precision', 'recall'))
    return scores

def early_vote_prediction(df, n_first_votes, n_folds=10, feature='progressive_mean'):
    """ Compute the scores of the results prediction for different quantities of first votes

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        n_first_votes (np.array): Array containing the number of first votes to use for the prediction
        n_folds (int, optional): Number of folds to use for the cross validation. Defaults to 10.

    Returns:
        scores (pd.DataFrame): Dataframe containing the scores of the results prediction for different quantities of first votes
    """
    scores = pd.DataFrame(index=pd.MultiIndex.from_product([n_first_votes, np.arange(n_folds)], names=['nb_first_votes', 'fold']), columns=[['accuracy', 'precision', 'recall']])
    for n in tqdm(n_first_votes):
        res = pd.DataFrame(predict_results(df, n, n_folds, feature))
        scores.loc[n, :] = res[['test_accuracy', 'test_precision', 'test_recall']].values
    scores.reset_index(inplace=True)
    return scores

def plot_prediction_scores(scores, plot=None):
    """ Plot the scores of the results prediction over different quantities of first votes

    Args:
        scores (pd.DataFrame): Dataframe containing the scores of the results prediction for different quantities of first votes
    """
    metrics = ['accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(1,3,figsize=(20, 6))
    # Get the data
    for metric, ax in zip(metrics, axes):
        # Plot the results
        sns.lineplot(y=scores[metric], x=scores.nb_first_votes, ax=ax, errorbar=('ci', 95))
        ax.set_ylim(0.5, 1.01)
        ax.set_xlim(scores.nb_first_votes.min(), scores.nb_first_votes.max())
        ax.set_title(metric + ' of the prediction')
        ax.set_xlabel('Number of first votes')
        ax.set_ylabel(metric)

    if plot is None:
        plt.savefig('Figures/prediction_scores.png', dpi=300)
        plt.show()
    else:
        plt.savefig('Figures/prediction_scores_' + plot + '.png', dpi=300)
        plt.close()

########## SOURCE ANALYSIS FUNCTIONS ##########

def plot_nb_votes_per_source(df, plot=None):
    """ Plot the histogram of the number of votes per sources

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    # Compute the number of votes per sources
    votes_per_source = df.groupby(['Source', 'Year'])
    votes_per_source = votes_per_source.size().reset_index()
    median = np.median(votes_per_source[0])

    # Plot the histogram of the number of votes per sources
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(votes_per_source, x=0, hue='Year', multiple='stack', ax=ax, log_scale=(False, True), bins=100, palette='CMRmap')
    ax.set_title('Histogram of the number of votes per sources and per year, median = {:.0f}'.format(median))
    ax.set_xlabel('Number of votes')
    ax.set_ylabel('Number of sources')
    ax.set_xlim(0, np.max(votes_per_source[0]))
    if plot is None:
        plt.savefig('Figures/hist_votes_per_source.png')
        plt.show()
    else:
        plt.savefig('Figures/hist_votes_per_source_' + plot + '.png')
        plt.close()

def get_source_df(df):
    """ Remove the sources with less than 3 votes (i.e. the median) and return the resulting dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes

    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the sources with less than 3 votes removed
    """
    # Compute the number of votes per sources
    votes_per_source = df.groupby(['Source']).size()
    median = np.median(votes_per_source)

    # Get the sources with more than the median number of votes (i.e. more than 3 votes)
    sources = votes_per_source[votes_per_source > median].index

    # Get the dataframe with only the sources with more than 3 votes
    df = df[df['Source'].isin(sources)]
    return df

def plot_mean_std_time(df, plot=None):
    """ Plot the distribution of the mean and standard deviation of voting time per source

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
    """
    # Remove the source 
    data = df.copy()
    data.Voting_time = time_to_float(data.Voting_time)
    means = data.groupby(['Source', 'Year'])['Voting_time'].mean().reset_index()
    stds = data.groupby(['Source', 'Year'])['Voting_time'].std().reset_index()
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    if means.Year.nunique() == 1:
        sns.histplot(data=means, x='Voting_time', ax=ax[0], palette='CMRmap')
    else:
        sns.histplot(data=means, x='Voting_time', hue='Year', ax=ax[0], palette='CMRmap', multiple='stack')
    ax[0].set_title('Mean voting time')
    sns.histplot(data=stds, x='Voting_time', hue='Year', ax=ax[1], palette='CMRmap', multiple='stack')
    ax[1].set_title('Standard deviation of voting time')
    fig.suptitle('Distribution of the mean and standard deviation of voting time per source')
    if plot is None:
        plt.savefig('Figures/mean_std_time.png')
        plt.show()
    else:
        plt.savefig('Figures/mean_std_time_' + plot + '.png')
        plt.close()
        

########## TIME SERIES AND SOURCE ANALYSIS ON COMMUNITY FUNCTIONS ##########
def add_comm_to_df(df):
    """Add the community column to the dataframe

    Args:
        df (DataFrame): DataFrame with the votes

    Returns:
        DataFrame: DataFrame with the votes and the community column
    """
    years = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
    df_by_year = pd.DataFrame()

    for year in years:
        # Assuming the CSV file naming convention is 'df_community_year.csv'
        filename = f'df_community_{year}.csv'

        try:
            # Load CSV into Pandas DataFrame
            df_comm = pd.read_csv(filename)
            # Remove the word '-source' from the Source column
            df_comm.Source = df_comm.Source.str.replace('-source', '')

            # Add the community column to the dataframe
            df_year = df[df['Year'] == year]
            df_year = df_year.merge(df_comm, how='left', left_on='Source', right_on='Source')          
                        
        except FileNotFoundError:
            # Handle the case where the file is not found
            print(f"File {filename} not found. Skipping.")

        df_by_year = pd.concat([df_by_year, df_year])
    return df_by_year

def get_progressive_mean_comm(df, communities):
    """ Compute the progressive mean of the votes of a specific community in each round (i.e. the mean of the votes at each time step)

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the progressive mean of the votes in each round
    """
    df_by_year = pd.DataFrame()

    for year, comm in communities:
        df_comm = df[(df['Year'] == year) & (df['Community'] == comm)]
        if len(df_comm.groupby(['Target', 'Round']).first()) == 1:
            df_comm['progressive_mean'] = df_comm.sort_values('Voting_time').Vote.cumsum() / np.arange(1, len(df_comm)+1)
            #Convert time in timedelta
            df_comm.Voting_time = pd.to_timedelta(df_comm.Voting_time, unit='h')
            df_comm.sort_values('Voting_time', inplace=True)
        else:
            df_comm = get_progressive_mean(df_comm)
        df_by_year = pd.concat([df_by_year, df_comm])

    return df_by_year 





########## COMMENTS ANALYSIS FUNCTIONS ##########

def get_parsed_comment(comment):
    """ Parse a comment using mwparserfromhell and return the text of the comment"""
    return mwparserfromhell.parse(comment).strip_code()

#For DF
def tokenize_one_comment(comment):
    return word_tokenize(comment.lower())

def get_bow_column(tokenized_column, stopwords=True, ponctuation=True, fine_tune_stopwords=True):
    tokenized_comments = tokenized_column.tolist()
    tokenized_comments = lemmatize_comments(tokenized_comments)
    if stopwords:
        tokenized_comments = remove_stopwords(tokenized_comments)
    if ponctuation:
        tokenized_comments = remove_ponctuation(tokenized_comments)
    if fine_tune_stopwords:
        tokenized_comments = remove_fine_tune_stopwords(tokenized_comments)
    dictionary = get_dict_representation(tokenized_comments)
    bow_corpus = get_bow_representation(tokenized_comments, dictionary)
    return bow_corpus

#For Pipeline
def tokenize_comments(comments_series):
    return [word_tokenize(comment.lower()) for comment in comments_series.tolist()]

def lemmatize_comments(tokenized_comments):
    lemmatizer = WordNetLemmatizer()
    outputs = []
    for comment in tokenized_comments:
        out = [lemmatizer.lemmatize(token) for token in comment]
        outputs.append(out)
    return outputs

def get_dict_representation(tokenized_comments):
    return corpora.Dictionary(tokenized_comments)

def remove_stopwords(tokenized_comments):
    return [[word for word in comment if word not in STOPWORDS] for comment in tokenized_comments]

def remove_ponctuation(tokenized_comments):
    return [[word for word in comment if word not in PUNCTUATION] for comment in tokenized_comments]

def remove_fine_tune_stopwords(tokenized_comments):
    return [[word for word in comment if word not in FINE_TUNE_STOPWORDS] for comment in tokenized_comments]

def get_bow_representation(tokenized_comments, dictionary):
    return [dictionary.doc2bow(comment) for comment in tokenized_comments]

def init_LDA_model(bow_corpus, dictionary, num_topics=3, passes=10):
    return models.LdaModel(bow_corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

def get_LDA_model_from_saved_file(path):
    return models.LdaModel.load(path)

#Not used
def get_LDA_topics(lda_model, num_words=10):
    return lda_model.print_topics(num_words=num_words)

#Pipeline
def get_LDA_model(comments_series, num_topics=3, passes=10,
                  stopwords=True, ponctuation=True, fine_tune_stopwords=True,
                  lemmatize=False):
    
    tokenized_comments = tokenize_comments(comments_series)
    if lemmatize:
        tokenized_comments = lemmatize_comments(tokenized_comments)
    if stopwords:
        tokenized_comments = remove_stopwords(tokenized_comments)
    if ponctuation:
        tokenized_comments = remove_ponctuation(tokenized_comments)
    if fine_tune_stopwords:
        tokenized_comments = remove_fine_tune_stopwords(tokenized_comments)
    dictionary = get_dict_representation(tokenized_comments)
    bow_corpus = get_bow_representation(tokenized_comments, dictionary)
    lda_model = init_LDA_model(bow_corpus, dictionary, num_topics=num_topics, passes=passes)
    return lda_model

# Loading DF with topics
def get_df_with_topics_from_csv(path='df_with_topics.csv'):
    df = pd.read_csv(path)
    #Parse the saved strings into lists
    df['Topics_from_3'] = df['Topics_from_3'].apply(ast.literal_eval).apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True))
    df['Topics_from_5'] = df['Topics_from_5'].apply(ast.literal_eval).apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True))
    df['Topics_from_7'] = df['Topics_from_7'].apply(ast.literal_eval).apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True))
    df['Topics_from_9'] = df['Topics_from_9'].apply(ast.literal_eval).apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True))
    return df

def get_stats_dict_from_df_with_topics(df, nb_topics, topic_positions):
    stats_dict = {}

    for from_topic in nb_topics:
        for pos in topic_positions:
            key = (from_topic, pos)
            value = df.groupby(f'Topics_from_{from_topic}_{pos}_topic')[f'Topics_from_{from_topic}_{pos}_topic_prob'].agg(
                                    count='count',  
                                    mean='mean', 
                                    std='std'
                                ).reset_index()
            value = value.rename(columns={f'Topics_from_{from_topic}_{pos}_topic': 'Topic', 
                                                                    'mean': f'mean_prob_when_{pos}_pos',
                                                                    'std': f'std_prob_when_{pos}_pos'})
            total_number_of_comments = value['count'].sum()
            value[f'prob_of_topic_to_be_{pos}'] = value['count'] / total_number_of_comments
            stats_dict[key] = value

    return stats_dict

def parse_topic_str(s, threshold=None):
    if threshold is None:
        return [[pair.split("*")[1][1:-2], float(pair.split('*')[0])] for pair in s.split('+')]
    else:
        output = [[pair.split("*")[1][1:-2], float(pair.split('*')[0])] for pair in s.split('+')]
        output = [pair for pair in output if pair[1] >= threshold]
        return output

def topic_dict_function(path_dir='./topic_dicts'):
    topic_dict = {}
    for filename in os.listdir(path_dir):
        if 'description' not in filename:
            with open(f'{path_dir}/{filename}', 'r') as f:
                topic_dict['from_' + filename.split('_')[1]] = json.load(f)
    for key, value in topic_dict.items():
        for k, v in value.items():
            topic_dict[key][k] = parse_topic_str(v, threshold=0.01)
    return topic_dict


def topic_str_plot(topic_dict):
    topic_description_dict = {}
    for model, themes_dict in topic_dict.items():
        build = ''
        counter = 0 
        for theme, tuples_list in themes_dict.items():
            build += f'Theme {theme}:'
            for x,y in tuples_list:
                build += f' {x} ({y:.4f}),'
                counter += 1
                if counter % 3 == 0:
                    build += '\n'
            build = build[:-1] + '\n\n'
        topic_description_dict[model] = build[:-2]
    return topic_description_dict

def mappprob_from_list(x, pos_idx, topic_or_prob):
    length_of_topics = len(x)
    if length_of_topics <= pos_idx:
        return np.nan
    return x[pos_idx][topic_or_prob]

def get_topic_stat(df_top_stat, nb_topics, topic_positions):
    for t_number in nb_topics:
        for i, pos in enumerate(topic_positions):
            df_top_stat[f'Topics_from_{t_number}_'+pos+'_topic'] = df_top_stat[f'Topics_from_{t_number}'].apply(lambda x: mappprob_from_list(x, i, 0))
            df_top_stat[f'Topics_from_{t_number}_'+pos+'_topic_prob'] = df_top_stat[f'Topics_from_{t_number}'].apply(lambda x: mappprob_from_list(x, i, 1))
    return df_top_stat

################### Community Analysis Functions #####################

def load_communities_dict_for_topic(path_dir='./community_anal_dfs/'):
    community_dict = {}
    for filename in os.listdir(path_dir):
        df = pd.read_csv(path_dir + filename)
        df.Source = df.Source.apply(lambda x: x.split('-')[0])
        community_dict[filename[-8:-4]] = df    
    return community_dict

def join_left_com_top(df_with_top_lem, communities_dcit):
    output_dict= {}
    for key, value in communities_dcit.items():
        df_to_merge = df_with_top_lem[df_with_top_lem['Year']==int(key)]
        df_merged = pd.merge(df_to_merge, value, how='left', on='Source')
        df_merged = df_merged[~df_merged.Community.isnull()]
        cur_communities = df_merged.Community.unique()
        output_dict[key] = {}
        for com in cur_communities:
            temp = df_merged[df_merged.Community == com]
            output_dict[key][com] = temp[['Source', 'BoW', 'Topics_from_3',
                                               'Topics_from_5', 'Topics_from_7',
                                               'Topics_from_9']]
    return output_dict

def get_src_sets_from_com_dict(com_dict):
    src_sets_dict = {}
    for year, df in com_dict.items():
        commu_possible = df.Community.unique()
        src_sets_dict[year] = {}
        for cur_commu in commu_possible:
            src_sets_dict[year][cur_commu] = set(df[df.Community == cur_commu].Source)
    return src_sets_dict

def jacquard_similarity(src_sets_dict, year1, year2):
    src_sets1 = src_sets_dict[year1]
    src_sets2 = src_sets_dict[year2]
    jacquard_sim = {}
    jacquard_sim_this_year = {}
    for com1, src_set1 in src_sets1.items():
        jacquard_sim_this_year[com1] = {}
        for com2, src_set2 in src_sets2.items():
            jacquard_sim_this_year[com1][com2] = len(src_set1.intersection(src_set2)) / len(src_set1.union(src_set2))
    jacquard_sim.update(jacquard_sim_this_year)
    return jacquard_sim

def jacquard_similarity_for_all_years(src_sets_by_comu_by_year, years_list):
    jacquard_similarities = {}
    for i,year in enumerate(years_list):
        year = int(year) 
        if i < len(years_list)-1:
            jacquard_similarities[f"{year}-{year+1}"] = jacquard_similarity(src_sets_by_comu_by_year, str(year), str(year+1))
    return jacquard_similarities

def max_jacquard_sim(jacquard_similarities):
    max_sim = {}
    for year, sim_dict in jacquard_similarities.items():
        max_sim[year] = {}
        for com1, sim_dict in sim_dict.items():
            key_max, value_max = max(sim_dict.items(), key=lambda x: x[1])
            max_sim[year][com1] = (key_max, round(value_max,3))
    return max_sim

def get_nodes_from_com_dict(com_dict):
    nodes = []
    for year, df in com_dict.items():
        for comu in df['Community'].unique():
            temp = df[df['Community'] == comu]
            size = len(temp)
            nodes.append((year, comu, size))
    return nodes

def get_edges_from_jacquard_similarities(jacquard_similarities):
    edges = set()
    for years, map in jacquard_similarities.items():
        x1 = years.split('-')[0]
        x2 = years.split('-')[1]
        for commu1, commuDictWeight in map.items():
            for commu2, weight in commuDictWeight.items():
                edges.add(((x1, commu1), (x2, commu2), weight))
    return edges

def plot_maxJac_connected_layers(nodes, max_jacquard_similarities):
    edges = set()
    for years, map in max_jacquard_similarities.items():
        x1 = years.split('-')[0]
        x2 = years.split('-')[1]
        for commu1, commu2Weight in map.items():
            commu2 = commu2Weight[0]
            edges.add(((x1, commu1), (x2, commu2), commu2Weight[1]))

    plt.figure(figsize=(20,10))
    for edge in edges:
        plt.plot([int(edge[0][0]), int(edge[1][0])], [int(edge[0][1]), int(edge[1][1])], linewidth=edge[2]*20, color='black')
    plt.scatter([int(node[0]) for node in nodes], [int(node[1]) for node in nodes], s=[int(node[2]) for node in nodes], color='blue', alpha=1)
    plt.show()
    
def plot_connected_layers(nodes, edges):
    plt.figure(figsize=(20,10))
    for edge in edges:
        plt.plot([int(edge[0][0]), int(edge[1][0])], [int(edge[0][1]), int(edge[1][1])], linewidth=edge[2]*20, color='black', alpha=min(edge[2]*9, 1))
    plt.scatter([int(node[0]) for node in nodes], [int(node[1]) for node in nodes], s=[int(node[2]) for node in nodes], color='blue', alpha=1)
    plt.show()

def plot_connected_with_topic(nodes, edges, model, y_offset=0.2):
    plt.figure(figsize=(20,10))
    plt.title(f'Model : {model}-topics, Communities and Topics evolution over the years')
    for edge in edges:
        plt.plot([int(edge[0][0]), int(edge[1][0])], [int(edge[0][1]), int(edge[1][1])], linewidth=edge[2]*20, color='black', alpha=min(edge[2]*9, 1))
    plt.scatter([int(node[0]) for node in nodes], [int(node[1]) for node in nodes], s=[int(node[2]) for node in nodes], color='blue', alpha=1)
    for node in nodes:
        plt.text(int(node[0]), int(node[1])+y_offset, f"T-{node[3]}-{node[4]}", color='orange', alpha=1, ha='center')
    plt.show()


################### Community on bipartite graph #####################

def create_bipartite_weight(sources, targets, weights):
    """ Create a bipartite weighted graph 
    Args:
        sources (list): list of Source name 
        targets (list): list of Target name
        weights (list): list of weight, either 0, 1 or -1 corresponding to the vote of the source for the target
    Returns:
        G (nx.Graph): bipartite graph created
    """
    
    #create bipartite graph
    G = nx.Graph()
    for i in range (len(sources)):
        G.add_node(sources[i], bipartite=0)
        G.add_node(targets[i], bipartite=1)
        G.add_edge(sources[i], targets[i], weight=weights[i])
    return G

def create_bipartite_weight_from_df(df):
    """" Create a bipartite weighted graph for one specific year
    Args:
        df (pd.Dataframe): Bipartite network containing source and target
    Returns:
        G (nx.Graph): bipartite graph created
    """
    #Extract source, target, weight
    sources = list(df['Source'])
    targets = list(df['Target'])
    weights = list(df['Vote']) 

    # A lot of target become source during the year, we want to gather only sources and then we don't want to take into account the connection a target, which became source,
    # has with sources when it was not yet elected as source
    #Change the name so that it does not happen
    sources= [source + '-source' for source in sources]

    G = create_bipartite_weight(sources, targets, weights)

    return G

def create_bipartite_weight_from_df_year(df, year):
    """ Create a bipartite weighted graph for one specific year
    Args:
        df (pd.Dataframe): Bipartite network containing source and target
        year (float): a specific year on which the graph needs to be built on
    Returns:
        G (nx.Graph): bipartite graph created
    """
    #Take data from the year
    df_year= df[df['Year']==year]

    #Extract source, target, weight
    sources_year = list(df_year['Source'])
    targets_year = list(df_year['Target'])
    weights_year = list(df_year['Vote']) 

    # A lot of target become source during the year, we want to gather only sources and then we don't want to take into account the connection a target, which became source,
    # has with sources when it was not yet elected as source
    #Change the name so that it does not happen
    sources_year = [source + '-source' for source in sources_year]

    G = create_bipartite_weight(sources_year, targets_year, weights_year)

    return G

def shared_neighbors_count(G, u, v):
    """ Count the number of neighbor (same target in network G and same weight) in common for node u and v
    Args:
        G (nx.Graph): Bipartite network containing source and target
        u, v: specific node of the graph
    Returns:
        len (int): number of shared neighbors
    """
    #Check if nodes u and v belong to the same side of the bipartite graph
    if G.nodes[u]['bipartite'] != G.nodes[v]['bipartite']:
        raise ValueError("Nodes should belong to the same side of the bipartite graph.")

    #Select the other part of the graph
    other_side = 1 - G.nodes[u]['bipartite']

    #u and v neighbor from the other part of the graph
    neighbors_u = set(n for n in G.neighbors(u) if G.nodes[n]['bipartite'] == other_side)
    neighbors_v = set(n for n in G.neighbors(v) if G.nodes[n]['bipartite'] == other_side)

    # Compute shared neighbor with the same weight
    shared_neighbors = neighbors_u.intersection(neighbors_v)

    return len(shared_neighbors)

def projected_weighted_bipartite(G, source):
    #give the projection of the bipartite weighted graph G on source
    return nx.algorithms.bipartite.generic_weighted_projected_graph(G, source, weight_function=shared_neighbors_count)

def extract_community_from_projected_graph(projected_G):
    com_dict = community.best_partition(projected_G)
    df_com = pd.DataFrame(list(com_dict.items()), columns=['Source', 'Community'])
    return df_com

################### Community analysis #####################

def load_com_csv(file_path):
    """ Load a csv file
    Args:
        file_path (string): path to the csv file to load
    Returns:
        df_com (pd.DataFrame): Dataframe containing the elements in our csv file
    """
    #load the file
    df_com = pd.read_csv(file_path)
    #remove the -source in each source name
    df_com['Source'] = df_com['Source'].str.replace('-source', '')
    
    return df_com

def compute_df_size_com (df_com, years):
    """ Compute the number of community per year, size and size proportion compared to all sources voting that year, for each community and each year
    Args:
        df_com (pd.DataFrame): Dataframe containing for each year a dataframe which matches for each source a community number
        years (list): years of interest
    Returns:
        df_com_stat (pd.DataFrame): DataFrame of the stat computed (nbr of community, community size, community proportion size)
    """
    #create our output df
    df_com_stat = pd.DataFrame(columns = ['Year'])
    df_com_stat['Year'] = years

    #compute the number of community and add it to the output df
    list_nbr_com = df_com['Community_df'].apply(lambda x: x['Community'].max()+1)
    df_com_stat['Nbr_of_com'] = list_nbr_com

    #compute the number of source in each community and add it to the output df
    list_nbr_in_com_ = df_com['Community_df'].apply(lambda x: x.groupby('Community').size())
    df_com_stat['Com_size'] = list(list_nbr_in_com_.values)

    #remove nan values when the number of community is 0
    list_nbr_in_com_filtered = df_com_stat['Com_size'].apply(lambda x: x[~np.isnan(x)])
    df_com_stat['Com_size'] = list_nbr_in_com_filtered

    #compute the size proportion in each community compared to the tot number of source for that year
    df_com_stat['Source_prop'] = df_com_stat['Com_size'].apply(lambda x: np.round(np.array(x) / np.sum(x), 3))
    return df_com_stat

def compute_df_source_prop_for_plot(df):
    """ Transform the df so that it is suitable for a plot
    Args:
        df (pd.DataFrame): Dataframe needing to be transformed
    Returns:
        df_result (pd.DataFrame): DataFrame transformed
    """
    df_expanded = df.apply(lambda row: pd.Series(row['Source_prop']), axis=1)

    # Add the column 'Year' to the new df
    df_expanded['Year'] = df['Year']

    # Rename the column to include 'Com_nbr'
    df_expanded.columns = [i if i != 'Year' else i for i in df_expanded.columns]

    # Use melt to reorganise the df
    df_result = pd.melt(df_expanded, id_vars=['Year'], var_name='Com_nbr', value_name='Source_prop')

    # Sort the results
    df_result = df_result.sort_values(by=['Year', 'Com_nbr']).reset_index(drop=True)
    df_result = df_result.dropna()
    return df_result

def plot_source_prop(df_result, year):
    """ Plot source proportion for each community and for specific years
    Args:
        df_result (pd.DataFrame): Dataframe containing the values to plot
        year (list): Specific years on which we want our plot
    """

    # Create the plot
    g = sns.catplot(x='Com_nbr', y='Source_prop', hue='Com_nbr', col='Year', data=df_result[df_result['Year'].isin(year)], kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'Size proportion')
    g.fig.suptitle('Size proportion per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=6, title=None, frameon=False)

    # Show the plot
    plt.show()

def com_vote(df_ref, df_com_year_, year):
    """ Compute the vote proportion for each vote type for each community and each year
    Args:
        df_ref (pd.DataFrame): Dataframe of reference, from which we will extract
        df_com_year (pd.DataFrame): Dataframe of the matching of sources with their community number for a year
        year (float): year of corresponding to the df_com_year
    Returns:
        df_com_stat (pd.DataFrame): DataFrame of the stat computed 
    """

    #filter the ref df
    df_ref_year = df_ref[df_ref['Year']==int(year)]
    #extract the source for the community df for 1 year
    source_per_com = df_com_year_.groupby('Community').apply(lambda x : x['Source'])
    pos_vote_prop = []
    neg_vote_prop = []
    neu_vote_prop = []
    #loop over all communities of this year
    for n in range (df_com_year_['Community'].max()+1):
        com = list(source_per_com[n].values)
        #extract rows of the ref df based on the source which are in the community n
        df_com = df_ref_year[df_ref_year['Source'].isin(com)].reset_index(drop=True)
        #extract proportion for theses sources
        prop_vote_pos_com = np.sum(df_com['Vote']==1)/len(df_com['Vote'])
        prop_vote_neg_com = np.sum(df_com['Vote']==-1)/len(df_com['Vote'])
        prop_vote_neu_com = np.sum(df_com['Vote']==0)/len(df_com['Vote'])
        pos_vote_prop.append(prop_vote_pos_com)
        neg_vote_prop.append(prop_vote_neg_com)
        neu_vote_prop.append(prop_vote_neu_com)
    
    #create the df
    years_ = [int(year)]*(df_com_year_['Community'].max()+1)
    com = list(range(df_com_year_['Community'].max()+1))
    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Pos_vote_prop', 'Neg_vote_prop', 'Neu_vote_prop'])
    df_stat_com['Year'] = years_
    df_stat_com['Com_nbr'] = com
    df_stat_com['Pos_vote_prop'] = pos_vote_prop
    df_stat_com['Neg_vote_prop'] = neg_vote_prop
    df_stat_com['Neu_vote_prop'] = neu_vote_prop
    
    return df_stat_com

def plot_dist_vote_per_com(df, years):
    """ Plot vote type proportion for each community and for specific years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
        year (list): Specific years on which we want our plot
    """
    # Transform the df in the adequate form
    df_melt = pd.melt(df[df['Year'].isin(years)], id_vars=['Com_nbr', 'Year'], value_vars=['Pos_vote_prop', 'Neg_vote_prop', 'Neu_vote_prop'],
                        var_name='Vote Type', value_name='Pourcentage')

    # Create the plot
    g = sns.catplot(x='Com_nbr', y='Pourcentage', hue='Vote Type', col='Year', data=df_melt, kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'Vote percentage')
    g.fig.suptitle('Vote percentage per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=3, title=None, frameon=False)

    # Show the plot
    plt.show()

def plot_vote_type_on_whole_year(df, type='Vote'):
    """ Plot vote type proportion per year
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
    """
    #Transform the df in adequate form for the plot
    melted_df = pd.melt(df, id_vars=['Year'], var_name='Statistic', value_name='Value')

    # Create a bar plot 
    plt.figure(figsize=(10, 6)) 
    sns.barplot(x='Year', y='Value', hue='Statistic', data=melted_df, palette='muted')

    # Add labels and title
    plt.xlabel('Year')
    if (type=='Results'): 
        plt.ylabel('Results proportion')
        plt.title('Results per year')
        plt.legend(title='Results', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    else:
        plt.ylabel('Vote Type proportion')
        plt.title('Vote type per year')
        plt.legend(title='Vote', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    # Show the plot
    plt.show()

def compute_vote_type_prop_on_whole_year(df_ref):
    """ Compute the vote proportion for each vote type per year
    Args:
        df_ref (pd.DataFrame): Dataframe containing vote type per year
    Returns:
        df_prop_vote_per_year (pd.DataFrame): DataFrame containing the proportion per vote type per year
    """
    prop_vote_per_year = df_ref.groupby('Year')['Vote'].value_counts(normalize=True).unstack(fill_value=0)

    # Extract proportions for each type of vote
    prop_vote_pos = prop_vote_per_year[1]
    prop_vote_neg = prop_vote_per_year[-1]
    prop_vote_neu = prop_vote_per_year[0]

    #create and fill in the df
    df_prop_vote_per_year = pd.DataFrame(columns=['Year', 'Pos_vote_prop', 'Neg_vote_prop', 'Neu_vote_prop'])
    df_prop_vote_per_year['Year'] = list(prop_vote_per_year.index)
    df_prop_vote_per_year['Pos_vote_prop'] = prop_vote_pos.values
    df_prop_vote_per_year['Neg_vote_prop'] = prop_vote_neg.values
    df_prop_vote_per_year['Neu_vote_prop'] = prop_vote_neu.values

    return df_prop_vote_per_year

def compute_results_type_prop_on_whole_year(df_ref):
    """ Compute the results proportion for each vote type per year
    Args:
        df_ref (pd.DataFrame): Dataframe containing vote type per year
    Returns:
        df_prop_results_per_year (pd.DataFrame): DataFrame containing the proportion per vote type per year
    """
    prop_results_per_year = df_ref.groupby('Year')['Results'].value_counts(normalize=True).unstack(fill_value=0)

    # Extract proportions for each type of vote
    prop_results_pos = prop_results_per_year[1]
    prop_results_neg = prop_results_per_year[-1]

    #create and fill in the df
    df_prop_results_per_year = pd.DataFrame(columns=['Year', 'Pos_results_prop', 'Neg_results_prop'])
    df_prop_results_per_year['Year'] = list(prop_results_per_year.index)
    df_prop_results_per_year['Pos_results_prop'] = prop_results_pos.values
    df_prop_results_per_year['Neg_results_prop'] = prop_results_neg.values

    return df_prop_results_per_year

def com_voting_time(df_ref, df_com_year_, year):
    """ Compute the voting time mean and median per year
    Args:
        df_ref (pd.DataFrame): Dataframe containing voting time per year
        df_com_year_ (DataFrame): Dataframe containing source and their matching community for a specific year
        year (float): specific year matching df_com_year_
    Returns:
        df_prop_vote_per_year (pd.DataFrame): DataFrame containing the proportion per vote type per year
    """
    #filter the ref df
    df_ref_year = df_ref[df_ref['Year']==int(year)]
    #extract the source for the community df for 1 year
    source_per_com = df_com_year_.groupby('Community').apply(lambda x : x['Source'])
    
    mean_voting_time = []
    median_voting_time = []
    #loop over all communities of this year
    for n in range (df_com_year_['Community'].max()+1):
        com = list(source_per_com[n].values)
        #extract rows of the ref df based on the source which are in the community n
        df_com = df_ref_year[df_ref_year['Source'].isin(com)].reset_index(drop=True)
        
        #extract proportion for theses sources
        mean_ = df_com['Voting_time'].mean()
        mean_voting_time.append(mean_)

        median_ = df_com['Voting_time'].median()
        median_voting_time.append(median_)
    
    #create the df
    years_ = [int(year)]*(df_com_year_['Community'].max()+1)
    com = list(range(df_com_year_['Community'].max()+1))
    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Mean_voting_time', 'Median_voting_time'])
    df_stat_com['Year'] = years_
    df_stat_com['Com_nbr'] = com
    df_stat_com['Mean_voting_time'] = mean_voting_time
    df_stat_com['Median_voting_time'] = median_voting_time
    
    return df_stat_com

def plot_voting_time_per_com(df, years):
    """ Plot mean and median voting time for each community and for specific years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
        year (list): Specific years on which we want our plot
    """
    # Tranform the df in an adequate form
    df_melt = pd.melt(df[df['Year'].isin(years)], id_vars=['Com_nbr', 'Year'], value_vars=['Mean_voting_time', 'Median_voting_time'],
                        var_name='Vote Type', value_name='Pourcentage')

    # Create the plot
    g = sns.catplot(x='Com_nbr', y='Pourcentage', hue='Vote Type', col='Year', data=df_melt, kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'Voting Time')
    g.fig.suptitle('Voting time per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=3, title=None, frameon=False)

    # Show the plot
    plt.show()

def plot_voting_time_on_whole_year(df):
    """ Plot voting mean and median per year
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
    """
    # Transform the df in an adequate form
    melted_df = pd.melt(df, id_vars=['Year'], var_name='Statistic', value_name='Value')

    # Create a bar plot 
    plt.figure(figsize=(10, 6)) 
    sns.barplot(x='Year', y='Value', hue='Statistic', data=melted_df, palette='muted')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Voting Time (hours)')
    plt.title('Mean and Median Voting Time per Year')

    # Add legend
    plt.legend(title='Statistic')

    # Show the plot
    plt.show()

def compute_df_recall_accuracy_precision(df_ref, df_com_year_, year):
    """ Compute the recall, accuracy and precision per year and per community
    Args:
        df_ref (pd.DataFrame): Dataframe containing voting time per year
        df_com_year_ (DataFrame): Dataframe containing source and their matching community for a specific year
        year (float): specific year matching df_com_year_
    Returns:
        df_stat_com (pd.DataFrame): DataFrame containing the recall, precision and accuracy
    """
    #filter the ref df
    df_ref_year = df_ref[df_ref['Year']==int(year)]
    #extract the source for the community df for 1 year
    source_per_com = df_com_year_.groupby('Community').apply(lambda x : x['Source'])

    accuracy = []
    recall = []
    precision = []
    specificity = []
    neutral_vote_prop = []
    #loop over all communities of this year
    for n in range (df_com_year_['Community'].max()+1):
        com = list(source_per_com[n].values)
        #extract rows of the ref df based on the source which are in the community n
        df_com = df_ref_year[df_ref_year['Source'].isin(com)].reset_index(drop=True)

        TP = np.sum((df_com['Vote']== df_com['Results']) & (df_com['Vote']==1))
        TN = np.sum((df_com['Vote']== df_com['Results']) & (df_com['Vote']==-1))
        FN = np.sum((df_com['Vote']!= df_com['Results']) & (df_com['Vote']==-1))
        FP = np.sum((df_com['Vote']!= df_com['Results']) & (df_com['Vote']==1))
        neutral = np.sum(df_com['Vote']==0)
        N = len(df_com['Vote'])
        N_accuracy = len(df_com['Vote']) - neutral
        accuracy_ = (TP + TN)/N_accuracy
        if (TP == 0 & FN == 0) : recall_ = 0
        else: recall_ = TP / (TP + FN)
        precision_ = TP / (TP + FP)
        specificity_ = TN/(TN+FN)
        neutral_vote_prop_ = neutral/N
        accuracy.append(accuracy_)
        recall.append(recall_)
        precision.append(precision_)
        specificity.append(specificity_)
        neutral_vote_prop.append(neutral_vote_prop_)

    #create the df
    years_ = [int(year)]*(df_com_year_['Community'].max()+1)
    com = list(range(df_com_year_['Community'].max()+1))
    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Neutral_vote_prop'])
    df_stat_com['Year'] = years_
    df_stat_com['Com_nbr'] = com
    df_stat_com['Accuracy'] = accuracy
    df_stat_com['Precision'] = precision
    df_stat_com['Recall'] = recall
    df_stat_com['Specificity'] = specificity
    df_stat_com['Neutral_vote_prop'] = neutral_vote_prop
   
    return df_stat_com
    

def plot_recall_accuracy_precision_per_com(df, years):
    """ Plot the recall, accuracy and precision for each community and for specific years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
        year (list): Specific years on which we want our plot
    """
    # Transform the df in an appropriate form
    df_melt = pd.melt(df[df['Year'].isin(years)], id_vars=['Com_nbr', 'Year'], value_vars=['Accuracy', 'Precision', 'Recall', 'Specificity', 'Neutral_vote_prop'],
                        var_name='Stat', value_name='Pourcentage')

    # Create the plot
    g = sns.catplot(x='Com_nbr', y='Pourcentage', hue='Stat', col='Year', data=df_melt, kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'Proportion')
    g.fig.suptitle('Accuracy, Recall, Precision, Specificity and proportion of neutral vote per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=5, title=None, frameon=False)

    # Show the plot
    plt.show()

def calculate_accuracy_recall_precision_on_whole_years(group):
    """ Plot the recall, accuracy and precision for a specific year
    Args:
        group (pd.DataFrame): Dataframe containing the vote and the results for one year
    """
    TP = np.sum((group['Vote'] == group['Results']) & (group['Vote'] == 1))
    TN = np.sum((group['Vote'] == group['Results']) & (group['Vote'] == -1))
    FN = np.sum((group['Vote'] != group['Results']) & (group['Vote'] == -1))
    FP = np.sum((group['Vote'] != group['Results']) & (group['Vote'] == 1))
    neutral = np.sum(group['Vote']==0)
    N = len(group['Vote'])
    N_accuracy = len(group['Vote'])/ - np.sum(group['Vote']==0)
    
    accuracy = (TP + TN) / N_accuracy if N_accuracy != 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    specificity = TN/(TN+FN) if (TN+FN) != 0 else np.nan
    neutral_vote_prop = neutral/N if N != 0 else np.nan

    return pd.Series({
        'Accuracy': np.round(accuracy,3),
        'Precision': np.round(precision,3),
        'Recall': np.round(recall,3),
        'Specificity' : np.round(specificity,3),
        'Neutral_vote_prop' : np.round(neutral_vote_prop, 3)
    })

def plot_accuracy_recall_precision_on_whole_years(df):
    """ Plot the recall, accuracy and precision per years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
    """
    # Transform the df in an appropriate form
    melted_df = pd.melt(df[['Year', 'Accuracy',  'Precision', 'Recall', 'Specificity', 'Neutral_vote_prop']], id_vars=['Year'], var_name='Statistic', value_name='Value')
    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 6))  
    sns.barplot(x='Year', y='Value', hue='Statistic', data=melted_df, palette='muted')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.title('Accuracy, Recall, Precision, Specificity and proportion of negative vote per year')

    # Add legend
    plt.legend(title='Vote', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)

    # Show the plot
    plt.show()
    
    
########## USER'S REVISION FUNCTIONS ##########
    
def process_elections(dataset):
    # Ensure 'Date' column is in datetime format
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

    # Group by 'Target' and get the first vote's date and result for each election
    result_df = dataset.groupby('Target').agg(
        First_Vote_Date=('Date', 'first'),
        Result=('Results', 'first')
    ).reset_index()

    # Extract year and month from the 'First_Vote_Date' column
    result_df['Year_Month'] = result_df['First_Vote_Date'].dt.to_period('M')

    return result_df

def process_elections_ts(dataset):
    # Ensure 'Date' column is in datetime format
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

    dataset = dataset.sort_values(['Target', 'Round', 'Date'])
    # Group by 'Target' and get the first vote's date and result for each election
    result_df = dataset.groupby(['Target', 'Round']).agg(
        First_Vote_Date=('Date', 'first'),
        Result=('Results', 'first')
    ).reset_index()

    # Extract year and month from the 'First_Vote_Date' column
    result_df['Year_Month'] = result_df['First_Vote_Date'].dt.to_period('M')

    return result_df

def generate_user_revision_dataset(revisions_dataset, election_dataset):
    # Ensure 'month' and 'Date' columns are in datetime format
    election_dataset['First_Vote_Date'] = pd.to_datetime(election_dataset['First_Vote_Date'], format='%Y-%m-%d')

    # Create a DataFrame with 'user_name', 'month', and 'revisions'
    result_df = pd.merge(revisions_dataset, election_dataset[['Target', 'First_Vote_Date', 'Result']], left_on='user_name', right_on='Target', how='left')
    # Calculate the relative months based on each user's election date
    result_df['Relative_Month'] = (result_df['month'].dt.year - result_df['First_Vote_Date'].dt.year) * 12 + result_df['month'].dt.month - result_df['First_Vote_Date'].dt.month
    # Filter rows based on the -12 to 12 range
    result_df = result_df[result_df['Relative_Month'].between(-12, 12, inclusive='both')]

    # Pivot the DataFrame to have columns for each month
    result_df_pivot = result_df.pivot_table(index='user_name', columns='Relative_Month', values='revisions', aggfunc='sum', fill_value=0)

    # Rename the columns to represent months
    result_df_pivot.columns = [f'Month_{m}' for m in result_df_pivot.columns]

    # Add the 'Results' column to the resulting DataFrame
    result_df_pivot['Result'] = result_df.groupby('user_name')['Result'].first()
    return result_df_pivot

def plot_average_edit(data, x, mean_col = 'center', var_cols = ['low', 'up']):

    # Plot the evolution of the edit
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=x, y=mean_col, data=data, hue='Result', palette='tab10', ax=ax)
    ax.fill_between(data[data.Result == -1][x], data[data.Result == -1][var_cols[0]], data[data.Result == -1][var_cols[1]], alpha=0.2, color='tab:blue')
    ax.fill_between(data[data.Result == 1][x], data[data.Result == 1][var_cols[0]], data[data.Result == 1][var_cols[1]], alpha=0.2, color='tab:orange')
    ax.set_ylabel('Mean edit')
    ax.legend(handles=[plt.Line2D([0], [0], color='tab:blue', lw=4, label='Rejected'),
                        plt.Line2D([0], [0], color='tab:orange', lw=4, label='Elected')])
    return ax

def revision_stat(dataset):
    dataset = dataset.reset_index()
    melted_df= dataset.melt(id_vars=['user_name', 'Result'], var_name='Month', value_name='Revisions')
    result_stat = melted_df.groupby(['Result', 'Month'])['Revisions'].agg(
    	mean=np.mean,
    	confidence_interval=lambda x: list(stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x)))
	).reset_index()
    result_stat['Month'] = result_stat['Month'].str.extract(r'(-?\d+)').astype(int)
    result_stat[['low', 'up']] = pd.DataFrame(result_stat['confidence_interval'].tolist())
    result_stat = result_stat.drop('confidence_interval', axis=1)

    return result_stat

def create_df_causal(election, revision):

  df = pd.DataFrame(columns=['Target','revisions','Result'])

  for index, row in election.iterrows():

    election_round = row['Round']
    election_month = row['Year_Month']
    election_target =  row['Target']
    election_result = row['Result']

    # Extract data for the specified time window before the election
    time_window_start =  (election_month.to_timestamp() - pd.DateOffset(months=1)).to_period('M')
    time_window_end =  election_month

    revisions_data = revision[
        (revision['user_name'] == election_target) &
        (revision['month'] >= time_window_start) &
        (revision['month'] < time_window_end)
    ]

    add =  pd.DataFrame({
    'Target': [election_target],
    'revisions': [revisions_data['revisions'].sum()],
    'Result': [election_result]
  })

    df = pd.concat([df,add], ignore_index=True)

  return df

def compute_stat(ts, topk_dict):
  ts['Year'] = ts['Year'].astype(int)
  stats=pd.DataFrame()

  for year, df in topk_dict.items():
    filter_ts = ts[ts['Year']==year]
    merged_df = pd.merge(df, filter_ts, on='Source', how='left')

    merged_df.reset_index(drop=True, inplace=True)
    grouped_df = merged_df.groupby(['Year', 'Community', 'Source'])

    average_stats = grouped_df[['Voting_time', 'Vote_number']].mean().round(2).reset_index()

    merged_df = merged_df.drop_duplicates(subset=['Source'])
    average_stats = pd.merge(average_stats, merged_df[['Community', 'Source', 'revisions']], on=['Community', 'Source'], how='left')

    stats=pd.concat([stats,average_stats], ignore_index=True)

  stats = stats.sort_values(by=['Year', 'Community', 'revisions'], ascending=[True, True, False])
  stats = stats.reset_index()
  stats = stats.drop('index', axis=1)
  return stats

def create_df_Community_edit(com, edit, year):

    filter = edit.groupby(['user_name','year'])['revisions'].sum()
    df_filter = filter.reset_index()
    df_filter.columns = ['user_name', 'year', 'revisions']

    filter_Source = df_filter[df_filter['year']==year]
    com['Source'] = com['Source'].str.replace('-source', '')
    df_Community_edit = pd.merge(com, filter_Source[['user_name', 'revisions']], left_on='Source', right_on='user_name', how='left')
    df_Community_edit = df_Community_edit.drop('user_name', axis=1)
    return df_Community_edit

def compute_stats_community(df_dict):

  stats_dict = {}

  for key in df_dict.keys():

    agg_data = df_dict[key].groupby('Community')['revisions'].agg(['sum', 'mean', 'size']).reset_index()
    agg_data['sum_normalized'] = agg_data['sum'] / agg_data['size']
    agg_data = agg_data.drop('size', axis=1)
    stats_dict[key] = agg_data

  return stats_dict

def compute_top_k(df_dict, k):
  stats_dict = {}

  for key in df_dict.keys():
    # Calculate the top 10 user_name for each community
    top_users_by_community = df_dict[key].groupby(['Community', 'Source'])['revisions'].sum().reset_index()
    top_users_by_community = top_users_by_community.sort_values(by='revisions', ascending=False)
    top_users_by_community = top_users_by_community.groupby('Community').head(k)
    stats_dict[key] = top_users_by_community
  return stats_dict

def load_datasets_com(edit):
    years = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
    df_edit_year_dict = {}

    for year in years:
        # Assuming the CSV file naming convention is 'df_community_year.csv'
        filename = f'df_community_{year}.csv'

        try:
            # Load CSV into Pandas DataFrame
            df = pd.read_csv(filename)

            df_edit = create_df_Community_edit(df, edit, year)
            # Append to the list
            df_edit_year_dict[year] = df_edit
        except FileNotFoundError:
            # Handle the case where the file is not found
            print(f"File {filename} not found. Skipping.")

    return df_edit_year_dict


########## SENTIMENT ANALYSIS FUNCTIONS ##########

def removing_comment_headers(df):
    df_sentiment = df.copy()

    support_pattern = re.compile(r'^\bsupport\w*\b', flags=re.IGNORECASE)
    df_sentiment["Parsed_Comment"] = df_sentiment["Parsed_Comment"].str.replace(support_pattern, '', regex=True)
    neutral_pattern = re.compile(r'^\bneutral\w*\b', flags=re.IGNORECASE)
    df_sentiment["Parsed_Comment"] = df_sentiment["Parsed_Comment"].str.replace(neutral_pattern, '', regex=True)
    oppose_pattern = re.compile(r'^\boppose\w*\b', flags=re.IGNORECASE)
    df_sentiment["Parsed_Comment"] = df_sentiment["Parsed_Comment"].str.replace(oppose_pattern, '', regex=True)
    
    return df_sentiment


def sentiment_analysis(df):
    #Parse the comments to a new column
    df['Parsed_Comment'] = df['Comment'].apply(get_parsed_comment)
    # Removing strings for sentiment analysis without headers (support, oppose, neutral)
    df = removing_comment_headers(df)

    analyzer = SentimentIntensityAnalyzer()
    df["sent_score"] = [analyzer.polarity_scores(sent)['compound'] for sent in df["Parsed_Comment"]]
    
    return df


def plot_compound_sentiment_score(df):

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    axs.hist(df["sent_score"], bins=15)
    axs.set_xlim([-1, 1])
    axs.set_xlabel('Compound sentiment')
    axs.set_ylabel('Number of comments')
    axs.set_title('Compound sentiment scores')
    
    
def pieChart_perYear_sentiment_score(df):

    pos_comments = df[df['sent_score'] >= 0.05].groupby('Year')['sent_score'].count().to_frame()
    neg_comments = df[df['sent_score'] <-0.05].groupby('Year')['sent_score'].count().to_frame()
    neu_comments = df[(df['sent_score'] > -0.05) & (df['sent_score'] < 0.05)].groupby('Year')['sent_score'].count().to_frame()

    comment_sentiment_count = pd.DataFrame()
    comment_sentiment_count["pos_comments"] = pos_comments
    comment_sentiment_count["neg_comments"] = neg_comments
    comment_sentiment_count["neu_comments"] = neu_comments
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(100, 50))

    for i in range(len(comment_sentiment_count)):
        plt.subplot(4, 3, i+1)
        year_data = comment_sentiment_count.iloc[i]
        plt.pie(year_data, labels=year_data.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 50})
        plt.title(f'Year {int(comment_sentiment_count.index[i])}', fontsize=70)

    plt.tight_layout()
    plt.show()
    

def comment_vectorizing(df):
    vectorizer = CountVectorizer(stop_words='english')
    df['Vectorized_Comment'] = df['Parsed_Comment'].apply(lambda x: vectorizer.build_analyzer()(x)) 
    df['Vector_Size'] = df['Vectorized_Comment'].apply(lambda x: len(x))
    
    return df


def print_vectorized_comment_stats(df):

    print('The minimun length of the comment is: ', np.min(df['Vector_Size']))
    print('The maximum length of the comment is: ', np.max(df['Vector_Size']))

    proportion_no_comment=np.sum(df['Vector_Size']==0)/len(df['Vector_Size'])*100
    print('The percentage of vote without comment is: ', proportion_no_comment)
    proportion_one_word=np.sum(df['Vector_Size']==1)/len(df['Vector_Size'])*100
    print('The percentage of comment with one word is: ', proportion_one_word)
    proportion_two_word=np.sum(df['Vector_Size']==2)/len(df['Vector_Size'])*100
    print('The percentage of comment with 2 words is: ', proportion_two_word)
    
    
def sentiment_remove_vectorSize_zero(df):
    df = df[df["Vector_Size"] != 0]
    
    return df


def com_vote_sentiment_analysis(df_ref, df_com_year_, year):
    #filter the ref df
    df_ref_year = df_ref[df_ref['Year']==int(year)]
    #extract the source for the community df for 1 year
    source_per_com = df_com_year_.groupby('Community').apply(lambda x : x['Source'])
    pos_vote_prop = []
    neg_vote_prop = []
    neu_vote_prop = []
    pos_sentScore_prop = []
    neg_sentScore_prop = []
    neu_sentScore_prop = []
    
    #loop over all communities of this year
    for n in range (df_com_year_['Community'].max()+1):
        com = list(source_per_com[n].values)
        #extract rows of the ref df based on the source which are in the community n
        df_com = df_ref_year[df_ref_year['Source'].isin(com)].reset_index(drop=True)
        #extract proportion for theses sources
        prop_vote_pos_com = np.sum(df_com['Vote']==1)/len(df_com['Vote'])
        prop_vote_neg_com = np.sum(df_com['Vote']==-1)/len(df_com['Vote'])
        prop_vote_neu_com = np.sum(df_com['Vote']==0)/len(df_com['Vote'])
        prop_sentScore_pos_com = sum(np.array(df_com['sent_score']>=0.05))/len(df_com['sent_score'])
        prop_sentScore_neg_com = sum(np.array(df_com['sent_score']<-0.05))/len(df_com['sent_score'])
        prop_sentScore_neu_com = sum(np.abs(df_com['sent_score'])<0.05)/len(df_com['sent_score'])
        pos_vote_prop.append(prop_vote_pos_com)
        neg_vote_prop.append(prop_vote_neg_com)
        neu_vote_prop.append(prop_vote_neu_com)
        pos_sentScore_prop.append(prop_sentScore_pos_com)
        neg_sentScore_prop.append(prop_sentScore_neg_com)
        neu_sentScore_prop.append(prop_sentScore_neu_com)
    #create the df
    years_ = [int(year)]*(df_com_year_['Community'].max()+1)
    com = list(range(df_com_year_['Community'].max()+1))
    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Pos_vote_prop', 'Neg_vote_prop', 'Neu_vote_prop', 'Pos_sentScore_prop', 'Neg_sentScore_prop', 'Neu_sentScore_prop'])
    df_stat_com['Year'] = years_
    df_stat_com['Com_nbr'] = com
    df_stat_com['Pos_vote_prop'] = pos_vote_prop
    df_stat_com['Neg_vote_prop'] = neg_vote_prop
    df_stat_com['Neu_vote_prop'] = neu_vote_prop
    df_stat_com['Pos_sentScore_prop'] = pos_sentScore_prop
    df_stat_com['Neg_sentScore_prop'] = neg_sentScore_prop
    df_stat_com['Neu_sentScore_prop'] = neu_sentScore_prop
    
    return df_stat_com


def community_sentiment_analysis_per_year(df):
    # Vectorizing the df    
    data = comment_vectorizing(df)
    # Removing comments of vector size = 0 for the sentiment analysis by community
    data = sentiment_remove_vectorSize_zero(data)

    years = data['Year'].unique()
    dict_com = {}
    for year in years:
        path = 'df_community_'+ str(int(year)) + '.csv'
        df_ = load_com_csv(path)
        dict_com[str(int(year))]=df_

    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Pos_vote_prop', 'Neg_vote_prop', 'Neu_vote_prop', 'Pos_sentScore_prop', 'Neg_sentScore_prop', 'Neu_sentScore_prop'])
    for year in dict_com.keys():
        df_com_year = dict_com[str(year)]
        stat_com = com_vote_sentiment_analysis(data, df_com_year, year)
        df_stat_com = pd.concat([df_stat_com, stat_com], ignore_index=True)
        
    return df_stat_com



def plot_dist_sentScore_per_com(df, years):
    # Melt the DataFrame to make it suitable for Seaborn
    df_melt = pd.melt(df[df['Year'].isin(years)], id_vars=['Com_nbr', 'Year'], value_vars=['Pos_sentScore_prop', 'Neg_sentScore_prop', 'Neu_sentScore_prop'],
                        var_name='Type of sentiment', value_name='Pourcentage')

    # Create a facet grid with a separate plot for each year
    g = sns.catplot(x='Com_nbr', y='Pourcentage', hue='Type of sentiment', col='Year', data=df_melt, kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'SentScore percentage')
    g.fig.suptitle('Vote percentage per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=3, title=None, frameon=False)

    # Show the plot
    plt.show()
    




   

################### End #####################