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
def pdf_voting_time(df):
    data = df[(df['Voting_time'] != 0) & (df['Voting_time'] < 24*8)]
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(data=data, x='Voting_time', ax=ax, bins=100, stat='percent', log_scale=(False, False), hue='Year', palette='tab10', multiple='stack')
    ax.set_title('Histogram of voting time')
    ax.set_xlabel('Voting time in the round (hours)')
    ax.set_ylabel('Percentage of votes')
    ax.set_xlim(0, 24*8)
    plt.savefig('Figures/distribution_voting_time.png', dpi=300)
    plt.show()

def cdf_voting_time(df):
    data = df[(df['Voting_time'] != 0) & (df['Voting_time'] < 24*8)]
    fig, ax = plt.subplots(figsize=(10,4))
    sns.ecdfplot(data=data, x='Voting_time', ax=ax, stat='percent', log_scale=(False, False), hue='Year', palette='tab10', complementary=False)
    ax.set_title('ECDF of voting time')
    ax.set_xlabel('Voting time in the round (hours)')
    ax.set_ylabel('Percentage of votes')
    ax.set_xlim(0, 24*8)
    plt.savefig('Figures/cdf_voting_time.png', dpi=300)
    plt.show()


def get_progressive_mean(df):
    """ Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the progressive mean of the votes in each round
    """
    # Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)
    progressive_mean = df.groupby(['Target', 'Round']).apply(lambda x: x.sort_values('Voting_time').Vote.cumsum() / np.arange(1, len(x)+1)).rename('progressive_mean')
    df = df.join(progressive_mean.droplevel([0,1]))

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
        stats = stats.join(grouper.Voting_time.median(), on=['Results', on])

    stats.Voting_time = time_to_float(stats.Voting_time)
    return stats

def get_quartiles(grouper, on):
    """ Compute the median, first and last quartile of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes 
        
    Returns:
        quartiles (pd.DataFrame): Dataframe containing the median, first and last quartile of the votes in each round
    """
    # Compute the median, first and last quartile of the votes in each round
    quartiles = grouper.progressive_mean.quantile([0.25, 0.5, 0.75]).unstack(level=2).reset_index()
    quartiles.rename(columns={0.25: 'lower', 0.5: 'center', 0.75: 'upper'}, inplace=True)
    quartiles = add_voting_time(grouper, quartiles, on)
    return quartiles

def get_confidence_interval(grouper, on):
    """ Compute the mean and 95% confidence interval of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes        
    
    Returns:
        ci (pd.DataFrame): Dataframe containing the mean and 95% confidence interval of the votes in each round
    """
    # Compute the mean and 95% confidence interval of the votes in each round
    ci = grouper.progressive_mean.agg(['mean', stats.sem]).reset_index()
    ci['lower'] = ci['mean'] - 1.96 * ci['sem']
    ci['upper'] = ci['mean'] + 1.96 * ci['sem']
    ci.rename(columns={'mean': 'center'}, inplace=True)
    ci = add_voting_time(grouper, ci, on)
    return ci

# Scatter plot of the progressive mean by voting time and vote number
def plot_time_distribution(df, x):
    data = df[df.Voting_time < pd.Timedelta('8 days')]
    data.Voting_time = time_to_float(data.Voting_time)
    fig, axes = plt.subplots(1,2,figsize=(20, 6))
    # Plot the histogram in 2D with log scale colorbar
    sns.histplot(data=data[data.Results==-1], x=x, y='progressive_mean', ax=axes[0], color='tab:blue', cbar=True, norm=LogNorm(), vmin=None, vmax=None, bins=100)
    sns.histplot(data=data[data.Results==1], x=x, y='progressive_mean', ax=axes[1], color='tab:orange', cbar=True, norm=LogNorm(), vmin=None, vmax=None, bins=100)
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
        fig.suptitle('Histogram of the progressive mean over time')
    elif x == 'Vote_number': 
        axes[0].set_xlabel('Number of votes casted')
        axes[1].set_xlabel('Number of votes casted')
        fig.suptitle('Histogram of the progressive mean over the number of votes casted')
    plt.savefig('Figures/hist_progressive_mean_over_' + x.lower() + '.png', dpi=300)
    plt.show()

def predict_results(df, n_first_votes, n_folds):
    """ Predict the results of the votes using the progressive mean of the votes in each round

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        n_first_votes (np.array): Array containing the number of first votes to use for the prediction
        n_folds (int): Number of folds to use for the cross validation

    Returns:
        (pd.DataFrame): Dataframe containing the results of the prediction
    """
    X = df[df.Vote_number <= n_first_votes][['Vote_number', 'progressive_mean', 'Target']]
    X.Target = X.Target.astype('category').cat.codes
    y = df[df.Vote_number <= n_first_votes]['Results']
    clf = GradientBoostingClassifier(random_state=0)
    # Perform cross validation by taking batches of 10 targets (and their votes) 
    scores = cross_validate(clf, X, y, cv=GroupKFold(n_splits=n_folds), groups=X.Target, scoring=('accuracy', 'precision', 'recall'))
    return scores

def early_vote_prediction(df, n_first_votes, n_folds=10):
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
        res = pd.DataFrame(predict_results(df, n, n_folds))
        scores.loc[n, :] = res[['test_accuracy', 'test_precision', 'test_recall']].values
    scores.reset_index(inplace=True)
    return scores

def plot_prediction_scores(scores):
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
        ax.set_ylim(0.6, 1.01)
        ax.set_xlim(scores.nb_first_votes.min(), scores.nb_first_votes.max())
        ax.set_title(metric + ' of the prediction')
        ax.set_xlabel('Number of first votes')
        ax.set_ylabel(metric)
    plt.savefig('Figures/prediction_scores.png', dpi=300)
    plt.show()


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

def edge_format_for_jsim(max_sim):
    ... 

def get_directed_graph_for_jsim(network):
    G = nx.DiGraph()
    for layer, connections in network.items():
        for node, (next_node, weight) in connections.items():
            G.add_edge((layer.split('-')[0], node), (layer.split('-')[1], next_node), weight=weight)
    return G

def plot_graph_jsim(G):
    
    return

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
    #loop over all communities of this year
    for n in range (df_com_year_['Community'].max()+1):
        com = list(source_per_com[n].values)
        #extract rows of the ref df based on the source which are in the community n
        df_com = df_ref_year[df_ref_year['Source'].isin(com)].reset_index(drop=True)

        TP = np.sum((df_com['Vote']== df_com['Results']) & (df_com['Vote']==1))
        TN = np.sum((df_com['Vote']== df_com['Results']) & (df_com['Vote']==-1))
        FN = np.sum((df_com['Vote']!= df_com['Results']) & (df_com['Vote']==-1))
        FP = np.sum((df_com['Vote']!= df_com['Results']) & (df_com['Vote']==1))
        N = len(df_com['Vote'])
        accuracy_ = (TP + TN)/N
        if (TP == 0 & FN == 0) : recall_ = 0
        else: recall_ = TP / (TP + FN)
        precision_ = TP / (TP + FP)
        specificity_ = TN/(TN+FN)
        accuracy.append(accuracy_)
        recall.append(recall_)
        precision.append(precision_)
        specificity.append(specificity_)
    
    #create the df
    years_ = [int(year)]*(df_com_year_['Community'].max()+1)
    com = list(range(df_com_year_['Community'].max()+1))
    df_stat_com = pd.DataFrame(columns=['Year', 'Com_nbr', 'Accuracy', 'Precision', 'Recall', 'Specificity'])
    df_stat_com['Year'] = years_
    df_stat_com['Com_nbr'] = com
    df_stat_com['Accuracy'] = accuracy
    df_stat_com['Precision'] = precision
    df_stat_com['Recall'] = recall
    df_stat_com['Specificity'] = specificity
   
    return df_stat_com

def plot_recall_accuracy_precision_per_com(df, years):
    """ Plot the recall, accuracy and precision for each community and for specific years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
        year (list): Specific years on which we want our plot
    """
    # Transform the df in an appropriate form
    df_melt = pd.melt(df[df['Year'].isin(years)], id_vars=['Com_nbr', 'Year'], value_vars=['Accuracy', 'Precision', 'Recall', 'Specificity'],
                        var_name='Stat', value_name='Pourcentage')

    # Create the plot
    g = sns.catplot(x='Com_nbr', y='Pourcentage', hue='Stat', col='Year', data=df_melt, kind='bar', palette='colorblind')

    # Add labels and title
    g.set_axis_labels('Community', 'Proportion')
    g.fig.suptitle('Accuracy, Recall, Precision and Specificity per community and per year', y=1.02)

    # Move legend to below the graph
    g.fig.subplots_adjust(bottom=0.2)
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.1), ncol=4, title=None, frameon=False)

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
    N = len(group['Vote'])
    
    accuracy = (TP + TN) / N if N != 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    
    return pd.Series({
        'Accuracy': np.round(accuracy,3),
        'Precision': np.round(precision,3),
        'Recall': np.round(recall,3)
    })

def plot_accuracy_recall_precision_on_whole_years(df):
    """ Plot the recall, accuracy and precision per years
    Args:
        df (pd.DataFrame): Dataframe containing the values to plot
    """
    # Transform the df in an appropriate form
    melted_df = pd.melt(df[['Year', 'Accuracy',  'Precision', 'Recall']], id_vars=['Year'], var_name='Statistic', value_name='Value')
    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 6))  
    sns.barplot(x='Year', y='Value', hue='Statistic', data=melted_df, palette='muted')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Voting Time (hours)')
    plt.title('Mean and Median Voting Time per Year')

    # Add legend
    plt.legend(title='Vote', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    # Show the plot
    plt.show()
################### End #####################