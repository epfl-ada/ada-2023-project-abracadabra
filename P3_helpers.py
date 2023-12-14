import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

import mwparserfromhell
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS

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
    with open(path, 'r') as file:
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
    df = df[df['Year'] != '2003'] # or df = df[df['Year'] != 2003]

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
    rounds = (df_timeserie.groupby('Target').apply(lambda x: compute_rounds(x, round_threshold))).rename('Round')
    df_timeserie = df_timeserie.join(rounds.droplevel(0))

    # Use the round number to compute the voting time in each round (i.e. the time between the current vote and the first vote of the round)
    Voting_time_round = df_timeserie.groupby(['Target', 'Round']).Voting_time.apply(lambda x: x - x.min())
    # Replace the column Voting_time by the voting time in each round
    df_timeserie = df_timeserie.drop(columns='Voting_time').join(Voting_time_round.droplevel([0,1]))

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
def get_progressive_mean(df):
    """ Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)

    Args:
        df (pd.DataFrame): Dataframe containing the data of the votes
        
    Returns:
        df (pd.DataFrame): Dataframe containing the data of the votes with the progressive mean of the votes in each round
    """
    # Compute the progressive mean of the votes in each round (i.e. the mean of the votes at each time step)
    progressive_mean = df.groupby(['Target', 'Round']).apply(lambda x: x.sort_values('Voting_time').Vote.cumsum() / np.arange(1, len(x)+1)).rename('progressive_mean')
    # Replace the column Vote by the progressive mean
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
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=x, y=mean_col, data=data, hue='Results', palette='tab10')
    plt.fill_between(data[data.Results == -1][x], data[data.Results == -1][var_cols[0]], data[data.Results == -1][var_cols[1]], alpha=0.2, color='tab:blue')
    plt.fill_between(data[data.Results == 1][x], data[data.Results == 1][var_cols[0]], data[data.Results == 1][var_cols[1]], alpha=0.2, color='tab:orange')
    plt.legend(loc='upper left')
    if x == 'Voting_time': plt.xlabel('Time (days)')
    elif x == 'rank': plt.xlabel('Number of votes')
    plt.ylabel('Progressive mean of the votes')
    plt.xlim(0, 24*8)
    plt.ylim(-1, 1.01)
    # Manually create legend 
    plt.legend(handles=[plt.Line2D([0], [0], color='tab:blue', lw=4, label='Rejected'),
                        plt.Line2D([0], [0], color='tab:orange', lw=4, label='Elected')])
    return plt.gca()
    
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

def get_quartiles(grouper):
    """ Compute the median, first and last quartile of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes 
        
    Returns:
        quartiles (pd.DataFrame): Dataframe containing the median, first and last quartile of the votes in each round
    """
    # Compute the median, first and last quartile of the votes in each round
    quartiles = grouper.quantile([0.25, 0.5, 0.75]).unstack(level=2).reset_index()
    quartiles.rename(columns={0.25: 'lower', 0.5: 'center', 0.75: 'upper'}, inplace=True)
    return quartiles

def get_confidence_interval(grouper):
    """ Compute the mean and 95% confidence interval of the votes in each round

    Args:
        grouper (pd.SeriesGroupBy): Groupby object containing the data of the votes        
    
    Returns:
        ci (pd.DataFrame): Dataframe containing the mean and 95% confidence interval of the votes in each round
    """
    # Compute the mean and 95% confidence interval of the votes in each round
    ci = grouper.agg(['mean', stats.sem]).reset_index()
    ci['lower'] = ci['mean'] - 1.96 * ci['sem']
    ci['upper'] = ci['mean'] + 1.96 * ci['sem']
    ci.rename(columns={'mean': 'center'}, inplace=True)
    return ci


########## COMMENTS ANALYSIS FUNCTIONS ##########

def get_parsed_comment(comment):
    """ Parse a comment using mwparserfromhell and return the text of the comment"""
    return mwparserfromhell.parse(comment).strip_code()

#Functions below are 
def tokenize_comments(comments_series):
    return [word_tokenize(comment.lower()) for comment in comments_series.tolist()]

def get_dict_representation(tokenized_comments):
    return corpora.Dictionary(tokenized_comments)

def remove_stopwords_from_dict(dictionary):
    dictionary.filter_tokens(bad_ids=[dictionary.token2id[word] for word in STOPWORDS])
    return dictionary

def get_bow_representation(tokenized_comments, dictionary):
    return [dictionary.doc2bow(comment) for comment in tokenized_comments]

def init_LDA_model(bow_corpus, dictionary, num_topics=3, passes=10):
    return models.LdaModel(bow_corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

def get_LDA_topics(lda_model, num_words=10):
    return lda_model.print_topics(num_words=num_words)

def get_LDA_topics_pipeline(comments_series, num_topics=3, num_words=10, passes=10):
    tokenized_comments = tokenize_comments(comments_series)
    dictionary = get_dict_representation(tokenized_comments)
    dictionary.filter_tokens(bad_ids=[dictionary.token2id[word] for word in STOPWORDS])
    bow_corpus = get_bow_representation(tokenized_comments, dictionary)
    lda_model = init_LDA_model(bow_corpus, dictionary, num_topics=num_topics, passes=passes)
    return get_LDA_topics(lda_model, num_words=num_words)
