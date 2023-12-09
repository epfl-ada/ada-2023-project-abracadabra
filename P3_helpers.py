import numpy as np
import pandas as pd

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
    perc_vote['Result'] = perc_vote.apply(lambda x: 1 if x['1'] >= 70 else -1, axis=1, result_type='reduce')
    # Replace results in duplicates with results in perc_vote
    duplicates['Results'] = duplicates.apply(lambda x: perc_vote.loc[(x['Target'], x['Year'])]['Result'].astype(int), axis=1)

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
    double_vote = df[df.duplicated(['Source', 'Target', 'Comment', 'Date'], keep=False) & df.Date.notnull() & (df.Vote == '0')]
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
    kde = np.histogram(voting_time, bins=100, density=True)
    deriv_kde_sign = np.sign(np.diff(kde[1]))
    local_mins = kde[0][np.append((np.roll(deriv_kde_sign, 1) - deriv_kde_sign) != 0, False)]
    y_mins = kde[1][np.append((np.roll(deriv_kde_sign, 1) - deriv_kde_sign) != 0, False)]
    # only keep the minima with a y value < 0.1 and a x value is between 10 and 1e4
    round_threshold = local_mins[(y_mins < 0.1) & (local_mins > 10) & (local_mins < 1e3)][0]

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
