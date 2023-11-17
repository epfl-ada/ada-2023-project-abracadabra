import numpy as np
import pandas as pd


# Start a new round each time the interval between the current voting and the first voting time of the round is greater than the round threshold
def compute_rounds(data, round_threshold):
    data = data.sort_values('Voting_time')
    voting_time = data.Voting_time.values
    rounds = np.array([1])
    for i in range(1,len(voting_time)):
        # If the candidate is elected, we stop the round but check if there is no other vote after more than 4 month (what we consider as a new election for a candidate that may have lost its admin rights)
        if data.Results.values[i] == 1 and voting_time.max()-voting_time[i] < 24*30*4:
            round_number = rounds[-1] + (voting_time[i] > round_threshold)
            rounds = np.append(rounds, round_number*np.ones(len(voting_time)-i).astype(int))
            break
        elif (voting_time[i] > round_threshold and data.Results.values[i] != 1 and voting_time[i]-voting_time[i-1]>=24) or \
            (data.Results.values[i] == 1 and (voting_time[i] > 24*30*4 or data.Results.values[i-1] != 1)):
            rounds = np.append(rounds, rounds[-1] + 1)
            voting_time = voting_time - voting_time[i]
        elif len(rounds) >= 2 and rounds[-2] != rounds[-1]:
            rounds = np.append(rounds, rounds[-1])
            voting_time = voting_time - voting_time[i]
        else:
            rounds = np.append(rounds, rounds[-1])
            
    rounds = pd.Series(rounds, index=data.index).astype(int)
    return rounds