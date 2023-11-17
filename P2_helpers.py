import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import igraph as ig 
import leidenalg as la
import networkx as nx


from sklearn.feature_extraction.text import TfidfVectorizer

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

def select_year(data, year):
  return data[data['Year'] == year].reset_index(drop=True)

#Helpers to compute to Community 
def compute_com_size(community_list):
    community_size=np.zeros(len(community_list), dtype=int)
    for n, com in enumerate(community_list):
        community_size[n]=len(com)
    return(community_size)

def extract_community_louvain(df, year, vote):
    #Data, Year, type of vote (-1,0,1) for negative, positive or neutral respectively required
    df_year=df[df['Year']==year]
    df_year_vote=df_year[df_year['Vote']==vote]
    df_year_vote=df_year_vote[['Source', 'Target']]
    #create the network
    G=nx.from_pandas_edgelist(df_year_vote, source='Source', target='Target')
    #extract communities with Louvain algorithm 
    G_community=nx.community.louvain_communities(G, seed=1234)
    
    return G_community

def extract_community_leiden(df, year, vote):
    #Data, Year, type of vote (-1,0,1) for negative, positive or neutral respectively required
    df_year=df[df['Year']==year]
    df_year_vote=df_year[df_year['Vote']==vote]
    df_year_vote=df_year_vote[['Source', 'Target']]

    #create the network
    G=nx.from_pandas_edgelist(df_year_vote, source='Source', target='Target')
    #convert into ig
    H=ig.Graph.from_networkx(G)

    #extract communities with Leiden algorithm 
    partition = la.find_partition(H, la.ModularityVertexPartition)
    
    return partition

def compute_partition_features(partition):
    #gives the number of community we have
    nbr_community=np.max(partition.membership)
    ind_community_size=[]
    for i in range (nbr_community):
        #gives the size of community i+1
        nbr=sum(partition.membership==np.full_like(partition.membership, fill_value=i+1)) 
        ind_community_size.append(nbr)
    return nbr_community, ind_community_size

def tf_idf_matrix(comments):
    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', max_features= 5000)
    tfidf_matrix = vectorizer.fit_transform(comments.values())
    return tfidf_matrix

def get_idx_lower_bound(keys):
    keys = [item[1] for item in keys]
    lower = min(keys)
    return lower

def get_idx_upper_bound(keys):
    keys = [item[1] for item in keys]
    upper = max(keys)
    return upper

def generate_cossim_pairs(x):
    dic = {}
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[1]):
            dic[(i, j)] = x[i, j]
    return dic


# Compare leiden and louvain algorithm by looking at the number of community for each year and and type of vote
def plot_comparaison_leiden_louvain_per_type_vote(vote, df_community, df_community_leiden):
    louvain_vote=df_community[df_community['Vote']==vote][['Year', 'Total nbr of community']]
    louvain_vote['Algo']='Louvain'
    leiden_vote=df_community_leiden[df_community_leiden['Vote']==vote][['Year', 'Total nbr of community']]
    leiden_vote['Algo']='Leiden'
    louvain_vote['Year'] = louvain_vote['Year'].astype('int')
    leiden_vote['Year'] = leiden_vote['Year'].astype('int')

    df_combined = pd.concat([louvain_vote, leiden_vote], ignore_index=True)
    max = df_combined['Total nbr of community'].max()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_combined, x='Year', y='Total nbr of community', hue='Algo')

    plt.title('Number of community per year and vote '+str(vote)+' for the Louvain and Leiden algorithm')
    plt.xlabel('Year')
    plt.ylabel('Number of community')
    plt.yticks(np.arange(0, max + 1, 4))
  
    plt.show()

#Extract communities with leiden algorithm using a directed graph
def extract_community_leiden_directed(df, year, vote):
    df_year=df[df['Year']==year]
    df_year_vote=df_year[df_year['Vote']==vote]
    df_year_vote=df_year_vote[['Source', 'Target']]

    #create the network
    G=nx.from_pandas_edgelist(df_year_vote, source='Source', target='Target', create_using=nx.DiGraph())

    #convert into ig
    H=ig.Graph.from_networkx(G)

    #extract communities with Leiden algorithm 
    partition = la.find_partition(H, la.ModularityVertexPartition)
    return partition
