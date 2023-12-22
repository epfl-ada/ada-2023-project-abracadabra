# Exploring Influence in Wikipedia Requests for Adminship
by Gaëlle Verdon, Jean-Daniel Rouveyrol, Emma Boehly, David Bekri and Alexandre Maillard

## Website:
https://epfl-ada.github.io/ada-2023-project-abracadabra/

## Abstract : 
This project aims to understand how people interact and form communities in the Wikipedia Requests for Adminship (RfA) dataset. Given the interesting fact that voters can see the votes and comments casted by other voters at all times, one can think it can result in their opinion being influenced. Instead of trying to predict if adminship requests will be successful or not, we're rather interested in understanding the cascading influence of early votes as well as comments, how certain patterns develop, and if influential groups emerge in the voting network.\
\
To grasp the complex interactions in the Wikipedia RfA dataset, we use different indicators that help us see if there are community-specific features and see how the community is influenced. We focus on two main things: the voting network and the comments that come with it.\
\
We look at what time users vote, and in particular how early votes can influence the votes of users voting later. By using network analysis, we look into potential communities formed by voters, trying to find links between those communities and spotting patterns. We also analyze user comments to understand the feelings, reasons, and discussions around adminship requests through sentiment analysis and topic modelling. We also managed to find an additional dataset that allows us to look at the number of edits of users over time.


## Research Question :
In this study, we would like to answer these questions:
- How an open, debated voting system can wield influence and shape collective decisions ? 
- How does a user's social network within Wikipedia influence their decision to vote for adminship?
- How can we find relationships of influence in a vote such as wikipedia RfA?

### Additional dataset :
- https://data.world/wikimedia/monthly-wikimedia-editor-activity : This dataset contains the monthly activity of all the wikipedia users from 2001 to 2015. We use it to measure the influence of the activity on vote results, while distinguishing recent activity and overall activity before the request for adminship.

## Methods :

### Voting time :
- We study the dynamics of voting behavior over time, to see if voting time relates to the final outcome of the election.
- We establish rounds of votes for a given target.
- We investigated the voting time of sources in conjunction with the number of vote they cast, as well as the voting time distribution for each election.
- We established a classification model to predict elections based on early votes, in particular through accuracy, precision, and recall metrics.


### Sentiment analysis of the comments :
- We performed a sentiment analysis on the comments using the NLTK module Vader, to see if the results of our analysis correlate with the actual votes casted by the voters, and understand if strongly opinionated comments can influence others. We also tried to categorized the comments into categories (positive, negative, and neutral) to see if the comments correlate with the casted votes for each year
- We looked at communities of voters to see to see if the sentiment analysis helps to correlate those communities' comments with the ones found through network analysis.


### Topic modelling and analysis of the comments :
- We performed topics detection to find the most frequently used topics and see if they have an order of appearance as the election progresses and any particular weight in the result. This analysis shows the positive or negative influence of a certain topic and its importance according to the order in which it appears. 

  
### Edits from users :
- With the help of an additional dataset, we look at how the number of edits is an important factor in the election of a user, as well as the evolution of edits of users over time, for example to see whether individuals making the most revisions were also among the first to vote, potentially indicating a greater influence.


### Network Analysis : 
- We build a weighted projected graph by grouping together sources that voted the same way for one or more targets, then we extract communities using Louvain's algorithm
- We analysed ths community based on several features: voting time, voting type, prediction of the result to see if we can extract any discriminating features between communities
- We looked at how communities evolved over years using Jaccard similarities. 


The primary objective of these various analyses is to examine whether distinct groups, drawn from both network structure and comments, exhibit shared characteristics, including common modes of expression and shared ideas. Furthermore, these analyses aim to uncover how influence is manifested within both the network dynamics and the qualitative content of comments.


## Team Contribution : 
- Network Analysis :
  - Data exploration, data handling, voting time analysis: Emma
  - Community extraction, community analysis based on voting time, vote type, results prediction : Gaëlle
  - Edit analysis globally and for communities : Alexandre
  - Evolution of communities over years : Jean-Daniel
- Comment Analysis :
  - Topics modelling : Jean-Daniel
  - Sentiment Analysis : David
  
