# Exploring Influence in Wikipedia Requests for Adminship
by Gaëlle Verdon, Jean-Daniel Rouveyrol, Emma Boehly, David Bekri and Alexandre Maillard



## Abstract : 
This project aims to understand how people interact and form communities in the Wikipedia Requests for Adminship (RfA) dataset. Given the interesting fact that voters can see the votes and comments casted by other voters at all times, one can think it can result in their opinion being influenced. Instead of trying to predict if adminship requests will be successful or not, we're rather interested in understanding the cascading influence of early votes as well as comments, how certain patterns develop, and if influential groups emerge in the voting network.\
\
To grasp the complex interactions in the Wikipedia RfA dataset, we use different indicators that help us see how the community is influenced. We focus on two main things: the voting network and the comments that come with it.\
\
We look at what time users vote, and in particular how early votes can influence the votes of users voting later. By using network analysis, we look into potential communities formed by voters, trying to find links between those communities and spotting patterns. We also analyze user comments to understand the feelings, reasons, and discussions around adminship requests through sentiment analysis and topic modelling. We also managed to find an additional dataset that allows us to look at the number of edits of users over time.


## Research Question :
In this study, we would like to answer these questions:
- How an open, debated voting system can wield influence and shape collective decisions ? 
- How does a user's social network within Wikipedia influence their decision to vote for adminship?
- How can we find relationships of influence in a vote such as wikipedia RfA?

### Additional datasets :
Not all of the following datasets have been found for the moment, but they remain potential sources for future analysis, which is why we have included them below. 
- https://data.world/wikimedia/monthly-wikimedia-editor-activity : This dataset contains the monthly activity of all the wikipedia users from 2001 to 2015. We’ll use it to measure the influence of the activity on vote results, while distinguishing recent activity and overall activity before the request for adminship.


## Methods :

### Voting time :
- We study the dynamics of voting behavior over time, to see if voting time relates to the final outcome of the election.
- We establish rounds of votes for a given target.
- We investigated the voting time of sources in conjunction with the number of vote they cast, as well as the voting time distribution for each election.
- We established a classification model to predict elections based on early votes, in particular through accuracy, precision, and recall metrics.

### Sentiment analysis of the comments :
- We performed a sentiment analysis on the comments using the NLTK module Vader, to see if the results of our analysis correlate with the actual votes casted by the voters, and understand if strongly opinionated comments can influence others more than others. We also tried to categorized the comments into categories (positive, negative, and neutral) to see if the comments correlate with the casted votes for each year
- We looked at communities of voters to see to see if the sentiment analysis helps to correlate those communities' comments with the ones found through network analysis.

### Topic modelling and analysis of the comments :
- We performed topics detection to find the most frequently used topics and see if they have an order of appearance as the election progresses and any particular weight in the result. This analysis shows the positive or negative influence of a certain topic and its importance according to the order in which it appears. 
  
### Edits from users :
- With the help of the supplementary dataset, we look into the evolution of users' edit over time to analyse the effect of the revisions made by a user in the decision making process of wikipedia-rfa.
- We checked the impact of the revisison in the influence between community and within community. 

### Network Analysis : 
- We build network graphs of the voters (Source) to have a visualization of how they are linked to each other and see if we can extract clusters from them. Ideally we would like to do this plot interactive to visualize the evolution over time and see if the clusters are stable/similar or not.
- Cluster positive and negative votes for each election : this point will enable us to see whether clusters overlap across elections and whether we can create groups of users who share the same votes/ideas.
- Extract communities for each vote (support, opposition, neutral) and assess their quality using modularity.
- Extract voting time pattern in the vote to see if some voters tend to vote early and others later, enabling us to infer who may have more chance to be the influencer (because voting at the beginning) or to get influenced (because voting at the end).
- Studying the evolution of votes over time to see how the supporting or opposing side becomes dominant in the timeline of voting. In particular, we would like to see if there is some wave of support or opposition that may swing some elections. This may also enable us to discriminate between heated debate and more homogeneous decisions.

  
### Combination of both
The final stage of this project will be to combine the two analyses (network and comment) to cross-check the clusters to see if the groups of people are the same and to see if the conclusions drawn from one method apply to the second. 

The primary objective of these various analyses is to examine whether distinct groups, drawn from both network structure and comments, exhibit shared characteristics, including common modes of expression and shared ideas. Furthermore, these analyses aim to uncover how influence is manifested within both the network dynamics and the qualitative content of comments.



## Proposed Timeline :
17.11.23 : P2 Milestone - Pre-processing our data and begin first analysis\
08.12.23 : Network and Comment Analysis complete\
15.12.23 : Combination Analysis complete\
22.12.23 : P3 Milestone - Final milestone: Data Story

## Team Contribution : 
- Network Analysis :
  - Visualisation and Studying the evolution of votes over time : Emma
  - First analysis of the community : Gaelle
- Comment Analysis :
  - Topics modelling : Jean-Daniel
  - Sentiment Analysis : David
- Edits from user : Alexandre
