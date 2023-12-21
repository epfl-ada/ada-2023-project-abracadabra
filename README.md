# Exploring Influence in Wikipedia Requests for Adminship
by Gaëlle Verdon, Jean-Daniel Rouveyrol, Emma Boehly, David Bekri and Alexandre Maillard
This is a test

## Abstract : 
This project aims to understand how people interact and form communities in the Wikipedia Requests for Adminship (RfA) dataset. An interesting fact is that, voters can see the votes and comments casted by other voters at all times, which can result in their opinion being influenced. Instead of trying to predict if adminship requests will be successful or not, we're interested in understanding the cascading influence of early votes, how certain patterns develop, and if there are influential groups in the voting network.\
\
To grasp the complex interactions in the Wikipedia RfA dataset, we use different indicators that help us see how the community is influenced. We focus on two main things: the voting network and the comments that come with it.\
\
By using network analysis, we look at how votes change over time, finding important nodes and spotting patterns in the RfA dataset. We also analyze user comments to understand the feelings, reasons, and discussions around adminship requests through sentiment analysis and categorizing themes.





## Research Question :
In this study, we would like to answer these questions:
- How an open, debated voting system can wield influence and shape collective decisions ? 
- How does a user's social network within Wikipedia influence their decision to vote for adminship?
- How can we find relationships of influence in a vote such as wikipedia RfA?

## Additional dataset :
Not all of the following datasets have been found for the moment, but they remain potential sources for future analysis, which is why we have included them below. 
- https://data.world/wikimedia/monthly-wikimedia-editor-activity : This dataset contains the monthly activity of all the wikipedia users from 2001 to 2015. We’ll use it to measure the influence of the activity on vote results, while distinguishing recent activity and overall activity before the request for adminship.
- https://snap.stanford.edu/data/wiki-meta.html : Those are 2 datasets on edits on wikipedia with users names. Those datasets, as the first one, will be useful to check the influence of the activity in the voting process. 
- A dataset on the subjects of articles published or modified by users.


## Methods : 
### Network Analysis : 
- Build network graphs of the voters (Source) to have a visualization of how they are linked to each other and see if we can extract clusters from them. Ideally we would like to do this plot interactive to visualize the evolution over time and see if the clusters are stable/similar or not.
- Cluster positive and negative votes for each election : this point will enable us to see whether clusters overlap across elections and whether we can create groups of users who share the same votes/ideas.
- Extract communities for each vote (support, opposition, neutral) and assess their quality using modularity.
- Extract voting time pattern in the vote to see if some voters tend to vote early and others later, enabling us to infer who may have more chance to be the influencer (because voting at the beginning) or to get influenced (because voting at the end).
- Studying the evolution of votes over time to see how the supporting or opposing side becomes dominant in the timeline of voting. In particular, we would like to see if there is some wave of support or opposition that may swing some elections. This may also enable us to discriminate between heated debate and more homogeneous decisions.

  
### Comment Analysis :
- Group the voters given the comment (similarity assessment) and try to extract trends in expression style, comment length and vocabulary to better qualify and describe each group of voters. For instance, we could try to determine if larger size commentaries tend to have a bigger influence, or trigger heated debates more frequently than smaller ones.
- Topics detection to find the most frequently used topics and see if they have an order of appearance as the election progresses and any particular weight in the result. This analysis shows the positive or negative influence of a certain topic and its importance according to the order in which it appears. 
- Perform a sentiment analysis on the comments and see if the results of this analysis correlate with the actual votes casted by the voters, and understand if strongly opinionated comments can influence others more than others. We can also Identify groups (with clustering) and try to see if these groups correspond  with the ones found through network analysis. 


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
  - Extract voting time pattern and Modularity to evaluate community quality : Gaelle
  - Cluster based on the vote : Alexandre
- Comment Analysis :
  - Topics modelling : Jean-Daniel
  - Sentiment Analysis : David
