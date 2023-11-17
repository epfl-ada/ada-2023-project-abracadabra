# Exploring Influence in Wikipedia Requests for Adminship
by Gaëlle Verdon, Jean-Daniel Rouveyrol, Emma Boehly, David Bekri and Alexandre Maillard

## Table of Contents 📋
1. [Abstract](#abstract)
2. [Research Questions](#research-questions)
3. [Proposed Additional Datasets](#proposed-additional-datasets)
4. [Methods](#methods)
5. [Proposed Timeline](#proposed-timeline--internal-milestones-im-)
6. [Organization Within the Team](#organization-within-the-team)
7. [Questions for TAs](#questions-for-tas-optional)


## Abstract : 
This project endeavors to unravel the intricate web of interactions and community dynamics inherent in the Wikipedia Requests for Adminship (RfA) dataset. Rather than aiming to predict the success or failure of adminship requests, our focus lies in understanding the cascading influence of early votes, the emergence of community-driven patterns, and the existence of influential clusters within the voting network.
To comprehend the intricate dynamics of interactions within the Wikipedia Requests for Adminship (RfA) dataset, we employ various indicators, each offering a distinct lens through which community influence is observed. Two primary facets of analysis emerge — the voting network and the accompanying comments.
Employing network analysis, we dissect the temporal evolution of votes, identifying influential nodes and discerning cascading patterns within the Wikipedia Requests for Adminship (RfA) dataset.
Qualitative insights are gleaned through sentiment analysis and thematic categorization of user comments, providing a nuanced understanding of sentiment, rationales, and discourse surrounding adminship requests.


## Research Question :
In this study, we would like to answer these questions:
- How an open, debated voting system can wield influence and shape collective decisions ? 
- How does a user's social network within Wikipedia influence their decision to vote for adminship?
- How can we find relationships of influence in a vote such as wikipedia RfA?

## Additional dataset :
Not all of the following datasets have been found for the moment, but they remain potential sources for future analysis, which is why we have included them below. 
- https://data.world/wikimedia/monthly-wikimedia-editor-activity : This dataset contains the activity by months of all the wikipedia’s user since 2001 until 2015. We’ll use it to mesure the influence of the activity on the result of the vote with the recent activity and the activity overall before the request for adminship.
- A dataset of the subjects of articles published or modified by users 

## Methods : 
### Network Analysis : 
- Build network graphs of the voters (source) to have a visualization of how they are linked to each other and see if we can extract clusters from them. Ideally we would like to do this plot interactive to visualize the evolution through time and see if the clusters are stable/similar or not (good starting point because we would like to have some stability to have a meaningful analysis).
- Cluster positive and negative votes for each election : this point will enable us to see whether clusters overlap across elections and whether we can create groups of users who share the same votes/ideas.
- Modularity to evaluate community quality (clustering the support votes, of opposition).
- Extract voting time pattern in the vote to see if some voters tend to vote early and others later, enabling us to infer who may have more chance to be the influencer (because voting at the beginning) or to get influenced (because voting at the end).
- Studying the evolution of votes over time to see how the supporting or opposing side becomes dominant in the timeline of voting. In theory, we would like to see if there is some wave of support or opposition that may swing some elections. This may also enable us to discriminate heated debate and more homogeneous decision (assuming that will we see more influential trend in the first case).
  
### Comment Analysis :
- Group the voters given the comment (similarity assessment) and try to extract trends in expression style, comment length and vocabulary to better qualify and describe each group of voters.
- Topics detection to find the most frequently used topics and see if they have an order of appearance as the election progresses and any particular weight in the result.
- Sentiment Analysis to create cluster.

### Combination of both
The final stage of this project will be to combine the two analyses (network and comment) to cross-check the clusters to see if the groups of people are the same and to see if the conclusions drawn from one method apply to the second. 

The main aim of these various analyses will be to see if there are groups of people both with the network and with the comments who share common points such as ways of expressing themselves, common ideas, etc...


## Proposed Timeline :
- 17.11.23 : P2 Milestone - Pre-processing our data and begin first analysis
- 08.12.23 : Network and Comment Analysis complete
- 15.12.23 : Combination Analysis complete
- 22.12.23 : P3 Milestone - Final milestone: Data Story
## Team Contribution : 
- Network Analysis :
  - Visualisation and Studying the evolution of votes over time : Emma
  - Extract voting time pattern and Modularity to evaluate community quality : Gaelle
  - Cluster based on the vote : Alexandre
- Comment Analysis :
  - Topics modelling : Jean-Daniel
  - Sentiment Analysis : David

## Questions for TAs :
-
-
-








##### First Version : 
Objective: study the dynamics of online communities and their decision-making processes. How an open, debated voting system can wield influence and shape collective decisions. (large scale project)
--> Se défendre contre des communautés extrémistes qui cherchent à renverser des élections en convaincant les autres de voter comme eux. Système auto-géré par les utilisateurs, large scale project qui permet 
Si on trouve pas d'influenceurs alors on peut en conclure que le système est plutôt stable et que les gens votent en fonction de leur opinion et non pas en fonction de ce que les autres pensent => smooth l'avis général et rend le résultat le plus objectif positif. => Notably, we explore the ability of Wikipedia's community to maintain stability despite scaling issues, which often afflict other online communities. And in fact, this could be accounted by the election system, so examining opposed votes and related comments especially can help us understand the self-organization and hierarchy-building processes within the Wikipedia community.
=> étude incrémental des votes et des commentaires pour voir si on peut prédire l'issue d'une élection

• trouver dataset qui contient les stats de Wikipedia au cours du temps (grosse croissance entre 2003 et 2013)

To this end we will mainly try to extract group dynamics in the vote in themselves and also in the comment. These are the research direction on which we will focus:
Build network graphs of the voters (source) to have a visualization of how they are linked to each other and see if we can extract clusters from them. Ideally we would like to do this plot interactive to visualize the evolution through time and see if the clusters are stable/similar or not (good starting point because we would like to have some stability to have a meaningful analysis) Emma
Cluster positive and negative votes for each election and compute Manhattan distance to compare the clusters Alexandre
Modularity to evaluate community quality (clustering the support votes, of opposition) Gaelle
Extract voting time pattern in the vote to see if some voters tend to vote early and others later, enabling us to infer who may have more chance to be the influencer (because voting at the beginning) or to get influenced (because voting at the end) Gaelle 
Studying the evolution of votes over time to see how the supporting or opposing side becomes dominant in the timeline of voting. In theory, we would like to see if there is some wave of support or opposition that may swing some elections. This may also enable us to discriminate heated debate and more homogeneous decision (assuming that will we see more influential trend in the first case)  Emma
Maybe try to do prediction (not sure if relevant)

Delving more into the comment analysis, our goal is to check whether the clusters and trends extracted using only the votes can be reproduced and refined with only the comment. We will notably:
Group the voters given the comment (similarity assessment). The resulting network will then be compared to the one we got in the first step to see if the clusters are similar (maybe extract subgroup from the one we got with only the votes) Jean-Daniel
From these clusters, try to extract trends in expression style, comment length and vocabulary to better qualify and describe each group of voters David (Sentiment) + Topic modelling (Jean-Daniel)
-> + clustering David + Jean-Daniel
See if these trends are involved in the influence patterns : for example if some type of words/vocabulary spread more and more in the comments, if comment became longer when elections is more debated (no clear decisions) indicating a desire to convince, check if the same type of concerns about the votees (target) are raised in certain group of voters, etc. P3




