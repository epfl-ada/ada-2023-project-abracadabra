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
Studying the evolution of votes over time to see how the supporting or opposing side becomes dominant is the timeline of voting. In theory, we would like to see if there is some wave of support or opposition that may swing some elections. This may also enable us to discriminate heated debate and more homogeneous decision (assuming that will we see more influential trend in the first case)  Emma
Maybe try to do prediction (not sure if relevant)

Delving more into the comment analysis, our goal is to check whether the clusters and trends extracted using only the votes can be reproduced and refined with only the comment. We will notably:
Group the voters given the comment (similarity assessment). The resulting network will then be compared to the one we got in the first step to see if the clusters are similar (maybe extract subgroup from the one we got with only the votes) Jean-Daniel
From these clusters, try to extract trends in expression style, comment length and vocabulary to better qualify and describe each group of voters David (Sentiment) + Topic modelling (Jean-Daniel)
-> + clustering David + Jean-Daniel
See if these trends are involved in the influence patterns : for example if some type of words/vocabulary spread more and more in the comments, if comment became longer when elections is more debated (no clear decisions) indicating a desire to convince, check if the same type of concerns about the votees (target) are raised in certain group of voters, etc. P3




