<link rel="stylesheet" href="/css/styles.css">

<h1 style="text-align: center;"> The Power of Online Communities and the Art of Large Scale Decision Process </h1>

How wonderful are the online communities! In the vast landscape of the digital era, the emergence of online communities stands as a demonstration of the transformative power of the internet. This virtual world offers a unique and powerful mean for people from every part of the globe to connect and exchange ideas, fostering the development of new tools and platforms from collaborative efforts, of which Wikipedia is a prime example. However, the growth of these communities brings along a new set of challenges related to the organization and management at such a large scale: How can thousands to millions of users make a decision together and agree on common rules and goals?

To address this question, we study the case of Wikipedia, the largest online encyclopedia, which has been built and maintained by millions of volunteers over the past two decades. In particular, we focus on the English Wikipedia, which is the largest Wikipedia edition with 7 million articles and 46 million registered users and for which we have access to a comprehensive dataset spanning from 2003 to 2013, focusing on Wikipedia elections, namely the : “Requests for Adminship” (RfA), as well as a dataset of the number of monthly edits per user over the same period.
The relevance of choosing Wikipedia as a case study is twofold. First, Wikipedia is a unique example of a large-scale online community that has been able to sustain itself over the years and despite its impressive growth. Second, Wikipedia has a well-defined and transparent process for electing administrators with publicly available data, which makes it an ideal case study for understanding the dynamics of collective decision-making in online communities.


--------------------------

## *Time Series and source EMMA* 

One first step towards understanding the process of collective decision-making is to study the dynamics of voting behavior over time. In particular, we are interested in the temporal patterns of votes and how they relate to the final outcome of the election.
To extract the timing of votes, we used the timestamps given in the raw data and defined the first vote casted for a target as the starting point of an election round. From there, we wondered if we could extract groups of voters that would vote earlier or later in the election which would be indicative of influential and influenced voters respectively.

PLOT DISTRIB VOTING TIME WITH YEAR TO ADD

By seeing the distribution of voting times, we were surprised to discover a bimodal distribution (when using a log scale) and in conjunction with our idea of splitting voters into two groups, we were tempted to explain this phenomenon as indicating the actual existence of two groups of voters separated by their voting time. However, after some investigation of the data and looking into the literature about the RfA rules, we quickly realized that this bimodal distribution can actually be easily explained by the fact that some targets that were not elected would be re-nominated and thus would have a second round of votes (or even more). And after delving deeper into the data and with the information we found on the RfA process, we were able to define properties of the voting time enabling us to distinguish multiple rounds of votes for a given target. We could also check that the resulting rounds were consistent with the data by extracting some comments coherent with our assumptions.
EXAMPLE ROUND NUMBER 7




### *Exemple how to add img* 
<iframe src="assets/img/avatar-icon.png" width="750px" height="530px" frameborder="0" position="relative">Title</iframe>

bislls

--------------------------

## *PART 2* 

hsjjs
<iframe src="assets/img/bgimage.png" width="750px" height="530px" frameborder="0" position="relative">Title</iframe>

test 

--------------------------

## *PART 3* 

jsjsjhh

--------------------------

## *PART 4* 

ksjhjxx

--------------------------

## *PART 5* 

ksjhh

--------------------------

## *Community analysis*

On the one hand, let's take relatively large communities, such as those in 2006. It includes 3 communities representing respectively 37%, 37% and 24% of the number of sources having voted that year.  
On the other hand, let's take smaller communities, for example, those of 2005, which includes 6 communities, 3 of which represent less than 5% of the total number of sources who voted that year (respectively 0.9%, 3.3% and 2.8% for communities 3, 4 and 5 of this year).

If we look at the percentage of vote by type of vote (positive, negative or neutral) for these communities, we observe a very large majority of positive votes, approaching 80%, a smaller proportion of negative votes, close to 15-20%, and a smaller proportion of neutral votes, close to 5% for the year 2006 and generally for the year 2005, in line with the general voting behavior observed previously. However, a closer look at community 3 in 2005 reveals a different pattern. This small community has a percentage of negative votes approaching 80%, while positive votes are close to 15%, suggesting a different voting dynamic.

<iframe src="assets/img/Figures_Gaelle/Vote_percentage_2005_2006.png = 250*250" width="1150px" height="530px" frameborder="0" position="relative">Vote percentage for each type of vote per community in 2005 and 2006</iframe>

Let's see if these differences are also observed in other voting features of these elections. 
Let's look at community voting time, for example. Generally speaking, the median voting time for 2006 varies between 15 and 25 hours. Similarly, for the larger communities in 2005 (communities 0, 1 and 2), the time varies between 20 and 30 hours, while it is less than 1 hour for the smaller community 3, indicating that it votes very quickly, and close to 40 hours for community 4, indicating a more pronounced voting delay. So, once again, it seems that smaller communities have a more fluctuating voting time dynamic than larger ones.

<iframe src="assets/img/Figures_Gaelle/Median_voting_time_2005_2006.png" width="1200px" height="530px" frameborder="0" position="relative">Median voting time in per community 2005 and 2006</iframe>

Finally, let's see if certain communities are more in agreement with the election result, in other words, if the vote of these communities is in agreement with the election result. To do this, we looked at a variety of metrics, including accuracy, precision, recall and specificity. For the year 2006, as well as for the large communities (0, 1, 2) of the year 2005, we observe that the value of each metric is relatively constant, between 0.8 and 0.9, across the communities of their year. 
On the other hand, for the smallest communities in 2005, the values are more dispersed. In particular, community 3 has a specificity of 1, an accuracy remaining close to 0.8, while recall and precision are equal to 0, indicating that each time this community votes negatively, the election result is also negative, while maintaining an accuracy in the same order of magnitude as that generally observed, meaning that the "quality" of their vote prediction is not altered. The zero value of recall and accuracy indicates that the positive votes of this community were not in agreement with the outcome of the election. Thus, it is possible to distinguish the small size community, 3, of the year 2005 by its characteristics, whereas the larger communities of the same year or those of the year 2006 seem to share more common features, and it therefore seems difficult to distinguish them based on the characteristics studied.

<iframe src="assets/img/Figures_Gaelle/prediction_vote_2005_2006.png" width="800px" height="530px" frameborder="0" position="relative">Result prediction per community in 2005 and 2006</iframe>

So far, we've only highlighted remarkable features of one small community, so we might ask whether the highlighting of these remarkable features is unique to that community. Let's see, for example, whether the prediction of election results stands out for certain communities, and whether these communities are small in size. 
Extracting the distinctive features, we can see that communities 2 and 4 in 2004 have a specificity equal to 1 (as does community 3 in 2005), meaning that when these sources vote against, the election result is also negative - these communities can be described as "negative but fair". Community 0 in 2009 has a precision of 1, meaning that as soon as it votes positively, the election result is also positive, so it could be described as "nice but fair". Finally, community 4 from 2004 has a low precision value, even though it has a high recall. This means that this community mostly votes positively, but too often for this to be done accurately. We could call it TROUVER UN NOM !!!!

<iframe src="assets/img/Figures_Gaelle/prediction_vote_2004_2009.png" width="350px" height="250px" frameborder="0" position="relative">Result prediction per comunity in 2005 and 2006</iframe>

Now let's see if the communities mentioned are indeed small. 

<iframe src="assets/img/Figures_Gaelle/community_size_proportion_2004_2009.png" width="300px" height="200px" frameborder="0" position="relative">Size proportion of the communties in 2004 and 2009</iframe>

We can see that all these communities are actually small, confirming the observation made earlier that more features can be extracted from small communities, as these features are smoother and therefore less visible in larger ones.

However, despite the distinctive features observed within these small communities, it is legitimate to question their representativity as voting sources. It could be that our community extraction algorithm has grouped together only the most extreme individuals within a given community in a given year, and with source votes spanning several years, it is pertinent to ask whether these communities are not in fact small entities independent of the voting process aimed at electing administrators for Wikipedia.


