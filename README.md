# LASI'18 WS and tutorial on Text Mining for learning content analysis

This repository stores materials for the **Text mining for learning content analysis** workshop and tutorial organized in the scope of the [Learning Analytics Summer Institute 2018 (LASI'18)](https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/), held at Teachers College Columbia University, New York City, NY, on June 12-13, 2018. 

The stored R scripts provide examples for four Text Mining (TM) tasks: classification, clustering, topic modelling, and keywords extraction. Each script presents the overall TM workflow for the respective task, starting with text preprocessing and ending with examination and evaluation of the results.

The four TM tasks are covered with R scripts as follows:
* Text classification:
  * TM_Intro_Newsgroup_Classifier.R (for WS)
  * NewsGroup_GloVe_Classifier.R (for WS)
  * TM_Tutorial_Newsgroup_Classifier.R (for tutorial)
  * UtilityFunctions.R (for WS and tutorial)
* Text Clustering:
  * NewsGroup_AP_Clustering.R (for WS)
  * ClustEvalUtil.R (for WS)
* Topic modelling with LDA method 
  * TopicModelingUseNetGroups.R (for WS)
  * UtilityFunctions.R (for WS)
* Keywords extraction using the [TextRank](http://www.aclweb.org/anthology/W04-3252) method
  * TextRankUsenetGroups.R (for WS)

Note also that some prebuilt models are available in the 'models' folder. Models are grouped into subfolders based on the TM task they are related to (that is, R scripts they are associated with). They are made available so that we do not need to wait for models to build during the WS/tutorial.

All examples are based on the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/).
This dataset is a collection of approximately 20,000 newsgroup documents (forum posts), partitioned (nearly) evenly across 20 different newsgroups, each corresponding to a different topic.
In case the term "newsgroup" is new to you: a newsgroup is an online discussion forum accessible through Usenet (a decentralized computer network, like Internet). Even though they are not 'mainstream' social networks, newsgroups are in active use (see [Newsgroup info](https://www.binsearch.info/groupinfo.php)).

Slides that introduce relevant concepts and methods can be downloaded from the links given below.
* Slides for the tutorial: 
  * [Text mining workflow for learning content analysis](https://1drv.ms/b/s!AjwXFgNk6IQbgheDJUZ5hViJPLos)
* Slides for the workshop:
  * [Introduction to Text mining and its workflow](https://1drv.ms/b/s!AjwXFgNk6IQbghkOxH6jIp9oakUG)
  * [Bagging and Random Forest](https://1drv.ms/b/s!AjwXFgNk6IQbgh5G-vQCyWnaXwZL)
  * [A Glimpse at Word Embeddings](https://1drv.ms/b/s!AjwXFgNk6IQbghXMfm0wTK6RLkqf)
  * [Affinity propagation for text clustering](https://1drv.ms/b/s!AjwXFgNk6IQbghvMSeEs9yqyar6q)
  * [Topic Modelling: LDA](https://1drv.ms/b/s!AjwXFgNk6IQbghGZ3aOHXoVQN-mz)
  * [Graph-based ranking methods: TextRank](https://1drv.ms/b/s!AjwXFgNk6IQbghB3pYLEEQ13iFQO)

