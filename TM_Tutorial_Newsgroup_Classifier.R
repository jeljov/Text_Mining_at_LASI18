## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Tutorial on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of the overall process of text classification, 
## including: 
## - preprocessing of textual data;
## - transformation of unstructured (textual) data into a structured data format (DTM)
##   that can be fed into a classification algorithm; this includes feature weighting 
##   and selection, as well as methods for reducing / transforming the feature space,
##   that is, turning a large number of sparse features into a significantly smaller 
##   number of dense features;
## - application of classification algorithms on the transformed textual data (that is,
##   the created feature set);
## - evaluation of the classification results.
##
## The example is based on a subset of the 20 Newsgroups dataset.

## ON THE DATASET
## 
## The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup 
## documents (forum posts), partitioned (nearly) evenly across 20 different newsgroups,
## each corresponding to a different topic.
##
## In case the term "newsgroup" is new to you: 
## A newsgroup is an online discussion forum accessible through Usenet.
## Usenet is a decentralized computer network, like Internet, created by two university 
## students in 1979, and initially primarily used by students and staff in universities  
## across the U.S. to communicate by sharing messages, news, and updates. It is still
## in active use with over 100K active newsgroups (see: https://www.binsearch.info/groupinfo.php)
##
## The dataset is publicly available from: http://qwone.com/~jason/20Newsgroups/
##
## Even though the 20 Newsgroup dataset allows for multiclass classification,
## to make this example easier to follow and understand, we will limit ourselves 
## to a binary classification task. In particular, we will focus on the groups 
## discussing guns ('talk.politics.guns') and Mideast ('talk.politics.mideast'), 
## within the broader 'politics' topic. The rationale for this selection:
## - It represents a case that we can expect in a discussion forum of a course  
##   dealing with current political issues and topics.  
## - Being topically closely related, the two groups will pose a challenge for a 
##   classifier - it is not an easy task to differentiate between groups of posts 
##   where we can expect a lot of shared vocabulary (at least far more than in the
##   case of differentiating between groups discussing e.g., space and medicine).
## Note that the same procedure is applicable to multiclass classification, only the 
## computation process would be more demanding and thus the computation time much longer. 


# The script makes use of the following R packages:
# - caret, e1071 - for various ML tasks
# - randomForest - for building classifiers based on the Random Forest algorithm
# - quanteda - for various text analytics tasks
# - stringr - for advanced string processing
# - irlba - for singular vector decomposition (SVD)
# - dplyr, tidyr - for general data analysis tasks 
# - ggplot2 - for visualization
# - doSNOW - for multi-core parallel process execution 
# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))

# Initially, we will load just a basic set of R packages 
# whereas the others will be loaded along the way, as we need them
library(dplyr)
library(tidyr)
library(ggplot2)

# Load a set of auxiliary functions
source("UtilityFunctions.R")

# Set the seed to be used in various computations that depend on random processes.
# This is to assure reproducibility of the results.
seed <- 28418

###############################
## LOAD CLEANED DATA FROM FILES
###############################

# We’ll start by reading in cleaned newsgroup posts. 
# The 'cleaning' process consisted of removing some extra text that 
# we do not want to include in our analysis. For example: 
# - every post has a header
# - many also have automated email signatures 
# - almost each post contains nested text representing quotes from other users
# This extra text has been removed using some simple heuristics, and
# the resulting 'cleaned' posts were saved in .csv files. 
# If interested in the steps of the 'cleaning' process, check the 
# classification script from the TM workshop (TM_Intro_Newsgroup_Classifier.R)

# Load training data
train_posts <- read.csv("data/2newsgroups-train.csv", 
                        stringsAsFactors = FALSE)
str(train_posts)
# Transform newsgroup into a factor variable, and use simpler labels
train_posts$newsgroup <- factor(train_posts$newsgroup, 
                                levels = unique(train_posts$newsgroup),
                                labels = c('guns', 'mideast'))

# Load test data
test_posts <- read.csv("data/2newsgroups-test.csv", 
                       stringsAsFactors = FALSE)
str(test_posts)
test_posts$newsgroup <- factor(test_posts$newsgroup, 
                               levels = unique(test_posts$newsgroup),
                               labels = c('guns', 'mideast'))

# Examine the class balance in the train and test sets 
table(train_posts$newsgroup)
table(test_posts$newsgroup)
# Both training and test sets are well balanced.
# If this was not the case, that is, if there was a prominent class imbalance,
# we would have had to apply a subsampling technique on the training set to 
# reduce the difference; these techniques are well covered in the *caret* R package:
# https://topepo.github.io/caret/subsampling-for-class-imbalances.html

# We will now use the training set to build a classifier.
# Test set will be used later, only for evaluation purposes.

####################################
## DATA (TEXT) WRANGLING: 
## TEXT CLEANING AND TRANSFORMATION 
####################################

# There are many packages in the R ecosystem for performing text analytics.
# One of the latest is *quanteda*. It has many useful functions for quickly
# and easily working with text data; they are well explained in the
# quanteda docs:
# http://docs.quanteda.io/index.html
library(quanteda)

#
# Tokenization of posts
#

# When tokenizing documents, a typical practice is to remove numbers, 
# punctuation marks, symbols, and urls. 
?tokens
train_tokens <- tokens(x = train_posts$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE,
                       remove_url = TRUE)

# Take a look at a specific post and see how it has been transformed
train_posts$post_txt[9]
train_tokens[[9]]

# The example post indicates that there are tokens with 1 or 2 characters only; 
# these should be removed as they rarely bear any meaning
train_tokens <- tokens_keep(x = train_tokens, min_nchar = 3)
train_tokens[[9]]

# The example post also indicates that there are email addresses among extracted tokens; 
# we will remove them, though this might not be necessary as they will probably be 
# removed later (after applying a word weighting sheme) due to their low relevance/weight 
train_tokens <- tokens_remove(x = train_tokens, 
                              pattern = "^[\\w-\\.]+@([\\w-]+\\.)+\\w{2,4}$",
                              valuetype = "regex", verbose = TRUE)
train_tokens[[9]]

## Note: regular expressions are very handy and often indispensable for text cleaning and
## transformation. If you feel you need to learn about regex or refresh your memory, 
## this tutorial is excellent: http://regex.bastardsbook.com/
## Also, the following R cheatsheet comes in useful:
## https://www.rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf 

# Next, we will reduce all tokens to lower letters to reduce the variability of 
# the token set (a part of the process known as text normalization)
train_tokens <- tokens_tolower(train_tokens)
train_tokens[[9]]

# Since forum posts, as well as messages exchanged in other kinds of online  
# communication channels (e.g. chat, status posts), tend to have misspelled 
# words, it might be useful to do spelling correction, as a part of the text 
# normalization step. A typical approach is to check the text against some of 
# the available misspelling corpora 
# (e.g. http://www.dcs.bbk.ac.uk/~ROGER/corpora.html).
# There is also an R package - spelling - for spell checking:
# https://cran.r-project.org/web/packages/spelling/index.html
# We will skip this step for now.

# Next, we will remove stopwords.
# To that end, we will use quanteda's default stopwords list for English.
?stopwords
# Note: depending on the task at hand, you might want to extend the built-in
# stopword list with additional, corpus specific 'stopwords' 
# (e.g. overly frequent words in the given corpus). 
# In addition, it is advised to inspect the default stopword list before 
# applying it to the problem at hand.
head(stopwords(), n = 50)
tail(stopwords(), n = 50)
train_tokens <- tokens_remove(train_tokens, stopwords())
train_tokens[[9]]

# Perform stemming on the tokens.
train_tokens <- tokens_wordstem(train_tokens, language = "english")
train_tokens[[9]]

###################################
# CREATE DOCUMENT TERM MATRIX (DTM)
###################################

# Now, we are ready to create DTM. 
# In quanteda's terminology DTM is referred to as "document feature matrix" or dfm
# By default, words are weighted with term frequencies (TF)
?dfm
train_dfm <- dfm(x = train_tokens, 
                 tolower = FALSE)  # we've already lower cased the tokens

train_dfm
# It's very sparse (sparsity = the proportion of cells that have zero counts); 
# we can get the precise level of sparsity with:
sparsity(train_dfm)

# Considering the large number of features (~12.4K) and the high level of
# sparsity, we should consider removing rare features, that is, words with 
# very low frequency.

# Examine total (corpus-level) TF for each word
summary(colSums(train_dfm))
# Considering the large number of features and their low corpus
# frequency, we will keep features with total TF above the median
tf_total <- colSums(train_dfm)
to_keep <- which(tf_total > median(tf_total))
train_dfm_reduced <- dfm_keep(train_dfm, pattern = names(to_keep),
                              valuetype = "fixed", verbose = TRUE)
train_dfm_reduced
# a significant reduction: from ~12.4K to ~5K features

# Also examine the number of documents each token appears in (i.e., document
# frequency - DF), to check if there might be those that are present in a 
# large proportion of documents (e.g. in over 50% of documents)
dfreq <- apply(train_dfm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
# Max presence is in approx. 1/3 of documents; hence, no need for this kind
# of trimming. 

# Use the (reduced) DTM to setup a feature data frame with labels.
# It will serve as the input for a classification algorithm. 
train_df <- create_feature_df(train_dfm_reduced, train_posts$newsgroup)


########################################################
# BUILD the 1st ML MODEL: RPART + UNIGRAMS + TF WEIGTHS
########################################################

# As per best practices, we will leverage cross validation (CV) for our
# modeling process. In particular, we will perform 5-fold CV to
# tune parameters and find the best performing model.

# Note that our data set is not trivial in size. As such, depending on the
# chosen ML algorithm, CV runs might take a long time to complete. 
# To cut down on total execution time, we use the *doSNOW* R package 
# to allow for cross-validation in parallel on multiple (logical) cores.

# Due to the size of the DTM, at this point, we will use a single decision
# tree (DT) algorithm to build our first model. We will use more powerful algorithms 
# later when we perform feature reduction to shrink the size of our feature set.

library(caret)
# Load also the *rpart* R package required for building DTs 
library(rpart)
# We will tune the cp parameter, which is considered the most important in the
# rpart function (the function used in the rpart package for building a DT).
# cp stands for the complexity parameter; any split that does not improve the overall
# fit of the model by at least cp is not attempted; default value is 0.01.
# Define the grid of values for the cp parameter to be examined during the CV process
cp_Grid = expand.grid( .cp = seq(from = 0.001, to = 0.02, by = 0.0005)) 
# Build a DT classifier through CV 
# (the cross_validate_classifier() function is defined in the UtilityFunctions.R script)
rpart_cv_1 <- cross_validate_classifier(seed,
                                        nclust = 5,
                                        train.data = train_df,
                                        ml.method = "rpart",
                                        grid.spec = cp_Grid)

# Check out the results:
rpart_cv_1
plot(rpart_cv_1)

# Since model building takes some time, save the model 
# to have a quick access to it later
saveRDS(rpart_cv_1, "models/tutorial/rpart_cv_1.RData")

# Load the saved model
# rpart_cv_1 <- readRDS("models/tutorial/rpart_cv_1.RData")

# First, take the cp value of the best performing model in CV
tf_best_cp <- rpart_cv_1$bestTune$cp
# Then, extract performance measures for the best cp value 
tf_best_results <- rpart_cv_1$results %>% filter(cp == tf_best_cp)
tf_best_results

# By inspecting the tree, we can get an idea of the terms (features)
# the DT model selected as the most important
print(rpart_cv_1$finalModel)

# We can also examine feature importance more directly
sort(rpart_cv_1$finalModel$variable.importance, decreasing = TRUE)[1:20]


###################################################################
## ADD BIGRAMS TO THE FEATURE SET AND APPLY TF-IDF WEIGHTING SCHEME
###################################################################

# Our first model can be characterized as being based on Bag-of-Words
# text representation, since we used only words (unigrams) and 
# estimated words' relevancy based on their frequency (TF).

# N-grams allow us to augment our DTM matrix with word ordering.
# This tends to lead to increased performance (e.g., accuracy)
# over ML models trained with unigrams only. On the down side,
# the inclusion of ngrams, even only bigrams and trigrams, 
# leads to an explosion in the number of features.

# Let's expand our feature set by adding just bigrams
train_tokens_2 <- tokens_ngrams(train_tokens, n = 1:2)
train_posts$post_txt[9]
train_tokens_2[[9]]

# Next, we should build DTM with the extended feature set.
# But, we should also reconsider how we estimate the term (feature)
# relevance, that is, which weighting scheme we want to use.

## The DTM used in the first model is based on the counts of terms, often known as 
## term frequency (TF). However, this is not always the best metric for estimating 
## the relevance of a word in a corpus.
## The Term Frequency-Inverse Document Frequency (TF-IDF) metric tends to provide
## better results. Specifically, TF-IDF accomplishes the following goals:
## - The TF metric does not account for the fact that longer documents will have 
##   higher individual term counts. By normalizing TF values, using, for example, 
##   L1 norm, that is, the document length expressed as the number of words,
##   we get a metric that is length independent.
## - The IDF metric accounts for the frequency of term appearance in all documents 
##   in the corpus. The intuition being that a term that appears in almost every  
##   document has practically no predictive power.
## - The multiplication of (normalized) TF by IDF allows for weighting each term 
##   based on both its specificity at the level of the overall corpus (IDF) and its 
##   specificity for a particular document (i.e. relatively high presence in the document).

?dfm_tfidf
# Create a TF-IDF weighted dfm
train_dfm_2 <- dfm(train_tokens_2, tolower = FALSE) %>%
  dfm_tfidf(scheme_tf = "prop") # compute TF-IDF w/ normalised TF weights
train_dfm_2
# Note the number of features (over 114K) and the level of sparsity!

# Again, we need to remove low weighted terms to reduce the feature space.
# Compute total TF-IDF weight per feature
tfidf_total <- colSums(train_dfm_2)
summary(tfidf_total)
# Note how skewed the distribution is
quantile(tfidf_total, probs = seq(0.9,1,0.025))
# Considering the number of features (ngrams), we will keep only ngrams 
# with TF-IDF weights in the top 2.5 percentile (~2.7K features)
# Note: setting threshold to top 10, top 5, and top 2.5 percentile
# resulted in almost the same performance metrics, so, the top 2.5 per
# is chosen as it allows for the largest reduction of the feature space. 
threshold <- quantile(tfidf_total, probs = 0.975)
to_keep <- which(tfidf_total > threshold)
train_dfm_topXperc <- dfm_keep(train_dfm_2, pattern = names(to_keep),
                               valuetype = "fixed", verbose = TRUE)
train_dfm_topXperc
# We've reduce the feature set to ~2.7K ngrams

# Make a clean data frame to be used as input for building a classifier
train_tfidf_df <- create_feature_df(train_dfm_topXperc, train_posts$newsgroup)

#####################################################################
# BUILD the 2nd ML MODEL: RPART + UNIGRAMS & BIGRAMS + TF-IDF WEIGHTS
#####################################################################

# Build a CV-ed model with the new feature set and all other settings unchanged
rpart_cv_2 <- cross_validate_classifier(seed, 
                                        nclust = 5,
                                        train.data = train_tfidf_df,
                                        ml.method = "rpart",
                                        grid.spec = cp_Grid)


# Check out the results:
rpart_cv_2
plot(rpart_cv_2)

# Save the model to have a quick access to it later
saveRDS(rpart_cv_2, "models/tutorial/rpart_cv_2.RData")

# Load the saved model
# rpart_cv_2 <- readRDS("models/tutorial/rpart_cv_2.RData")

# Extract and store evaluation metrics for the best performing model
tfidf_best_cp <- rpart_cv_2$bestTune$cp
tfidf_best_results <- rpart_cv_2$results %>% filter(cp == tfidf_best_cp)

# Compare the performance of the two classification models built so far
data.frame(rbind(tf_best_results, tfidf_best_results), 
           row.names = c("TF", "TF-IDF"))
# The model achieved slightly weaker performance than the previous one.

# Before proceeding, let's examine the terms (features) used for building the model
sort(rpart_cv_2$finalModel$variable.importance, decreasing = TRUE)[1:30]

# In the next step, we will apply a more sophisticated feature reduction method.
# In particular, we'll apply Singular Value Decomposition (SVD) to the DTM of
# TF-IDF weighted unigrams and bigrams.

####################################
# SINGULAR VALUE DECOMPOSITION (SVD)
# FOR REDUCING THE FEATURE SPACE
####################################

# We will now use Singular Value Decomposition (SVD) to reduce the number 
# of features (ngrams) to a smaller set that explains a large portion of 
# variability in the data.

# Suggested reading for SVD and Latent Semantic Analysis, the latter being, in fact,
# SVD applied to text analytics:
# - Landauer, T. K., Foltz, P. W., & Laham, D. (1998). Introduction to Latent 
#   Semantic Analysis. Discourse Processes, 25, 259-284. 
#   URL: http://lsa.colorado.edu/papers/dp1.LSAintro.pdf


# First, we need to setup the data to which SVD will be applied.  
# What we need is TF-IDF weighted TDM (Term Document Matrix). 
# Note: TDM is nothing more than transposed DTM.
# Since our DTM with bigrams included is very large, we need to 
# reduce its size, though not as much as we did above 
# (taking just top 2.5 percentile) - we need somewhat larger 
# base so that we allow more ngrams to contribute to the new 
# feature space (ie., singular vectors).
# So, let's re-examine the distribution of TF-IDF weights
summary(tfidf_total)
# Considering the number of features and the very uneven distribution 
# of TF-IDF values, remove tokens with TF-IDF below the 3rd quartile
to_remove <- which(tfidf_total < quantile(tfidf_total, probs = 0.75))
train_dfm_4_svd <- dfm_remove(train_dfm_2, pattern = names(to_remove), 
                              valuetype = "fixed", verbose = TRUE)
train_dfm_4_svd  

# Next, we need to set the number of the most important singular vectors we wish 
# to calculate and retain as features (in SVD terms, it is the rank the original 
# matrix is to be reduced to).
# How to determine the "right" number of singular vectors is still an open issue.
# Some useful links on that topic:
# - https://stackoverflow.com/questions/9582291/how-do-we-decide-the-number-of-dimensions-for-latent-semantic-analysis 
# - https://irthoughts.wordpress.com/2008/02/13/lsi-how-many-dimensions-to-keep/

# We will reduce the dimensionality down to 300 columns. This number is chosen as it
# is often recommended (based on the experience in practice).
# To get the best results, the number of dimensions would have to be experimentally 
# determined, by trying several different values and comparing the performance of 
# the resulting models.

# We'll use the *irlba* R package for SVD
library(irlba)
set.seed(seed)
svd_res <- irlba(t(as.matrix(train_dfm_4_svd)), # SVD / LSA requires TDM (not DTM) as its input 
                 nv = 300, # the number of dimensions (singular vectors) to estimate
                 maxit = 600) # maxit is set to be twice larger than nv 

# Examine the result:
str(svd_res)
# d - corresponds to singular values (values on the diagonal of the sigma matrix)
# u - corresponds to the left singular vector and respresents relation between
#     the extracted dimensions and the ngrams 
# v - corresponds to the right singular vector and respresents relation between
#     the extracted dimensions and the documents

# Save the SVD result to have a quick access to it later
saveRDS(svd_res$d, "models/tutorial/svd_d.RData")
saveRDS(svd_res$u, "models/tutorial/svd_u.RData")
saveRDS(svd_res$v, "models/tutorial/svd_v.RData")

# Take a glimpse at the new feature set (the right singular vector):
View(svd_res$v[1:20,1:50])

# Create a new feature data frame using the 300 features obtained by applying
# SVD to TF-IDF weighted DTM (i.e. the V matrix produced by SVD)
train_svd_df <- cbind(Label = train_posts$newsgroup, data.frame(svd_res$v))

# Next, we will examine the predictive power of the model with singular 
# vectors as features.

## Before proceeding to the creation of a classifier, note that there is
## an alternative approach to data preparation for SVD / LSA.
## It was suggested in the original paper on SVD / LSA by Landauer, Foltz, 
## & Laham (1998):
##    "Before the SVD is computed, it is customary in LSA to subject the data 
##    in the raw word-by-context matrix to a two-part transformation. 
##    First, the word frequency (+ 1) in each cell is converted to its log. 
##    Second, the information-theoretic measure, entropy, of each word 
##    is computed as: -p*logp over all entries in its row, 
##    and each cell entry then divided by the row [word] entropy value. 
##    The effect of this transformation is to weight each word-type occurrence  
##    directly by an estimate of its importance in the passage [document] and 
##    inversely with the degree to which knowing that a word occurs provides  
##    information about which passage [document] it appeared in."
##
## So, instead of TF-IDF, transform the original DTM (train_dfm_2) in the 
## manner suggested above, apply SVD on thus transformed DTM, and build 
## a RF model, as we do below. Compare the results with those of rf_cv_1 
## (given below). 

###############################################
# BUILD the 3rd ML MODEL: RANDOM FOREST + 
# SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM)
###############################################

# We have reduced the dimensionality of our data using SVD. Now, we can use a more 
# complex and powerful classification algorithm. In particular, we will build a 
# Random Forest (RF) model.

## For a brief introduction to the Random Forest algorithm, 
## see the "Bagging and Random Forest" slides (made available as part of the WS materials).
## For more details and an excellent explanation of Random Forest and related algorithms,
## see chapter 8.2 of the Introduction to Statistical Learning book
## http://www-bcf.usc.edu/~gareth/ISL/ 

# We will build a RF model with 1000 trees. We'll also try different 
# values of the mtry parameter to find the value that gives the best result. 
# The mtry parameter stands for the number of variables randomly sampled as 
# candidates at each split. 
# For the mtry parameter, we will consider 10 different values between the minimum
# (1 variable) and the maximum possible value (all variables). 
mtry_Grid = expand.grid( .mtry = seq(from = 1, to = (ncol(train_svd_df)-1), length.out = 10))

# NOTE: The following code takes a long time to run. Here is why:
# We are performing 5-fold CV. That means we will examine each model configuration 
# 5 times. We will have 10 configurations as we are asking caret to try 10 different
# values of the mtry parameter. In addition, we are asking RF to build 1000 trees. 
# Lastly, when the best value for mtry is chosen, caret will use it to build the 
# final model using all the training data. So, the number of trees we're building is:
# (5 * 10 * 1000) + 1000 = 51,000 trees!

# Build a RF classifier
rf_cv_1 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train.data = train_svd_df,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid)

# As the model building takes too much time, save the model
saveRDS(rf_cv_1, "models/tutorial/rf_cv_1.RData")

# load the model
# rf_cv_1 <- readRDS("models/tutorial/rf_cv_1.RData")

# Check out the results
rf_cv_1
plot(rf_cv_1)

# Extract evaluaton measures for the best performing model 
svd_best_mtry <- rf_cv_1$bestTune$mtry
svd_best_res <- rf_cv_1$results %>% filter(mtry==svd_best_mtry)

# Compare the results with the previously CV-ed models
comparison <- data.frame(rbind(tf_best_results[,-1], # exclude the cp parameter 
                               tfidf_best_results[,-1], 
                               svd_best_res[,-1]), # exclude the mtry parameter 
                         row.names = c("RPART_TF", "RPART_TF-IDF", "RF_SVD"))
# Add a column with feature number
comparison$NFeatures <- c(ncol(train_df),
                          ncol(train_tfidf_df),
                          ncol(train_svd_df))
comparison
# The combined use of the new feature set and a more powerful algorithm significantly 
# improved the results, in terms of both accuracy and kappa measures. In addition,
# the number of features is over 10 times smaller than in the second best model; this
# is highly important as it makes the model less prone to overfitting.

##################
# TEST THE MODEL
##################

# Now that we've built predictive models, it is time to verify the one that proved  
# the best (the 3rd one: RF + Singular Vectors) using the test holdout data 
# we set aside at the beginning of the script.  
# The first stage of the evaluation process is running the test data through our 
# text transformation pipeline of:
# - Tokenization
# - Removing tokens less than 3 characters long
# - Removing email addresses
# - Lower casing
# - Stopword removal
# - Stemming
# - Adding bigrams
# - Creating DTM and ensuring the test DTM has the same features (ngrams) 
#   as the train DTM
# - Computing TF-IDF weights 
# - Feature set transformation / reduction using SVD  

test_tokens <- tokens(x = test_posts$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE,
                       remove_url = TRUE)

test_tokens <- tokens_keep(x = test_tokens, min_nchar = 3) %>%
  tokens_remove(pattern = "^[\\w-\\.]+@([\\w-]+\\.)+\\w{2,4}$", valuetype = "regex") %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_wordstem(language = "english") %>%
  tokens_ngrams(n = 1:2)
  
test_dfm <- dfm(test_tokens, tolower = FALSE)

# Compare the train and test DTMs
train_dfm_4_svd
test_dfm
# The two DTMs differ in the feature set. This is expected as features
# are ngrams from two different sets of posts (training and test).
# However, we have to ensure that the test DTM has the same n-grams 
# (features) as the training DTM.
# The rationale: once a classifier is deployed and used, we should 
# expect that new posts will contain n-grams that did not exist in 
# the training data set. In spite of this, we need to represent any 
# new post in the feature space that our classifier 'is aware of', 
# and that is the feature space of the training set.

# Transform test_dfm so that it has the same features as the one that  
# served as the input for SVD
test_dfm <- dfm_select(test_dfm, 
                       pattern = train_dfm_4_svd,
                       selection = "keep")
test_dfm
# Now, test dfm seems to have the same features as the train dfm.
# Let's check if those are really the same features
setdiff(colnames(test_dfm), colnames(train_dfm_4_svd))
setdiff(colnames(train_dfm_4_svd), colnames(test_dfm))
# No difference -> they are exactly the same.

# The next step is to 'project' the test DTM into the same 
# TF-IDF vector space we built for our training data. 
# This requires the following steps:
# 1 - Normalize term counts in each document (i.e, each row)
# 2 - Perform IDF multiplication using training IDF values
# (we use the IDF values from the training set, as we represented
# our test data using the features (ngrams) of the training set)

# Normalize term counts in all test posts
test_tf <- dfm_weight(test_dfm, scheme = "prop")

# Next, get IDF values of the training set features, in particular,
# of features included in the train_dfm_4_svd
# (Note: the inverse_doc_freq() f. is defined in the UtilityFunctions.R script)
train_idf <- apply(train_dfm_4_svd, 2, inverse_doc_freq)

# Next, calculate TF-IDF using IDF of our training corpus
# (Note: the tf_idf() f. is defined in the UtilityFunctions.R script)
test_tfidf <-  apply(as.matrix(test_tf), 1, function(x) tf_idf(x, idf = train_idf))
dim(test_tfidf)
# Transpose the matrix (so that the documents are in the rows)
test_tfidf <- t(test_tfidf)

# With the test data projected into the TF-IDF vector space of the training
# data, we can now do the final projection into the training SVD space
# (i.e. apply the SVD matrix factorization).

##############################################
## APPLYING SVD PROJECTION ON A NEW DATA SET
##############################################

# The formula to be used for this projection (of a particular document - d): 
#
# d_hat = sigma_inverse * transposed_U_matrix %*% d_TF-IDF_vector
#
# d_hat is the representation of the given document d in the SVD space of 
# the training dataset; more precisely, it is the representation of d in
# terms of the dimensions of the V matrix (right singular vector).
# 
# Before applying this formula, let us examine why and how do we use it

# As an example, let's use the first document from the training set, 
# that is, the TF-IDF representation of the first post in the train set
example_doc <- as.matrix(train_dfm_4_svd)[1,]

# For convenience, we'll introduce:
sigma_inverse <- 1 / svd_res$d
u_transpose <- t(svd_res$u)

# The projection of the example document in the SVD space:
example_doc_hat <- as.vector(sigma_inverse * u_transpose %*% example_doc)
# Look at the first 10 components of projected document...
example_doc_hat[1:10]
# ... and the corresponding row in the document space produced by SVD (the V matrix)
svd_res$v[1, 1:10]
# The two vectors are almost identical (note the values are expressed in e-04, e-05,...).
# In fact, the differences are so tiny that when we compute cosine similarity 
# between the two vectors, the similarity turns to be equal to 1:
library(lsa)
cosine(as.vector(example_doc_hat), as.vector(svd_res$v[1,]))
#
# Why is this useful?
# It shows that using the above given formula, we can transform any document into
# the singular vector space of the training set, using the computed sigma_inverse 
# and transposed_U_matrix; this further means that we can take a new, unseen 
# document (a post in our case), compute TF-IDF values for it and transform it 
# into singular vector space so that it can be classified by our prediction model.


# So, we will use the above given formula to represent posts from the test set in 
# the singular vector space. As we have multiple documents, we need to replace 
# d_TF-IDF_vector (3rd element on the left), with a matrix of TF-IDF values 
# (the matrix should have terms in rows and documents in columns)
test_svd_hat <- sigma_inverse * u_transpose %*% t(test_tfidf)
dim(test_svd_hat)

###################################################
## MAKE PREDICTIONS ON THE (TRANSFORMED) TEST DATA
###################################################

# With the feature set ready, we can now build the test data frame to 
# feed into our trained machine learning model for predictions. 
test_svd_df <- data.frame(Label = test_posts$newsgroup, 
                          t(test_svd_hat)) # need to transpose it, to place documents in rows

# Now we can make predictions on the test data set using our best classifer  
# (rf_cv_1)
preds <- predict(rf_cv_1, newdata = test_svd_df)

# Examine the results
test_eval <- confusionMatrix(data = preds, reference = test_svd_df$Label)
get_eval_measures(test_eval)

# Let's compare these results with those obtained on the training data 
train_eval <- confusionMatrix(data = rf_cv_1$finalModel$predicted,
                                reference = train_svd_df$Label)
get_eval_measures(train_eval)

# All metrics are somewhat lower on the test set, which is,  
# often the case.

# Note: the fact that we got slightly better results on CV...
comparison[3,]
# ... than when applying the final model on the whole training set; this is because
# the performance always slightly varies depending on the sample that is used for
# model building, and in CV the performance metrics are averaged over CV runs.

