## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of applying Latent Dirchelet Allocation (LDA)
## algorithm to identify latent topics discussed in a subset of the 20 Newsgroups dataset.
## The example includes:
## - preparation of textual data for topic modelling
## - application of the LDA algorithm on the transformed textual data
## - visual inspection of the LDA results
## - application of the built LDA model on a new (test) dataset
## - different methods for determining the 'correct' number of topics

# Load the required packages
# (more will be added along the way)
library(dplyr)
library(tidyr)
library(ggplot2)

library(stringr)
library(quanteda)

###############################################
# CHOOSE A SUBSET OF 20 NEWSGROUPS TO FOCUS ON
###############################################

# Since we have already done the pre-processing of the 20 Newsgroups dataset,
# we will use the pre-processed data
cleaned_posts <- read.csv("data/20news-bydate/20news-bydate-train.csv", 
                          stringsAsFactors = FALSE)

# Examine the available newsgroups
unique(cleaned_posts$newsgroup)

# In what follows, we will focus on a subset of newsgroups. This is to reduce 
# the corpus, and make the required computations less demanding. The same 
# procedure would apply if we were to use the entire dataset, that is, all the
# newsgroups. 
# We will choose a subset of newsgroups that share a common broader theme. 
# In particular, we will select the 4 newsgroups discussing scientific topics.
# This is to make the topic modeling task not a trivial one, as it will not be 
# that easy for the topic model to distinguish between closely related topics
# discussed within and across the science related newsgroups.
sci_posts <- cleaned_posts %>%
  filter(str_detect(newsgroup, "^sci\\."))
# Check the newsgroups in sci_post to be sure that we have done the selection
# correctly
unique(sci_posts$newsgroup)

######################################
# PREPARE DATA FOR TOPIC MODELLING
######################################

# Tokenise the posts, and keep tokens that are 
# not numbers, symbols, punctuation marks, or URLs
sci_tokens <- tokens(sci_posts$post_txt, 
                     what = "word",
                     remove_punct = TRUE, 
                     remove_symbols = TRUE, 
                     remove_numbers = TRUE,
                     remove_url = TRUE)

# Do some basic kinds of text preprocessing to reduce 
# the variability of the token set:  
# - reduce all tokens to lower letters
# - remove stopwords
# - remove tokens with just 1 char
# - stem words
sci_tokens <- sci_tokens %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_keep(min_nchar = 2) %>%
  tokens_wordstem(language = "english")


## NOTE:
## An additional pre-processing step may be Part-of-Speech (POS) tagging, and
## then keeping only certain categories of words, e.g., nouns and/or verbs.
## In particular, it is often the case that the use of only nouns leads 
## to better (more coherent) topic models.
## POS tagging can be easily done using the *udpipe* R package:
## https://github.com/bnosac/udpipe 
## POS tagging is exemplified in the TextRankNewsGroups.R script
## (part of the WS materials).


# Create DTM
sci_dfm <- dfm(sci_tokens, tolower = FALSE)
sci_dfm
# Huge number of features (~21K) and very sparse DTM (0.997) 

# We should remove words with very low freq. This will not only reduce the sparsity,
# but will also reduce the 'noise' in the input for topic modeling.
# To decide on which low-freq words to remove, we need to first examine the distribution
# of words in the corpus, that is, to examine term frequency and document frequency
# for all the words in the corpus.

# Total term frequency for each word can be obtained by summing columns of the DTM 
sci_word_freqs <- colSums(sci_dfm)
summary(sci_word_freqs)
# Note that at least half of the words appear in the overall corpus only once; 
# 3rd quartile is just 4; on the other hand, some words are highy frequent.
# Typical power-law distribution.

# Let's also examine document frequency, that is, the number of documents a word appears in.
# We can get this by counting, for each word (column), the number of documents (rows)
# with value greater than zero.
sci_word_doc_count <- apply(as.matrix(sci_dfm), 2, function(x) sum(x > 0))
summary(sci_word_doc_count)
# Again, very uneven distribution, with at least half of the words appearing in only 1 post,
# 75% of words appearing in at most 3 posts, and maximum appearance in 1/3 of all posts.

# So, for example, we can trim the DTM by keeping words appearing in at least 2 posts
# (i.e. having above the median document frequency)
sci_dfm_reduced <- dfm_trim(sci_dfm, min_docfreq = 2)
sci_dfm_reduced
# Significant reduction in the number of features (words): 
# from over 21.1K to ~ 8.8K

# It is also wise to remove frequent but non-informative words
# as they are not much useful for topic-wise differentiation of the corpus.    
# To detect such words, we will first assess the relevance of the words 
# (in the DTM) by computing their TF-IDF weights
sci_dfm_reduced_tfidf <- dfm_tfidf(sci_dfm_reduced, scheme_tf = "prop")
sci_word_tfidf <- colSums(sci_dfm_reduced_tfidf)
summary(sci_word_tfidf)

# We will remove words with TF-IDF value below the 75th percentile.
# Note: other thresholds were examined (median, 90% of the median),
# but this one gave the best results, that is, the most meaningful
# (descriptive) words among the top 15 words for each topic. 
# Another option for choosing the optimal threshold is to examine 
# models' *perplexity* for different threshold value. The use of 
# perplexity for model tuning will be shown later.
threshold <- quantile(sci_word_tfidf, probs = 0.75) 
# then, get the indices of the words to be removed 
words_to_remove <- which(sci_word_tfidf < threshold)

# Note: we have used TF-IDF weighted DTM just to detect words
# to be removed. For building an LDA model, we will use 
# TF-weighted DTM as required by the LDA topic modeling method.
# So, we remove the selected words from the TF-weighted DTM:
sci_dfm_reduced <- dfm_remove(sci_dfm_reduced, 
                              pattern = names(sci_word_tfidf[words_to_remove]),
                              valuetype = "fixed",
                              verbose = TRUE)   
# Examine the DTM after the words removal
sci_dfm_reduced
# further significant reduction in the number of features (words): 
# from over 8.8K to ~ 2.2K

# After building a topic model and inspecting the top 15 words for each topic, 
# some common, not topic-discriminative words have been observed; these were
# selected for removal as 'custom' (corpus-specific) stopwords
custom_stopwords <- c('one', 'two', 'can', 'get', 'just', 'much', 'even', 
                      'also', 'may', 'want', 'use', 'good')
sci_dfm_reduced <- dfm_remove(sci_dfm_reduced, 
                              pattern = custom_stopwords,
                              valuetype = "fixed",
                              verbose = TRUE)

# Before proceeding, we will remove large objects that are no longer needed
remove(sci_word_tfidf, sci_word_freqs, sci_word_doc_count, 
       sci_tokens, sci_dfm_reduced_tfidf, sci_dfm, cleaned_posts)

############################
## APPLY LDA TO MINE TOPICS
############################
# install.packages('topicmodels')
library(topicmodels)

# We will use the LDA function with Gibbs sampling
?LDA

# Setting the parameters for Gibbs sampling

## The Gibbs sampling method starts by choosing model parameters at random and
## then iteratively updates (improves) the parameter estimates. The overall method
## can be thought of as a random walk in the parameter space, 'directed' by the  
## optimisation function. Because the starting point of the walk is chosen at random, 
## it is necessary to discard certain number of iterations at the start of the walk 
## (as these do not correctly reflect the properties of distributions). 
## This is referred to as the burn-in period.
burnin <- 1000
# After the burn-in period, we will perform 2000 iterations (the default value)...
iter <- 2000 
# ... omitting 2000 iterations (default value) in-between two subsequent Gibbs iterations; 
# this is to avoid correlations between samples
thin <- 2000 
# We will make five independent runs, and need a seed for each one
nstart <- 5
seeds <- list(5, 10, 6, 11, 2018)

# Set the number of topics to the number of science-related newsgroups 
k = 4
sci_lda <- LDA(x = convert(sci_dfm_reduced, "topicmodels"), # convert quanteda's dfm into DTM required by 
                                                            # the 'topicmodels' R package
               k = k, 
               method = "Gibbs",
               control = list(nstart=nstart, seed = seeds, 
                              burnin = burnin, iter = iter, thin=thin))

# Save the model as the model building process lasts long
saveRDS(sci_lda, "models/LDA/Sci_NewsGroup_4_topic_LDA.RData")

# load the saved model
# sci_lda <- readRDS("models/LDA/Sci_NewsGroup_4_topic_LDA.RData")

# Examine the model's hyperparameters alpha and beta
sci_lda@alpha
# By default, alpha value is set to 50/k, while beta is estimated
str(sci_lda@beta)
# a 4 x 2177 matrix defining parameter values for each word (in columns) in each topic (in rows) 

## You can tune the model by changing the value of alpha; to do that, include the parameter
## 'alpha' in the 'control' list of the LDA function:
## LDA(... control = list(alpha = <new_value>, ...))
## Setting a smaller value for alpha will force the algorithm to associate documents with
## smaller number of topics (i.e. few topics w/ high probability and others with (very) low
## probability).

## To tune the parameters of the model, including the number of topics, hyperparameters, and
## parameters specific for Gibbs sampling (e.g., burin, iter, thin), we can make use of the 
## *perplexity* measure. Perplexity is a measure of how well a probability model fits a new 
## set of data. The lower its value, the better. The use of this metric for determining an 
## optimal number of topics will be demonstrated later.

######################
# EXAMINE THE RESULTS
######################

#
# Examine how terms are distributed across topics
#

# Extract top 15 words for each topic
sci_top_topic_terms <- terms(sci_lda, k = 15)
sci_top_topic_terms

# Extract newsgroup labels to compare them with topics (top15 terms lists)
sci_labels <- unique(sci_posts$newsgroup)
sci_labels

# Let's examine term probability distribution for each topic in more detail
# by plotting the prob. dist. of the top 15 words in each topic

# First, we need term (posterior) probabilities for each topic 
sci_term_prob <- posterior(sci_lda)$terms
# Transpose the matrix so that words are given in rows and topics
# are in columns, to be easier for inspection and further processing 
sci_term_prob <- t(sci_term_prob)
head(sci_term_prob, n=10)

# Next, to do the plotting, we need to:
# 1) transform the terms probility matrix into a data frame 
sci_term_prob_df <- as.data.frame(sci_term_prob) %>%
  mutate(word = row.names(sci_term_prob)) 
colnames(sci_term_prob_df)[1:4] <- paste0("topic_", 1:4)
# 2) transform the term probabilities df (from wide) to a long format with 
# the top 15 words (in terms of posterior probability) for each topic
sci_term_prob_long <- sci_term_prob_df %>%
  gather(key = topic, value = posterior, ... = topic_1:topic_4, factor_key = TRUE) %>% 
  group_by(topic) %>%
  top_n(n = 15, wt = posterior) %>%
  ungroup() 
# 3) plot the term probabilities per topic
sci_term_prob_long %>%
  mutate(word = reorder(word, posterior)) %>%  # to plot words in decreasing order of prob.
  ggplot(aes(x = word, y = posterior, fill = topic)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales = "free_y") +
  ylab("Term (posterior) probability") + xlab("Terms") +
  coord_flip() +
  theme_bw()

# check against the group names
sci_labels

#
# Examine how topics are distributed across newsgroups
#

# First, we need to get (posterior) probabilities for topics across documents
sci_topics_prob <- posterior(sci_lda)$topics
head(sci_topics_prob)

# Associate documents (posts) with the newsgroup they originate from
sci_topics_prob_df <- as.data.frame(sci_topics_prob) %>%
  mutate(newsgroup = sci_posts$newsgroup)
colnames(sci_topics_prob_df)[1:4] <- paste0("topic_", 1:4)

# Transform the topics probability df (from wide) to a long format, and 
# plot the distribution of topics for each newsgroup separately
sci_topics_prob_df %>%
  gather(key = topic, value = posterior, ... = topic_1:topic_4, factor_key = TRUE) %>%
  group_by(newsgroup) %>%
  ggplot(aes(x = topic, y = posterior)) +
  geom_boxplot() +
  facet_wrap(~newsgroup) +
  ylab("topic distribution") +
  theme_bw()

# reminder of the top topic terms
sci_top_topic_terms

# Another way to examine topics against newsgroups is to juxtapose 
# topics vs newsgroups based on the documents' dominant topic 
# (ie. topic with the highest probability for a given document)

# Take the dominant topic for each document
sci_top_topics <- topics(sci_lda, k = 1)

# Add dominant topics to the sci_posts df
sci_post_main_topic <- sci_posts %>%
  mutate(main_topic = as.integer(sci_top_topics))

# For each newsgroup, plot the number of documents per (dominant) topic 
sci_post_main_topic %>%
  ggplot(aes(x = main_topic, fill = newsgroup)) +
  geom_bar(show.legend = FALSE, width = 0.5) +
  facet_wrap(~newsgroup) +
  ylab("Number of posts per main topic") +
  xlab("Main (= most probable) topic") +
  theme_bw()

# reminder of the top topic terms
sci_top_topic_terms

# Each group has one clearly dominant topic, except for the sci.space group
# where besides dominant topic 2, there is a notable presence of topic 1. 


##########################################
# APPLY THE MODEL ON A NEW (TEST) DATASET
##########################################

# Load the (pre-processed) 20 Newsgroups test data
cleaned_test_posts <- read.csv("data/20news-bydate/20news-bydate-test.csv", 
                               stringsAsFactors = FALSE)
# Select science-related newsgroups
sci_test_posts <- cleaned_test_posts %>% 
  filter(str_detect(newsgroup, "^sci\\."))

# Apply the same text transformation steps as done on the training data
sci_test_tokens <- tokens(sci_test_posts$post_txt, what = "word",
                          remove_punct = TRUE, remove_symbols = TRUE, 
                          remove_numbers = TRUE, remove_url = TRUE)

sci_test_tokens <- tokens_tolower(sci_test_tokens) %>%
  tokens_remove(stopwords()) %>%
  tokens_keep(min_nchar = 2) %>%
  tokens_wordstem(language = "english")

# Create DTM
sci_test_dfm <- dfm(sci_test_tokens, tolower = FALSE)
sci_test_dfm
# Now, we have to assure that test DTM has the same set of features (words)
# as the training DTM; otherwise, our LDA model would not be able to work 
# with it. Recall that we did the same thing when evaluating our RF 
# classifier on the test dataset (TM_Intro_Newsgroup_Classifier.R script)
sci_test_dfm <- dfm_select(sci_test_dfm, 
                           pattern = sci_dfm_reduced,
                           selection = "keep")
sci_test_dfm

# Considering that we have represented posts from the test dataset using
# the features (words) from the training set, it might happen that some
# of the test posts contain none of the words from the new feature set, 
# that is, all elements of the corresponding test DTM rows are zeros. 
# Let's check:
which(rowSums(sci_test_dfm)==0)
# 7 such posts (rows); we need to remove them to be able to proceed
sci_test_dfm <- dfm_subset(sci_test_dfm, 
                           subset = (rowSums(sci_test_dfm)!= 0)) 
sci_test_dfm

# Now that we have the same features as in our LDA model and have removed
# all-zero rows from the test DTM, we can use the LDA model to estimate 
# - topic probabilities for posts from the test dataset
# - term probabilities for each topic 
test_estimates <- posterior(sci_lda, 
                            newdata = sci_test_dfm,
                            control = list(seed = 2018))

# Topic probability distribution for each test post
test_topic_probs <- test_estimates$topics
head(test_topic_probs)

# Term probability distribution for each topic
test_term_probs <- test_estimates$terms
head(t(test_term_probs))
# Notice that these distributions are the same as before
# (compare the results of the previous line with: head(t(sci_term_prob)))
# This is expected since the 'structure' of our topics did not change.
# What is new is the distribution of topics over the test documents (posts).

# Now you can, for example, use the same kinds of visualizations as we used 
# above to examine distribution of topics across (test) newsgroups. 

# Before proceeding, we will remove large objects that are not needed any more
remove(sci_test_dfm, sci_test_tokens, cleaned_test_posts)

##################################
## CHOOSING THE NUMBER OF TOPICS
##################################

seed <- 2018

# If we didn't know the newsgroups the examined posts originated from, 
# we wouldn't be able to suggest 4 as the number of topics. So, we would 
# have to guess the number of topics or apply a statistical approach to 
# choosing the number of topics.

#
# USING TUNING FUNCTIONS FROM THE ldatuning R PACKAGE
#

# install.packages('ldatuning')
library(ldatuning)

## The package implements 4 methods for tuning the number of topics, 
## named after the papers the methods were published in: 
## Griffiths2004, CaoJuan2009, Arun2010, Deveaud2014. 
## References to papers that describe these methods are given in
## the package documentation:
## https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html
## Since all tuning methods are computationally intensive, as they 
## require training of multiple LDA models (to select the best performing one), 
## ldatuning uses parallelism to improve the performance. Still, even with the
## parallelisation, the process might take long time to complete.

# Do the tuning
k_tuning <- FindTopicsNumber(
  dtm = convert(sci_dfm_reduced, to = "topicmodels"),
  topics = seq(from = 3, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = seed, burnin = burnin, iter = iter, thin=thin),
  mc.cores = 3L, # number of logical cores to be used for parallelisation
  verbose = TRUE
)
k_tuning

# Save the results as the tuning process tends to last long
# saveRDS(k_tuning, "models/LDA/lda_tuning_results.RData")

# Load the saved results
# k_tuning <- readRDS("models/LDA/lda_tuning_results.RData")

# It will be far easier to examine the results visually
FindTopicsNumber_plot(k_tuning)

# Combining the results of the four plotted metrics, k = 20 seems to be the best solution.
# As that may be too high number for practical consideration, it may be be better to
# choose k = 7 as the second best option. 

#
# APPROACH BASED ON THE PERPLEXITY MEASURE 
#

# One way to computationally estimate the number of topics is to make use of the 
# perplexity measure. It is well exemplified in this StackOverflow thread:
# http://stackoverflow.com/questions/21355156/topic-models-cross-validation-with-loglikelihood-or-perplexity/
# (the 1st answer)

# Perplexity is a measure of how well a probability model fits a new set of data.
# The lower its value, the better.

# This approach requires splitting data into train and validation sets
# (perplexity is computed on the validation set)
set.seed(seed)
# Randomly choose 75% of posts for model building; the rest will be used for 
# computing the perplexity
n_docs <- nrow(sci_dfm_reduced)
train_indices <- sample(1:n_docs, round(n_docs * 0.75))
# when creating training and validation sets, transform them from 
# quanteda dfm format to the one used by the topicmodels package
sci_dtm_train <- convert(sci_dfm_reduced[train_indices, ], to = "topicmodels")
sci_dtm_valid <- convert(sci_dfm_reduced[-train_indices, ], to = "topicmodels")

# Compute perplexity for a range of k values (number of topics)
k_range <- seq(3, 20, by = 1)
m_perplexity <- lapply(k_range, 
                       function(k){
                         set.seed(seed)
                         # build a model
                         m <- LDA(sci_dtm_train, 
                                  k = k, method = "Gibbs",
                                  control = list(seed = seed, burnin = burnin, 
                                                 iter = iter, thin=thin))
                         # compute perplexity of the model on the validation set
                         perplexity(m, newdata = sci_dtm_valid) })

# Save the computed perplexity values as the computation lasts long
# saveRDS(m_perplexity, "models/LDA/topic_num_perplexity_results.RData")

# Load the saved results
# m_perplexity <- readRDS("models/LDA/topic_num_perplexity_results.RData")

# Plot the computed perplexity values
m_perplexity_df <- data.frame(topics = k_range, perplexity = unlist(m_perplexity))
ggplot(m_perplexity_df, aes(x = topics, y = perplexity)) +
  geom_point() +
  geom_line() +
  labs(x = "Candidate number of topics", 
       y = "Perplexity on the validation set") +
  scale_x_continuous(breaks = k_range)

# Again, the graph suggests high number of topics (20). An alternative may be to 
# look for the number of topics where the decrease in perplexity starts to diminish.
# In this case, k = 8 seem to be the point where drops in the perplexity start to 
# be smaller and smaller. So, k = 8 may be considered the optimal number of topics.

## The advantage of this method over the metrics implemented in the ldatuning package 
## is that perplexity is computed on the validation dataset, and not on the dataset
## used for building the model.
## A further improvement - in terms of model stability - can be achieved by using
## cross-validation for parameter tuning. An illustrative example is given here:
## http://freerangestats.info/blog/2017/01/05/topic-model-cv

#
# APPROACH BASED ON THE HARMONIC MEAN OF LOG LIKELIHOOD
#

## Another approach is based on the Harmonic Mean of Log Likelihood.
## In particular, it suggests choosing k that maximizes harmonic mean of LogLikelihood.
## It is well explained and exemplified at:
## http://stackoverflow.com/questions/21355156/topic-models-cross-validation-with-loglikelihood-or-perplexity/
## (the 2nd answer)

# We are introducing a new parameter: keep. Log-likelihood is evaluated and stored 
# in every 'keep' iteration 
keep <- 100 
# As before, we will build models with various number of topics (3-20)
sci_lda_models <- lapply(k_range, 
                         function(k){LDA(convert(sci_dfm_reduced, "topicmodels"), 
                                         k = k, method = "Gibbs",
                                         control = list(seed = seed, burnin = burnin, 
                                                        iter = iter, thin=thin,
                                                        keep = keep)) })

# Extract the log likelihood of the built models.
# Note: since by setting the keep parameter, we requested to keep log-likelihood in every '
# keep' iteration, the built LDA models contain all log-likelihood values including those 
# computed during the burnin iterations; log-likelihoods computed during burn-in period 
# need to be omitted
last_keep_in_burnin <- round(burnin/keep)
models_logLik <- lapply(sci_lda_models, function(m)  m@logLiks[-c(1:last_keep_in_burnin)])

# Save the computed log likelihood values as the computation lasts long
saveRDS(models_logLik, "models/LDA/topic_num_loglike_results.RData")

# Load the saved results
# models_logLik <- readRDS("models/LDA/topic_num_loglike_results.RData")

# Compute harmonic mean of LogLike for each model
source("UtilityFunctions.R")
models_hm <- sapply(models_logLik, function(h) harmonicMean(h))

# Plot the computed harmonic means
models_hm_df <- data.frame(topics = k_range, logLikHM = models_hm)
ggplot(models_hm_df, aes(x = topics, y = logLikHM)) + 
  xlab("Number of topics") + 
  ylab("Harmonic Mean of models' Log likelihood") + 
  geom_point() + geom_line() + 
  scale_x_continuous(breaks = k_range) +
  theme_bw()

# The Harmonic Mean (HM) of LogLike keeps incresing as the number of topic increases; 
# hence, it is unclear what would be the best number of topics. One option may be 
# to choose the number of topics where HM of LogLike has the last highest increase, 
# which would be k = 7.