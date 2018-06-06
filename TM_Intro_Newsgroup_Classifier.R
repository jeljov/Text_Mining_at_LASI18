## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of the overall process of text classification, 
## including: 
## - preprocessing of textual data;
## - transformation of unstructured (textual) data into a structured data format (DTM)
##   that can be fed into a classification algorithm; this includes different methods
##   for feature weighting and selection, as well as methods for reducing / transforming
##   the feature space (i.e. turning a large number of sparse features into a much
##   smaller number of dense features);
## - application of classification algorithms on the transformed textual data (that is,
##   the built feature set);
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
## in active use with over 100K newsgroups (see: https://www.binsearch.info/groupinfo.php)
##
## The dataset is publicly available from: http://qwone.com/~jason/20Newsgroups/
##
## We will use the recommended, "bydate" version of the data set (20news-bydate.tar.gz),
## which consists of 18,846 documents, sorted by date into training(60%) and test(40%) sets. 
## Each subdirectory in both train and test bundles represents a newsgroup; 
## each file in a subdirectory is a message posted to that newsgroup.

# The script makes use of the following R packages:
# - caret, e1071 - for various ML tasks
# - randomForest - for building classifiers based on the Random Forest algorithm
# - quanteda - for various text analytics tasks
# - stringr - for advanced string processing
# - irlba - for singular vector decomposition (SVD)
# - dplyr, tidyr, purrr, reader - for general data analysis tasks 
# - ggplot2 - for visualization
# - doSNOW - for multi-core parallel process execution 
# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))

# Initially, we will load just a basic set of packages; 
# the others will be loaded along the way, as we need them.
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(stringr)

# Load a set of auxiliary functions
source("UtilityFunctions.R")

# Choose a seed to be used in various computations that depend on random processes.
# This is to assure reproducibility of the results.
seed <- 28418

########################
## READ DATA FROM FILES
########################

# We’ll start by reading in posts from the "20news-bydate-train" and 
# "20news-bydate-test" folders, which are organized in sub-folders, 
# each corresponding to one newsgroup, with one file for each post

training_folder <- "data/20news-bydate/20news-bydate-train/"
test_folder <- "data/20news-bydate/20news-bydate-test/"

# Read in the contents of all posts from the train dataset
# Note: read_folder() is a utility function defined in the UtilityFunctions.R 
raw_train_data <- data_frame(folder = dir(training_folder, full.names = TRUE)) %>%
  unnest(map(folder, read_folder)) %>%  # map results in a list-column; 
                                        # unnest transforms each element of a list into a row
  transmute(newsgroup = basename(folder), id, text)
# Do the same for the test dataset
raw_test_data <- data_frame(folder = dir(test_folder, full.names = TRUE)) %>%
  unnest(map(folder, read_folder)) %>%
  transmute(newsgroup = basename(folder), id, text)

# Examine the newsgroups that are included in the training and test datasets, 
# and the number of posts in each one
raw_train_data %>%
  group_by(newsgroup) %>%
  summarise(post_count = n_distinct(id)) %>%
  arrange(post_count)
raw_test_data %>%
  group_by(newsgroup) %>%
  summarise(post_count = n_distinct(id)) %>%
  arrange(post_count)


################################
## PRE-PROCESS (CLEAN) THE DATA
################################

# Each post has some extra text that we do not want to include in our analysis. 
# For example, every post has a header, containing fields such as 'from:' or 'in_reply_to:' 
# Some also have automated email signatures, which occur after a line containing just dashes 
# (e.g. '--' or '---')
# As an example, examine post 103128
with(raw_train_data, text[id==103128])

# Remove the post header and the automated signature
# Note: the remove_header_and_signature() f. is defined in the UtilityFunctions.R
cleaned_train_data <- remove_header_and_signature(raw_train_data)
cleaned_test_data <- remove_header_and_signature(raw_test_data)

# Check the 103128 post after the first cleaning step
with(cleaned_train_data, text[id==103128])

# Many lines also have nested text representing quotes from other users, 
# typically starting with a line like:
# "In article <snelson3.8.0@uwsuper.edu> snelson3@uwsuper.edu (SCOTT R. NELSON) writes:"
# or ">The rotation has changed due to..." 
# or "}first I thought it was an 'RC31'.."
# Remove such quotes
# (remove_quoted_text() is defined in the UtilityFunctions.R)
cleaned_train_data <- remove_quoted_text(cleaned_train_data)
cleaned_test_data <- remove_quoted_text(cleaned_test_data)

# Check the 103128 post after the 2nd cleaning step
with(cleaned_train_data, text[id==103128])

# We will also remove 3 posts from the training set (9704, 9985, and 14991)
# that contain a large amount of strange, non-text content
cleaned_train_data <- cleaned_train_data %>% filter(!(id %in% c(9704, 9985, 14991)))

# Now, merge all lines of text that belong to the same post
# (the merge_post_text() f. is defined in UtilityFunctions.R)
cleaned_train_posts <- merge_post_text(cleaned_train_data)
str(cleaned_train_posts)
cleaned_test_posts <- merge_post_text(cleaned_test_data)

# Store the cleaned train and test data into .csv files so that 
# they can be used later more easily
cleaned_train_posts %>%
  write_csv("data/20news-bydate/20news-bydate-train.csv")
cleaned_test_posts %>%
  write_csv("data/20news-bydate/20news-bydate-test.csv")


#########################################################
## SELECT TWO NEWSGROUPS FOR A BINARY CLASSIFICATION TASK  
#########################################################

## Even though the 20 Newsgroup dataset allows for multiclass classification,
## to make this example easier to follow and understand, we will limit ourselves 
## to a binary classification task. Note that the same procedure is applicable  
## to multiclass classification, only the computation process would be more 
## demanding and thus the computation time much longer. 

# For this example, we will choose the group discussing guns ('talk.politics.guns')
# and the one on Mideast ('talk.politics.mideast'), both within the broader 'politics'  
# topic. The rationale for this selection:
# - It represents a case that we can expect in a discussion forum of a course dealing 
#   with current political issues and topics.  
# - Being topically closely related, the two groups will pose a challenge for a 
#   classifier - it is not an easy task to differentiate between groups of  
#   posts where we can expect a lot of shared vocabulary (at least far more than
#   in the case of differentiating between posts on, e.g., space and medicine).     

selected_ngs <- c('talk.politics.guns', 'talk.politics.mideast') 
selected_lbls <- c('guns', 'mideast')

train_2cl <- cleaned_train_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, # transform newsgroup into a factor w/ shorter labels
                            levels = selected_ngs,
                            labels = selected_lbls))
test_2cl <- cleaned_test_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, 
                            levels = selected_ngs,
                            labels = selected_lbls))

# Examine the class balance in the train and test sets 
table(train_2cl$newsgroup)
table(test_2cl$newsgroup)
# Both training and test sets are well balanced.
# If this was not the case, that is, if there was a prominent class imbalance,
# we would have to apply a subsampling technique to reduce the difference;
# these techniques are well covered in the *caret* R package:
# https://topepo.github.io/caret/subsampling-for-class-imbalances.html

# We will now use the training set to build a classifier.
# Test set will be used later, only for evaluation purposes.

################################################
## DATA (TEXT) WRANGLING: 
## TEXT TRANSFORMATION AND FEATURE EXTRACTION
################################################

# There are many packages in the R ecosystem for performing text analytics.
# One of the latest is *quanteda*. It has many useful functions for
# quickly and easily working with text data; they are well explained in the
# quanteda docs:
# http://docs.quanteda.io/index.html
library(quanteda)

#
# Computing and examining corpus statistics
#

# We'll start by creating a corpus out of the newsgroups posts, 
# as this will allow us to easily obtain some basic corpus statistics
newsgroups_corpus <- corpus(x = train_2cl$post_txt)
# Add newsgroup as a document level variable
docvars(newsgroups_corpus, field = "newsgroup") <- train_2cl$newsgroup
# Inspect the corpus summary that quanteda makes available
summary(newsgroups_corpus, n = 10)
# Compute summary statistics for the whole corpus
corpus_stats <- newsgroups_corpus %>% 
  summary(n = nrow(train_2cl)) %>%  # include all posts in the summary
  mutate(TTR = Types/Tokens)        # add TTR = Type Token Ratio as a measure of lexical diversity

# Note: for more sophisticated measures of lexical diversity, check the
# textstat_lexdiv() f. of the *quanteda* package, or the *koRpus* package
# https://reaktanz.de/?c=hacking&s=koRpus

# Compute some simple statistics per newsgroup.
# Note: omitting to inlcude max_TTR, min_token, and min_sent as these have
# the same value (1) for both groups 
corpus_stats %>% 
  group_by(newsgroup) %>%
  summarize(min_TTR = round(min(TTR), digits = 2),
            median_TTR = round(median(TTR), digits = 2),
            median_token = round(median(Tokens), digits = 2),
            max_token = max(Tokens),
            median_sent = round(median(Sentences), digits = 2),
            max_sent = max(Sentences)) 

# While the two newsgroups seem to be similar in terms of TTR (should be 
# checked further), they appear to differ in terms of post length (token and 
# sentence count). We can inspect this further visually.
library(ggplot2)
# First, compare token counts between the two newsgroups
# (the plotting f. is defined in the UtilityFunctions.R)
plot_ng_feature_comparison(corpus_stats, 'Tokens', "Token count")

# A few extreme outliers; let's examine outliers closer ...
outliers <- sort(boxplot.stats(corpus_stats$Tokens)$out, decreasing = TRUE)
outliers
# Examine the content of extreme outliers to see if these are regular posts
# (just unusually long) or they contain some erroneous text
extreme_outlier_indices <- which(corpus_stats$Tokens %in% outliers[1:7])
train_2cl$post_txt[extreme_outlier_indices]
# The 2nd among these outliers is a post with strange content and should be
# removed. 
# But, before that, for the sake of better comparison of the two newsgroups
# with respect to the post length, we will replot the distribution of token 
# counts without the extreme outliers
plot_ng_feature_comparison(corpus_stats[-extreme_outlier_indices,], 'Tokens', "Token count")
# It seems that posts on Mideast tend to be longer than those discussing guns.

# Now, we can remove the identified errenous post 
corpus_stats <- corpus_stats[-(extreme_outlier_indices[2]),] 
train_2cl <- train_2cl[-extreme_outlier_indices[2],]

# Next, compare the two groups based on the sentence count
plot_ng_feature_comparison(corpus_stats, 'Sentences', "Sentence count")
# Again, a few extreme outliers that prevent us from comparing the distribution of the
# sentence number in the two newsgroups. Let's examine these outliers
outliers_2 <- sort(boxplot.stats(corpus_stats$Sentences)$out, decreasing = TRUE)
outliers_2
extreme_outlier_indices <- which(corpus_stats$Sentences %in% outliers_2[1:6])
# Examine the content of the extreme outliers
train_2cl$post_txt[extreme_outlier_indices]
# These posts look correct, just unusually long; so, no real reason to remove any of them.
# Just for the sake of better comparison of the two newsgroups with respect to the post
# length, we will replot the distribution of sentence counts without the extreme outliers
plot_ng_feature_comparison(corpus_stats[-extreme_outlier_indices,], 'Sentences', 
                           "Sentence count")
# It seems that posts on Mideast tend to be longer also in terms of the number of sentences.

# Finally, compare the two newsgroups based on the TTR
plot_ng_feature_comparison(corpus_stats, 'TTR', "Type Token Ratio (TTR)")
# Again, a few extreme outliers that we should look into
outliers_3 <- sort(boxplot.stats(corpus_stats$TTR)$out)
outliers_3
extreme_outlier_indices <- which(corpus_stats$TTR %in% outliers_3)
# Examine the content of the extreme outliers
train_2cl$post_txt[extreme_outlier_indices]
# Only the 2nd should be removed, as is not meaningful
# But, before that, for the sake of better comparison of the two newsgroups
# with respect to the TTR, we will replot the distribution of TTR without 
# the extreme outliers
plot_ng_feature_comparison(corpus_stats[-extreme_outlier_indices,], 'TTR', 
                           "Type Token Ratio (TTR)")
# The difference in TTR is less pronounced, though might turn to be relevant; 
# we can also check it using a statistical test:
kruskal.test(corpus_stats$TTR ~ corpus_stats$newsgroup)
# The test suggests that TTR might be a relevant feature.
# Before proceeding, we just need to remove that one outlier we detected
# as not meaningful post
train_2cl <- train_2cl[-extreme_outlier_indices[2],]
corpus_stats <- corpus_stats[-extreme_outlier_indices[2],]

# Add the 3 examined corpus statistics to the training dataset as
# potential features for classification
train_2cl$TokenCnt <- corpus_stats$Tokens
train_2cl$SentCnt <- corpus_stats$Sentences
train_2cl$TTR <- corpus_stats$TTR

#
# Tokenization of posts
#

# When tokenizing documents, a typical practice is to remove numbers, 
# punctuation marks, symbols, and urls. 
?tokens
train_tokens <- tokens(x = train_2cl$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE,
                       remove_url = TRUE)

# Take a look at a specific post and see how it has been transformed
train_2cl$post_txt[9]
train_tokens[[9]]

# The example post indicates that there are tokens with 1 or 2 characters only; 
# these should be removed as they rarely bear any meaning
train_tokens <- tokens_keep(x = train_tokens, min_nchar = 3)
train_tokens[[9]]

# The example post also indicates the presence of email addresses; we will remove them,
# though this might not be necessary as they will probably be removed later due to 
# their low relevance / weight 
train_tokens <- tokens_remove(x = train_tokens, 
                              pattern = "^[\\w\\.-]+@([\\w-]+\\.)+\\w{2,4}$",
                              valuetype = "regex", verbose = TRUE)
train_tokens[[9]]

## Note: regular expressions are very handy and often indispensable for text cleaning.
## If in need to learn about them or refresh your memory, this tutorial is excellent:
## http://regex.bastardsbook.com/
## Also, the following R cheatsheet comes in useful:
## https://www.rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf 

# Next, we will reduce all tokens to lower letters to reduce the variability of 
# the token set (a part of the process known as text normalization)
train_tokens <- tokens_tolower(train_tokens)
train_tokens[[9]]

# Since forum posts, as well as messages exchanged in other kinds of online  
# communication channels (e.g. chat, status posts), tend to have misspelled 
# words, it might be useful to do spelling correction, as a part of the text 
# normalization step. Typical approach is to check the text against some of 
# the available misspelling corpora 
# (e.g. http://www.dcs.bbk.ac.uk/~ROGER/corpora.html).
# There is also an R package - spelling - for spell checking:
# https://cran.r-project.org/web/packages/spelling/index.html
# We will skip this step for now.

# Next, we will remove stopwords.
# To that end, we will use quanteda's built-in stopword list for English.
?stopwords
# Note: depending on the task at hand, you might want to extend the built-in
# stopword list with additional, corpus specific 'stopwords' 
# (e.g. overly frequent words in the given corpus). Or, you might want to 
# consider a larger stopwords list (than the default one); in that case, 
# check the stopwords lists offered by the stopwords R package:
# stopwords::stopwords_getsources()
# In any case, it is advised to inspect a stopword list before 
# applying it to the problem at hand.
head(stopwords(), n = 30)
tail(stopwords(), n = 30)
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
# By default, words are weighted with their term frequencies (TF)
?dfm
train_dfm <- dfm(x = train_tokens, 
                 tolower = FALSE)  # we've already lower cased the tokens

train_dfm
# It's very sparse (sparsity = the proportion of cells that have zero counts).  
# We can get the precise level of sparsity with:
sparsity(train_dfm)

# Considering the large number of features (~12.35K) and the high level of
# sparsity, we should consider removing rare features, that is, words with 
# very low corpus frequency.

# Examine total (corpus) TF for each word:
summary(colSums(train_dfm))
# Considering the large number of features and their low corpus
# frequency, we will keep only those features (words) with 
# above median total TF
tf_total <- colSums(train_dfm)
to_keep <- which(tf_total > median(tf_total))
train_dfm_reduced <- dfm_keep(train_dfm, pattern = names(to_keep),
                              valuetype = "fixed", verbose = TRUE)
train_dfm_reduced
# a significant reduction: from ~12.35K to ~5K features

# It is also recommended to remove words that are overly frequent,
# that is, present in a large proportion of corpus documents, since
# such words are not informative for differentiating bwtween document groups.

# So, we'll examine the number of documents each word appears in (document
# frequency - DF), to check if there might be those that are present in a 
# large proportion of documents
dfreq <- apply(train_dfm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
# Word w/ max presence is in about 1/3 of the corpus documents; hence, 
# no need for this kind of trimming. 

# Use the (reduced) DTM to setup a feature data frame with labels.
# It will serve as the input for a classification algorithm 
train_df <- cbind(Label = train_2cl$newsgroup, data.frame(train_dfm_reduced))


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
cp_Grid = expand.grid( .cp = seq(from = 0.0005, to = 0.01, by = 0.0005)) 
# Build a DT classifier through CV 
# (cross_validate_classifier() function is defined in the UtilityFunctions R script)
rpart_cv_1 <- cross_validate_classifier(seed,
                                        nclust = 5,
                                        train.data = train_df,
                                        ml.method = "rpart",
                                        grid.spec = cp_Grid)

# Since model building takes some time, save the model 
# to have a quick access to it later
saveRDS(rpart_cv_1, "models/classification/rpart_cv_1.RData")

# Load the saved model
# rpart_cv_1 <- readRDS("models/classification/rpart_cv_1.RData")

# Check out the results:
rpart_cv_1
plot(rpart_cv_1)

# First, take the cp value of the best performing model
tf_best_cp <- rpart_cv_1$bestTune$cp
# Then, extract performance measures for the best cp value 
tf_best_results <- with(rpart_cv_1, results[results$cp == tf_best_cp,])
tf_best_results

# We can also examine the words (features) that the DT algorithm selected as the most important
sort(rpart_cv_1$finalModel$variable.importance, decreasing = TRUE)[1:30]


################################
## APPLY TF-IDF WEIGHTING SCHEME
################################

# Our first model can be characterized as being based on Bag-of-Words
# text representation, since we used only words (unigrams) and 
# estimated words' relevancy based on their frequency (TF).

# However, TF is not always the best metric for estimating the relevance of a word 
# in a corpus. Term Frequency-Inverse Document Frequency (TF-IDF) tends to provide
# better results, and here is why:
# - The TF metric does not account for the fact that longer documents will have 
#   higher individual term counts. By normalizing TF values, using, for example, 
#   L1 norm, that is, the document length expressed as the number of words,
#   we get a metric that is length independent.
# - The IDF metric accounts for the frequency of term appearance in all documents 
#   in the corpus. The intuition being that a term that appears in almost every  
#   document has practically no predictive power.
# - The multiplication of (normalized) TF by IDF allows for weighting each term 
#   based on both its specificity at the level of the overall corpus (IDF) and its 
#   specificity for a particular document (TF).

train_tfidf <- dfm_tfidf(train_dfm, scheme_tf = "prop")
train_tfidf

# Again, we need to remove low weighted terms to reduce the feature space.
# Examine total TF-IDF for each word
tfidf_total <- colSums(train_tfidf)
summary(tfidf_total)
# Considering the number of words (features), we need to make a 
# significant reduction, so we will keep only words with TF-IDF 
# above the 75th percentile
# Note: the initial threashold was set to the median value, but 
# subsequent experimentation proved that using the 75th percentile 
# led to the same classifier performance, with significantly 
# less features 
threshold <- quantile(tfidf_total, probs = 0.75)
to_remove <- which(tfidf_total <= threshold)
train_tfidf_reduced <- dfm_remove(train_tfidf, pattern = names(to_remove),
                                  valuetype = "fixed", verbose = TRUE)
train_tfidf_reduced
# We have reduced the feature set from ~12.35K to ~3K words

# Make a data frame, to serve as input for a classifier, using the same process as before
train_tfidf_df <- cbind(Label = train_2cl$newsgroup, data.frame(train_tfidf_reduced))
# Use this data frame as the input for the next classification model

############################################################
# BUILD the 2nd ML MODEL: RPART + UNIGRAMS + TF-IDF WEIGHTS
############################################################

# Build a classifier through CV, using the new version of the DTM 
# (DTM w/ TF-IDF weights) and all other settings unchanged
rpart_cv_2 <- cross_validate_classifier(seed,
                                        nclust = 5,
                                        train.data = train_tfidf_df,
                                        ml.method = "rpart",
                                        grid.spec = cp_Grid)

# Save the model to have a quick access to it later
saveRDS(rpart_cv_2, "models/classification/rpart_cv_2.RData")

# Load the saved model
# rpart_cv_2 <- readRDS("models/classification/rpart_cv_2.RData")

# Check out the results
rpart_cv_2
plot(rpart_cv_2)

# Extract and store evaluation metrics for the best performing model
tfidf_best_cp <- rpart_cv_2$bestTune$cp
tfidf_best_results <- with(rpart_cv_2, results[results$cp == tfidf_best_cp,])

# Compare the performance of the two classification models built so far
data.frame(rbind(tf_best_results, tfidf_best_results), 
           row.names = c("TF", "Norm_TF-IDF"))
# Somewhat better performance, both in terms of Accuracy and Kappa measures

# Before proceeding, let's examine the words (features) used for building the model
# by making use of the estimated feature importance:
sort(rpart_cv_2$finalModel$variable.importance, decreasing = TRUE)[1:30]

#####################################
## ADDING BIGRAMS TO THE FEATURE SET
#####################################

# N-grams allow us to augment our DTM matrix with word ordering.
# This tends to lead to increased performance (e.g., accuracy)
# over ML models trained with unigrams only. On the down side,
# the inclusion of ngrams, even only bigrams and trigrams, 
# leads to an explosion in the number of features.

# Let's expand our feature set by adding just bigrams, and 
# apply TF-IDF metric to the expanded DTM to see if the 
# performance would improve.

# Include bigrams in the token set
train_tokens_2 <- tokens_ngrams(train_tokens, n = 1:2)
train_2cl$post_txt[9]
train_tokens_2[[9]]

# Create a TF-IDF weighted dfm
train_dfm_2 <- dfm(train_tokens_2, tolower = FALSE) %>%
  dfm_tfidf(scheme_tf = "prop")
train_dfm_2
# Note the (huge) number of features and the level of sparsity

# As before, we will compute total TF-IDF weight per feature so 
# that we can remove features that are expected to bring in 
# more noise than signal (information)
tfidf_2_total <- colSums(train_dfm_2)
summary(tfidf_2_total)
# Note how skewed the distribution is
quantile(tfidf_2_total, probs = seq(0.9,1,0.025))
# Considering the number of features (over 114K), we will
# keep only ngrams with TF-IDF weights in the top 2.5 percentile 
# Note: setting threshold to top 10, top 5, and top 2.5 percentile
# resulted in almost the same performance metrics, so, the top 2.5 per.
# is chosen as it allows for the largest reduction of the feature space 
threshold <- quantile(tfidf_2_total, probs = 0.975)
to_remove <- which(tfidf_2_total <= threshold)
train_dfm_topXperc <- dfm_remove(train_dfm_2, pattern = names(to_remove),
                                 valuetype = "fixed", verbose = TRUE)
train_dfm_topXperc
# We've reduce the feature set to ~2.7K ngrams 

# Make a clean data frame to be used as input for building a classifier
train_tfidf_df_2 <- cbind(Label = train_2cl$newsgroup, data.frame(train_dfm_topXperc))

#####################################################################
# BUILD the 3rd ML MODEL: RPART + UNIGRAMS & BIGRAMS + TF-IDF WEIGHTS
#####################################################################

# Build a CV-ed model with the new feature set and all other settings unchanged
rpart_cv_3 <- cross_validate_classifier(seed, 
                                        nclust = 5,
                                        train.data = train_tfidf_df_2,
                                        ml.method = "rpart",
                                        grid.spec = cp_Grid)

# Save the model to have a quick access to it later
saveRDS(rpart_cv_3, "models/classification/rpart_cv_3.RData")

# Load the saved model
# rpart_cv_3 <- readRDS("models/classification/rpart_cv_3.RData")

# Check out the results:
rpart_cv_3
plot(rpart_cv_3)

# Extract and store evaluation metrics for the best performing model
tfidf_2_best_cp <- rpart_cv_3$bestTune$cp
tfidf_2_best_results <- with(rpart_cv_3, results[results$cp == tfidf_2_best_cp,])

# Compare the performance of the three classification models built so far
data.frame(rbind(tf_best_results, tfidf_best_results, tfidf_2_best_results), 
           row.names = c("TF", "Norm_TF-IDF", "Top2.5p_Ngram"))
# The model has notably weaker performance than the previous two. This suggests  
# that, in this case, the introduction of bigrams has brought in more noise 
# than signal.

# Before proceeding, let's examine the terms (features) used for building the model
sort(rpart_cv_3$finalModel$variable.importance, decreasing = TRUE)[1:30]

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
# feature space (singular vectors)
# So, let's re-examine the distribution of TF-IDF weights
summary(tfidf_2_total)
# Considering the number of features and very low TF-IDF values,
# we'll keep tokens with TF-IDF above the 3rd quartile
to_keep <- which(tfidf_2_total > quantile(tfidf_2_total, probs = 0.75))
train_dfm_4_svd <- dfm_keep(train_dfm_2, pattern = names(to_keep), 
                            valuetype = "fixed", verbose = TRUE)
train_dfm_4_svd  

# Next, we need to set the number of the most important singular vectors we wish 
# to calculate and retain as features (in SVD terms, it is the rank (k) the original 
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

# Save the SVD result to have a quick access to it later
saveRDS(svd_res, "models/classification/svd_res.RData")

# Load the saved model
# svd_res <- readRDS("models/classification/svd_res.RData")

# Examine the result:
str(svd_res)
# d - corresponds to singular values (values on the diagonal of the sigma matrix)
# u - corresponds to the left singular vector and respresents relation between
#     the extracted dimensions and the ngrams 
# v - corresponds to the right singular vector and respresents relation between
#     the extracted dimensions and the documents

# A glimpse into the new feature set (the right singular vector):
View(svd_res$v[1:20,1:50])

# Next, we will examine the predictive power of the model with singular 
# vectors as features

## Before proceeding to the creation of a classifier, note that there is
## an alternative approach to data preparation for SVD / LSA.
## It was suggested in the original paper on SVD / LSA by Landauer, Foltz, 
## & Laham (1998):
##    "Before the SVD is computed, it is customary in LSA to subject the data 
##    in the raw word-by-context matrix to a two-part transformation. 
##    First, the word frequency (+ 1) in each cell is converted to its log. 
##    Second, the information-theoretic measure, entropy, of each word 
##    is computed as: -p*logp over all entries in its row, 
##    and each cell entry then divided by the row entropy value. 
##    The effect of this transformation is to weight each word-type occurrence  
##    directly by an estimate of its importance in the passage and inversely 
##    with the degree to which knowing that a word occurs provides information 
##    about which passage [document] it appeared in."
##
## So, instead of TF-IDF, transform the original DTM (train_dfm_2) in the 
## manner suggested above, apply SVD on thus transformed DTM, and build 
## a RF model, as we do below. Compare the results with those of rf_cv_1 
## (given below, model 5). 


#############################################################################
# BUILD the 4th ML MODEL: RPART + SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM)
#############################################################################

# Create a new feature data frame using the 300 features obtained by applying
# SVD to TF-IDF weighted DTM (i.e. the V matrix produced by SVD)
train_svd_df <- cbind(Label = train_2cl$newsgroup, data.frame(svd_res$v))

# Build a DT-model with an expanded grid search space - as we now have
# significantly smaller number of features, CV will be far more efficient 
cpGrid_2 = expand.grid( .cp = seq(from = 0.0005, to = 0.02, by = 0.0005)) 
rpart_cv_4 <- cross_validate_classifier(seed, 
                                        nclust = 5,
                                        train.data = train_svd_df,
                                        ml.method = "rpart",
                                        grid.spec = cpGrid_2)

# Save the model
saveRDS(rpart_cv_4, "models/classification/rpart_cv_4.RData")

# Load the saved model
# rpart_cv_4 <- readRDS("models/classification/rpart_cv_4.RData")

# Check out the results
rpart_cv_4
plot(rpart_cv_4)

# Extract and store evaluation metrics for the best performing model
svd_best_cp <- rpart_cv_4$bestTune$cp
svd_best_results <- with(rpart_cv_4, results[results$cp==svd_best_cp,])

# Compare the performance of the present model with the performance of the previous models
comparison <- data.frame(rbind(tf_best_results %>% select(-cp), # exclude the cp parameter 
                               tfidf_best_results %>% select(-cp), 
                               tfidf_2_best_results %>% select(-cp), 
                               svd_best_results %>% select(-cp)),
                         row.names = c("TF", "Nom_TF-IDF", "Top2.5p_Ngram", "SVD"))
# add the number of features for each model
comparison$NFeatures <- c(ncol(train_df),
                          ncol(train_tfidf_df),
                          ncol(train_tfidf_df_2),
                          ncol(train_svd_df))
comparison
# We got significantly weaker performance, especially in terms of Kappa. 
# Still, note that we have almost 8-16 times less features than in the other models, 
# which makes our model more robust, that is, less prone to overfitting.

# We will try to improve the performance by using a more sophisticated ML algorithm.
# Now that we have a reasonable number of features, we can do that.

###############################################
# BUILD the 5th ML MODEL: RANDOM FOREST + 
# SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM)
###############################################

# Having reduced the dimensionality of our data, we can use a more complex and 
# powerful classification algorithm. We will build a Random Forest (RF) model.

## For a brief introduction to the Random Forest algorithm, 
## see the "Bagging and Random Forests" slides (made available as part of the WS materials).
## For more details and an excellent explanation of Random Forest and related algorithms,
## see chapter 8.2 of the Introduction to Statistical Learning book:
## http://www-bcf.usc.edu/~gareth/ISL/ 

# We will build a RF model with 1000 trees. We'll also try different 
# values of the mtry parameter to find the value that gives the best result. 
# The mtry parameter stands for the number of variables randomly sampled as 
# candidates at each split. 
# We will consider 10 different values for mtry, between the minimum
# (1 feature) and the maximum possible value (all features). 
n_features <- ncol(train_svd_df)-1
mtry_Grid = expand.grid( .mtry = seq(from = 1, to = n_features, length.out = 10))

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
saveRDS(rf_cv_1, "models/classification/rf_cv_1.RData")

# Load the saved model
# rf_cv_1 <- readRDS("models/classification/rf_cv_1.RData")

# Check out the results
rf_cv_1
plot(rf_cv_1)

# Extract evaluaton measures for the best performing model 
rf_1_best_mtry <- rf_cv_1$bestTune$mtry
rf_1_best_res <- with(rf_cv_1, results[results$mtry==rf_1_best_mtry,])

# Compare the results with the previously CV-ed models
comparison <- data.frame(rbind(comparison, 
                               c(rf_1_best_res %>% select(-mtry), 
                                 NFeatures=ncol(train_svd_df))),
                         row.names = c(row.names(comparison), "RF_SVD"))
comparison
# Obviously, the use of a more powerful algorithm significantly improved the results, 
# in terms of both Accuracy and Kappa measures.


###############################################
# BUILD the 6th ML MODEL: RANDOM FOREST + 
# SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM) +
# "HAND-CRAFTED" DENSE FEATURES 
###############################################

# Now let's add the features we created at the beginning
# of the script (hopefully, you haven't forgoten them:)) 
# to check if they will improve the model
str(train_2cl)

# Add TokenCnt, SentCnt, and TTR to the feature set:
train_svd_df <- cbind(train_svd_df, 
                      TokenCount = train_2cl$TokenCnt,
                      SentCount = train_2cl$SentCnt,
                      TTR = train_2cl$TTR)
tail(colnames(train_svd_df), n=5)

# Build a model with the new features added and all the other settings unchanged
rf_cv_2 <- cross_validate_classifier(seed,
                                     nclust = 5,
                                     train.data = train_svd_df,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid)

# Save the model to have it available for later
saveRDS(rf_cv_2, "models/classification/rf_cv_2.RData")

# Load the saved model
# rf_cv_2 <- readRDS("models/classification/rf_cv_2.RData")

# Examine the results
rf_cv_2
plot(rf_cv_2)

# Extract evaluaton measures for the best performing model (among the CV-ed ones)
rf_2_best_mtry <- rf_cv_2$bestTune$mtry
rf_2_best_results <- with(rf_cv_2, results[results$mtry==rf_2_best_mtry,])

# Compare the results with the previously built models
comparison <- data.frame(rbind(comparison, 
                               c(rf_2_best_results %>% select(-mtry), 
                                 NFeatures=ncol(train_svd_df))),
                         row.names = c(row.names(comparison), "RF_SVD_DenseFeat"))
comparison
# There is a tiny decrease in both Accuracy and Kappa, suggesting that the added
# features are not particularly relevant when compared to the other features. 
# But, can we estimate how (un)important they are?

# The *randomForest* R package provides handy functions for seamless exploration 
# of feature importance
library(randomForest)

varImpPlot(rf_cv_2$finalModel, type = 2) 
# (the type argument specifies the importance measure)
# The plot makes use of the Gini index to assess and depict feature importance.
# Gini index measures the total decrease in node impurity from splitting on a
# particular feature, averaged over all the trees. So, the larger the Gini value for a
# feature, the more relevant the feature is. This further implies that a large drop in Gini
# due to the feature removal, indicates the importance of the respective feature. 

# By default, the plot shows top 30 features; so, our 'dense' features are
# not among the top 30. We can look for them in another way:
rf_2_f_imp <- importance(rf_cv_2$finalModel, type = 2)
head(rf_2_f_imp)
rf_2_f_imp <- data.frame(Feature=row.names(rf_2_f_imp),
                         Importance=as.numeric(rf_2_f_imp))
# Order the features based on the estimated importance
rf_2_f_imp <- arrange(rf_2_f_imp, desc(Importance))
# Check the position of the 3 hand-crafted features
which(rf_2_f_imp$Feature %in% c("TokenCount", "SentCount", "TTR"))
# 51th,  78th, and 211th place

##################
# TEST THE MODEL
##################

# Now that we've built predictive models, it is time to evaluate the one that   
# proved the best (the 5th one: RF + Singular Vectors) using the test dataset
# we set aside at the beginning of the script.  
# The first stage of the evaluation process is running the test data through 
# our text transformation pipeline of:
# - Tokenization
# - Removing tokens less than 3 characters long 
# - Removing email addresses
# - Lower casing
# - Stopword removal
# - Stemming
# - Adding bigrams
# - Creating DTM and ensuring the test DTM has the same features 
#   (ngrams) as the train DTM
# - Computing TF-IDF weights 
# - Feature set transformation / reduction using the SVD matrices
#   computed on the training set

test_tokens <- tokens(x = test_2cl$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE,
                       remove_url = TRUE)

test_tokens <- tokens_keep(x = test_tokens, min_nchar = 3) %>%
  tokens_remove(pattern = "^[\\w\\.-]+@([\\w-]+\\.)+\\w{2,4}$", valuetype = "regex") %>%
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

# Transform test_dfm so that it has the same features as the DTM that  
# served as the input for SVD
test_dfm <- dfm_select(test_dfm, 
                       pattern = train_dfm_4_svd,
                       selection = "keep", verbose = TRUE)
test_dfm
# Now, test dfm seems to have the same features as the train dfm
# Let's check if those are really the same features
setdiff(colnames(test_dfm), colnames(train_dfm_4_svd))
setdiff(colnames(train_dfm_4_svd), colnames(test_dfm))
# no difference -> they are exactly the same

# With the initial test DTM in place, the next step is to 'project' 
# the test features into the same TF-IDF vector space we built for our 
# training data. This requires the following steps:
# 1 - Normalize term counts (TF) in each document (i.e, each row of DTM)
# 2 - Perform IDF multiplication using training IDF values
# (Note: we use the IDF values from the training set, as we represented
# our test data using the features (ngrams) of the training set; besides,
# a large proportion of the features would have DF=0 on the test set)

# Normalize term counts in all test posts
test_tf <- dfm_weight(test_dfm, scheme = "prop")

# Next, get IDF values of the training set features, in particular,
# of features included in the train_dfm_4_svd DTM
# (Note: the inverse_doc_freq() f. is defined in the UtilityFunctions R script)
train_idf <- apply(train_dfm_4_svd, 2, inverse_doc_freq)

# Next, calculate TF-IDF using IDF of our training corpus
# (Note: the tf_idf() f. is defined in the UtilityFunctions R script)
test_tfidf <-  apply(as.matrix(test_tf), 1, function(x) tf_idf(x, idf = train_idf))
dim(test_tfidf)
# Transpose the matrix (so that the documents are in the rows)
test_tfidf <- t(test_tfidf)

# With the test data projected into the TF-IDF vector space of the training
# set, we can now do the final projection into the training SVD space.

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
# Before applying this formula, let us examine why and how we use it

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
# singular vector space using the computed sigma_inverse and transposed_U_matrix; 
# this further means that we can take a new, unseen document (a post in our case),
# compute TF-IDF values for it and transform it into singular vector space
# so that it can be classified by our prediction model.


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
# feed into our trained machine learning model for predictions 
test_svd_df <- data.frame(Label = test_2cl$newsgroup, 
                          t(test_svd_hat)) # need to transpose it, to place documents in rows

# Now we can make predictions on the test data set using our best classifer  
# (rf_cv_1)
preds <- predict(rf_cv_1, newdata = test_svd_df)

# Examine the results
rf_1_test_eval <- confusionMatrix(data = preds, reference = test_svd_df$Label)
# (Note: the get_eval_measures() f. is defined in the UtilityFunctions.R script)
get_eval_measures(rf_1_test_eval)

# Let's compare these results with those obtained on the training data 
rf_cv_1_eval <- confusionMatrix(data = rf_cv_1$finalModel$predicted,
                                reference = train_svd_df$Label)
get_eval_measures(rf_cv_1_eval)

# All examined performance measures are notably better on the training data then 
# on the test data. This is expected, as ML models generally perform better on the 
# training than on the test set.

# Note: the fact that we got somewhat better results on CV...
comparison[5,]
# ... than when applying the final model on the whole training set; this is because
# the performance always slightly varies depending on the sample that is used for
# model building, and in CV the performance metrics are averaged over CV runs. 
