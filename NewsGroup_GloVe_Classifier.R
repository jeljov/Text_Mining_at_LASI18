## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of the text classification task using a subset of  
## the 20 Newsgroups dataset and word vectors derived from a pre-trained GloVe model.
##
## GloVe stands for Global Vectors for Word Representation.
## It is an unsupervised learning method for obtaining vector representation of words.
## It was introduced in the paper:
## Pennington, J., Socher, R. & Manning, C. D. (2014). Glove: Global Vectors for Word 
## Representation. EMNLP.(pp.1532-1543). URL: http://nlp.stanford.edu/pubs/glove.pdf
## 
## Pre-trained GloVe models are available for download from:
## https://nlp.stanford.edu/projects/glove/ 
## We will use one of the models bundled within the "glove.6B.zip" data file. 
## Some facts about this model:
## - it was trained on the corpus that combined a 2014 Wikipedia dump and Gigaword5,
##   and consisted of 6 billion tokens (hence 6B in the file name)
## - after tokenizing and lowercasing the corpus, 400K most frequent words were used
##   to build the vocabulary
## - the zip includes models with 50, 100, 200, and 300 dimension vectors 
##   (each one in a separate .txt file) 

## We will examine some simple ways of using pre-trained word vectors 
## to create document features that can be subsequently used for 
## document classification. Specifically, we will examine:
## - computing TF weighted average of word vectors
## - computing TF-IDF weighted average of word vectors
## - taking min and max values of TF-IDF weighted word vectors and concatenating them.
## These methods are found to perform well on short texts. This means that they can
## be applied e.g. on posts exchanged in online communication channels that are often 
## used in e-learning contexts (e.g. forums, chats, Twitter and similar social networks).
##
## We will also examine the use of Word Mover Distance (WMD) for computing document 
## similarity by leveraging vector representation (word vectors / embeddings) of
## words the documents consist of. The computed document similarities are used as
## the input for the K Nearest Neighbours (kNN) classification algorithm. This method
## is also found to perform well on short texts.

## For the document classification task, we will use a subset of the 20 Newsgroups dataset.
## As we will attempt a binary classification task, we'll choose 2 newsgroups from the
## original dataset. In particular, newsgroups on politics (talk.politics.misc) and 
## religion (talk.religion.misc). The rationale for this selection:
## - the topics are general ones and can be expected to be well represented
##   in Wikipedia and Gigaword5 (sources for the GloVe model)
## - based on the token-related stats, the posts are not overly long
##   (recall: the methods to be applied proved to work well for short/shorter texts)
## - the groups are topically related; thus, it will not be an easy task for a
##   classifier to distinguish between posts of the two groups 
## - these could easily be topics discussed in a forum of an online / blended course


# Load the required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(quanteda)
library(stringr)
library(caret)

# Load auxiliary functions 
source("UtilityFunctions.R")

# Since we have already done the pre-processing of the 20 Newsgroups dataset,
# we will use the pre-processed data
# (for pre-processing details see TM_Intro_Newsgroup_Classifier R script)
usenet_data <- read.csv(file = "data/20news-bydate/20news-bydate-train.csv",
                        stringsAsFactors = FALSE)
str(usenet_data)

# Note: we will use just the training part of the 20 Newsgroups dataset, 
# and (later) split it into training and test sets, since we will need to
# perform several computationally demanding tasks; limiting our work to 
# smaller datasets will reduce the computation time.  

# Select groups on politics (talk.politics.misc) and religion (talk.religion.misc)
usenet_data <- usenet_data %>% 
  filter(newsgroup %in% c('talk.politics.misc', 'talk.religion.misc'))

# Create a corpus out of the newsgroups posts
usenet_corpus <- corpus(x = usenet_data$post_txt)
# Add newsgroup and post id as document level variables
docvars(usenet_corpus, field = "newsgroup") <- usenet_data$newsgroup
docvars(usenet_corpus, field = "post_id") <- usenet_data$id

# Compute some basic statistics about the token counts per newsgroup.
# This is to assure that we do not have overly long posts
# (recall that the methods we'll use work well with shorter texts)
usenet_corpus %>% 
  summary(n = nrow(usenet_data)) %>% # include all documents in the summary
  group_by(newsgroup) %>%
  summarize(doc_cnt = n(),
            tot_token = sum(Tokens),
            avg_token = mean(Tokens),
            median_token = median(Tokens),
            min_token = min(Tokens),
            max_token = max(Tokens))

# According to the examined token-related statistics, the two selected groups 
# have some overly long posts. We will check this more closely and remove 
# posts that are excessively long 
corpus_stats <- usenet_corpus %>% 
  summary(n=nrow(usenet_corpus$documents))
# Examine the distribution of post length (= token count) visually
# (plot_ng_feature_comparison() is defined in the UtilityFunctions.R script)
plot_ng_feature_comparison(corpus_stats, 'Tokens', 'Token count')
# A lot of outliers... Let's examine them closer
sort(boxplot.stats(corpus_stats$Tokens)$out, decreasing = TRUE)
# 1100 tokens seems to be a reasonable threshold: 
# - 29 (3.5%) posts will be removed 
# - texts of up to 1100 tokens (tokens are not only words but also punctuation marks, symbols
#   numbers...), while not short, still cannot be considered long 

# Removing, from the corpus, posts (documents) with more than 1100 tokens
posts_to_remove <- corpus_stats$post_id[corpus_stats$Tokens > 1100]
usenet_corpus <- corpus_subset(usenet_corpus, subset = !post_id %in% posts_to_remove)

#
# Transform the posts' content into a set of features to be used for classification
#

# Extract tokens from the corpus
post_tokens <- tokens(x = usenet_corpus, 
                      what = "word", 
                      remove_numbers = TRUE, 
                      remove_punct = TRUE,
                      remove_symbols = TRUE,
                      remove_twitter = TRUE, # removes leading '#' and '@' characters
                      remove_url = TRUE)
# Normalize tokens (set them to lower case), remove stopwords and 
# tokens with only 1 or 2 letters
post_tokens <- post_tokens %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_keep(min_nchar = 3) 
# Note that we are not stemming the tokens since words in the GloVe 
# model were not stemmed, and we need to match against those words.

# Create DTM
post_dtm <- dfm(post_tokens, tolower=FALSE)
post_dtm

# Extract words (features) from the DTM since we need to match these 
# against the words in the pre-trained GloVe model
post_words <- colnames(post_dtm)
# ... and examine them
head(post_words, n = 100)
tail(post_words, n = 100)

# Notice the presence of words ending with "'s" (such as "kaiser's"). 
# Replace such words with their version without "'s" 
end_with_s <- str_detect(post_words, "(\\w+)'s$")
end_with_s <- post_words[which(end_with_s)]
words_no_s <- str_replace(end_with_s, "(\\w+)'s", "\\1")
# Replace, in the tokens object, tokens that end with "'s" 
# with their 'cleaned' version
post_tokens <- tokens_replace(post_tokens, 
                              pattern = end_with_s, 
                              replacement = words_no_s) 

# Create again dtm
post_dtm <- dfm(post_tokens, tolower = FALSE)
post_dtm
# Create again a vector of vocabulary terms
post_words <- colnames(post_dtm)


# Load the pre-trained GloVe word vectors
# In particular, take the glove.6B.300d model 
# (model with 300 dimension word vectors)

# Read in the the model 
# (Note: change the 'glove_6B_300d_file' variable to the path of the 
#  "glove.6B.300d.txt" file on your computer)
glove_6B_300d_file <- "~/R Studio Projects/Large datasets/glove.6B/glove.6B.300d.txt"
g6b_300d <- scan(file = glove_6B_300d_file, what="", sep="\n")

# What we have read - g6b_300d - is in fact a huge character vector, 
# consisting of 400K entries - one entry per word from the vocabulary. 
g6b_300d[1]
# Each entry is given as a string that consists of 301 items
# delimited by a space: the 1st item is a word and the rest (300 items)
# are the estimated values of the 300 dimensions of that word

# Create a data frame out of the large vector read from the file
# (get_word_vectors_df() is defined in the UtilityFunctions.R script)
g6b_300d_df <- get_word_vectors_df(g6b_300d, verbose = TRUE)
dim(g6b_300d_df)
View(g6b_300d_df[1:20, 100:120])

# Remove g6b_300d to release memory
remove(g6b_300d)

# Take the words from the GloVe model - we need these words to 
# match them against the features (words) from the corpus DTM
glove_words <- colnames(g6b_300d_df)


# The next step is to match words from the post_dtm to
# the corresponding word vectors in the loaded GloVe model,
# and keep only those words that are present both in 
# post_dtm and in the GloVe model. 
words_to_keep <- intersect(post_words, glove_words)
# check the 'level' of matching
length(words_to_keep)/length(post_words)
# 89.24% of words from our DTM have their vectors in GloVe

# Let's briefly inspect words from post_dtm that are not in GloVe
setdiff(post_words, glove_words)[1:100]
# Mostly abbreviations, misspelled words, and compound words

# Create a new DTM that will keep only those words (columns)
# from the original DTM (post_dtm) that are present in the GloVe model  
dtm_reduced <- dfm_keep(post_dtm, pattern=words_to_keep, 
                        valuetype="fixed", verbose=TRUE)
dtm_reduced

# Likewise, from GloVe, select word vectors that will be used for building 
# a feature set, that is, vectors of the words present in the dtm_reduced
glove_to_keep_indices <- which(glove_words %in% words_to_keep)
g6b_300d_df_reduced <- g6b_300d_df[,glove_to_keep_indices]
# Remove the original glove df (g6b_300d_df)
remove(g6b_300d_df)

# Order the columns (words) in the g6b_300d_df_reduced, to be the same as in
# the dtm_reduced
g6b_300d_df_reduced <- g6b_300d_df_reduced[,colnames(dtm_reduced)]

# Before proceeding, remove large objects that are no longer needed
remove(usenet_data, corpus_stats, post_tokens, glove_words, post_words, post_dtm)

############################################################################
## Create features - to be used as the input for a classifier - by computing 
## weighted average (mean) of word vectors; TF is used as the weight
############################################################################

# Compute feature values for each post as the (coordinate-wise) TF-weighted mean 
# value across all the word vectors.
#
# Note that after the above reduction of DTM and GloVe to the common set of 
# features (words), the two matrices have the same number of columns.
# Now, we will take each post (row) from the DTM and multiply it with the transposed 
# GloVe matrix, thus, in fact weighting word vectors in GloVe with the post-specific 
# TF weights of the corresponding words. As the result, we will get a matrix of
# TF-weighted word vectors (words in rows, dimensions in columns) for each post. 
# Next, we take the mean value (across words) for each dimension (columns), to obtain
# a new feature vector for each post; these vectors have the same number of features 
# as there are dimensions in the GloVe model (300). This way, we are, in fact,  
# translating the existing feature space (words in DTM) into a new feature space 
# (dimensions of the GloVe word vectors).
word_vec_features <- data.frame()
for(i in 1:nrow(dtm_reduced)) {
  doc <- as.matrix(dtm_reduced)[i,]  
  doc_word_vecs <- doc * t(g6b_300d_df_reduced) 
  doc_features <- apply(doc_word_vecs, 2, mean)  
  word_vec_features <- as.data.frame(rbind(word_vec_features, doc_features))
}
colnames(word_vec_features) <- paste0("dim_",1:ncol(word_vec_features))
dim(word_vec_features)

# Add class label to the feature matrix
lbl <- ifelse(docvars(usenet_corpus, field = "newsgroup") == "talk.politics.misc",
              yes = "politics", no = "religion")
word_vec_features$Label <- as.factor(lbl)

# Check the class proportion
table(word_vec_features$Label)
prop.table(table(word_vec_features$Label))
# We have a slight class imbalance, but not at the level that 
# would require subsampling. 

# Split the data into training (80%) and test (20%) sets

## NOTE:
## Since the purpose of this example is just to illustrate the use of word vectors
## for document classification, to focus on that objective, we do not follow the 
## fully correct procedure of feature creation on the training set only, building a model, 
## and then representing the test set using the feature space of the training set 
## (as was done in the classification example in TM_Intro_Newsgroup_Classifier.R). 
## Instead, we simply split the created document-feature matrix into training and test parts. 
## However, if / when applying the above approach to feature creation in your own  
## work - be it research or practice - you are advised to follow the approach
## presented in the TM_Intro_Newsgroup_Classifier script. 

seed <- 6518
set.seed(seed)
train_indices <- createDataPartition(y = word_vec_features$Label, 
                                     p = 0.80, list = FALSE)
train_data <- word_vec_features[train_indices,]
test_data <- word_vec_features[-train_indices,]

# Create a Random Forest (RF) classifier.
# Use cross-validation to tune the mtry parameter;
# examine 10 different, equally spread, values for mtry
n_features <- ncol(word_vec_features) - 1
mtry_Grid = expand.grid( .mtry = seq(from = 1, to = n_features, length.out = 10))
rf_cv_1 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train.data = train_data,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid)

# Save the model to have it available for later
saveRDS(rf_cv_1, "models/glove/rf_cv_1.RData")

# Load the saved model
rf_cv_1 <- readRDS("models/glove/rf_cv_1.RData")

# Check out the CV results
rf_cv_1
plot(rf_cv_1)

# Examine the model's performance in more detail
rf_cv_1_eval <- confusionMatrix(reference = train_data$Label, 
                                data = rf_cv_1$finalModel$predicted)
get_eval_measures(rf_cv_1_eval)

# Make predictions on the test data
preds <- predict(rf_cv_1, newdata = test_data)
# Evaluate the model's performance on the test set...
rf_1_test_eval <- confusionMatrix(data = preds, reference = test_data$Label)
# ... and compare the evaluation metrics obtained on the train and the test data 
data.frame(rbind(get_eval_measures(rf_cv_1_eval), 
                 get_eval_measures(rf_1_test_eval)),
           row.names = c("TRAIN", "TEST"))

# Interestingly, we got somewhat better results on the test set than on the training set. 
# This can be due to following:
# - Training and test datasets originate from the same feature set (word_vec_features). 
#   That is, instead of creating separate feature sets for the training and test data 
#   (as we did in the TM_Intro_Newsgroup_Classification script), for convenience reasons,
#   we built one feature set (word_vec_features) and then split it into training and test
#   subsets. This could be the most likely reason for unusually good performance on the 
#   test set.
# - This might also be caused by relatively small dataset and large number of features.
# - It can also be the case that the model is not sufficiently 'stable', that is, it 
#   might be susceptible to the variance problem. 


# Before proceeding, remove large objects that are no longer needed
remove(usenet_corpus, doc_word_vecs, word_vec_features, train_data, test_data)

########################################################################
## Create document features by computing weighted average (mean) of 
## word vectors; TF-IDF is used as the weight
########################################################################

# Create TF-IDF weighted DTM
tf_idf_dtm_reduced <- dfm_tfidf(dtm_reduced, 
                                scheme_tf = "prop") # for TF, use normalized counts (ie. proportions)
tf_idf_dtm_reduced

# Compute feature values for each document as weighted (coordinate-wise) mean 
# value across all the word vectors, using word's TF-IDF value as the weight; 
# (it is the same procedure as the above one when TF weights were used)
word_vec_tf_idf_features <- data.frame()
for(i in 1:nrow(tf_idf_dtm_reduced)) {
  doc <- as.matrix(tf_idf_dtm_reduced)[i,]  
  doc_word_vecs <- doc * t(g6b_300d_df_reduced) 
  doc_features <- apply(doc_word_vecs, 2, mean)  
  word_vec_tf_idf_features <- as.data.frame(rbind(word_vec_tf_idf_features, doc_features))
}
colnames(word_vec_tf_idf_features) <- paste0("dim_",1:ncol(word_vec_tf_idf_features))
dim(word_vec_tf_idf_features)

# Add class label to the feature set
word_vec_tf_idf_features$Label <- as.factor(lbl)

# Split data into training and test sets in the same way as before
train_tf_idf <- word_vec_tf_idf_features[train_indices,]
test_tf_idf <- word_vec_tf_idf_features[-train_indices,]

# Create a RF classifier through CV
# Tune the mtry parameter as was done for the first model
rf_cv_2 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train.data = train_tf_idf,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid)

# Save the model to have it available for later
saveRDS(rf_cv_2, "models/glove/rf_cv_2.RData")

# Load the saved model
# rf_cv_2 <- readRDS("models/glove/rf_cv_2.RData")

# Check out the CV results
rf_cv_2
plot(rf_cv_2)

# Examine the model's performance in more detail
rf_cv_2_eval <- confusionMatrix(reference = train_tf_idf$Label, 
                                data = rf_cv_2$finalModel$predicted)
get_eval_measures(rf_cv_2_eval)

# Make predictions on the test data
preds <- predict(rf_cv_2, newdata = test_tf_idf)
# Evaluate the model's performance on the test set
rf_2_test_eval <- confusionMatrix(data = preds, reference = test_tf_idf$Label)
# ... and compare the evaluation metrics obtained on the train and the test data 
data.frame(rbind(get_eval_measures(rf_cv_2_eval), 
                 get_eval_measures(rf_2_test_eval)),
           row.names = c("TRAIN", "TEST"))

# All performance metrics are notably better on the test set.
# As this is no longer a small difference it requires further investigation.
# Suggested steps to investigate and remedy the problem:
# For the 2 selected newsgroups, use all the posts from the training portion of
# the 20 Newsgroups dataset (20news-bydate-train.csv) to create a feature set
# (as done above), and build a model. Then, use the posts of the two newsgroups
# from the test portion of the dataset (20news-bydate-test.csv) to test the model
# analogous to the way it was done in the TM_Intro_Newsgroup_Classifier script.
# This should help in 2 ways: 
# 1) by increasing the size of the dataset; 
# 2) by making the test set truly 'unseen' by the model.
# If after applying the suggested procedure the problem remains present, 
# it would mean that the model really 'suffers' from the variance problem  
# and has to be reconsidered, that is, created in a different way. 

# Considering the observed lack of 'stability' in the results, instead of comparing
# the models based on their performance on the test set, we will compare them based 
# on their cross-validation (CV) results. This is because CV results were obtained 
# by averaging the results (performance measures) over K iterations (K=5 in this case).
get_cv_results <- function(cv_model) {
  with(cv_model, results[results$mtry == bestTune$mtry,])
}
lapply(list(rf_cv_1, rf_cv_2), get_cv_results)

# Before proceeding, remove large objects that are no longer needed
remove(word_vec_tf_idf_features, train_tf_idf, test_tf_idf)

############################################################################
## Create document features by taking min and max of TF weighted 
## word vectors and concatenating them; thus, the number of features 
## will be 2 times larger than the number of dimensions of the word vectors
############################################################################

# Create the feature set by taking and concatenating (coordinate-wise) 
# min and max values across TF weighted word vectors; 
# so, instead of one feature per word, there will be 2 features
word_vec_min_max <- data.frame()
for(i in 1:nrow(dtm_reduced)) {
  doc <- as.matrix(dtm_reduced)[i,]  
  doc_word_vecs <- doc * t(g6b_300d_df_reduced) 
  vec_min <- apply(doc_word_vecs, 2, min)
  vec_max <- apply(doc_word_vecs, 2, max)
  word_vec_min_max <- as.data.frame(rbind(word_vec_min_max, c(vec_min, vec_max)))
}
colnames(word_vec_min_max) <- paste0("dim_",1:ncol(word_vec_min_max))
dim(word_vec_min_max)

# Add class label to the feature set
word_vec_min_max$Label <- as.factor(lbl)

# Split data into training and test in the same manner as done before
train_min_max <- word_vec_min_max[train_indices,]
test_min_max <- word_vec_min_max[-train_indices,]

# Create a RF classifier as in the case of the two previous models.
# Note: since we now have two times more features than in the previous 2 models, 
# to keep the same level of granularity of the search grid for the mtry parameter
# (as in the previous two cases), we need to expand the search grid:
n_f_min_max <- ncol(word_vec_min_max) - 1
mtry_Grid_2 = expand.grid( .mtry = seq(from = 1, to = n_f_min_max, length.out = 20))
rf_cv_3 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train.data = train_min_max,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid_2)

# Save the model to have it available for later
saveRDS(rf_cv_3, "models/glove/rf_cv_3.RData")

# Load the saved model
# rf_cv_3 <- readRDS("models/glove/rf_cv_3.RData")

# Check out the results
rf_cv_3
plot(rf_cv_3)

# Compare the performance of all three models, using CV performance measures 
lapply(list(rf_cv_1, rf_cv_2, rf_cv_3), get_cv_results)
# Based on the CV results, the first model (TF-weighted average) has the best performance, 
# whereas the 3rd one (min-max) is the weakest.

# Before proceeding, remove large objects that are no longer needed
remove(word_vec_min_max, train_min_max, test_min_max)

########################################################################
## Compare the obtained results with the performance of a classifier
## built on the feature set obtained by applying SVD on the traditional 
## VSM (TF-IDF weighted unigrams)
########################################################################

# Use the already computed TF-IDF DTM matrix
dim(tf_idf_dtm_reduced)

# Considering the number of words (features), creating a RF classifier 
# would last forever. Hence, we'll do dimensionality 
# reduction using SVD. This should also bring about better results.

library(irlba)
# Reduce the dimensionality down to 300 features (columns).
# 300 is chosen as the number that is often recommended (based on the experience in practice),
# and also because word vectors used in the above examined configurations had 300 dimensions.
set.seed(seed)
svd_res <- irlba(t(tf_idf_dtm_reduced), # it is transposed as SVD / LSA requires TDM as an input 
                 nv = 300,  # number of singular vectors to estimate
                 maxit = 600) # maxit is recommended to be twice larger than nv 

# Save the SVD result, to have it readily available
saveRDS(svd_res, "models/glove/svd_res.RData")

# Load the saved SVD object
# svd_res <- readRDS("models/glove/svd_res.RData")

# Examine the structure of the result 
str(svd_res)
# We will use the v matrix - right singular vector - as the feature set

# Create the feature set by extending the right singular vector (v)
# with the class label
svd_df <- data.frame(svd_res$v) %>%
  mutate(Label = as.factor(lbl))

# Split the data into training and test in the same manner as done above
train_svd <- svd_df[train_indices,]
test_svd <- svd_df[-train_indices,]

# Create a RF classifier through CV, in exactly the same manner as done above
rf_cv_4 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train.data = train_svd,
                                     ml.method = "rf",
                                     grid.spec = mtry_Grid)

# Save the model to have it available for later
saveRDS(rf_cv_4, "models/glove/rf_cv_4.RData")

# Load the saved model
# rf_cv_4 <- readRDS("models/glove/rf_cv_4.RData")

# Check out the results
rf_cv_4
plot(rf_cv_4)

# Compare CV performance with the previous three models
lapply(list(rf_cv_1, rf_cv_2, rf_cv_3, rf_cv_4), get_cv_results)

# Based on the CV results, the classifiers that rely on word vectors 
# performed better than the one built on the SVD reduced set of (weighted) unigrams. 

# Before proceeding, remove large objects that are no longer needed
remove(svd_res, tf_idf_dtm_reduced, svd_df, train_svd, test_svd)

########################################################################
## Create another classifier based on: 
## - (Relaxed) Word Mover Distance (WMD) for computing the distance / 
##   dissimilarity of documents (posts) 
## - kNN classification algorithm
## Compare the performance with the other, previously built, classifiers
########################################################################

## WMD and Relaxed WMD are proposed in the paper:
## Kusner, M. J., Sun, Y., Kolkin, N. I., & Weinberger, K. Q. (2015). 
## From Word Embeddings to Document Distances. In Proc. of the 32Nd Int'l Conf. 
## on Machine Learning - Vol. 37 (pp. 957–966). Lille, France: JMLR.org.
## URL: http://proceedings.mlr.press/v37/kusnerb15.pdf
##
## Check Figure 1 (p.1) in the paper to quickly get an intuition (general idea)
## this metric is based upon.

# We'll start by computing Relaxed WMD between each document pair using the 
# appropriate functions from the *text2vec* R package
# install.packages('text2vec')
library(text2vec)

# Create a Relaxed WMD (RWMD) object by specifying 2 input parameters:
# - word vector matrix with words given in rows and dimensions of the 
#   embedding space in columns; rows should have word names.
# - the method to be used for computing the distance between word vectors
rwmd_model = RWMD$new(wv = t(g6b_300d_df_reduced), method = "cosine")

## Note: in the original paper on WMD, authors use Euclidean distance to compute
## distance between word vectors; however, the author of text2vec suggests using
## Cosine similarity as in their experience, it results in better performance.
## In that case, distance is computed as: 1 - cosine_between_wv

# Now, we use the RWMD object and our DTM to compute distances between 
# each document pair. However, before that, we need to normalize TFs in
# the DTM matrix (required by the WMD algorithm; see the original paper)
dtm_norm <- dfm_weight(dtm_reduced, scheme = "prop")
rwmd_dist = dist2(x = dtm_norm, method = rwmd_model, norm = 'none')
dim(rwmd_dist)


# To build a KNN classifier, we will use the *FastKNN* R package.
# The reason for choosing this package is that it allows for building a KNN classifier
# using precomputed distances, which is not the case with the often used knn() f. from 
# the class package (and many other packages).
# install.packages("FastKNN")
library(FastKNN)

# First, get indices of training and test observations.
# We will use the same split into training and test (sub)sets as was done for
# the other classifiers. Recall that train indices are stored in the train_indices 
# variable; so, we just need to compute test indices
test_indices <- setdiff(1:nrow(dtm_norm), train_indices)

# Use the training portion of the dataset to find the best value for K
knn_eval_df <- data.frame()
train_labels <- as.factor(lbl[train_indices])
set.seed(seed)
for(k in seq(from = 5, to = 35, by=2)) {
  knn_res <- knn_training_function(dataset = as.matrix(dtm_norm)[train_indices,],
                                   distance = rwmd_dist[train_indices, train_indices],
                                   label = train_labels,
                                   k = k)
  knn_eval <- confusionMatrix(data = as.factor(knn_res), reference = train_labels)
  knn_eval <- get_eval_measures(knn_eval)
  knn_eval <- c(k, knn_eval)
  knn_eval_df <- as.data.frame(rbind(knn_eval, knn_eval_df))
}
colnames(knn_eval_df) <- c("K", "Precision", "Recall", "F1", "Accuracy", "Kappa")
knn_eval_df
# Sort the results based on accuracy
arrange(knn_eval_df, desc(Accuracy))
# k=23 is the best k value based on all the metrics except recall. 
# It might also be wise to consider k=17 as the optimal solution since it offers fairly 
# good performance, very close to the best one, and a lower risk of overfitting 
# (the higher the value for k, the higher susceptibility to overfitting).

# Now, evaluate the model, with k=23, on the test set

# The function we will use to test the kNN classifier requires a matrix with distances
# between each observation of the test set and the training set
test_train_dist <- rwmd_dist[test_indices, train_indices]
knn_pred <- knn_test_function(dataset = as.matrix(dtm_norm)[train_indices,],
                              test = as.matrix(dtm_norm)[test_indices,],
                              distance = test_train_dist,
                              labels = train_labels,
                              k = 23)
# Use the computed predictions and the test set labels to evaluate the model
knn_eval <- confusionMatrix(data = as.factor(knn_pred), 
                            reference = as.factor(lbl[test_indices]))
get_eval_measures(knn_eval)

# Recall the performance of the other examined classifiers
lapply(list(rf_cv_1, rf_cv_2, rf_cv_3, rf_cv_4), get_cv_results)

# While we cannot make any firm conclusions - considering that the performance metrics
# are computed in different ways (CV vs test set) - the comparison suggests that this 
# model (RWMD + kNN) is of comparable performance to the classifiers based on weighted 
# average of word vectors and better than that of the SVD-based classifier.