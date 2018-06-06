## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of the overall process of text clustering, including: 
## - transformation of textual data into a feature set
## - computation of document similarity (using different similarity measures)
## - application of a clustering algorithm, specifically, Affinity Propagation (AP)
## - evaluation of the clustering results  
##
## The example is based on a subset of the 20 Newsgroups dataset. In particular,
## it uses a collection of documents (posts) from the science-related newsgroups. 

# Load an initial set of required packages
# (more will be added along the way)
library(tidyr)
library(dplyr)
library(quanteda)
library(stringr)

# Read in a (pre-processed) subset of the 20 Newsgroups dataset
# (for the preprocessing details, see the TM_Intro_Newsgroup_Classifier R script)
ng_data <- read.csv(file = "data/20news-bydate/20news-bydate-train.csv",
                    stringsAsFactors = FALSE)

# Examine the available newsgroups
unique(ng_data$newsgroup)

# Select only posts that originate from the science-related newsgroups
sci_data <- ng_data %>%
  filter(str_detect(newsgroup, "^sci\\.")) %>%
  mutate(newsgroup = factor(newsgroup))
# Remove ng_data as no longer needed
remove(ng_data)

######################################
# TRANSFORM TEXT INTO FEATURE VECTORS
######################################

# Extract ngrams (unigrams and bigrams) from the corpus
# Do some basic text cleaning along the way
sci_tokens <- tokens(x = sci_data$post_txt, 
                     what = "word", 
                     remove_numbers = TRUE, 
                     remove_punct = TRUE,
                     remove_symbols = TRUE,
                     remove_url = TRUE,
                     ngrams = 1:2)

# Reduce token (ngram) variability and remove low informative ones
# - normalize tokens (set them to lower case); 
# - remove stopwords; 
# - remove tokens less than 3 character long; 
# - stem the tokens 
sci_tokens <- sci_tokens %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_keep(min_nchar = 3L) %>%
  tokens_wordstem(language = "english")

# Create DTM
sci_dfm <- dfm(sci_tokens, tolower = FALSE)
sci_dfm
# Huge number of features (195K) coupled with extremely high sparsity

# Remove the tokens object as it is very large and we won't need it any more.
# Note: we will keep removing large objects as they become no longer needed. 
remove(sci_tokens)

# Considering the huge number of features (195K), we need to apply
# some form of feature selection. 
# Let's start by examining ngrams' corpus frequency
ngram_tf_tot <- colSums(sci_dfm)
summary(ngram_tf_tot)
# As expected, highly uneven (power-law) distribution. 
# At least half of the ngrams appear only once in the overall corpus; as such,
# they cannot be informative for the clustering task. So, we'll start by removing 
# ngrams with only one occurrence at the corpus level  
sci_dfm <- dfm_trim(sci_dfm, min_termfreq = 2)
sci_dfm
# A huge reduction: from 195K to ~54.4K features (34.2% of the initial feature set)
# Remove the large object that is of no further use
remove(ngram_tf_tot)

# To further reduce the feature set by removing ngrams of low to no relevance,
# we will estimate the ngrams' importance by computing their TF-IDF weights
sci_tfidf_dfm <- dfm_tfidf(x = sci_dfm, scheme_tf = "prop")
# Examine ngrams' total TF-IDF weights
ngram_tfidf_tot <- colSums(sci_tfidf_dfm)
summary(ngram_tfidf_tot)
# Again, highly uneven distribution. 
# Considering the still very large number of features (54.4K) and their low weights
# we will keep only features (ngrams) with TF-IDF above the 75th percentile
ngrams_to_keep <- which(ngram_tfidf_tot > quantile(ngram_tfidf_tot, probs = 0.75))
sci_tfidf_dfm <- dfm_keep(sci_tfidf_dfm, 
                          pattern = names(ngrams_to_keep),
                          valuetype = "fixed", verbose = TRUE)
sci_tfidf_dfm
# Further reduction to ~13.6K features

# Do some clean up of the memory
remove(ngrams_to_keep, ngram_tfidf_tot)

# Next, we use SVD to transform our feature space (large and sparse) into a smaller and
# more 'compact' (denser) one: a feature space with significantly less dimensions, 
# but important information preserved

# Reduce the dimensionality down to 300 features (columns)
# (Note: see the TM_Intro_Newsgroup_Classifier R script for an explanation of why
# 300 dimensions are used and how to choose the number of dimensions)
library(irlba)
sci_svd <- irlba(t(sci_tfidf_dfm), nv = 300,  maxit = 600)

# As the computation lasts long, save the SVD result, to have it later for a quick access
# saveRDS(sci_svd, file = "models/AP/sci_svd.RData")
# Load the saved SVD result
# sci_svd <- readRDS(file = "models/AP/sci_svd.RData")

# Use the right singular vector as the new feature set
sci_sv_features <- sci_svd$v 
dim(sci_sv_features)

# Remove the large (SVD result) object from the memory, as we won't need it any more
remove(sci_svd)

# So, instead of ngrams, each document (post) is now represented using the 
# set of features (dimensions) extracted by applying the SVD procedure 
# over the original feature set (ngrams).

# The next step is to compute document similarity as the similarity of
# feature vectors the documents are represented with. 

##############################
# COMPUTE DOCUMENT SIMILARITY
##############################

## The *proxy* R package allows for seamless computation of numerous
## similarity and distance / dissimilarity measures. Supported measures 
## are listed in the proxy's documentation, specifically, here:
## https://cran.r-project.org/web/packages/proxy/vignettes/overview.pdf
# install.packages('proxy')
library(proxy)

# We will compute document similarity using 3 similarity measures:
# - Cosine similarity
# - Pearson correlation coefficient
# - Extended Jaccard index or Tanimoto index
# All three similarity measures are covered in the slides prepared for the WS
# ("Affinity propagation for Text Clustering")

#
# Start with the Cosine similarity
#
sci_cos_sim <- simil(x = sci_sv_features, method = "cosine", by_rows = TRUE)
dim(sci_cos_sim)
# Note the result is a squared matrix with dimensions equal to the number of
# documents; cell ij of the matrix is the similarity of the document in row i 
# and the one in column j.

#
# Next, compute correlation based similarity
#
sci_cor_sim <- simil(x = sci_sv_features, method = "correlation", by_rows = TRUE)

#
# Finally, compute extended Jaccard index
#
# Extended Jaccard index (eJaccard) is an extension / adaptation of 
# the 'original' Jaccard index to continuous variables
sci_ejaccard_sim <- simil(x = sci_sv_features, 
                          method = "eJaccard",
                          by_rows = TRUE)

# Now that we have computed document similarities, we can proceed to clustering.
# But, before that, we'll remove large objects that won't be needed any more
remove(sci_corpus, sci_dfm, sci_sv_features, sci_tfidf_dfm)

################################
# CLUSTERING OF NEWSGROUP POSTS
################################

## We will do the clustering using the Affinity Propagation (AP) algorithm.
## For a brief overview of the AP algorithm, see the slides on text clustering
## that are part of the WS materials. 
## For more details, check the original paper on AP by Frey & Dueck:
## Frey, B. J., & Dueck, D. (2007). Clustering by Passing Messages Between Data Points.
## Science, 315(5814), 972–976. https://doi.org/10.1126/science.1136800

## The *apcluster* R package provides full support for AP clustering.
## Check the following Web page for relevant materials (doc, tutorial)
## about this package: http://www.bioinf.jku.at/software/apcluster/
# install.packages("apcluster")
library(apcluster)

## The key method is apcluster()
?apcluster
## It has several input parameters among which, from practical perspective, 
## the following two are the most important:
## - s - similarity matrix
## - q - allows for easily setting / controling the value of the so-called
##  'input preferences', and thus, implicitly, determines the number of clusters
##  (input preferences largely control the number of clusters). Specifically,
##  this parameter (q), sets the value of input preferences to the q-th quantile 
##  of input similarities (matrix s), and thus directs the algorith to create more 
##  or less clusters. Default value for q is 0.5 (median of input similarities, as
##  recommended by AP authors); setting a smaller value will result in a smaller 
##  number of clusters.

#
# Run the AP algorithm w/ Cosine similarity
#

set.seed(2018)
sci_cos_apclust <- apcluster(s = as.matrix(sci_cos_sim, diag = 1), 
                             details = TRUE, q = 0)
# Note 1: the reason for assigning value 1 to the 'diag' parameter of the as.matrix()
# function is that in matrices produced by the simil() function values on the diagonal
# are NAs, and it is left to the user to set these values when transforming a simil 
# object to a matrix.
# Note 2: q is set to 0 since the default setting (q=0.5) led to very large 
# number of clusters (292).

# Check the number of exemplars (cluster 'representatives'), and thus, also, the number
# of clusters
length(sci_cos_apclust@exemplars)
# Still very large number of clusters (202), even with the preferences set in such 
# a way that minimum number of clusters should be generated.

## Note: this example demonstrates that setting input preferences to the minimum of
## input similarities (q=0) does not necessarily result in a small number of clusters.
## This is because input preferences need not be in the range of the input
## similarities. To get a precise range for input preferences, one can call the 
## preferenceRange() f., and use thus obtained values to control the number of
## clusters to be produced by the AP algorithm (via parameter p). Later, we will
## examine a more convenient way of controling for the number of clusters.

# To get more interpretable and useful results, we can try merging the obtained 
# clusters by applying Exemplar-based Agglomerative Clustering on top of the 
# AP clustering results.

## Exemplar-based Agglomerative Clustering is the same as 'traditional' agglomerative 
## clustering in all aspects but one: the merging objective. Here, the merging objective
## is based on the idea that two clusters should be joined if they are best described 
## by a joint exemplar (= a sample that maximizes the average similarity to all samples 
## in the two clusters considered for a merge).

## Apply Exemplar-based aggl. clustering to the clustering results
sci_cos_agg_ap <- aggExCluster(x = sci_cos_apclust, 
                               s = as.matrix(sci_cos_sim, diag = 1))
plot(sci_cos_agg_ap)
# Does not look helpful...

# Another option is to apply Exemplar-based aggl. clustering directly 
# on the similarity matrix (instead on the AP clustering results):
sci_cos_agg <- aggExCluster(s = as.matrix(sci_cos_sim, diag = 1))
plot(sci_cos_agg)
# Not useful at all in this case...
# Remove the three large objects that haven't proved useful
remove(sci_cos_agg, sci_cos_agg_ap, sci_cos_apclust)


# We can also 'request' from the algorithm to produce certain number of clusters. 
# This is enabled by the apclusterK() f. and its parameter K.
# It is an alternative to directly manipulating input preference values, in those 
# cases when we have an idea about the expected / desirable number of clusters.
# Here, we choose 4 clusters, as is the number of newsgroups.
set.seed(2018)
sci_cos_ap_4cl <- apclusterK(s = as.matrix(sci_cos_sim, diag = 1), 
                             K = 4, details = TRUE)
print(sci_cos_ap_4cl)

# Save the model since the model building is a lenghty process
# saveRDS(sci_cos_ap_4cl, file = "models/AP/sci_cos_ap_4cl.RData")
# Load the saved model
# sci_cos_ap_4cl <- readRDS(file = "models/AP/sci_cos_ap_4cl.RData")

# Let's examine the exemplars by associating them with the newsgroup labels.
sci_cos_ap_4cl_exemplars <- sci_data[sci_cos_ap_4cl@exemplars,]
sci_cos_ap_4cl_exemplars$newsgroup
# One exemplar per newsgroup - this suggests that the obtained clusters do,
# at least partially, match the newsgroups.
# Add exemplar (cluster) association to each document in the initial data frame
sci_data$cos_sim_4cl <- sci_cos_ap_4cl@idx

# Examine how well clusters match newsgroups by creating a matrix with
# cluster assignments given in rows and newsgroup membership in columns
cos_sim_ng_match <- matrix(data = with(sci_data, table(cos_sim_4cl, newsgroup)), 
                           nrow = 4, ncol = 4, 
                           dimnames = list(sci_cos_ap_4cl_exemplars$newsgroup, # row names
                                           unique(sci_data$newsgroup)))   # column names
cos_sim_ng_match
# Check how this looks in terms of proportions (of newsgroups within a cluster)
round(prop.table(cos_sim_ng_match, margin = 1), digits = 2)
# Not bad... A bit later we'll do more formal evaluation


#
# Run the AP algorithm w/ Correlation similarity
#

set.seed(2018)
sci_cor_apclust <- apcluster(s = as.matrix(sci_cor_sim, diag = 1), 
                             details = TRUE, q = 0)
length(sci_cor_apclust@exemplars)
# w/ default settings 293 clusters
# w/ q=0 202 clusters
# Large number of clusters, even with the preferences set in such a way that 
# minimum number of clusters is generated.
remove(sci_cor_apclust)

# Force the number of clusters: set 4 clusters as is the number of newsgroups
sci_cor_ap_4cl <- apclusterK(s = as.matrix(sci_cor_sim, diag = 1), 
                             K = 4, details = TRUE)

# Save the model since the model building is a lenghty process
saveRDS(sci_cor_ap_4cl, file = "models/AP/sci_cor_ap_4cl.RData")
# Load the saved model
# sci_cor_ap_4cl <- readRDS(file = "models/AP/sci_cor_ap_4cl.RData")

# Examine the exemplars by associating them with the newsgroup labels
sci_cor_ap_4cl_exemplars <- sci_data[sci_cor_ap_4cl@exemplars,]
sci_cor_ap_4cl_exemplars$newsgroup
# Each exemplar is associated with a different (unique) newsgroup
# Add exemplar association (i.e. cluster membership) to each document
sci_data$cor_sim_4cl <- sci_cor_ap_4cl@idx

# Examine how well clusters match newsgroups by creating cluster assignment (rows)
# by newsgroup (columns) matrix
cor_sim_ng_match <- matrix(data = with(sci_data, table(cor_sim_4cl, newsgroup)), 
                           nrow = 4, ncol = 4, 
                           dimnames = list(sci_cor_ap_4cl_exemplars$newsgroup, 
                                           unique(sci_data$newsgroup)))
cor_sim_ng_match
# Compute proportions (across rows) to check how newsgroups are spread across clusters
round(prop.table(cor_sim_ng_match, margin = 1), digits = 2)
# Very similar to the one w/ Cosine similarity


#
# Run the AP algorithm w/ (extended) Jaccard index as similarity measure
#

set.seed(2018)
sci_ejaccard_apclust <- apcluster(s = as.matrix(sci_ejaccard_sim, diag = 1), 
                                 details = TRUE, q = 0)
length(sci_ejaccard_apclust@exemplars)
# w/ default settings 314 clusters
# w/ q=0 settings 250 cluster
# Obviously, controling the clustering process via the q parameter will 
# not lead us to a useful clustering solution
remove(sci_ejaccard_apclust)

# So, as done in the two previous cases, set the number of clusters to 4, 
# hoping to match the 4 newsgroups
set.seed(2018)
sci_ejaccard_ap_4cl <- apclusterK(s = as.matrix(sci_ejaccard_sim, diag = 1), 
                                  K = 4, details = TRUE, 
                                  maxit = 2000, convits = 200)
# Note 1: maxit and convits parameters were increased as the algorithm didn't
# converge with the default parameter values
# Note 2: by setting the parameter 'prc', you can make the algorithm stop if it 
# finds the number of clusters that does not deviate more than prc percent from 
# the desired value K. This is a good option when having a rough idea about the
# desired number of clusters or when the algorithm does not converge for the 
# specified number of clusters.

# Save the model
# saveRDS(sci_ejaccard_ap_4cl, file = "models/AP/sci_jaccard_ap_4cl.RData")
# Load the saved model
# sci_ejaccard_ap_4cl <- readRDS(file = "models/AP/sci_jaccard_ap_4cl.RData")

# Examine the exemplars by associating them with the newsgroup labels
sci_ejacc_ap_4cl_exemplars <- sci_data[sci_ejaccard_ap_4cl@exemplars,]
sci_ejacc_ap_4cl_exemplars$newsgroup
# Each exemplar is associated with a different (unique) newsgroup, suggesting
# that the clusters might, at least partially, match the newsgroups 
# Associate each document with its exemplar (cluster) 
sci_data$ejacc_sim_4cl <- sci_ejaccard_ap_4cl@idx
# Examine how well clusters match the 4 newsgroups
ejacc_sim_ng_match <- matrix(data = with(sci_data, table(ejacc_sim_4cl, newsgroup)), 
                           nrow = 4, ncol = 4, 
                           dimnames = list(sci_ejacc_ap_4cl_exemplars$newsgroup, 
                                           unique(sci_data$newsgroup)))
ejacc_sim_ng_match
# Examine proportions (across rows) to check how newsgroups are spread 
# across each cluster (row)
round(prop.table(ejacc_sim_ng_match, margin = 1), digits = 2)
# Very similar, maybe even slightly better than the previous two results

#################################################
# EVALUATION AND COMPARISON OF CLUSTERING RESULTS
#################################################

## When available, pre-assigned labels (categories) are very useful 
## for cluster validation. Since our dataset comes with human assigned 
## labels (newsgroups), we will use them to measure the level of 
## matching / consistency between the resulting clusters and 
## the categories assigned by humans.
## Specifically, we will evaluate the clusters with often used measures
## of purity and entropy:
## - purity measures the coherence of a cluster, that is, the degree 
##   to which a cluster contains documents from a single category
## - entropy evaluates the distribution of categories in a given cluster

# Functions for computing purity and entropy, as well as some other 
# utility functions for cluster evaluation are given in the
# ClustEvalUtil script
source("ClustEvalUtil.R")

# Compute and compare purity for clusters obtained by applying the 
# AP algorithm with the 3 similarity measures 
# (the higher the purity, the better)
data.frame(rbind(clust_purity(cos_sim_ng_match),
                 clust_purity(cor_sim_ng_match),
                 clust_purity(ejacc_sim_ng_match)),
           row.names = c("Cosine", "Pearson", "eJaccard"))

# Compute and compare entropy
# (the smaller the value (closer to 0), the better)
data.frame(rbind(clust_entropy(cos_sim_ng_match),
                 clust_entropy(cor_sim_ng_match),
                 clust_entropy(ejacc_sim_ng_match)),
           row.names = c("Cosine", "Pearson", "eJaccard"))

# Compute and compare average entropy of the overall clustering solution 
list(cosine = avg_clust_entropy(cos_sim_ng_match),
     pearson = avg_clust_entropy(cor_sim_ng_match),
     ejaccard = avg_clust_entropy(ejacc_sim_ng_match))

# Based on purity and entropy, the 3 clustering results are very similar.
# The same conclusion was reported by Huang (2008) in the paper
# "Similarity Measures for Text Document Clustering"
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.4480&rep=rep1&type=pdf
# (though different clustering algorithm - k-means - was used)


## If human-assigned labels are not available, which is often the case,
## other evaluation measures need to be used. Note: even though numerous  
## measures have been defined, they still provide only a partial insight 
## into the quality of the clusters and some form of qualitative (manual)
## validation is often needed (e.g. inspecting a sample of documents from
## each cluster).

## The cluster.stats() function from  the *fpc* R package allows for
## comparing two or more clustering solutions along numerous evaluation 
## measures. The function is agnostic of the clustering method, that is,
## it can be used to compare clustering results obtained using different 
## clustering methods.

# install.packages("fpc")
library(fpc)

# Check the documentation of the cluster.stats() function to examine
# the statistics that it offers for cluster comparison
?cluster.stats
# Select the criteria to be used for the comparison
comparison_criteria <- c("average.between", "average.within", 
                         "max.diameter", "min.separation", 
                         "within.cluster.ss",
                         "clus.avg.silwidths")  

# Compute stats for the Cosine similarity + AP clustering.
# Since the cluster.stats() f. requires distance matrix as its input,
# we need to 'transform' the computed similarities into distances
sci_cos_dist <- 1 - as.matrix(sci_cos_sim, diag = 1)
cos_eval_stats <- clust_compare_stats(dist_matrix = sci_cos_dist, 
                                      exemplars =  sci_data$cos_sim_4cl,
                                      compare_criteria = comparison_criteria)
cos_eval_stats

# Stats for the Correlation-based similarity + AP clustering
sci_cor_dist <- 1 - as.matrix(sci_cor_sim, diag = 1)
cor_eval_stats <- clust_compare_stats(dist_matrix = sci_cor_dist,
                                      exemplars =sci_data$cor_sim_4cl,
                                      compare_criteria = comparison_criteria)
cor_eval_stats

# Stats for the Jaccard similarity + AP clustering
sci_ejaccard_dist <- 1 - as.matrix(sci_ejaccard_sim, diag = 1)
ejacc_eval_stats <- clust_compare_stats(dist_matrix = sci_ejaccard_dist, 
                                        exemplars = sci_data$ejacc_sim_4cl,
                                        compare_criteria = comparison_criteria)
ejacc_eval_stats

# Now that we have computed evaluation stats for each clustering result, we can compare 
# them either on individual measures or a an set of measures. For example, let's compare 
# them using average within and average between measures
data.frame(cbind(avg.within = c(cos_eval_stats$average.within, 
                                cor_eval_stats$average.within, 
                                ejacc_eval_stats$average.within),
                 avg.between = c(cos_eval_stats$average.between, 
                                 cor_eval_stats$average.between,
                                 ejacc_eval_stats$average.between)),
           row.names = c("Cosine", "Pearson", "EJaccard"))
# Based on these two measures, one cannot say what clustering solution is better:
# the first two are better in terms of average within, and the 3rd is better in 
# terms of average between. Other computed measures should be compared, as well.
# Try also other evaluation measures available from the cluster.stats() function.  
