## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The function computes the purity measure for a clustering solution
## based on the matrix of clusters (in rows) vs human-labelled categories (in columns).
## The purity measure estimates the coherence of a cluster, that is, the degree to which 
## a cluster contains items (documents) from a single category.
clust_purity <- function(clust_newsgr_matrix) {
  purity <- apply(clust_newsgr_matrix, 1, function(x) max(x)/sum(x))
  names(purity) <- paste0("cluster_", 1:4)
  purity
}


## The function computes the entropy measure for a clustering solution
## based on the matrix of clusters (in rows) vs human-labelled categories (in columns).
## The entropy measure evaluates the distribution of categories in a given cluster.
clust_entropy <- function(clust_newsgr_matrix) {
  entropy <- apply(clust_newsgr_matrix, 1, function(x){
    clust_tot <- sum(x)
    sum((x/clust_tot) * log10(x/clust_tot))
  })
  entropy <- (-1/log10(ncol(clust_newsgr_matrix))) * entropy
  names(entropy) <- paste0("cluster_", 1:4)
  entropy
}


## The function computes averaged entropy of the overall clustering solution
## as the weighted sum of the entropy value of individual clusters. 
## Weight is computed as: num_of_samples_in_cluster / tot_num_of_samples
## The input is a matrix of clusters (in rows) vs human-labelled categories 
## (in columns).
avg_clust_entropy <- function(clust_newsgr_matrix) {
  cl_entropy <- as.numeric(clust_entropy(clust_newsgr_matrix))
  n <- sum(clust_newsgr_matrix)
  cl_count <- as.integer(rowSums(clust_newsgr_matrix))
  avg_ent <- 0
  for(i in 1:nrow(clust_newsgr_matrix))
    avg_ent <- avg_ent + (cl_count[i]/n) * cl_entropy[i]
  avg_ent
}


## The function computes a selection of cluster comparison statistics based on the
## following input parameters:
## - the distance matrix
## - a vector of cluster assignments represented with indices of cluster exemplars
## - a vector of statistics to compute, named as defined in the cluster.stats() f.
## 
## Note that the 'clustering' parameter of the cluster.stats() f. has to be given as 
## an integer vector with values from 1 to the number of clusters; in our case, 
## cluster assignments are stored as indices of exemplars; hence, we need to 're-label' 
## them with integer values
clust_compare_stats <- function(dist_matrix, exemplars, compare_criteria) {
  n_clust <- length(unique(exemplars))
  clust_assignments <- factor(x = exemplars, levels = unique(exemplars), labels = 1:n_clust)
  clust_stats <- cluster.stats(d = dist_matrix, 
                              clustering = as.integer(clust_assignments))
  sapply(compare_criteria, function(x) clust_stats[[x]])
}

