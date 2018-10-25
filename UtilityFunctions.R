## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The function reads all files from the given folder (infolder) 
## into a data frame and returns the created data frame
read_folder <- function(infolder) {
  data_frame(file = dir(infolder, full.names = TRUE)) %>%
    mutate(text = map(file, read_lines)) %>%
    transmute(id = basename(file), text) %>%
    unnest(text)  # text is a list-column; unnest transforms each element of the list into a row
}

## For each post in the input data frame, the function removes the header and the automated 
## signature. It returns a data frame of the same structure, but with cleaned post text.
## Note: the 1st cumsum() removes all lines before an 'empty' line;
## since, in most of the posts, an empty line delimits the header from the rest of 
## the post content, this way, the header will be removed.    
## The 2nd cumsum() keeps all the lines until the line starting with one or more hyphens ('-')
## This way, the part of the post with automatic signature is removed.
remove_header_and_signature <- function(newsgroup_df) {
  newsgroup_df %>%
    group_by(newsgroup, id) %>%  # each group contains lines of text from one post
    filter(cumsum(text == "") > 0, 
           cumsum(str_detect(text, "^-+")) == 0) %>%
    ungroup()
}

## The function does further cleaning of the post content. In particular, 
## it removes lines that:
## 1) start with 1 or more ">" or "}" characters (1st filter statement)
## 2) end with "writes:" or "writes..." (2nd filter statement)
## 3) start with "In article <" (3rd filter statement)
## 4) are empty (4th filter statement)
remove_quoted_text <- function(newsgroup_df) {
  newsgroup_df %>%
    filter(str_detect(text, "^[^[>|\\}]]+[A-Za-z\\d]"), 
           !str_detect(text, "writes(:|\\.\\.\\.)$"), 
           !str_detect(text, "^In article <"), 
           text != "") 
}
  

## The function transforms the input data frame by merging all pieces of text 
## that belong to one post and associating the merged content with the post id
# Note: since posts with the same id may appear in different newsgroups, to have
# a unique post identifier (post_id), we need to combine 'newsgroup' and 'id' 
# variables. After the post content is merged, we split the post_id back to 
# 'newsgroup' and 'id' variables (the last line)
merge_post_text <- function(newsgroup_df) {
  newsgroup_df %>%
    transmute(post_id = str_c(newsgroup, id, sep = "_"), text) %>%
    group_by(post_id) %>%
    summarise(post_txt = str_c(text, collapse = " ")) %>%
    ungroup() %>%
    separate(col = post_id, into = c("newsgroup", "id"), sep = "_") 
}

## The function uses boxplots to plot the distribution for the
## given feature; a separate boxplot is drawn for each newsgroup
plot_ng_feature_comparison <- function(df, feature, f_name) {
  require(ggplot2)
  ggplot(mapping = aes(x = df[['newsgroup']], 
                       y = df[[feature]], 
                       fill = df[['newsgroup']])) +
    geom_boxplot() +
    labs(x = "Newsgroups", y = f_name) +
    scale_fill_discrete(name="Newsgroups") +
    theme_bw()
}


## Function for creating a feature data frame out of
## - a DTM, represented in the form of quanteda's dfm, and 
## - a vector of class labels
create_feature_df <- function(train_dfm, class_labels) {
  
  train_df <- convert(train_dfm, "data.frame")
  # The 'convert' f. from quanteda adds 'document' as the 1st feature (column)
  # in the resulting data frame. It needs to be removed before the data frame 
  # is used for training.
  if ((names(train_df)[1] == 'document') & (class(train_df[,1])=='character'))
    train_df <- train_df[, -1]
  
  # Check if there are documents that have 'lost' all their words, that is,
  # if there are rows with all zeros
  doc_word_cnt <- rowSums(train_df)
  zero_word_docs <- which(doc_word_cnt == 0)
  # If there are zero-word rows, remove them
  if (length(zero_word_docs) > 0) {
    print(paste("Number of documents to remove due to sparsity:", length(zero_word_docs)))
    train_df <- train_df[-zero_word_docs,]
    class_labels <- class_labels[-zero_word_docs]
  }
  
  # Assure that column names are regular R names
  require(janitor)
  train_df <- clean_names(train_df)
  
  # Combine class labels and the features 
  cbind(Label = class_labels, train_df)
  
}


## Function for performing 5-fold cross validation on the given training data set
## (train.data) using the specified ML algorithm (ml.method). 
## Cross-validation is done in parallel on the specified number (nclust) of logical cores.
## The grid.spec serves for passing the grid of values to be used in tuning the parameter
## of the ML method; it is assumed that only one parameter will be tuned.
## Accuracy is used as the default evaluation metric for the selection of the best model.
## The ntree parameter can be used to set the number of trees in case Random Forest is used.
cross_validate_classifier <- function(seed,
                                      nclust, 
                                      train.data, 
                                      ml.method,
                                      grid.spec,
                                      ntree = 1000,
                                      metric = "Accuracy") { 
  require(caret)
  require(doSNOW)
  
  # Setup the CV parameters
  cv.cntrl <- trainControl(method = "cv", 
                           number = 5, 
                           search = "grid")
  
  # Create a cluster to work on nclust logical cores;
  # what it means (simplified): create nclust instances of RStudio and 
  # let caret use them for the processing 
  cl <- makeCluster(nclust, 
                    type = "SOCK") # SOCK stands for socket cluster
  registerDoSNOW(cl)
  
  # Track the time of the code execution
  start.time <- Sys.time()
  
  set.seed(seed)
  if (ml.method=="rpart")
    model.cv <- train(x = train.data[,names(train.data) != 'Label'],
                      y = train.data$Label,
                      method = 'rpart', trControl = cv.cntrl, 
                      tuneGrid = grid.spec, metric = metric)
  if (ml.method=="rf")
    model.cv <- train(x = train.data[,names(train.data) != 'Label'],
                      y = train.data$Label, 
                      method = 'rf', trControl = cv.cntrl, 
                      tuneGrid = grid.spec, metric = metric,
                      ntree = ntree)
  
  # Processing is done, stop the cluster
  stopCluster(cl)
  
  # Compute and print the total time of execution
  total.time <- Sys.time() - start.time
  print(paste("Total processing time:", total.time))
  
  # Return the built model
  model.cv
  
}

## Function for calculating relative (normalized) term frequency (TF)
relative_term_frequency <- function(row) { # in DTM, each row corresponds to one document 
  row / sum(row)
}

## Function for calculating inverse document frequency (IDF)
## Formula: log(corpus.size/doc.with.term.count)
inverse_doc_freq <- function(col) { # in DTM, each column corresponds to one term (feature) 
  corpus.size <- length(col) # the length of a column is in fact the number of rows (documents) in DTM
  doc.count <- length(which(col > 0)) # number of documents that contain the term
  log10(corpus.size / doc.count)
}

## Function for calculating TF-IDF
tf_idf <- function(x, idf) {
  x * idf
}


## The function extracts some basic evaluation metrics from the model evaluation object
## produced by the confusionMatrix() f. of the caret package
get_eval_measures <- function(model_eval) {
  metrics <- c("Precision", "Recall", "F1", "Accuracy", "Kappa")
  eval_measures <- model_eval$byClass[metrics[1:3]]
  eval_measures <- c(eval_measures, model_eval$overall[metrics[4:5]])
  eval_measures
}


## The function creates a data frame out of the word vectors 
## that originate from a pre-trained GloVe model (m_glove)
get_word_vectors_df <- function(m_glove, verbose = FALSE) {
  
  # initialize space for values and the names of each word in the model
  n_words <- length(m_glove)
  vals <- list()
  names <- character(n_words)
  
  # loop through to gather values and names of each word
  for(i in 1:n_words) {
    if (verbose) {
      if(i %% 5000 == 0) {print(i)}
    }
    this_vec <- m_glove[i]
    this_vec_unlisted <- unlist(strsplit(this_vec, " "))
    this_vec_values <- as.numeric(this_vec_unlisted[-1])  
    this_vec_name <- this_vec_unlisted[1]
    
    vals[[i]] <- this_vec_values
    names[i] <- this_vec_name
  }
  
  # convert the list to a data frame and attach the names
  glove_df <- data.frame(vals)
  names(glove_df) <- names
  
  glove_df
}

## The function computes harmonic mean for the given input vector
harmonicMean <- function(values, precision=2000L) {
  require("Rmpfr")
  valMed <- median(values)
  as.double(valMed - log(mean(exp(-mpfr(values, prec = precision) + valMed))))
}


