## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of applying the TextRank algorithm, 
## using the *textrank* R package, to extract keywords (and key phrases) 
## from science-related newsgroups.
##
## The TextRank algorithm was originally proposed by Mihalcea & Tarau, in:
## Mihalcea, R. & Tarau, P. (2004). TextRank: Bringing order into texts. 
## In D. Lin & D. Wu (Eds.), Proc. of Empirical Methods in Natural Language 
## Processing (EMNLP) 2004 (pp. 404–411), Barcelona, Spain, July. 
## URL: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
##
## The textrank R package is available at:
## https://cran.r-project.org/web/packages/textrank/vignettes/textrank.html
##
## The dataset used in the example is derived from the publicly available
## 20 Newsgroups dataset, http://qwone.com/~jason/20Newsgroups/ 

# Load the initial set of required packages
# (more will be added along the way)
library(tidyr)
library(dplyr)
library(quanteda)
library(stringr)

# Read in a (pre-processed) subset of the 20 Newsgroups dataset
# (for the preprocessing details, see the TM_Intro_Newsgroup_Classifier R script)
usenet_data <- read.csv(file = "data/20news-bydate/20news-bydate-train.csv",
                        stringsAsFactors = FALSE)

# Examine the available newsgroups
unique(usenet_data$newsgroup)

# Select only messages related to science-focused groups
sci_data <- usenet_data %>%
  filter(str_detect(newsgroup, "^sci\\."))
# Remove usenet_data as no longer needed
remove(usenet_data)

# Keywords extraction using the TextRank algorithm requires: 
# - Part-of-Speach (POS) tagged words, and 
# - word lemmas 
# To do both things, we will use the *udpipe* R package
# https://github.com/bnosac/udpipe
# (quanteda package does not support POS tagging)
# install.packages('udpipe')
library(udpipe)

# Load the appropriate language model (the one for English language)
# Note: the udpipe package has language models for over 50 languages
tagger <- udpipe_download_model("english", model_dir = "models/textrank")
tagger <- udpipe_load_model(file = "models/textrank/english-ud-2.0-170801.udpipe")
# Annotate the text of the posts using the loaded model (tagger).
# This will produce several linguistic annotations for each word, 
# including the appropriate POS tags and lemmas
sci_posts_annotated <- udpipe_annotate(tagger, sci_data$post_txt)
# To be able to use the udpipe object easily, we'll transform it 
# into a data frame
sci_posts_ann_df <- as.data.frame(sci_posts_annotated)
# and remove the large udpipe object, to release memory
remove(sci_posts_annotated)

# Store the data frame with annotations since the annotation taks 
# takes a lot of time to complete
# saveRDS(sci_posts_ann_df, file = "models/textrank/sci_posts_ann_df.RData")

# Load the saved annotations
# sci_posts_ann_df <- readRDS(file = "models/textrank/sci_posts_ann_df.RData")

# Let's quickly inspect the results of the annotation process
str(sci_posts_ann_df)
View(sci_posts_ann_df[1:20, ])
# The meaning of all the variables is explained at:
# http://universaldependencies.org/format.html

# Next, we'll examine a sample of lemmas to check what they look like 
# and if some 'cleaning' is required
unique(sci_posts_ann_df$lemma)[1:100]
unique(sci_posts_ann_df$lemma)[2000:2100]
# Note the presence of stopwords ('the', 'and',...), as well as 
# single-char lemmas, numbers, and punctuation marks.

# To clean up the data, we will remove stopwords, numbers, punctuation, ...
sci_posts_ann_filtered <- sci_posts_ann_df %>%
  filter(!lemma %in% stopwords()) %>%
  filter(nchar(lemma) > 1) %>%
  filter(str_detect(lemma, "^[a-z0-9'-]+$")) %>%  # allow only letters, digits, single quotes, and hyphens in lemmas 
  filter(!str_detect(lemma, "^(-?)\\d+$")) %>%    # do not allow digits optionally preceded by a hyphen 
  filter(!str_detect(lemma, "^'[a-z]{1,2}$"))     # do not allow lemmas consisting of a single quote followed by up to 2 letters (eg. 's, 'll)  

# Check the results of the 'cleaning' steps
unique(sci_posts_ann_filtered$lemma)[1:100]
unique(sci_posts_ann_filtered$lemma)[2000:2100]

# Note: the above cleaning steps were not really necessary since we will later 
# filter lemmas based on their POS tags (selecting only nouns and adjectives),
# but it was a good opportunity to practice text cleaning :-)

# Now that the text is preprocessed, we can proceed to keywords extraction
# using the textrank package
# install.packages('textrank')
library(textrank)

# Check the function for the extraction of keywords
?textrank_keywords

# To extract keywords, we need to provide:
# - a vector of words (lemmas)
# - a vector of logicals indicating which words are potentially relevant, that is, 
#   which words are to be used to build a word collocation graph; we will select 
#   nouns and adjectives
# - the proportion or the number of words to keep
# - the longest n-gram to consider when forming keyphrases

# To determine the proportion of words to keep, we will
# check the number of nouns and adjectives (as relevant word types)
rel_word_types <- c("NOUN", "ADJ")
sci_posts_ann_filtered %>%
  filter(upos %in% rel_word_types) %>%
  select(lemma) %>%
  n_distinct()
# ~12K nouns and adjectives
# Considering this large number of potentially relevant words,
# we will tell the function to keep just 2.5% (~300) as relevant

# Run the textrank algorithm
sci_keywords <- 
  textrank::textrank_keywords(x = sci_posts_ann_filtered$lemma,
                              relevant = sci_posts_ann_filtered$upos %in% rel_word_types,
                              p = 0.025,  
                              ngram_max = 3,
                              sep = "_")
# Examine the structure of the result
str(sci_keywords)
# The 1st component of the result list is the 'terms' vector with top ranked words
# based on the computed PageRank score: 
head(sci_keywords$terms, n=10)
# The disadvantage is that we do not have PageRank score associated with these words

# To get PageRank score, we should take the 2nd component of the result (pagerank):
head(sci_keywords$pagerank$vector, n=10)
# Note that this vector contains PageRank score for all the considered words, not 
# only for the proportion we specified to be kept (0.025 in this case):
length(sci_keywords$pagerank$vector)
# Sort the keywords, based on the PageRank
sci_keyw_pagerank <- sort(sci_keywords$pagerank$vector, decreasing = TRUE)
head(sci_keyw_pagerank, n = 25)
# Overall the keywords look fine, though seem to be overly general. This is, in fact, 
# expected considering the broad range of topics dicussed in the science-related newsgroups. 
# Hence, it might be better to focus on individual newsgroups instead.


# To get keywords of individual science newsgroups and then compare them, 
# we need to call the textrank_keywords() f. for each newsgroup separatelly.

# First, since the data frame with annotated documents (sci_posts_ann_filtered) 
# does not have a connection between the documents (posts) and newsgroups, we need 
# to establish that connection. We'll do that in 2 steps:
# 1) add to sci_data df document id in the form used in the sci_posts_ann_filtered df  
sci_data$doc_id <- paste0("doc", 1:nrow(sci_data))
# 2) merge sci_data and sci_posts_ann_filtered data frames
sci_posts_ann_filtered <- merge(x = sci_posts_ann_filtered,
                                y = sci_data %>% select(doc_id, newsgroup),
                                by = "doc_id", 
                                all.x = TRUE, all.y = FALSE) 

# Next, we'll create a function that for the given set of word lemmas, POS tags,
# and proportion of words to keep as keywords, applies the TextRank algorithm and 
# returns the results (lemmas w/ their PageRank scores and freqs) as a data frame.
# It will be used for extracting keywords for each newsgroup individually
tr_keywords <- function(word_lemmas, pos_tags, prop_to_keep) {
  keywords <- textrank::textrank_keywords(x = word_lemmas,
                                          relevant = pos_tags %in% rel_word_types,
                                          p = prop_to_keep,
                                          ngram_max = 3,
                                          sep = "_")
  keyw_pagerank <- sort(keywords$pagerank$vector, decreasing = TRUE)
  keyw_pagerank_df <- data.frame(lemma = names(keyw_pagerank),
                                 pagerank = as.numeric(keyw_pagerank),
                                 stringsAsFactors = FALSE) 
  # keep only those words that are selected by the algorithm as relevant
  keyw_pagerank_df <- keyw_pagerank_df %>% filter(lemma %in% keywords$terms)
  # add frequencies
  keyw_pagerank_df <- merge(x = keyw_pagerank_df, 
                            y = keywords$keywords_by_ngram %>% 
                                filter(ngram==1) %>% select(keyword, freq),
                            by.x = "lemma", by.y = "keyword", all.x = TRUE, all.y = FALSE)
  keyw_pagerank_df
}

# Before calling our tr_keywords() function on each newsgroup, we have to decide
# on how many words to keep as keywords. To do that, we'll check the number of 
# candidate keywords, that is, nouns and adjectives per newsgroup
sci_posts_ann_filtered %>% 
  filter(upos %in% rel_word_types) %>%
  group_by(newsgroup) %>% 
  summarise(rel_word_tot = n_distinct(lemma))
# All newsgroups except one have well above 4.5K words (ranges from 3.99K to 5.44K).
# So, it might be good to keep 5% of the top ranked words in each group.

# For each science-related newsgroup, call the tr_keywords() f. to execute the TextRank
# algorithm and combine the output (df with words and their TextRank scores and frequencies) 
# into one df (newsgroup_keywords_df)
newsgroup_keywords_df <- data.frame()
newsgroups <- unique(sci_posts_ann_filtered$newsgroup)
for(ngroup in newsgroups) {
  ngroup_posts <- sci_posts_ann_filtered %>% filter(newsgroup == ngroup) 
  tr_keyw_df <- tr_keywords(word_lemmas = ngroup_posts$lemma, 
                            pos_tags = ngroup_posts$upos, 
                            prop_to_keep = 0.05) 
  tr_keyw_df$newsgroup <- rep(ngroup, times = nrow(tr_keyw_df))
  newsgroup_keywords_df <- rbind(tr_keyw_df, newsgroup_keywords_df)
}
str(newsgroup_keywords_df)
newsgroup_keywords_df$newsgroup <- as.factor(newsgroup_keywords_df$newsgroup)

# To visually inspect the results, plot for each newsgroup 
# the top 15 words (based on the TextRank score) 
library(ggplot2)
newsgroup_keywords_df %>% 
  group_by(newsgroup) %>%
  top_n(n = 15, wt = pagerank) %>%
  ungroup() %>%
  mutate(lemma = reorder(lemma, pagerank)) %>%
  ggplot(mapping = aes(x = lemma, y = pagerank, fill = newsgroup)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~newsgroup, scales = "free") + 
  coord_flip() +
  theme_bw()
  
# Let's see how we can use wordclouds as another way to visually inspect
# keywords from a newsgroup  
# install.packages('wordcloud')
library(wordcloud)

# For example, we can generate a word cloud for the sci.cript group
# Check the number of extracted keywords
newsgroup_keywords_df %>% filter(newsgroup=="sci.crypt") %>% n_distinct()
# 221
# Considering the number of keywords, we cannot present them all;
# instead, we'll select those with above median TextRank
crypt_selection <- newsgroup_keywords_df %>%
  filter(newsgroup == "sci.crypt") %>%
  filter(pagerank > median(pagerank))
# Since word clouds typically work with word frequencies (counts),
# to use them to visualise words based on their TextRank scores (very small floats),
# we need to rescale those scores
set.seed(2018)
wordcloud(words = crypt_selection$lemma, 
          freq = round(crypt_selection$pagerank * 10000), # instead of freq, we are using rescalled pagerank score
          random.order=FALSE,                             # plot words in decreasing freq order
          rot.per=0.35,                                   # prop. of words plotted vertically
          colors=c('#d4b9da','#c994c7','#df65b0','#dd1c77','#980043')) # color pallet

# In the same way word clouds can be created for the other 3 newsgroups.

## To learn more about different options to customise word clouds, and also
## some specific kinds of word clouds (e.g. comparison cloud), check this RPub:
## https://rpubs.com/brandonkopp/creating-word-clouds-in-r
## For the selection of a color pallet, consider: http://colorbrewer2.org/


# So far, we have only examined keywords (unigrams), but not key phrases
# (bigrams, trigrams). Let's call again the textrank_keywords() f. and check 
# bigrams and trigrams, as well. As an example, we'll take the sci.med group
med_posts_ann <- sci_posts_ann_filtered %>% filter(newsgroup=="sci.med")
med_keywords <- 
  textrank::textrank_keywords(x = med_posts_ann$lemma,
                              relevant = med_posts_ann$upos %in% rel_word_types,
                              p = 0.05,
                              ngram_max = 3,
                              sep = "_")

str(med_keywords)
# The keywords_by_ngram component of the result gives us the selected (5%) 
# top ranked words (unigrams), as well as, bigrams and trigrams from the corpus 
# that are formed by the selected words (unigrams).

# Take only bigrams and trigrams
med_phrases <- med_keywords$keywords_by_ngram %>% filter(ngram > 1)

# Examine the most frequent bigrams and trigrams (built out of the selected unigrams) 
# Top 15 sci.med bigrams
med_phrases %>% filter(ngram == 2 ) %>%
  top_n(n=15)
# Note that even though we asked for top 15, more bigrams are presented
# due to the presence of ties

# Top 15 sci.med trigrams
med_phrases %>% filter(ngram == 3) %>%
  top_n(n=15)
# Bigrams seem to be far more useful as key phrases than trigrams
