##### Sample code for text-mining lecture #####
##### Feng Mai 2015/8 #######

## Building a predictive model for movie review sentiment##
# Data source: http://www.cs.cornell.edu/people/pabo/movie-review-data/

install.packages(c('tm', 'SnowballC', 'wordcloud', 'topicmodels'))
library(tm)
library(SnowballC)
library(wordcloud)

reviews = read.csv("movie_reviews.csv", stringsAsFactors = F, row.names = 1)
# A collection of text documents is called a Corpus
review_corpus = Corpus(VectorSource(reviews$content))

# Change to lower case, not necessary here
review_corpus = tm_map(review_corpus, content_transformer(tolower))
# Remove numbers
review_corpus = tm_map(review_corpus, removeNumbers)
# Remove punctuation marks and stopwords
review_corpus = tm_map(review_corpus, removePunctuation)
review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
# Remove extra whitespaces
review_corpus =  tm_map(review_corpus, stripWhitespace)

# Sometimes stemming is necessary
# review_corpus =  tm_map(review_corpus, stemDocument)

inspect(review_corpus[1])

# Document-Term Matrix: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Notice the dimension of the matrix
review_dtm <- DocumentTermMatrix(review_corpus)
review_dtm
inspect(review_dtm[500:505, 500:505])

# Simple word cloud
findFreqTerms(review_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(review_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

# Remove the less frequent terms such that the sparsity is less than 0.95
review_dtm = removeSparseTerms(review_dtm, 0.99)
review_dtm
# The first document
inspect(review_dtm[1,1:20])

# tf–idf(term frequency–inverse document frequency) instead of the frequencies of the term as entries, tf-idf measures the relative importance of a word to a document
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf
# The first document
inspect(review_dtm_tfidf[1,1:20])

# A new word cloud
freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

# Precompiled list of words with positive and negative meanings
# Source: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
neg_words = read.table("http://homepages.uc.edu/~maifg/DataMining/tm/negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("http://homepages.uc.edu/~maifg/DataMining/tm/positive-words.txt", header = F, stringsAsFactors = F)[, 1]

# neg, pos contain the number of positive and negative words in each document
reviews$neg = sapply(review_corpus, tm_term_score, neg_words)
reviews$pos = sapply(review_corpus, tm_term_score, pos_words)
# remove the actual texual content for statistical models
reviews$content = NULL
# construct the dataset for models
reviews = cbind(reviews, as.matrix(review_dtm_tfidf))
reviews$polarity = as.factor(reviews$polarity)

# Split to testing and training set
id_train <- sample(nrow(reviews),nrow(reviews)*0.80)
reviews.train = reviews[id_train,]
reviews.test = reviews[-id_train,]

# Now you know how to do the rest
library(rpart)
library(rpart.plot)
install.packages("rpart.plot")
library(e1071) # for Support Vector Machine
library(nnet)


reviews.tree = rpart(polarity~.,  method = "class", data = reviews.train);  
prp(reviews.tree)
reviews.glm = glm(polarity~ ., family = "binomial", data =reviews.train, maxit = 100);  
reviews.svm = svm(polarity~., data = reviews.train);
reviews.nnet = nnet(polarity~., data=reviews.train, size=1, maxit=500)


pred.tree = predict(reviews.tree, reviews.test,  type="class")
table(reviews.test$polarity,pred.tree,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$polarity != pred.tree, 1, 0))


pred.glm = as.numeric(predict(reviews.glm, reviews.test, type="response") > 0.5)
table(reviews.test$polarity,pred.glm,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$polarity != pred.glm, 1, 0))

pred.svm = predict(reviews.svm, reviews.test)
table(reviews.test$polarity,pred.svm,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$polarity != pred.svm, 1, 0))


prob.nnet= predict(reviews.nnet,reviews.test)
pred.nnet = as.numeric(prob.nnet > 0.5)
table(reviews.test$polarity, pred.nnet, dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$polarity != pred.nnet, 1, 0))




### Topic modeling example ###
library("topicmodels")
#Associated Press data from the First Text Retrieval Conference (TREC-1) 1992.
data("AssociatedPress", package = "topicmodels")
# Fitting a 10-topic model with variational EM
topics <- LDA(AssociatedPress[1:200,], k = 5)
# Print the representative terms for each topic
terms(topics, 10)

