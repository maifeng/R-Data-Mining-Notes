library(e1071)
spam <- read.csv("spam.csv")

myvars<-names(spam)%in% c("isuid","id","spampct")
spam<-spam[!myvars]



Etrain<-vector(mode="numeric")
Etest<-vector(mode="numeric")

for (i in 1:50){
  
  train.ind <- sample(1:nrow(spam), ceiling(nrow(spam)*0.5), replace=FALSE)
  
  model <- naiveBayes(spam ~ ., data=spam[train.ind,],laplace=1)
  
  pred<- predict(model, spam[train.ind,])
  confusion.mat <- table(pred, spam[train.ind,"spam"])
  Etrain[i]<-1-(sum(diag(confusion.mat))/sum(confusion.mat))
  
  
  pred<- predict(model, spam[-train.ind,])
  confusion.mat <- table(pred, spam[-train.ind,"spam"])
  Etest[i]<-1-(sum(diag(confusion.mat))/sum(confusion.mat))
}

boxplot(Etrain, Etest, names = c('train', 'test'))
predict(model, spam[-train.ind,],type = 'raw')



searchgrid = seq(0.9, 0.99, 0.0001)
result = cbind(searchgrid, NA)

cost1<-function(actual,prob){
  weight1=1
  weight0=2
  c1=(actual=="yes")&(prob<cutoff)
  c0=(actual=="no")&(prob>cutoff)
  return (mean(weight1*c1+weight0*c0))
}

model <- naiveBayes(spam ~ ., data=spam[train.ind,],laplace=1)
predopt<-predict(model,spam[train.ind,],type="raw")[,2]
for (i in 1:length(searchgrid)){
  cutoff<-result[i,1]
  result[i,2]<-cost1(spam[train.ind,]$spam,predopt)
}
plot(result)

