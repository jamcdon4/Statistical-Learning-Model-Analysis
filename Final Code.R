library(caret)
library(mlbench)
library(dplyr)
library(e1071)
library(glmnet)
library(pls)
library(class)
library(tidyverse)
library(tree)

rm(list = ls())


#set the working directory
setwd("D:/Financial Mathematics Semester 3/ST 563 Statistical Learning/")

# Load MODIS data
wine  <- read.csv2("winequality-red.csv")

head(wine)

wine[] <- lapply(wine, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})


# =============== LINEAR REGRESSION ==================
set.seed(5)
train_list = sample(dim(wine)[1], dim(wine)[1]*(4/5))
train_set<-wine[train_list,]
test_set<-wine[-train_list,]

shapiro.test(wine$quality)
# p-value is less than 0.05 hence the test fails and our dependent
# variable is not normal


Model0 <- lm(quality ~ ., data = train_set)
car::vif(Model0)

Model.training <-predict(Model0, train_set) # Apply model to make prediction on Training set
Model.testing <-predict(Model0, test_set) # Apply model to make prediction on Testing set
linear_test_MSE <- mean((Model.testing - test_set$quality)^2)
linear_test_MSE


Model <- lm(quality~.-fixed.acidity-density, data = train_set)
Model.training <-predict(Model, train_set) # Apply model to make prediction on Training set
Model.testing <-predict(Model, test_set) # Apply model to make prediction on Testing set

linear_test_MSE <- mean((Model.testing - test_set$quality)^2)
linear_test_MSE

summary(Model)
par(mfrow = c(1, 2))
plot(train_set$quality,Model.training, col = "blue" )
plot(test_set$quality,Model.testing, col = "blue" )
par(mfrow = c(2, 2))
plot(Model)


# =============== Ridge REGRESSION ==================

X<-model.matrix(quality~.,wine)[,-1] 
Y<-wine$quality


#ridge analysis
grid=10^seq(10,-2, length =100) 
ridge.mod=glmnet (X,Y,alpha=0, lambda=grid)


cv.out_ridge=cv.glmnet(X[train_list ,],Y[ train_list],alpha=0) 
plot(cv.out_ridge)

bestlam_ridge =cv.out_ridge$lambda.min 
bestlam_ridge


ridge.pred=predict(ridge.mod ,s=bestlam_ridge ,newx=X[-train_list,]) 
ridge_test_MSE<-mean((ridge.pred -Y[-train_list])^2)
ridge_test_MSE

n<-length(train_list)
k<-ncol(X)
#set R squre function
rsquare <- function(true, predicted) {
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- (1 - (sse / sst)*(n-1)/(n-k-1))
}

ridge.pred_train=predict(ridge.mod ,s=bestlam_ridge ,newx=X[train_list,]) 

ridge_adj_rsq  <- rsquare(Y[train_list], ridge.pred_train)
ridge_adj_rsq



# =============== Lasso REGRESSION ==================

cv.out_lasso=cv.glmnet(X[train_list ,],Y[ train_list],alpha=1) 
plot(cv.out_lasso)

bestlam_lasso =cv.out_lasso$lambda.min
bestlam_lasso

lasso.pred=predict(ridge.mod ,s=bestlam_lasso ,newx=X[-train_list,]) 
lasso_test_MSE_<-mean((lasso.pred-Y[-train_list])^2) 
lasso_test_MSE_

lasso.pred_train=predict(ridge.mod ,s=bestlam_lasso ,newx=X[train_list,]) 

lasso_adj_rsq  <- rsquare(Y[train_list], lasso.pred_train)
lasso_adj_rsq

# =============== PCR REGRESSION ==================

pcr.fit=pcr(quality~., data=wine, subset=train_list, scale=TRUE,validation ="CV")
validationplot(pcr.fit,val.type="MSEP")


M<-10 
pcr.pred=predict(pcr.fit ,X[-train_list ,],ncomp =M) 
pcr_test_MSE_<-mean((pcr.pred -Y[-train_list])^2)
pcr_test_MSE_


pcr.train_pred=predict(pcr.fit ,X[train_list ,],ncomp =M)
pcr_adj_rsq  <- rsquare(Y[train_list], pcr.train_pred)
pcr_adj_rsq

# =============== PLS REGRESSION ==================

pls.fit=plsr(quality~., data=wine, subset=train_list, scale=TRUE,validation ="CV") 


validationplot(pls.fit,val.type="MSEP")


M=4
pls.pred=predict (pls.fit ,X[-train_list ,],ncomp =M) 
pls_test_MSE<-mean((pls.pred -Y[-train_list])^2)
pls_test_MSE


pls.train_pred=predict(pls.fit ,X[train_list ,],ncomp =M)
pls_adj_rsq  <- rsquare(Y[train_list], pls.train_pred)
pls_adj_rsq


# ===================POLYNOMIAL=================================
Model <- lm(quality~poly(volatile.acidity,2) + 
              poly(citric.acid,2) +
              poly(chlorides,2) + 
              poly(total.sulfur.dioxide,2) + 
              poly(pH,2) + 
              poly(sulphates,2) + 
              alcohol, data = train_set)

car::vif(Model)

Model.training <-predict(Model, train_set) # Apply model to make prediction on Training set
Model.testing <-predict(Model, test_set) # Apply model to make prediction on Testing set

poly_test_MSE <- mean((Model.testing - test_set$quality)^2)
poly_test_MSE

summary(Model)
par(mfrow = c(1, 2))
plot(train_set$quality,Model.training, col = "blue" )
plot(test_set$quality,Model.testing, col = "blue" )
par(mfrow = c(2, 2))
plot(Model)




# ======================== LOGISTIC REGRESSION =======================


wine$rating <-ifelse(wine$quality > 5 , 'high','low')
wine$rating <- factor(wine$rating)

wine <- wine[ , !(names(wine) %in% c("quality"))]

set.seed(5)
train_list = sample(dim(wine)[1], dim(wine)[1]*(4/5))
train_set<-wine[train_list,]
test_set<-wine[-train_list,]

# We see 53.47% accuracy as baseline 
prop.table(table(wine$rating))


Model <- train(rating ~ ., data = train_set, method = 'glm', family = "binomial")

# train error
1 - Model$results$Accuracy

# test error
Model.testing <- predict(Model, newdata = test_set)
mean(Model.testing != test_set$rating)
summary(Model)

confusionMatrix(data = Model.testing, reference = test_set$rating, positive = "1")


# ======================== KNN =======================


knn<-knn(train=train_set, test=test_set, cl=train_set$rating, k=30)
confusionMatrix(knn, test_set$rating)
mean(knn==test_set$rating)
table(knn, test_set$rating)

# ======================== Decision Tree =======================

wine.tree <- tree(rating~., data = train_set)
tree.pred <- predict(wine.tree, test_set,type ="class")

#confusion table
with(test_set, table(tree.pred, rating))
summary(wine.tree)
plot(wine.tree)
