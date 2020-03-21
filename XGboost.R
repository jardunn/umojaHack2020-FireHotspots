# XG Boost setup

library(xgboost)
library(dplyr)


# Load the Data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train = read.csv("train.csv")
test  = read.csv("test.csv")

# Date variables
train$date <- as.Date(train$date)
train$month <- as.numeric(format(train$date, "%m"))
train$year  <- as.numeric(format(train$date, "%Y"))

# Split data for validation

train_all <- train
trainmodel <- train_all[train_all$date <= '2011-01-01',]
validmodel <- train_all[train_all$date > '2011-01-01',]

# Define input and output columns
in_cols <- names(train[,-c(1:7,31,33)]) #-c(1:7)
target_col <- "burn_area"
in_cols

# Get our X and y training and validation sets ready
X_train <- trainmodel[,c(target_col,in_cols)]

X_valid <- validmodel[,in_cols]
Y_valid <- validmodel[,target_col]

bst <- xgboost(data = data.matrix(X_train),
               label = Y_train,
               nrounds=40)

xg_pred <- predict(bst, data.matrix(X_train))
xg_pred <- data.frame(xg_pred)
# Score
sqrt( mean( (preds$preds-Y_valid)^2 , na.rm = TRUE ) ) # RMSE lower is better
