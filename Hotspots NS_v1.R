## HotSpots Starter R Script
# This script should help you get started with the Hotspots data. In it we cover:
  
# Loading the data
# Simple EDA and an example of feature enginnering
# Suggestions for validation split
# Creating a simple model
# Making a submission
# Some tips for improving your score

# Install Packages
library(Metrics)
library(h2o)
library(dplyr)
library(ggplot2)
library(readr)
library(ggcorrplot)
library(dplyr)
library(tidyverse)
library(glmnet)
library(mltools)
library(data.table)

# Load the Data

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train_orig = read.csv("train.csv")
test  = read.csv("test.csv")

# Making an extra copy of the data to limit the amount of times it needs to be loaded
train <- train_orig

str(train)
head(test)
#summary(train)
view(train[1:10,])

#############################################
##EDA + Feature Engineering

# Change the NA values in the population density column to 0
train$population_density[is.na(train$population_density)] = 0

# Date variables
train$date <- as.Date(train$date)
train$month <- as.numeric(format(train$date, "%m"))
train$year  <- as.numeric(format(train$date, "%Y"))

# Turn month in categorical
month_var = data.table(as.factor(train$month))
month_cat = one_hot(month_var)
head(month_cat)
names(month_cat) <- c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
dim(month_cat)

# Replace the current month variable with the categorical columns
train <- cbind(train, month_cat)
train <- train %>% select(-c(month))

# Add hot, dry, window category [0,1] when all three variables are above the 3rd quartile
train <- train %>% 
  mutate(hot_dry_windy = ifelse(climate_vpd > 101 & climate_vs > 170 & climate_tmmx > 310, 1, 0))

# Add urban and populated categorical [0,1]

# Month-lagged PDSI variable
# Select the PDSI variable and move it forward by a month; the downside here is that the first
# entry for every area no longer has a data point
PDSI_orig <- train %>% arrange(area) %>% select(climate_pdsi)

n_sites <- unique(train$area)

no_entries = dim(PDSI_orig)[1]
PDSI_new = numeric()
PDSI_new[1] = 0
PDSI_new[2:no_entries] = PDSI_orig[1:no_entries-1,]

for (i in 1:length(n_sites)){
  PDSI_new[1+((i-1)*164)] = 0
}
# Check -- commented out
#a <- data.frame(cbind(PDSI_new, train %>% arrange(area) %>% select(climate_pdsi)))
#a[164:166,]

# Add the data in
train <- train %>% arrange(area)
train <- train %>% mutate(lagged_PDSI = PDSI_new)


# The first data entry for every area has this set to "0" because obviously there isn't a lagged
# value for the first month; unsure of whether to remove this or not
train <- train %>% filter(PDSI_lagged != 0)

# Remove variables that are unneccesary for prediction 
train <- train %>% select(-c("ID", "date", "area"))

view(train[1:10,])
#############################################



## Adding more features - some ideas
# Read the list of climate variables and what they mean. See if you can combine them in interesting ways - perhaps a 'hot_and_dry' metric...
# Fire depends on some processes that take a long time - for example, there may be more fuel if the previous growing season was a good one. Consider some lagged variables to give the model some inputs for what came before the current month.
# Make some categorical features - 'dominant_land_type' or 'is_peak_rainfall'.


#############################################
####Data Split for Validation

# We don't want to just split randomly. Instead, let's use the last 3 years of the dataset for validation to more closely match the test configuration.
train_all <- train
trainmodel <- train_all[train_all$date <= '2011-01-01',]
validmodel <- train_all[train_all$date > '2011-01-01',]

dim(trainmodel)
dim(validmodel)

#############################################
######Simple Model
# Define input and output columns
in_cols <- names(train[,-c(1:7)])
target_col <- "burn_area"
in_cols

# Get our X and y training and validation sets ready
X_train <- trainmodel[,c(target_col,in_cols)]

X_valid <- validmodel[,in_cols]
Y_valid <- validmodel[,target_col]

# Create and fit the model
model <- lm(burn_area ~ ., X_train)

# Make predictions
preds <- predict(model, X_valid)
preds <- as.data.frame(preds)

# Score
sqrt( mean( (preds$preds-Y_valid)^2 , na.rm = TRUE ) ) # RMSE lower is better

# Exercise. Try a RandomForestRegressor model. Use n_estimators=10 if the default takes too long to run, and experiment with the max_depth parameter.
# With some tweaking, you should be able to get scores ~0.042 or lower.

train <- cbind(train, month_cat)
train <- train %>% select(-c(month))
# The first data entry for every area has this set to "0" because obviously there isn't a lagged
# value for the first month; unsure of whether to remove this or not
train <- filter(PDSI_lagged != 0)

train_all <- train
trainmodel <- train_all[train_all$date <= '2011-01-01',]

################################################################################
                     ## -- FEATURE ENGINEERING ENDS HERE -- ##
################################################################################
### Doing the ridge regression
x_var <- as.matrix(trainmodel %>% select(-c("area","burn_area")))
y_var <- as.matrix(trainmodel %>% select(( "burn_area")))
lambda_seq <- 10^seq(3,-3, by = -0.1)
fit <- glmnet(x_var, y_var, alpha = 0, lambda  = lambda_seq)
str(as.double(x_var))

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda

head(train)


##############################################################################
######Making a Submission
# Once you've got some features and a model you're happy with, it's time to submit!
# Look at the sample submission file
ss = read.csv('hot_spots_drc_ss.csv')
head(ss)

# And the test data
head(test)

# So we need to predict the burn area for each row in test.
# Add the same features to test as we did to train
test$date <- as.Date(test$date)
test$month <- as.numeric(format(test$date, "%m"))
test$year  <- as.numeric(format(test$date, "%Y"))

# Get predictions
pred.test <- predict(model, test)

# Create submission df
ss$Prediction <- pred.test

# Save submission file in csv format
write_csv(ss,"mysubmission.csv")



##########################################################
### --- feature engineering from the original data --- ###
# Look at correlation with target

cordb = cor(train[,!names(train) %in% c("X","date","ID")], train[,"burn_area"])
cordb <- as.data.frame((cordb))
names(cordb) <- "burn_area"
cordb$index = c(rownames(cordb))

cordb <- cordb[order(cordb$burn_area),]
barplot(cordb[,"burn_area"],names.arg=cordb[,"index"],col="blue", las=2)
colnames(train)
# --- Correlation between variables --- #
corVar = cor(train[,!names(train) %in% c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")], train[,!names(train) %in% c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")])
corVar <- as.data.frame((corVar))
head(corVar)
ggcorrplot(corVar, tl.cex = 10)

# Look at some scatter plots 

ggplot(train, aes(x=climate_vap, y=burn_area))+
  geom_point(size=3,col="blue", alpha=I(0.3))+theme_classic()+theme(legend.position="top")

ggplot(train, aes(x=climate_tmmx, y=burn_area))+
  geom_point(size=3,col="blue", alpha=I(0.3))+theme_classic()+theme(legend.position="top")
