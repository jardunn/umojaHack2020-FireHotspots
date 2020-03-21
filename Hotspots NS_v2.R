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
library(xgboost)

# Load the Data

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train_orig = read.csv("train.csv")
test  = read.csv("test.csv")

#############################################
##EDA + Feature Engineering
processDataSet <- function(dataset){
  # Change the NA values in the population density column to 0
  dataset$population_density[is.na(dataset$population_density)] = 0
  
  # Date variables
  dataset$date <- as.Date(dataset$date)
  dataset$month <- as.numeric(format(dataset$date, "%m"))
  dataset$year  <- as.numeric(format(dataset$date, "%Y"))
  
  # Turn month in categorical
  month_var = data.table(as.factor(dataset$month))
  month_cat = one_hot(month_var)
  head(month_cat)
  names(month_cat) <- c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
  dim(month_cat)
  
  # Replace the current month variable with the categorical columns
  dataset <- cbind(dataset, month_cat)
  dataset <- dataset %>% select(-c(month))
  
  # Add hot, dry, window category [0,1] when all three variables are above the 3rd quartile
  dataset <- dataset %>% 
    mutate(hot_dry_windy = ifelse(climate_vpd > 101 & climate_vs > 170 & climate_tmmx > 310, 1, 0))
  dataset <- dataset %>%
    mutate(heat_wind_index = climate_vpd * climate_vs * climate_tmmx)
  dataset <- dataset %>%
    mutate(heat_wind_index = (heat_wind_index-mean(heat_wind_index))/sd(heat_wind_index))
  
  # Add a variable for treecover and remove the tree-related variables
  dataset <- dataset %>% 
    mutate(tree_cover = (landcover_1 + landcover_2 + landcover_3 + landcover_4 + landcover_5))
  dataset <- dataset %>% 
    select(-c(landcover_1,landcover_2,landcover_3,landcover_4,landcover_5))
  dataset <- dataset %>%
    mutate(non_veg = (landcover_0 + landcover_7 + landcover_8))
  dataset <- dataset %>% select(-c(landcover_0,landcover_7,landcover_8))
  
  dataset <- dataset %>% 
    mutate(climate_vapour = climate_vap * climate_vpd)
  dataset <- dataset %>%
    mutate(climate_vapour = (climate_vapour-mean(climate_vapour))/sd(climate_vapour))
  dataset <- dataset %>% select(-c(climate_vpd, climate_vap))
  
  # The first data entry for every area has this set to "0" because obviously there isn't a lagged
  # value for the first month; unsure of whether to remove this or not -- Note: can't remove
  # this because the data sizes don't match up then
  # dataset <- dataset %>% filter(lagged_PDSI != 0)
  
  # Remove variables that are unneccesary for prediction 
  dataset <- dataset %>% select(-c("ID", "area", "year", "climate_swe", "lat"))
  
  # Remove more variables
  dataset <- dataset %>% select(-c(population_density, climate_pdsi))
  return(dataset)
}

trainData <- processDataSet(train_orig)
testData <- processDataSet(test)

#############################################


################################################################################
## -- FEATURE ENGINEERING ENDS HERE -- ##
################################################################################.



# Define input and output columns
in_cols <- names(trainData[,-c(1,3)])
target_col <- "burn_area"


# Get our X and y training and validation sets ready
X_trainXG <- trainData[,in_cols]
Y_trainXG <- trainData[,target_col]

### --- XGBoost --- ###
bst <- xgboost(data = data.matrix(X_trainXG),
               label = Y_trainXG,
               nrounds=15)


##############################################################################
######Making a Submission
# Once you've got some features and a model you're happy with, it's time to submit!
# Look at the sample submission file
ss = read.csv('SampleSubmission.csv')
head(ss)

in_cols <- names(testData[,-c(1,3)])

# Get our X and y training and validation sets ready
X_test_data <- testData[,in_cols]


# Get predictions
xg_pred <- predict(bst, data.matrix(X_test_data))
xg_pred[xg_pred < 0] = 0

# Create submission df
ss$Prediction <- xg_pred

# Save submission file in csv format
write_csv(ss,"mysubmission.csv")



##########################################################
# --- Correlation between variables --- #
# str(dataset)
# corVar = cor(dataset[,!names(dataset) %in% 
#                        c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug",
#                          "Sep","Oct","Nov","Dec","date")], dataset[,!names(dataset) 
#                           %in% c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug",
#                                  "Sep","Oct","Nov","Dec","date")])
# corVar <- as.data.frame((corVar))
# #head(corVar)
# ggcorrplot(corVar, tl.cex = 10)

