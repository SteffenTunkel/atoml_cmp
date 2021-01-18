# Title     : TODO
# Objective : TODO
# Created by: Steffen Tunkel
# Created on: 12.01.2021

#install.packages("caret")
library(caret)
library(foreign)
library(testthat)
suppressPackageStartupMessages(install.packages("stepPlr"))
test_that("LogReg",{
    set.seed(825)

    testdata <- read.arff("smokedata/Uniform_1_test.arff")
    traindata <- read.arff("smokedata/Uniform_1_training.arff")

    target_index <- ncol(traindata)
    train_x <- traindata[,-target_index]
    train_y <- traindata[,target_index]
    test_x  <- testdata[,-target_index]
    test_y  <- testdata[,target_index]


    control <- trainControl(method = "none")
    paramGrid <-  expand.grid(lambda = 0.0, cp="bic")


    model <- train(x = train_x,
                 y = train_y,
                 method = "plr",
                 tuneGrid = paramGrid,
                 trControl = control)

    predictions <- predict(model, test_x)
    probabilities <- predict(model, test_x, type = "prob")

    confusionMatrix(data= predictions,
                  reference = test_y)

    pred_class_numbers <- as.integer(predictions) - 1
    test_y <- as.integer(test_y) - 1

    csv_df <- cbind(actual = test_y,
                  prediction = pred_class_numbers,
                  prob_0 = probabilities[,1],
                  prob_1 = probabilities[,2])

    write.csv(x = csv_df,
            file = file.path("../log/pred_CARET_LogisticRegression_Uniform.csv"),
            row.names = FALSE)
    expect_true(TRUE) # expect statement needed, because otherwise skipped
})
