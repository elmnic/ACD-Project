library(ggplot2)
library(tm)
library(dplyr)
library(caret)
library(doParallel)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# Read and process data
data <- readxl::read_excel("Data/brexit_blog_corpus.xlsx")
keepCols <- c("Utterance", "Stance category")
data <- as.data.frame(data[keepCols])
names(data) <- c("Utterance", "Stance_category")

# Create corpus and pre-process the text
corpus <- VCorpus(VectorSource(data$Utterance))
dtm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert dtm to data.frame and assign class
data_prep <- as.matrix(dtm)
data_prep <- cbind(data_prep, data$Stance_category)
colnames(data_prep)[ncol(data_prep)] <- 'Class'
data_prep <- as.data.frame(data_prep)
data_prep$Class <- as.factor(data_prep$Class)
data_prep <- mutate(data_prep, id = rownames(data_prep)) # Add id column

# Sample up and down
upSampled_data <- upSample(data_prep[,-((ncol(data_prep)-2):ncol(data_prep))], data_prep$Class)
upSampled_data <- mutate(upSampled_data, id = rownames(upSampled_data)) # Add id column

downSampled_data <- downSample(data_prep[,-((ncol(data_prep)-2):ncol(data_prep))], data_prep$Class)
downSampled_data <- mutate(downSampled_data, id = rownames(downSampled_data)) # Add id column


# Split untouched data into train and test
normal_train <- data_prep %>%
                group_by(Class) %>%
                sample_frac(size = .75)
normal_test <- anti_join(data_prep, normal_train, by = c("id"))
normal_train$id <- NULL
normal_test$id <- NULL

# Split up sampled data into train and test
upSampled_train <- upSampled_data %>%
                    group_by(Class) %>%
                    sample_frac(size = .75)
upSampled_test <- anti_join(upSampled_data, upSampled_train, by = c("id"))
upSampled_train$id <- NULL
upSampled_test$id <- NULL

# Split down sampled data into train and test
downSampled_train <- downSampled_data %>%
                      group_by(Class) %>%
                      sample_frac(size = .75)
downSampled_test <- anti_join(downSampled_data, downSampled_train, by = c("id"))
downSampled_train$id <- NULL
downSampled_test$id <- NULL

# Fit with k-fold cross validation
ctrl <- caret::trainControl(method = "cv",
                            number = 5,
                            verboseIter = TRUE,
                            summaryFunction = multiClassSummary)

############### SVM ########################
start.time <- Sys.time()
model_svm_normal <- caret::train(Class ~ ., 
                           data = normal_train,
                           method = "svmRadial",
                           trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_svm_normal, "model_svm_normal_PC.rds")
time.svm_normal <- end.time - start.time
cat(sprintf("SVM_normal time: %f", time.svm_normal))


start.time <- Sys.time()
model_svm_up <- caret::train(Class ~ ., 
                          data = upSampled_train,
                          method = "svmRadial",
                          trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_svm_up, "model_svm_up_PC.rds")
time.svm_up <- end.time - start.time
cat(sprintf("SVM_up time: %f", time.svm_up))


start.time <- Sys.time()
model_svm_down <- caret::train(Class ~ ., 
                          data = downSampled_train,
                          method = "svmRadial",
                          trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_svm_down, "model_svm_down_PC.rds")
time.svm_down <- end.time - start.time
cat(sprintf("SVM_down time: %f", time.svm_down))

################ Random Forest ###################
start.time <- Sys.time()
model_rf_normal <- caret::train(Class ~ ., 
                           data = normal_train,
                           method = "rf",
                           trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_rf_normal, "model_rf_normal_PC.rds")
time.rf_normal <- end.time - start.time
cat(sprintf("RF_normal time: %f", time.rf_normal))


start.time <- Sys.time()
model_rf_up <- caret::train(Class ~ ., 
                            data = upSampled_train,
                            method = "rf",
                            trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_rf_up, "model_rf_up_PC.rds")
time.rf_up <- end.time - start.time
cat(sprintf("RF_up time: %f", time.rf_up))


start.time <- Sys.time()
model_rf_down <- caret::train(Class ~ ., 
                            data = downSampled_train,
                            method = "rf",
                            trControl = ctrl)
end.time <- Sys.time()
saveRDS(model_rf_down, "model_rf_down_PC.rds")
time.rf_down <- end.time - start.time
cat(sprintf("RF_down time: %f", time.rf_down))
################ Finish training ##################
stopCluster(cl)

# Test RF and SVM models
rf_pred_normal <- predict(model_rf_normal, newdata = normal_test)
rf_pred_up <- predict(model_rf_up, newdata = upSampled_test)
rf_pred_down <- predict(model_rf_down, newdata = downSampled_test)

svm_pred_normal <- predict(model_svm_normal, newdata = normal_test)
svm_pred_up <- predict(model_svm_up, newdata = upSampled_test)
svm_pred_down <- predict(model_svm_down, newdata = downSampled_test)

# Get Confusion Matrix
cm_rf_pred_normal <- confusionMatrix(rf_pred_normal, normal_test$Class, dnn = c("Prediction", "Reference"))
cm_rf_pred_up <- confusionMatrix(rf_pred_up, upSampled_test$Class, dnn = c("Prediction", "Reference"))
cm_rf_pred_down <- confusionMatrix(rf_pred_down, downSampled_test$Class, dnn = c("Prediction", "Reference"))

cm_svm_pred_normal <- confusionMatrix(svm_pred_normal, normal_test$Class, dnn = c("Prediction", "Reference"))
cm_svm_pred_up <- confusionMatrix(svm_pred_up, upSampled_test$Class, dnn = c("Prediction", "Reference"))
cm_svm_pred_down <- confusionMatrix(svm_pred_down, downSampled_test$Class, dnn = c("Prediction", "Reference"))

# Get overall accuracy
rf_acc_normal <- cm_rf_pred_normal$overall["Accuracy"]
rf_acc_up <- cm_rf_pred_up$overall["Accuracy"]
rf_acc_down <- cm_rf_pred_down$overall["Accuracy"]

svm_acc_normal <- cm_svm_pred_normal$overall["Accuracy"]
svm_acc_up <- cm_svm_pred_up$overall["Accuracy"]
svm_acc_down <- cm_svm_pred_down$overall["Accuracy"]

# Retrieve Precision, Recall and Balanced Accuracys
rf_prb_normal <- cm_rf_pred_normal[["byClass"]][,c(5,6,11)]
rf_prb_up <- cm_rf_pred_up[["byClass"]][,c(5,6,11)]
rf_prb_down <- cm_rf_pred_down[["byClass"]][,c(5,6,11)]

svm_prb_normal <- cm_svm_pred_normal[["byClass"]][,c(5,6,11)]
svm_prb_up <- cm_svm_pred_up[["byClass"]][,c(5,6,11)]
svm_prb_down <- cm_svm_pred_down[["byClass"]][,c(5,6,11)]

# Print RF accuracy
cat(sprintf("Accuracy\nRF: %f\nRF Up: %f\nRF Down: %f\n", 
            rf_acc_normal, rf_acc_up, rf_acc_down))
cat(sprintf("Accuracy\nSVM: %f\nSVM Up: %f\nSVM Down: %f\n", 
            svm_acc_normal, svm_acc_up, svm_acc_down))

# Save and plot results
results <- data.frame(c(rf_acc_normal, rf_acc_up, rf_acc_down, svm_acc_normal, svm_acc_up, svm_acc_down),
                      c("RF normal", "RF up", "RF down", "SVM normal", "SVM up", "SVM down"),
                      row.names = 2)
colnames(results) <- c("Accuracy")
ggplot(results, aes(x=row.names(results), y=Accuracy, fill=Accuracy)) + 
  geom_col() + 
  ylim(0.0, 1.0) +
  labs(x="")

# Print table
cm_rf_pred_normal['table']
cm_rf_pred_up['table']
cm_rf_pred_down['table']

cm_svm_pred_normal['table']
cm_svm_pred_up['table']
cm_svm_pred_down['table']

#
f <- list(
  family = "Courier New, monospace",
  size = 20,
  color = "#000000"
)
x <- list(
  title = "Reference",
  titlefont = f
)
y <- list(
  title = "Predicted",
  titlefont = f
)
plot_data <- cm_rf_pred_down$table
plot_ly(x = rownames(plot_data), 
        y = colnames(plot_data), 
        z = plot_data, 
        type = "heatmap",
        colors = colorRamp(c("black", "cyan"))) %>%
  layout(xaxis = x, yaxis=y, title="RF down")


# Print Precision, Recall and Balanced Accuracy per class
rf_prb_normal
rf_prb_up
rf_prb_down

svm_prb_normal
svm_prb_up
svm_prb_down


# Plot the class frequency
stanceFreq <- as.data.frame(table(data$Stance_category))
ggplot(stanceFreq, aes(x=reorder(Var1, Freq), y=Freq, fill=Freq)) + 
  geom_col() + 
  labs(x="Stance", y="Frequency") + 
  coord_flip()

# Plot distribution after Upsampling
up_freq <- as.data.frame(table(upSampled_data$Class))
ggplot(up_freq, aes(x=reorder(Var1, Freq), y=Freq, fill=Freq)) + 
  geom_col() + 
  labs(x="Stance", y="Frequency") + 
  coord_flip()

# Plot distribution after Downsampling
down_freq <- as.data.frame(table(downSampled_data$Class))
ggplot(down_freq, aes(x=reorder(Var1, Freq), y=Freq, fill=Freq)) + 
  geom_col() + 
  labs(x="Stance", y="Frequency") + 
  coord_flip()
#####################################################################


model <- svm(y ~ ., data = train, type = 'C-classification')
test_pred <- predict(model, test)
test_cm <- confusionMatrix(test_pred, test$y, dnn = c("Prediction", "Reference"))
test_cm[['byClass']]
table(test_pred, test$y)

stances <- unique(data$Stance_category)
numericStances <- match(data$Stance_category, stances) - 1
slda_model <- lda.collapsed.gibbs.sampler(
                lex$documents, 
                K = 10, 
                vocab = lex$vocab, 
                num.iterations = 2,
                alpha = 1.0,
                eta = 0.1)
slda.predict(lex$documents, slda_model['topics'], slda_model, alpha=1.0, eta=0.1)


lex <- lexicalize(data$Utterance)



