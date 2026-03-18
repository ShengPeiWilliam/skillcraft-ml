library(ggplot2)
library(rpart)      
library(rpart.plot)
library(cowplot)
library(caret)
library(randomForest)
library(tidyr)
library(GGally)
library(corrplot)
library(pROC)
library(nnet)
library(car)
library(dplyr)
library(smotefamily)
library(themis)
library(nnet)
options(warn = -1)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv"
skillcraft.data <- read.csv(URL)

str(skillcraft.data)

skillcraft.data$LeagueIndex <- as.factor(skillcraft.data$LeagueIndex)
skillcraft.data$GameID <- NULL
skillcraft.data$Age <- as.numeric(skillcraft.data$Age)
skillcraft.data$HoursPerWeek <- as.numeric(skillcraft.data$HoursPerWeek)
skillcraft.data$TotalHours <- as.numeric(skillcraft.data$TotalHours)

summary(skillcraft.data)

ggplot(data = skillcraft.data, mapping = aes(x = LeagueIndex)) +
  geom_bar(fill = "steelblue") +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  scale_x_discrete(labels = c("1" = "Bronze", "2" = "Silver", "3" = "Gold",
                               "4" = "Platinum", "5" = "Diamond", "6" = "Master",
                               "7" = "GM", "8" = "Pro")) +
  labs(title = "Player Distribution by League",
       x = "League", y = "Count")

features <- c("Age", "TotalHours", "WorkersMade", "MinimapAttacks")

df_long <- pivot_longer(skillcraft.data[, c("LeagueIndex", features)],
                        cols = -LeagueIndex,
                        names_to = "feature",
                        values_to = "value")

ggplot(df_long, aes(x = LeagueIndex, y = value, fill = LeagueIndex)) +
  geom_boxplot(outlier.size = 0.5, alpha = 0.7) +
  facet_wrap(~ feature, scales = "free_y", ncol = 2) +
  labs(title = "Feature Distribution by League",
       x = "League Index", y = "Value") +
  theme(legend.position = "none")

percentiles <- quantile(skillcraft.data$TotalHours, 
                        probs = c(0.90, 0.95, 0.99, 0.999, 1.0), 
                        na.rm = TRUE)
print(percentiles)

cap_99 <- quantile(skillcraft.data$TotalHours, probs = 0.99, na.rm = TRUE)

ggplot(skillcraft.data[skillcraft.data$TotalHours <= cap_99, ], 
       aes(x = TotalHours)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  labs(title = "TotalHours Distribution (Capped at 99th Percentile)", 
       x = "TotalHours", y = "Count")

show_missing <- function(league_id) {
  rows <- skillcraft.data[skillcraft.data$LeagueIndex == league_id, ]
  counts <- colSums(is.na(rows))
  counts <- counts[counts > 0]
  if (length(counts) == 0) return(NULL)
  data.frame(
    League       = paste0("League ", league_id),
    Feature      = names(counts),
    Missing      = as.integer(counts),
    Total        = nrow(rows),
    Missing_Rate = paste0(round(counts / nrow(rows) * 100, 1), "%")
  )
}

result <- as.data.frame(rbind(show_missing(5), show_missing(8)))
print(result, row.names = FALSE)

numeric_features <- c("LeagueIndex", "APM", "ActionLatency", 
                      "UniqueHotkeys", "AssignToHotkeys")

corrplot(cor(skillcraft.data[, numeric_features] %>% 
               mutate(LeagueIndex = as.numeric(LeagueIndex)), 
             use = "complete.obs"),
         method = "color",
         type = "upper",
         tl.cex = 0.8,
         addCoef.col = "black",
         number.cex = 0.6)

skillcraft.data <- skillcraft.data[skillcraft.data$LeagueIndex != 8, ]
skillcraft.data <- skillcraft.data[skillcraft.data$LeagueIndex != 7, ]
skillcraft.data$LeagueIndex <- droplevels(skillcraft.data$LeagueIndex)

cap_99 <- quantile(skillcraft.data$TotalHours, probs = 0.99, na.rm = TRUE)
skillcraft.data$TotalHours <- pmin(skillcraft.data$TotalHours, cap_99)

set.seed(42)
skillcraft.data.imputed <- rfImpute(LeagueIndex ~ ., data = skillcraft.data, iter = 6)

set.seed(42)
train.index <- sample(1:nrow(skillcraft.data.imputed), 0.8 * nrow(skillcraft.data.imputed))
train.data <- skillcraft.data.imputed[train.index, ]
test.data  <- skillcraft.data.imputed[-train.index, ]

lm.proxy <- lm(as.numeric(LeagueIndex) ~ ., data = train.data)
vif.result <- vif(lm.proxy)

vif.df <- data.frame(
  Feature = names(vif.result),
  VIF     = round(vif.result, 3)
)
print(vif.df, row.names = FALSE)

invisible(capture.output(
  step.result <- stats::step(lm.proxy, direction = "backward")
))

step.result

selected.features <- names(coef(step.result))[-1]
all.features      <- colnames(train.data[, -1])
removed.features  <- all.features[!all.features %in% selected.features]

cat("Removed", length(removed.features), "features:", paste(removed.features, collapse = ", "), "\n")

set.seed(42)
ctrl <- trainControl(method = "cv", number = 5)

invisible(capture.output(
  cv.model <- train(LeagueIndex ~ HoursPerWeek + TotalHours +
                      SelectByHotkeys + AssignToHotkeys + UniqueHotkeys +
                      MinimapAttacks + NumberOfPACs + GapBetweenPACs +
                      ActionLatency + ActionsInPAC + TotalMapExplored +
                      WorkersMade + UniqueUnitsMade + ComplexUnitsMade,
                    data = train.data,
                    method = "multinom",
                    trControl = ctrl,
                    maxit = 300,
                    trace = FALSE)
))

cv.model

cv.results <- data.frame(
  Decay    = cv.model$results$decay,
  Accuracy = round(cv.model$results$Accuracy, 4),
  Kappa    = round(cv.model$results$Kappa, 4)
)
print(cv.results, row.names = FALSE)

calculate_class_weights <- function(data) {
  class.counts <- table(data$LeagueIndex)
  max.count <- max(class.counts)
  class.weights <- max.count / class.counts
  sample.weights <- as.numeric(class.weights[as.character(data$LeagueIndex)])
  return(sample.weights)
}

set.seed(42)

train.smote <- SMOTE(train.data[, -1], train.data$LeagueIndex, K = 5, dup_size = 5)
train.data.smote <- train.smote$data
names(train.data.smote)[names(train.data.smote) == "class"] <- "LeagueIndex"
train.data.smote$LeagueIndex <- as.factor(train.data.smote$LeagueIndex)

train.data.smote.tomek <- tomek(train.data.smote, var = "LeagueIndex")

before.dist <- table(train.data$LeagueIndex)
before.pct <- round(before.dist / sum(before.dist) * 100, 2)

after.dist <- table(train.data.smote.tomek$LeagueIndex)
after.pct <- round(after.dist / sum(after.dist) * 100, 2)

resampling.summary <- data.frame(
  League = names(before.dist),
  Before_Count = as.numeric(before.dist),
  Before_Pct = paste0(before.pct, "%"),
  After_Count = as.numeric(after.dist),
  After_Pct = paste0(after.pct, "%"),
  row.names = NULL
)

print(resampling.summary, row.names = FALSE)

train_lr <- function(train.data) {
  set.seed(42)
  multinom(LeagueIndex ~ HoursPerWeek + TotalHours +
             SelectByHotkeys + AssignToHotkeys + UniqueHotkeys +
             MinimapAttacks + NumberOfPACs + GapBetweenPACs +
             ActionLatency + ActionsInPAC + TotalMapExplored +
             WorkersMade + UniqueUnitsMade + ComplexUnitsMade,
           data = train.data, maxit = 500, trace = FALSE)
}

train_lr_weighted <- function(data) {
  set.seed(42)
  weights <- calculate_class_weights(data)
  multinom(LeagueIndex ~ HoursPerWeek + TotalHours +
             SelectByHotkeys + AssignToHotkeys + UniqueHotkeys +
             MinimapAttacks + NumberOfPACs + GapBetweenPACs +
             ActionLatency + ActionsInPAC + TotalMapExplored +
             WorkersMade + UniqueUnitsMade + ComplexUnitsMade,
           data = data, weights = weights, maxit = 500, trace = FALSE)
}

train_dt <- function(data) {
  set.seed(42)
  model   <- rpart(LeagueIndex ~ ., data = data, method = "class")
  best.cp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]
  prune(model, cp = best.cp)
}

train_dt_weighted <- function(data) {
  set.seed(42)
  weights <- calculate_class_weights(data)
  model   <- rpart(LeagueIndex ~ ., data = data, method = "class", weights = weights)
  best.cp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]
  prune(model, cp = best.cp)
}

train_rf <- function(data) {
  set.seed(42)
  randomForest(LeagueIndex ~ ., data = data, ntree = 1000, proximity = TRUE)
}

train_rf_weighted <- function(data) {
  set.seed(42)
  weights <- calculate_class_weights(data)
  randomForest(LeagueIndex ~ ., data = data, ntree = 1000, proximity = TRUE,
               classwt = table(data$LeagueIndex) / nrow(data))
}

eval_model <- function(model, test.data) {
  pred <- predict(model, newdata = test.data, type = "class")
  cm   <- confusionMatrix(pred, test.data$LeagueIndex)
  print(cm$table)
  min_class <- names(which.min(table(test.data$LeagueIndex)))
  results <- data.frame(
    Accuracy = round(cm$overall["Accuracy"], 4),
    Recall_minority = round(cm$byClass[paste0("Class: ", min_class), "Recall"], 4)
  )
  cat("Minority class: League", min_class, "\n")
  print(results, row.names = FALSE)
  invisible(results)
}

lr.model <- train_lr(train.data)
lr.results <- eval_model(lr.model, test.data)

lr.model.weighted <- train_lr_weighted(train.data)
lr.results.weighted <- eval_model(lr.model.weighted, test.data)

lr.model.smote.tomek <- train_lr(train.data.smote.tomek)
lr.results.smote.tomek <- eval_model(lr.model.smote.tomek, test.data)

comparison.lr <- data.frame(
  Method = c("Baseline", "Class Weights", "SMOTE + Tomek"),
  Accuracy = c(lr.results$Accuracy, 
               lr.results.weighted$Accuracy, 
               lr.results.smote.tomek$Accuracy),
  Recall_minority = c(lr.results$Recall_minority, 
                      lr.results.weighted$Recall_minority, 
                      lr.results.smote.tomek$Recall_minority)
)
print(comparison.lr, row.names = FALSE)

dt.model <- train_dt(train.data)
dt.results <- eval_model(dt.model, test.data)

rpart.plot(dt.model)

dt.model.weighted <- train_dt_weighted(train.data)
dt.results.weighted <- eval_model(dt.model.weighted, test.data)

dt.model.smote.tomek <- train_dt(train.data.smote.tomek)
dt.results.smote.tomek <- eval_model(dt.model.smote.tomek, test.data)

comparison.dt <- data.frame(
  Method = c("Baseline", "Class Weights", "SMOTE + Tomek"),
  Accuracy = c(dt.results$Accuracy, 
               dt.results.weighted$Accuracy, 
               dt.results.smote.tomek$Accuracy),
  Recall_minority = c(dt.results$Recall_minority, 
                      dt.results.weighted$Recall_minority, 
                      dt.results.smote.tomek$Recall_minority)
)
print(comparison.dt, row.names = FALSE)

rf.model <- train_rf(train.data)
rf.results   <- eval_model(rf.model, test.data)

oob.error.data <- data.frame(
    Trees = rep(1:nrow(rf.model$err.rate), times = 7), 
    Type  = rep(c("OOB", "1", "2", "3", "4", "5", "6"), 
                each = nrow(rf.model$err.rate)),
    Error = c(rf.model$err.rate[, "OOB"],
              rf.model$err.rate[, "1"],
              rf.model$err.rate[, "2"],
              rf.model$err.rate[, "3"],
              rf.model$err.rate[, "4"],
              rf.model$err.rate[, "5"],
              rf.model$err.rate[, "6"])
)

ggplot(data = oob.error.data, aes(x = Trees, y = Error)) +
    geom_line(aes(color = Type)) +
    labs(title = "OOB Error Convergence (Baseline RF)",
         x = "Number of Trees", y = "Error Rate")

rf.model.weighted <- train_rf_weighted(train.data)
rf.results.weighted <- eval_model(rf.model.weighted, test.data)

rf.model.smote.tomek <- train_rf(train.data.smote.tomek)
rf.results.smote.tomek <- eval_model(rf.model.smote.tomek, test.data)

varImpPlot(rf.model.smote.tomek)

distance.matrix <- dist(1 - rf.model.smote.tomek$proximity)
mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

mds.values <- mds.stuff$points

mds.data <- data.frame(Sample = rownames(mds.values),
    X = mds.values[, 1],
    Y = mds.values[, 2],
    Status = train.data.smote.tomek$LeagueIndex)

ggplot(data = mds.data, aes(x = X, y = Y, label = Sample)) +
    geom_text(aes(color = Status)) + 
    theme_bw() + 
    xlab(paste("MDS1 - ", mds.var.per[1], "%", sep = "")) +
    ylab(paste("MDS2 - ", mds.var.per[2], "%", sep = "")) +
    ggtitle("MDS plot using (1 - Random Forest Proximities)")

comparison.rf <- data.frame(
  Method = c("Baseline", "Class Weights", "SMOTE + Tomek"),
  Accuracy = c(rf.results$Accuracy, 
               rf.results.weighted$Accuracy, 
               rf.results.smote.tomek$Accuracy),
  Recall_minority = c(rf.results$Recall_minority, 
                      rf.results.weighted$Recall_minority, 
                      rf.results.smote.tomek$Recall_minority)
)
print(comparison.rf, row.names = FALSE)

run_pairwise_rf <- function(league_a, league_b) {
  
  pair.data <- skillcraft.data.imputed[
    skillcraft.data.imputed$LeagueIndex %in% c(league_a, league_b), ]
  pair.data$LeagueIndex <- droplevels(pair.data$LeagueIndex)
  
  set.seed(42)
  idx        <- sample(1:nrow(pair.data), 0.8 * nrow(pair.data))
  pair.train <- pair.data[idx, ]
  pair.test  <- pair.data[-idx, ]
  
  model <- train_rf_weighted(pair.train)
  pred  <- predict(model, newdata = pair.test)
  acc   <- round(mean(pred == pair.test$LeagueIndex) * 100, 2)
  
  return(acc)
}

acc.bg <- run_pairwise_rf(1, 3)
acc.sp <- run_pairwise_rf(2, 4)
acc.gd <- run_pairwise_rf(3, 5)
acc.pm <- run_pairwise_rf(4, 6)

paper.acc <- c(82.32, 78.27, 79.01, 80.70)
our.acc   <- c(acc.bg, acc.sp, acc.gd, acc.pm)

pairwise.summary <- data.frame(
  Pair         = c("Bronze-Gold", "Silver-Platinum", "Gold-Diamond", "Platinum-Masters"),
  Paper_Forest = paper.acc,
  Our_LR       = our.acc,
  Beat_Paper   = our.acc > paper.acc
)
print(pairwise.summary, row.names = FALSE)

all.results <- data.frame(
  Algorithm = rep(c("Logistic Regression", "Decision Tree", "Random Forest"), each = 3),
  Method    = rep(c("Baseline", "Class Weights", "SMOTE + Tomek"), 3),
  Accuracy  = c(comparison.lr$Accuracy,
                comparison.dt$Accuracy,
                comparison.rf$Accuracy),
  Recall_minority = c(comparison.lr$Recall_minority,
                      comparison.dt$Recall_minority,
                      comparison.rf$Recall_minority)
)

print(all.results, row.names = FALSE)
