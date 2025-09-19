# ==========================================================

# ==========================================================

rm(list = ls())

library(dplyr)
library(survival)
library(randomForestSRC)
library(gbm)
library(CoxBoost)
library(glmnet)
library(timeROC)
library(ggplot2)
library(survcomp)
library(rmda)
library(pec)   

# ==========================================================

# ==========================================================

setwd("D:\\DuanWork\\D1lungColon\\3data")

trainSet <- read.csv('trainSet.csv', header = T, row.names = 1)
testSet <-read.csv('testSet.csv', header = T, row.names = 1)

names(trainSet)

selevaraibles <- c("OS", "Time", 
                   "Hist_lung",  "TStage_lung", "NStage_lung" ,
                   "Surgery_lung", "Chemotherapy_lung",  "Radiation_lung",
                   "Age",   "Sex", "Hist_colon" ,  
                   "TStage_colon",  "NStage_colon", "MStage_colon",
                   "Surgery_colon" , "Chemotherapy_colon", "Radiation_colon")

trainSet <- trainSet[,which(colnames(trainSet) %in% selevaraibles)]
testSet <- testSet[,which(colnames(testSet) %in% selevaraibles)]


trainSet <- trainSet[, !names(trainSet) %in% c(
  "Hist_lung", "Laterality", "Stage_lung", "TStage_lung",
  "NStage_lung", "Surgery_lung", "Chemotherapy_lung", "Radiation_lung"
)]


testSet <- testSet[, !names(testSet) %in% c(
  "Hist_lung", "Laterality", "Stage_lung", "TStage_lung",
  "NStage_lung", "Surgery_lung", "Chemotherapy_lung", "Radiation_lung"
)]


feature_cols <- setdiff(colnames(trainSet), c("Time","OS"))

setwd('D:\\DuanWork\\D1lungColon\\6OSmodel')
set.seed(42)
# ---------- 训练各模型（参数尽量简单） ----------
# 1) CoxPH
fit_cox <- coxph(Surv(Time, OS) ~ ., data = trainSet)

# 2) RSF
fit_rsf <- rfsrc(Surv(Time, OS) ~ ., data = trainSet,
                 ntree = 50, mtry = 3,
                 importance = FALSE)



# 3) CoxBoost
x_train_cb <- as.matrix(trainSet[, feature_cols])
fit_cb <- CoxBoost(time = trainSet$Time, status = trainSet$OS,
                   x = x_train_cb, stepno = 100)

# 4) GBM 
n_trees <- 100
fit_gbm <- gbm(Surv(Time, OS) ~ ., data = trainSet,
               distribution = "coxph",
               n.trees = n_trees, interaction.depth = 3,
               shrinkage = 0.01, n.minobsinnode = 10, bag.fraction = 0.7,
               train.fraction = 1.0, verbose = FALSE)

# 5) Stepwise Cox
full_cox <- coxph(Surv(Time, OS) ~ ., data = trainSet)
fit_step <- step(full_cox, direction = "both", trace = FALSE)


# ---------- 预测风险分 ----------
scores_df <- data.frame(row_id = rownames(testSet))


scores_df$OS <- testSet$OS
scores_df$Time <- testSet$Time

scores_df$CoxPH   <- as.numeric(predict(fit_cox,  newdata = testSet, type = "lp"))
scores_df$RSF     <- as.numeric(predict(fit_rsf,  newdata = testSet)$predicted)
scores_df$CoxBoost<- as.numeric(predict(fit_cb,   newdata = as.matrix(testSet[, feature_cols]), 
                                        type = "lp"))
scores_df$GBM     <- as.numeric(predict(fit_gbm,  newdata = testSet, n.trees = n_trees, 
                                        type = "link"))
scores_df$stepCox <- as.numeric(predict(fit_step, newdata = testSet, type = "lp"))

min(scores_df$Time)
max(scores_df$Time)

head(scores_df)

write.csv(scores_df, file = 'scores_without.csv')

# ==========================================================

# ==========================================================

algorithms <- c("CoxPH", "RSF", "CoxBoost", "GBM", "stepCox")
timepoints <- seq(0.5, 10, by = 0.5)

auc_matrix <- matrix(NA, nrow = length(timepoints), ncol = length(algorithms),
                     dimnames = list(timepoints, algorithms))

for (alg in algorithms) {
  roc_result <- timeROC(
    T = scores_df$Time,
    delta = scores_df$OS,
    marker = scores_df[[alg]],
    cause = 1,
    times = timepoints
  )
  
  auc_matrix[, alg] <- roc_result$AUC
}

auc_df <- as.data.frame(auc_matrix)
print(auc_df)
rownames(auc_df) <- paste0('Year', rownames(auc_df))

getwd()

write.csv(auc_df, file = 'AUC_withoutLung.csv')


# ==========================================================

# ==========================================================

cindex_df <- data.frame(algorithm = character(), cindex = numeric())

for (alg in algorithms) {
  cindex_val <- concordance.index(
    x = scores_df[[alg]],
    surv.time = scores_df$Time,
    surv.event = scores_df$OS
  )$c.index
  
  cindex_df <- rbind(cindex_df, data.frame(
    algorithm = alg,
    cindex = round(cindex_val, 3)
  ))
}

cindex_df

write.csv(cindex_df, file = 'Cindex_withoutLung.csv')


# ==========================================================

# =========================================================


ibs_times <- timepoints   

model_cols <- algorithms

safe_norm <- function(x){
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) rep(0.5, length(x)) else (x - rng[1]) / (rng[2] - rng[1])
}

brier_at_time <- function(norm_score, t, time_vec, event_vec){
  y <- as.numeric(time_vec <= t & event_vec == 1)
  mean((norm_score - y)^2, na.rm = TRUE)
}

brier_long <- do.call(rbind, lapply(model_cols, function(mn){
  ns <- safe_norm(scores_df[[mn]])
  data.frame(
    model = mn,
    time  = ibs_times,
    brier = sapply(ibs_times, function(tt)
      brier_at_time(ns, tt, time_vec = testSet$Time, event_vec = testSet$OS)
    )
  )
}))

ibs_df <- brier_long |>
  dplyr::group_by(model) |>
  dplyr::summarise(IBS = mean(brier, na.rm = TRUE)) |>
  dplyr::arrange(IBS)

ibs_df <- as.data.frame(ibs_df)
print(ibs_df)

write.csv(ibs_df, file = 'ibs_withoutLung.csv')



#===============================================================

#===============================================================


dca_data <- data.frame(OS_event = scores_df$OS)

for (model in algorithms) {
  risk_score <- scores_df[[model]]
  norm_score <- (risk_score - min(risk_score)) / (max(risk_score) - min(risk_score))
  dca_data[[paste0(model, "_norm")]] <- norm_score
}


dca_list <- list()
for (model in algorithms) {
  formula <- as.formula(paste0("OS_event ~ ", model, "_norm"))
  dca_res <- decision_curve(formula,
                            data=dca_data,
                            thresholds=seq(0.01,0.99,by=0.01),
                            confidence.intervals=0.95)
  dca_list[[model]] <- dca_res
}

custom_colors <- c("#1F77B4",  "#FF7F0E",  "#2CA02C", "#D62728", 
                   "#9467BD",   "#8C564B", "#E377C2"  
)


pdf("DCAwithoutLung.pdf", width=8, height=8)
plot_decision_curve(dca_list,
                    curve.names=algorithms,
                    col=custom_colors,
                    xlab="Threshold Probability",
                    ylab="Net Benefit",
                    lty=1,
                    lwd=2,
                    cost.benefit.axis=FALSE,
                    legend.position="topright",
                    confidence.intervals=FALSE)
dev.off()
