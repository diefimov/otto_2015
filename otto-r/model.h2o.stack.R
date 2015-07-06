source("utils.R")
load(paste0(output.r.path, "data.all.RData"))
nfolds <- 3
localH2O <- h2o.init(nthread=16,Xmx="60g")

set.seed(2015)
cv.folds <- cvFolds(nrow(data.all[!is.na(target)]), K = nfolds)
cvs <- cv.folds$which
cat("Instance CV distribution: \n")
print(table(cv.folds$which))

for (k in 1:(nfolds+1)) {
  if (k > nfolds) {
    data.train <- data.all[!is.na(target)]
    data.test <- data.all[is.na(target)]
  } else {
    data.train <- data.all[!is.na(target)][which(cvs!=k)]
    data.test <- data.all[!is.na(target)][which(cvs==k)]
  }
  data.train <- data.train[sample(c(1:nrow(data.train)), nrow(data.train))]
  data.actual <- as.data.frame(data.train[,paste0("Class_",c(1:9)),with=F])
  data.target <- data.train$target
  ids.test <- data.test$id
  flist <- setdiff(colnames(data.train), c("id", "target", paste0("Class_", c(1:9))))
  data.train <- as.data.frame(data.train[,flist,with=F])
  data.test <- as.data.frame(data.test[,flist,with=F])
  data.train <- sqrt(data.train+(3/8))
  data.test <- sqrt(data.test+(3/8))
  data.train$target <- data.target

  data.train <- as.h2o(localH2O,data.train)
  data.test <- as.h2o(localH2O,data.test)
  
  predictors <- 1:(ncol(data.train)-1)
  response <- ncol(data.train)

  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=data.train,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=50,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  data.pred <- as.data.frame(h2o.predict(model,data.test))[,2:10]
  data.pred <- as.data.table(data.pred)
  data.pred[, id := ids.test]
  if (k == 1) {
    data.pred.train <- copy(data.pred)
  } else {
    if (k <= nfolds) {
      data.pred.train <- rbind(data.pred.train, data.pred)
    } else {
      data.pred.test <- copy(data.pred)
    }
  }
}
write.csv(as.data.frame(data.pred.train),
          file=paste0(output.r.path, "train/model.h2o.csv"),
          row.names=FALSE) 

write.csv(as.data.frame(data.pred.test),
          file=paste0(output.r.path, "test/model.h2o.csv"),
          row.names=FALSE) 
