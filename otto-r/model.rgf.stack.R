source("utils.R")
load(paste0(output.r.path, "data.all.RData"))
nfolds <- 7

set.seed(2015)
cv.folds <- cvFolds(nrow(data.all[!is.na(target)]), K = nfolds)
cvs <- cv.folds$which
cat("Instance CV distribution: \n")
print(table(cv.folds$which))

fn.register.wk(6)
rgf.pred <- foreach(k=1:(nfolds+1),.errorhandling='stop',.combine='rbind') %dopar% {
  source("utils.parallel.R")
  library(data.table)
  
  # define train and test set
  if (k > nfolds) {
    data.train <- data.all[!is.na(target)]
    data.test <- data.all[is.na(target)]
  } else {
    data.train <- data.all[!is.na(target)][which(cvs!=k)]
    data.test <- data.all[!is.na(target)][which(cvs==k)]
  }
  data.train <- data.train[sample(c(1:nrow(data.train)), nrow(data.train))]
  data.actual <- as.data.frame(data.train[,paste0("Class_",c(1:9)),with=F])
  ids.test <- data.test$id
  flist <- setdiff(colnames(data.train), c("id", "target", paste0("Class_", c(1:9))))
  data.train <- as.data.frame(data.train[,flist,with=F])
  data.test <- as.data.frame(data.test[,flist,with=F])
  
  # settings of rgf
  prefix <- paste0("fold",k)
  prefix.model <- paste0("rgf.model.", prefix)
  train.x.file <- paste0(output.rgf.path, "train.x.", prefix, ".csv")
  train.y.file <- paste0(output.rgf.path, "train.y.", prefix, ".csv")
  test.x.file <- paste0(output.rgf.path, "test.x.",prefix,".csv")
  model.file <- paste0(output.rgf.path, "rgf.model.", prefix)
  test.y.file <- paste0(output.rgf.path, "test.y.", prefix,".csv")
  train.settings.file <- paste0(output.rgf.path,"train.rgf.settings.fold",k)
  test.settings.file <- paste0(output.rgf.path,"test.rgf.settings.fold",k)
  fn.init.worker(paste0(output.rgf.path, prefix))
  
  trainSettings <- file(paste0(train.settings.file, ".inp"))
  pars <- paste0("train_x_fn=",train.x.file,"\n",
                 "train_y_fn=",train.y.file,"\n",
                 "model_fn_prefix=",model.file,"\n",
                 "reg_L2=", 0.01, "\n",
                 #"reg_sL2=", 0.1, "\n",
                 #"reg_depth=", 1.01, "\n",
                 "algorithm=","RGF","\n",
                 "loss=","Log","\n",
                 "num_iteration_opt=", 7, "\n",
                 "num_tree_search=", 5, "\n",
                 "min_pop=", 8, "\n",
                 #"opt_stepsize=", 0.7, "\n",
                 #"opt_interval=", 50, "\n",
                 "test_interval=",10000,"\n",
                 "max_leaf_forest=",10000,"\n",
                 "Verbose","\n")
  writeLines(pars, trainSettings)
  close(trainSettings)
  
  write.table(
    data.train,
    file=train.x.file,
    row.names = F, quote = F, na = "", sep = " ",
    append = F, col.names = F
  )
  
  write.table(
    data.test,
    file=test.x.file,
    row.names = F, quote = F, na = "", sep = " ",
    append = F, col.names = F
  )
  
  data.pred <- c()
  for (target in paste0("Class_", c(1:9))) {
    cat("Processing target:", target, "...\n")
    write.table((data.actual[[target]]-0.5)*2,
                file = train.y.file,
                quote = FALSE,
                row.names = FALSE,
                col.names = FALSE)
    
    system(paste("perl ../rgf1.2/test/call_exe.pl",
                 "../rgf1.2/bin/rgf train",
                 train.settings.file,
                 ">>", paste0(output.rgf.path, "rgf.", prefix, ".log"), "2>&1"))
    
    models <- list.files(output.rgf.path, pattern=paste0("^",prefix.model))
    ix <- which.max(sapply(strsplit(models, "-"), function(x) as.numeric(x[2])))
    model <- models[ix]
    
    testSettings<-file(paste0(test.settings.file, ".inp"))
    pars <- paste0("test_x_fn=",test.x.file,"\n",
                   "model_fn=",output.rgf.path, model,"\n",
                   "prediction_fn=", test.y.file,"\n")
    writeLines(pars, testSettings)
    close(testSettings)
    
    system(paste("perl ../rgf1.2/test/call_exe.pl",
                 "../rgf1.2/bin/rgf predict",
                 test.settings.file,
                 ">>", paste0(output.rgf.path, "rgf.", prefix, ".log"), "2>&1"))
    pred <- 1/(1+exp(-scan(test.y.file)))
    data.pred <- cbind(data.pred, pred)
  }
  colnames(data.pred) <- paste0("Class_", c(1:9))
  data.pred <- as.data.table(data.pred)
  data.pred[, id := ids.test]
  if (k > nfolds) {
    data.pred[, train := 0]
  } else {
    data.pred[, train := 1]
  }
  #fn.clean.worker()
  data.pred
}
fn.kill.wk()

setkey(rgf.pred, train, id)
data.train <- rgf.pred[train==1]
data.train[, train := NULL]
data.test <- rgf.pred[train==0]
data.test[, train := NULL]

write.csv(as.data.frame(data.train),
          file=paste0(output.r.path, "train/model.rgf.csv"),
          quote=FALSE,
          row.names=FALSE)

write.csv(as.data.frame(data.test),
          file=paste0(output.r.path, "test/model.rgf.csv"),
          quote=FALSE,
          row.names=FALSE)

# checking multilogloss
data.actual <- data.all[!is.na(target)][, paste0("Class_", c(1:9)),with=F]
data.actual <- as.matrix(data.actual)
data.pred <- copy(data.train)
data.pred[, id := NULL]
data.pred <- as.matrix(data.pred)
data.pred <- data.pred/rowSums(data.pred)
cat("Logloss for rgf:", fn.multilogloss(data.actual, data.pred), "\n")
