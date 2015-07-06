source("utils.R")
load(paste0(output.r.path, "data.all.RData"))
data.actual <- as.matrix(data.all[!is.na(target)][,paste0("Class_", c(1:9)),with=F])

data.train.linear <- fread(paste0(output.py.path, "train/model.linear.csv"))
data.test.linear <- fread(paste0(output.py.path, "test/model.linear.csv"))
data.train.linear.red <- fread(paste0(output.py.path, "train/model.linear.red.csv"))
data.test.linear.red <- fread(paste0(output.py.path, "test/model.linear.red.csv"))
data.train.nne <- fread(paste0(output.py.path, "train/model.nne.csv"))
data.test.nne <- fread(paste0(output.py.path, "test/model.nne.csv"))
data.train.xgbx <- fread(paste0(output.py.path, "train/model.xgbx.csv"))
data.test.xgbx <- fread(paste0(output.py.path, "test/model.xgbx.csv"))
data.train.xgbz <- fread(paste0(output.py.path, "train/model.xgbz.csv"))
data.test.xgbz <- fread(paste0(output.py.path, "test/model.xgbz.csv"))
data.train.svc <- fread(paste0(output.py.path, "train/model.svc.csv"))
data.test.svc <- fread(paste0(output.py.path, "test/model.svc.csv"))
data.train.et <- fread(paste0(output.py.path, "train/model.et.csv"))
data.test.et <- fread(paste0(output.py.path, "test/model.et.csv"))
data.train.nnb <- fread(paste0(output.py.path, "train/model.nnb.csv"))
data.test.nnb <- fread(paste0(output.py.path, "test/model.nnb.csv"))
data.train.nnc <- fread(paste0(output.py.path, "train/model.nnc.csv"))
data.test.nnc <- fread(paste0(output.py.path, "test/model.nnc.csv"))
data.train.nnd <- fread(paste0(output.py.path, "train/model.nnd.csv"))
data.test.nnd <- fread(paste0(output.py.path, "test/model.nnd.csv"))
data.train.knn2 <- fread(paste0(output.py.path, "train/model.knn2.csv"))
data.test.knn2 <- fread(paste0(output.py.path, "test/model.knn2.csv"))
data.train.knn2.red <- fread(paste0(output.py.path, "train/model.knn2.red.csv"))
data.test.knn2.red <- fread(paste0(output.py.path, "test/model.knn2.red.csv"))
data.train.knn4 <- fread(paste0(output.py.path, "train/model.knn4.csv"))
data.test.knn4 <- fread(paste0(output.py.path, "test/model.knn4.csv"))
data.train.knn4.red <- fread(paste0(output.py.path, "train/model.knn4.red.csv"))
data.test.knn4.red <- fread(paste0(output.py.path, "test/model.knn4.red.csv"))
data.train.knn5 <- fread(paste0(output.py.path, "train/model.knn5.csv"))
data.test.knn5 <- fread(paste0(output.py.path, "test/model.knn5.csv"))
data.train.knn5.red <- fread(paste0(output.py.path, "train/model.knn5.red.csv"))
data.test.knn5.red <- fread(paste0(output.py.path, "test/model.knn5.red.csv"))
data.train.knn6 <- fread(paste0(output.py.path, "train/model.knn6.csv"))
data.test.knn6 <- fread(paste0(output.py.path, "test/model.knn6.csv"))
data.train.knn6.red <- fread(paste0(output.py.path, "train/model.knn6.red.csv"))
data.test.knn6.red <- fread(paste0(output.py.path, "test/model.knn6.red.csv"))
data.train.knn8 <- fread(paste0(output.py.path, "train/model.knn8.csv"))
data.test.knn8 <- fread(paste0(output.py.path, "test/model.knn8.csv"))
data.train.knn8.red <- fread(paste0(output.py.path, "train/model.knn8.red.csv"))
data.test.knn8.red <- fread(paste0(output.py.path, "test/model.knn8.red.csv"))
data.train.rgf <- fread(paste0(output.r.path, "train/model.rgf.csv"))
data.test.rgf <- fread(paste0(output.r.path, "test/model.rgf.csv"))
data.train.h2o <- fread(paste0(output.r.path, "train/model.h2o.csv"))
data.test.h2o <- fread(paste0(output.r.path, "test/model.h2o.csv"))

##############################
###### XGBOOST ENSEMBLING ####
##############################
data.train <- copy(data.train.linear)
data.train <- merge(data.train, data.train.nne, by="id")
data.train <- merge(data.train, data.train.xgbx, by="id")
data.train <- merge(data.train, data.train.et, by="id")
data.train <- merge(data.train, data.train.svc, by="id")
data.train <- merge(data.train, data.train.nnb, by="id")
data.train <- merge(data.train, data.train.nnd, by="id")
data.train <- merge(data.train, data.train.nnc, by="id")
data.train <- merge(data.train, data.train.knn2, by="id")
data.train <- merge(data.train, data.train.knn4, by="id")
data.train <- merge(data.train, data.train.knn5, by="id")
data.train <- merge(data.train, data.train.knn6, by="id")
data.train <- merge(data.train, data.train.knn8, by="id")
data.train <- merge(data.train, data.train.rgf, by="id")
#data.train <- merge(data.train, data.train.h2o, by="id")
cols <- setdiff(colnames(data.train), "id")
for (col in cols) {
  setnames(data.train, col, "feature")
  data.train[feature<0.0000001, feature := 0.0000001]
  data.train[feature>0.9999999, feature := 0.9999999]
  data.train[, feature := -log((1-feature)/feature)]
  setnames(data.train, "feature", col)
}
data.train <- merge(data.train, data.all[!is.na(target)][,c("id", "target"),with=F], by="id")

data.test <- copy(data.test.linear)
data.test <- merge(data.test, data.test.nne, by="id")
data.test <- merge(data.test, data.test.xgbx, by="id")
data.test <- merge(data.test, data.test.et, by="id")
data.test <- merge(data.test, data.test.svc, by="id")
data.test <- merge(data.test, data.test.nnb, by="id")
data.test <- merge(data.test, data.test.nnd, by="id")
data.test <- merge(data.test, data.test.nnc, by="id")
data.test <- merge(data.test, data.test.knn2, by="id")
data.test <- merge(data.test, data.test.knn4, by="id")
data.test <- merge(data.test, data.test.knn5, by="id")
data.test <- merge(data.test, data.test.knn6, by="id")
data.test <- merge(data.test, data.test.knn8, by="id")
data.test <- merge(data.test, data.test.rgf, by="id")
#data.test <- merge(data.test, data.test.h2o, by="id")
cols <- setdiff(colnames(data.test), "id")
for (col in cols) {
  setnames(data.test, col, "feature")
  data.test[feature<0.0000001, feature := 0.0000001]
  data.test[feature>0.9999999, feature := 0.9999999]
  data.test[, feature := -log((1-feature)/feature)]
  setnames(data.test, "feature", col)
}

fn.write.batches.csv(data.train, paste0(output.r.path, "train.stack.csv"), col.names=T, sep=",", nchunks=1)
fn.write.batches.csv(data.test, paste0(output.r.path, "test.stack.csv"), col.names=T, sep=",", nchunks=1)

system(
  paste(
    "cd ../otto-py && python -u model.xgb.ens.py",
    "--train", paste0(output.r.path, "train.stack.csv"),
    "--test", paste0(output.r.path, "test.stack.csv"),
    "--pred", "model.xgb.ens.csv",
    " >> ", paste0("../data/log/xgb.ens.log"), " 2>&1"
  )
)  

##################################
##### NEURAL NET ENSEMBLING ######
##################################

data.train <- copy(data.train.linear.red)
data.train <- merge(data.train, data.train.nne, by="id")
data.train <- merge(data.train, data.train.xgbx, by="id")
data.train <- merge(data.train, data.train.et, by="id")
data.train <- merge(data.train, data.train.nnb, by="id")
data.train <- merge(data.train, data.train.nnd, by="id")
data.train <- merge(data.train, data.train.nnc, by="id")
data.train <- merge(data.train, data.train.rgf, by="id")
data.train <- merge(data.train, data.train.xgbz, by="id")
data.train <- merge(data.train, data.train.h2o, by="id")
cols <- setdiff(colnames(data.train), "id")
for (col in cols) {
  setnames(data.train, col, "feature")
  data.train[feature<0.0000001, feature := 0.0000001]
  data.train[feature>0.9999999, feature := 0.9999999]
  data.train[, feature := -log((1-feature)/feature)]
  setnames(data.train, "feature", col)
}
data.train <- merge(data.train, data.train.knn2.red, by="id")
data.train <- merge(data.train, data.train.knn4.red, by="id")
data.train <- merge(data.train, data.train.knn5.red, by="id")
data.train <- merge(data.train, data.train.knn6.red, by="id")
data.train <- merge(data.train, data.train.knn8.red, by="id")
data.train <- merge(data.train, data.all[!is.na(target)][,c("id", "target"),with=F], by="id")

data.test <- copy(data.test.linear.red)
data.test <- merge(data.test, data.test.nne, by="id")
data.test <- merge(data.test, data.test.xgbx, by="id")
data.test <- merge(data.test, data.test.et, by="id")
data.test <- merge(data.test, data.test.nnb, by="id")
data.test <- merge(data.test, data.test.nnd, by="id")
data.test <- merge(data.test, data.test.nnc, by="id")
data.test <- merge(data.test, data.test.rgf, by="id")
data.test <- merge(data.test, data.test.xgbz, by="id")
data.test <- merge(data.test, data.test.h2o, by="id")
cols <- setdiff(colnames(data.test), "id")
for (col in cols) {
  setnames(data.test, col, "feature")
  data.test[feature<0.0000001, feature := 0.0000001]
  data.test[feature>0.9999999, feature := 0.9999999]
  data.test[, feature := -log((1-feature)/feature)]
  setnames(data.test, "feature", col)
}
data.test <- merge(data.test, data.test.knn2.red, by="id")
data.test <- merge(data.test, data.test.knn4.red, by="id")
data.test <- merge(data.test, data.test.knn5.red, by="id")
data.test <- merge(data.test, data.test.knn6.red, by="id")
data.test <- merge(data.test, data.test.knn8.red, by="id")

fn.write.batches.csv(data.train, paste0(output.r.path, "train.stack.csv"), col.names=T, sep=",", nchunks=1)
fn.write.batches.csv(data.test, paste0(output.r.path, "test.stack.csv"), col.names=T, sep=",", nchunks=1)

system(
  paste(
    "cd ../otto-py && python -u model.nn.ens.py",
    "--train", paste0(output.r.path, "train.stack.csv"),
    "--test", paste0(output.r.path, "test.stack.csv"),
    "--pred", "model.nn.ens.csv",
    "--epoch", 12,
    " >> ", paste0("../data/log/nn.ens.log"), " 2>&1"
  )
)  

####################################################################
############## COMBINED DATASET IDEA ###############################
####################################################################

data.train <- copy(data.train.linear)
data.train <- merge(data.train, data.train.nne, by="id")
data.train <- merge(data.train, data.train.xgbx, by="id")
data.train <- merge(data.train, data.train.et, by="id")
data.train <- merge(data.train, data.train.svc, by="id")
data.train <- merge(data.train, data.train.nnb, by="id")
data.train <- merge(data.train, data.train.nnd, by="id")
data.train <- merge(data.train, data.train.nnc, by="id")
data.train <- merge(data.train, data.train.rgf, by="id")
data.train <- merge(data.train, data.train.xgbz, by="id")
cols <- setdiff(colnames(data.train), "id")
for (col in cols) {
  setnames(data.train, col, "feature")
  data.train[feature<0.0000001, feature := 0.0000001]
  data.train[feature>0.9999999, feature := 0.9999999]
  data.train[, feature := -log((1-feature)/feature)]
  setnames(data.train, "feature", col)
}
data.train <- merge(data.train, data.train.knn2, by="id")
data.train <- merge(data.train, data.train.knn4, by="id")
data.train <- merge(data.train, data.train.knn5, by="id")
data.train <- merge(data.train, data.train.knn6, by="id")
data.train <- merge(data.train, data.train.knn8, by="id")
data.train <- merge(data.train, data.all[!is.na(target)][,c("id", "target"),with=F], by="id")

data.test <- copy(data.test.linear)
data.test <- merge(data.test, data.test.nne, by="id")
data.test <- merge(data.test, data.test.xgbx, by="id")
data.test <- merge(data.test, data.test.et, by="id")
data.test <- merge(data.test, data.test.svc, by="id")
data.test <- merge(data.test, data.test.nnb, by="id")
data.test <- merge(data.test, data.test.nnd, by="id")
data.test <- merge(data.test, data.test.nnc, by="id")
data.test <- merge(data.test, data.test.rgf, by="id")
data.test <- merge(data.test, data.test.xgbz, by="id")
cols <- setdiff(colnames(data.test), "id")
for (col in cols) {
  setnames(data.test, col, "feature")
  data.test[feature<0.0000001, feature := 0.0000001]
  data.test[feature>0.9999999, feature := 0.9999999]
  data.test[, feature := -log((1-feature)/feature)]
  setnames(data.test, "feature", col)
}
data.test <- merge(data.test, data.test.knn2, by="id")
data.test <- merge(data.test, data.test.knn4, by="id")
data.test <- merge(data.test, data.test.knn5, by="id")
data.test <- merge(data.test, data.test.knn6, by="id")
data.test <- merge(data.test, data.test.knn8, by="id")

data.class.list <- list()
for (i in 1:9) {
  predicted.class <- paste0("Class_",i)
  ix <- which(grepl(predicted.class, colnames(data.train)))
  cols <- colnames(data.train)[ix]
  data.class <- data.train[,cols,with=F]
  cols <- gsub(paste0("_",i,"_"), "_", colnames(data.class))
  setnames(data.class, colnames(data.class), cols)
  #data.class <- -log(data.class)
  data.class[, id := data.train$id]
  data.class <- data.class[,c("id",cols),with=F]
  
  for (j in 1:9) {
    if (i == j) {
      data.class[, indicator := 1]
    } else {
      data.class[, indicator := 0]
    }
    setnames(data.class, "indicator", paste0("Class_",j,"_ind"))
  }
  data.class[, target := 0]
  ix <- which(data.train$target == predicted.class)
  data.class[ix, target := 1]
  data.class.list[[length(data.class.list)+1]] <- data.class
}
data.train.ext <- rbindlist(data.class.list, use.names=F, fill=F)


data.class.list <- list()
for (i in 1:9) {
  predicted.class <- paste0("Class_",i)
  ix <- which(grepl(predicted.class, colnames(data.test)))
  cols <- colnames(data.test)[ix]
  data.class <- data.test[,cols,with=F]
  cols <- gsub(paste0("_",i,"_"), "_", colnames(data.class))
  setnames(data.class, colnames(data.class), cols)
  #data.class <- -log(data.class)
  data.class[, id := data.test$id]
  data.class <- data.class[,c("id",cols),with=F]
  
  for (j in 1:9) {
    if (i == j) {
      data.class[, indicator := 1]
    } else {
      data.class[, indicator := 0]
    }
    setnames(data.class, "indicator", paste0("Class_",j,"_ind"))
  }
  data.class.list[[length(data.class.list)+1]] <- data.class
}
data.test.ext <- rbindlist(data.class.list, use.names=F, fill=F)

fn.write.batches.csv(data.train.ext, paste0(output.r.path, "train.stack.csv"), col.names=T, sep=",", nchunks=1)
fn.write.batches.csv(data.test.ext, paste0(output.r.path, "test.stack.csv"), col.names=T, sep=",", nchunks=1)

system(
  paste(
    "cd ../otto-py && python -u model.xgbx.ens.py",
    "--train", paste0(output.r.path, "train.stack.csv"),
    "--test", paste0(output.r.path, "test.stack.csv"),
    "--pred", "model.xgbx.ens.csv",
    " >> ", paste0("../data/log/xgbx.ens.log"), " 2>&1"
  )
)  

