source("utils.R")
load(paste0(output.r.path, "data.all.RData"))
data.actual <- as.matrix(data.all[!is.na(target)][,paste0("Class_", c(1:9)),with=F])

#####################################
############ LINEAR MODELS ##########
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.linear.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.linear.csv",
    "--epoch", 1,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/linear.stack.log"), " 2>&1"
  )
)
combine.preds.linear("train_raw/model.linear")

system(
  paste(
    "cd ../otto-py && python -u model.linear.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.linear.csv",
    "--epoch", 1,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/linear.stack.log"), " 2>&1"
  )
)
combine.preds.linear("test_raw/model.linear")

#####################################
########### NEURAL NET E ############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.nne.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nne.csv",
    "--epoch", 50,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/nne.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.nne", 50)

system(
  paste(
    "cd ../otto-py && python -u model.nne.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nne.csv",
    "--epoch", 50,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/nne.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.nne", 50)

#####################################
########### XGBOOST X ###############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.xgbx.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.xgbx.csv",
    "--epoch", 10,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/xgbx.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.xgbx", 10)

system(
  paste(
    "cd ../otto-py && python -u model.xgbx.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.xgbx.csv",
    "--epoch", 10,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/xgbx.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.xgbx", 10)

#####################################
########### XGBOOST Z ###############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.xgbz.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.xgbz.csv",
    "--epoch", 10,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/xgbz.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.xgbz", 10)

system(
  paste(
    "cd ../otto-py && python -u model.xgbz.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.xgbz.csv",
    "--epoch", 10,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/xgbz.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.xgbz", 10)

#####################################
########### EXTRA TREES #############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.extratrees.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.et.csv",
    "--epoch", 5,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/et.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.et", 5)

system(
  paste(
    "cd ../otto-py && python -u model.extratrees.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.et.csv",
    "--epoch", 5,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/et.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.et", 5)

#####################################
############### SVC #################
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.svc.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.svc.csv",
    "--epoch", 1,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/svc.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.svc", 1)

system(
  paste(
    "cd ../otto-py && python -u model.svc.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.svc.csv",
    "--epoch", 1,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/svc.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.svc", 1)

#####################################
########### NEURAL NET B ############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.nnb.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnb.csv",
    "--epoch", 30,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/nnb.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.nnb", 30)

system(
  paste(
    "cd ../otto-py && python -u model.nnb.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnb.csv",
    "--epoch", 30,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/nnb.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.nnb", 30)

#####################################
########### NEURAL NET C ############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.nnc.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnc.csv",
    "--epoch", 30,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/nnc.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.nnc", 30)

system(
  paste(
    "cd ../otto-py && python -u model.nnc.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnc.csv",
    "--epoch", 30,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/nnc.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.nnc", 30)

#####################################
########### NEURAL NET D ############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.nnd.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnd.csv",
    "--epoch", 30,
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/nnd.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.nnd", 30)

system(
  paste(
    "cd ../otto-py && python -u model.nnd.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.nnd.csv",
    "--epoch", 30,
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/nnd.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.nnd", 30)

#####################################
################ KNN 2 ##############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.knn2.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn2.csv",
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/knn2.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.knn2", 11, save.original=T)
combine.preds("train_raw/model.knn2", 11, suffix=".red")

system(
  paste(
    "cd ../otto-py && python -u model.knn2.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn2.csv",
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/knn2.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.knn2", 11, save.original=T)
combine.preds("test_raw/model.knn2", 11, suffix=".red")

#####################################
################ KNN 4 ##############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.knn4.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn4.csv",
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/knn4.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.knn4", 11, save.original=T)
combine.preds("train_raw/model.knn4", 11, suffix=".red")

system(
  paste(
    "cd ../otto-py && python -u model.knn4.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn4.csv",
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/knn4.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.knn4", 11, save.original=T)
combine.preds("test_raw/model.knn4", 11, suffix=".red")

#####################################
################ KNN 5 ##############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.knn5.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn5.csv",
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/knn5.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.knn5", 11, save.original=T)
combine.preds("train_raw/model.knn5", 11, suffix=".red")

system(
  paste(
    "cd ../otto-py && python -u model.knn5.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn5.csv",
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/knn5.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.knn5", 11, save.original=T)
combine.preds("test_raw/model.knn5", 11, suffix=".red")

#####################################
################ KNN 6 ##############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.knn6.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn6.csv",
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/knn6.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.knn6", 11, save.original=T)
combine.preds("train_raw/model.knn6", 11, suffix=".red")

system(
  paste(
    "cd ../otto-py && python -u model.knn6.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn6.csv",
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/knn6.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.knn6", 11, save.original=T)
combine.preds("test_raw/model.knn6", 11, suffix=".red")

#####################################
################ KNN 8 ##############
#####################################

system(
  paste(
    "cd ../otto-py && python -u model.knn8.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn8.csv",
    "--cv", 1,
    "--folds", 10,
    " >> ", paste0("../data/log/knn8.stack.log"), " 2>&1"
  )
)
combine.preds("train_raw/model.knn8", 11, save.original=T)
combine.preds("train_raw/model.knn8", 11, suffix=".red")

system(
  paste(
    "cd ../otto-py && python -u model.knn8.stack.py",
    "--train", paste0(data.path, train.file),
    "--test", paste0(data.path, test.file),
    "--pred", "model.knn8.csv",
    "--cv", 0,
    "--folds", 0,
    " >> ", paste0("../data/log/knn8.stack.log"), " 2>&1"
  )
)
combine.preds("test_raw/model.knn8", 11, save.original=T)
combine.preds("test_raw/model.knn8", 11, suffix=".red")

#####################################
################ RGF ################
#####################################
source("model.rgf.stack.R")

#####################################
################ H2O ################
#####################################
source("model.h2o.stack.R")
