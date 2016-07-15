source("utils.parallel.R")

#memory.limit(1000000)

library(data.table)
library(verification)
library(cvTools)
library(Matrix)
library(h2o)

data.path <- "../data/input/"
output.r.path <- "../data/output-r/"
output.py.path <- "../data/output-py/"
output.rgf.path <- "../data/output-rgf/"
train.file <- "train.csv"
test.file <- "test.csv"

#############################################################
# tic toc
#############################################################
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")) {
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function() {
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}

fn.stats <- function(data.all, flist.col, target.col) {
  setnames(data.all, target.col, "target")
  data.stats <- c()
  for (f in flist.col) {
    setnames(data.all, f, "feature")
    stats.row <- c(length(unique(data.all[,feature])),
                   length(unique(data.all[!is.na(target),feature])),
                   length(unique(data.all[is.na(target),feature])),
                   max(data.all[,feature]),
                   max(data.all[!is.na(target),feature]),
                   max(data.all[is.na(target),feature]),
                   min(data.all[,feature]),
                   min(data.all[!is.na(target),feature]),
                   min(data.all[is.na(target),feature]),
                   mean(data.all[,feature]),
                   mean(data.all[!is.na(target),feature]),
                   mean(data.all[is.na(target),feature]),
                   median(data.all[,feature]),
                   median(data.all[!is.na(target),feature]),
                   median(data.all[is.na(target),feature]),
                   sd(data.all[,feature]),
                   sd(data.all[!is.na(target),feature]),
                   sd(data.all[is.na(target),feature]))
    data.stats <- rbind(data.stats, stats.row)
    setnames(data.all, "feature", f)
  }
  colnames(data.stats) <- c("unique","unique_tr","unique_test",
                            "max","max_tr","max_test",
                            "min","min_tr","min_test",
                            "mean","mean_tr","mean_test",
                            "median", "median_tr", "median_test",
                            "sd", "sd_tr", "sd_test")
  data.stats <- as.data.table(data.stats)
  data.stats[, feature := flist.col]
  setnames(data.all, "target", target.col)
  return (data.stats)
}

fn.multilogloss <- function(data.actual, data.predicted) {
  actual <- as.matrix(data.actual)
  predicted <- as.matrix(data.predicted)
  probs <- rowSums(actual*predicted)
  probs[which(probs>0.999999)] <- 0.999999
  probs[which(probs<0.000001)] <- 0.000001  
  return(-(1/nrow(actual))*sum(log(probs)))
}

fn.logloss <- function(actual, predicted, pred.min=0.000001, pred.max=0.999999) {
  predicted[which(predicted > pred.max)] <- pred.max
  predicted[which(predicted < pred.min)] <- pred.min
  error <- sum(-actual*log(predicted) - (1-actual)*log(1-predicted))/length(actual)
  return (error)
}

fn.mcrmse <- function(actual, predicted) {
  if (is.vector(predicted) & is.vector(actual)) {
    ix <- which(!is.na(actual))
    nsamples <- length(ix)
    return (sqrt(sum((actual[ix] - predicted[ix])^2)/nsamples))
  }
  if (ncol(actual) != ncol(predicted)) return (NULL)
  if (nrow(actual) != nrow(predicted)) return (NULL)
  ix <- which(!is.na(actual[,1]))
  nsamples <- length(ix)
  error <- 0
  #cat("Errors by targets:")
  errors <- c()
  for (i in 1:ncol(actual)) {
    error.col <- sqrt(sum((actual[ix,i] - predicted[ix,i])^2)/nsamples)
    errors <- c(errors, error.col)
    error <- error + error.col
    #cat(colnames(actual)[i],":",error.col,";")
  }
  #cat("\n")
  errors <- c(errors, error/ncol(actual))
  return (errors)
}

fn.memory.usage <- function() {
  return (sum(sort( sapply(ls(globalenv()),function(x){object.size(get(x))}))))
}

fn.write.libsvm <- function(
  data.tr, 
  data.test, 
  name,
  fn.y.transf = NULL, 
  dir = "libfm",
  col.y = "click",
  col.x = colnames(data.tr)[!(colnames(data.tr) %in% c(col.y, col.qid))],
  col.qid = NULL,
  feat.start = 1,
  vw.mode = F,
  data.val = NULL,
  y.def = min(data.tr[[col.y]]))
{
  options(scipen=999)
  library("data.table")
  cat("Building feature map ...")
  model.dts <- list()
  val.xi <- feat.start
  col.x.groups <- NULL
  feat.size <- 0
  for (i in (1:length(col.x))) {
    col <- col.x[i]
    if (is.factor(data.tr[[col]])) {
      
      col.ids.tr   <- as.factor(levels(data.tr[[col]]))
      model.dts[[col]] <- data.table(ID = col.ids.tr, key = "ID")
      if (!is.null(data.val)) {
        col.ids.val <- as.factor(levels(data.val[[col]]))
        model.dts[[col]] <- merge(model.dts[[col]], 
                                  data.table(ID = col.ids.val, key = "ID"))
      }
      if (!is.null(data.test)) {
        col.ids.test <- as.factor(levels(data.test[[col]]))
        model.dts[[col]] <- merge(model.dts[[col]], 
                                  data.table(ID = col.ids.test, key = "ID"))
      }
      feat.size <- feat.size + nrow(model.dts[[col]])
      
      model.dts[[col]]$X.Idx <- val.xi:(val.xi+nrow(model.dts[[col]])-1)
      
      val.xi <- val.xi+nrow(model.dts[[col]])
      
    } else {
      model.dts[[col]] <- val.xi
      val.xi <- val.xi + 1
    }
  }
  map.name <- paste(name, ".map", sep="")
  assign(map.name, model.dts)
  save(list = map.name, file = paste0(dir, "/", map.name, ".RData"))
  cat("done \n")
  
  if (!exists("cmpfun")) {
    cmpfun <- identity
  }
  write.file <- cmpfun(function (data, file) { 
    col.chunk <- col.x
    if (!is.null(col.qid)) {
      col.chunk <- c(col.chunk, col.qid)
    }
    if (!is.null(data[[col.y]])) {
      col.chunk <- c(col.y, col.chunk)
    }
    unlink(file)
    cat("Saving ", file, "...")
    fileConn <- file(file, open="at")
    
    data.chunk <- data[, col.chunk]
    if (is.null(data.chunk[[col.y]])) {
      data.chunk[[col.y]] <- 0
    }
    data.chunk[[col.y]][is.na(data.chunk[[col.y]])] <- y.def
    if (!is.null(fn.y.transf)) {
      data.chunk[[col.y]] <- fn.y.transf(data.chunk[[col.y]])
    }
    data.chunk[[col.y]][data.chunk[[col.y]] == Inf] <- y.def
    
    for (col in col.x) {
      if (is.numeric(data.chunk[[col]])) {
        data.chunk[[col]][data.chunk[[col]] == 0] <- NA
      }
    }
    
    for (col in col.x) {
      if (is.factor(data.chunk[[col]])) {
        data.chunk[[col]] <-  paste(
          model.dts[[col]][J(data.chunk[[col]])]$X.Idx,
          c(1), sep = ":")
      } else {
        data.chunk[[col]] <- paste(
          rep(model.dts[[col]], nrow(data.chunk)),
          data.chunk[[col]], sep = ":")
      }
    }
    
    if (!is.null(col.qid)) {
      data.chunk[[col.qid]] <- paste("qid", data.chunk[[col.qid]], sep = ":")
    }
    
    data.chunk <- do.call(paste, data.chunk[, c(col.y, col.qid, col.x)])
    chunk.size <- as.numeric(object.size(data.chunk))
    chunk.size.ch <- T
    while (chunk.size.ch) {
      data.chunk <- gsub(" [0-9]+\\:?NA", "", data.chunk)
      data.chunk <- gsub(" NA\\:-?[0-9]+", "", data.chunk)
      chunk.size.ch <- chunk.size != as.numeric(object.size(data.chunk))
      chunk.size <- as.numeric(object.size(data.chunk))
    }
    data.chunk <- gsub("\\s+", " ", data.chunk)
    data.chunk <- gsub("^([0-9]+(\\.[0-9]+)?)\\s*$", paste0("\\1 ", val.xi, ":1"), data.chunk)
    
    if (vw.mode) {
      data.chunk <- gsub("^([-]?[0-9]+(\\.[0-9]+)?)\\s+", "\\1 | ", data.chunk)
    }
    
    writeLines(c(data.chunk), fileConn)
    
    close(fileConn)
    cat("done.\n")
  })
  #     debug(write.file)
  write.file(data.tr, paste(dir, "/", name, ".tr.libsvm", sep=""))
  if (!is.null(data.val)) {
    write.file(data.val, paste(dir, "/", name, ".val.libsvm", sep=""))
  }
  if (!is.null(data.test)) {
    write.file(data.test, paste(dir, "/", name, ".test.libsvm", sep=""))
  }
}

fn.write.batches.csv <- function(data, train.file, col.names, sep, nchunks = 4, continue.chunks=FALSE) {
  options(scipen=999)
  if (nchunks == 1) {
    write.table(
      data,
      file=train.file,
      row.names = F, quote = F, na = "", sep = sep,
      append = FALSE, col.names = col.names
    )
  } else {
    nr <- nrow(data)
    ix <- seq(1, nr, round(nr/nchunks))
    if (ix[length(ix)] != nr) {
      ix <- c(ix, nr+1)
    } else {
      ix[length(ix)] <- nr+1
    }
    gc()
    for (i in 1:(length(ix)-1)) {
      cat("Processing chunk", i, "...\n")
      if (i == 1 & !continue.chunks) {
        write.table(
          data[ix[i]:(ix[i+1]-1),],
          file=train.file,
          row.names = F, quote = F, na = "", sep = sep,
          append = FALSE, col.names = col.names
        )
      } else {
        write.table(
          data[ix[i]:(ix[i+1]-1),],
          file=train.file,
          row.names = F, quote = F, na = "", sep = sep,
          append = TRUE, col.names = FALSE
        )
      } 
      invisible(gc())
    }
  }
}

fn.optim <- function(y, x) {
  
  x <- as.matrix(x)
  pars0 <- rep(0.0, ncol(x))
  
  #error to minimize
  fn.loss <- function(pars) {
    y.pred <- 1 / (1 + exp(-as.numeric(x %*% pars)))
    y.pred <- pmax(y.pred, 10^(-6))
    y.pred <- pmin(y.pred, 1-10^(-6))
    sum(-y*log(y.pred) - (1-y)*log(1-y.pred))/length(y)
  } 
  
  cat ("Initial loss:", fn.loss(pars0), "\n")
  opt.result <- optim(pars0, 
                      fn.loss, 
                      #method = "Brent",
                      #method = "L-BFGS-B",
                      #lower = 0.0,
                      #upper = 10.0,
                      control = list(trace = T,maxit=5000))
  return (opt.result$par)
}

combine.preds <- function(file.name, nepoch, save.original=F, suffix="") {
  is.train.part <- grepl("train", file.name)
  model.name <- unlist(strsplit(unlist(strsplit(file.name, "/"))[2], "\\."))[2]
  
  if (nepoch<2) {
    data.pred.epoch <- fread(paste0(output.py.path, file.name, ".epoch0.csv"))
    if ("set" %in% colnames(data.pred.epoch)) {
      data.pred.epoch <- data.pred.epoch[set==1]
      data.pred.epoch[, set := NULL]
    }
    setkey(data.pred.epoch, id)
    if (is.train.part) {
      data.pred <- data.pred.epoch[,paste0("Class_",c(1:9), "_", model.name),with=F]
      data.pred <- as.matrix(data.pred)
      data.pred <- data.pred/rowSums(data.pred)
      cat("Logloss for", model.name, fn.multilogloss(data.actual, data.pred), "\n")
      result.file <- paste0("train/model.", model.name, ".csv")
    } else {
      result.file <- paste0("test/model.", model.name, ".csv")
    }
    fn.write.batches.csv(data.pred.epoch, paste0(output.py.path, result.file), col.names=T, sep=",", nchunks=1)
  } else {
    for (i in 0:(nepoch-1)) {
      cat("Reading epoch", i, "...\n")
      data.pred.epoch <- fread(paste0(output.py.path, file.name, ".epoch",i,".csv"))
      if ("set" %in% colnames(data.pred.epoch)) {
        data.pred.epoch <- data.pred.epoch[set==1]
        data.pred.epoch[, set := NULL]
      }
      setkey(data.pred.epoch, id)
      cols <- setdiff(colnames(data.pred.epoch), "id")
      setnames(data.pred.epoch, cols, paste0(cols,"_epoch", i+1))
      if (i == 0) {
        ids <- data.pred.epoch$id
        data.pred.epoch[, id := NULL]
        data.pred <- as.matrix(data.pred.epoch)
      } else {
        data.pred.epoch[, id := NULL]
        data.pred <- cbind(data.pred, as.matrix(data.pred.epoch))
      }
    }
    if (save.original) {
      cols <- colnames(data.pred)
      data.pred.summary <- data.pred
      cols.summary <- paste0(cols, "_", model.name)
    } else {
      cols <- colnames(data.pred)
      data.pred.summary <- c()
      cols.summary <- c()
    }
    for (i in 1:9) {
      class.name <- paste0("Class_",i)
      class.cols <- cols[which(grepl(class.name, cols))]
      data.pred.summary <- cbind(data.pred.summary, apply(data.pred[,class.cols], 1, median))
      data.pred.summary <- cbind(data.pred.summary, apply(data.pred[,class.cols], 1, mean))
      data.pred.summary <- cbind(data.pred.summary, apply(data.pred[,class.cols], 1, max))
      cols.summary <- c(cols.summary, paste0(class.name, c("_median_", "_mean_", "_max_"), model.name))
    }
    colnames(data.pred.summary) <- cols.summary
    data.pred.summary <- as.data.table(data.pred.summary)
    data.pred.summary[, id := ids]
    
    # checking log loss
    if (is.train.part) {
      data.pred <- data.pred.summary[,paste0("Class_",c(1:9), "_mean_", model.name),with=F]
      data.pred <- as.matrix(data.pred)
      data.pred <- data.pred/rowSums(data.pred)
      cat("Logloss for", model.name, fn.multilogloss(data.actual, data.pred), "\n")
      result.file <- paste0("train/model.", model.name, suffix, ".csv")
    } else {
      result.file <- paste0("test/model.", model.name, suffix, ".csv")
    }
    fn.write.batches.csv(data.pred.summary, paste0(output.py.path, result.file), col.names=T, sep=",", nchunks=1)
  }
}

combine.preds.linear <- function(file.name) {
  is.train.part <- grepl("train", file.name)
  
  data.linear0 <- fread(paste0(output.py.path, file.name, ".epoch0.csv"))
  setkey(data.linear0, id)
  for (col in colnames(data.linear0)) {
    nas <- sum(is.na(data.linear0[[col]]))
    if (nas>0) {
      print (col)
      setnames(data.linear0, col, "col")
      data.linear0[is.na(col), col := 1]
      setnames(data.linear0, "col", col)
    }
  }
  
  ix1 <- which(grepl(paste0("LinModel",1,"$"), colnames(data.linear0)))
  ix2 <- which(grepl(paste0("LinModel",5,"$"), colnames(data.linear0)))
  ix3 <- which(grepl(paste0("LinModel",9,"$"), colnames(data.linear0)))
  cols <- c("id",
            colnames(data.linear0)[ix1],
            colnames(data.linear0)[ix2],
            colnames(data.linear0)[ix3])
  data.linear01 <- data.linear0[,cols,with=F]
  
  for (i in 1:9) {
    cols <- colnames(data.linear01)[grepl(paste0("Class_",i), colnames(data.linear01))]
    data.pred0 <- as.matrix(data.linear01[,cols,with=F])
    data.pred <- data.pred0
    data.pred <- cbind(data.pred, apply(data.pred0, 1, median))
    data.pred <- cbind(data.pred, apply(data.pred0, 1, mean))
    data.pred <- cbind(data.pred, apply(data.pred0, 1, max))
    colnames(data.pred) <- c(cols, 
                             paste0("Class_",i,"_median_LinModel"), 
                             paste0("Class_",i,"_mean_LinModel"),
                             paste0("Class_",i,"_max_LinModel"))
    data.pred <- as.data.table(data.pred)
    if (i == 1) {
      data.linear1 <- copy(data.pred)
    } else {
      data.linear1 <- cbind(data.linear1, data.pred)
    }
  }
  
  data.linear1[,id := data.linear01$id]

  # checking log loss
  if (is.train.part) {
    data.pred <- data.linear1[,paste0("Class_",c(1:9),"_mean_LinModel"),with=F]
    data.pred <- as.matrix(data.pred)
    data.pred <- data.pred/rowSums(data.pred)
    cat("Logloss for linear model:", fn.multilogloss(data.actual, data.pred), "\n")
    result.file <- paste0("train/", unlist(strsplit(file.name, "/"))[2], ".csv")
    result.file.red <- paste0("train/", unlist(strsplit(file.name, "/"))[2], ".red.csv")
  } else {
    result.file <- paste0("test/", unlist(strsplit(file.name, "/"))[2], ".csv")
    result.file.red <- paste0("test/", unlist(strsplit(file.name, "/"))[2], ".red.csv")
  }

  # save to files
  fn.write.batches.csv(data.linear1, paste0(output.py.path, result.file), col.names=T, sep=",", nchunks=1)
  ix <- which(grepl("mean|median|max", colnames(data.linear1)))
  cols <- colnames(data.linear1)[ix]
  data.linear1.reduced <- data.linear1[,c("id",cols),with=F]
  fn.write.batches.csv(data.linear1.reduced, paste0(output.py.path, result.file.red), col.names=T, sep=",", nchunks=1)
}

