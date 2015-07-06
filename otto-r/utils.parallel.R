require("compiler")
enableJIT(3) 
setCompilerOptions(suppressUndefined = T)
path.wd <- getwd()
options(max.print = 1000)
options(stringsAsFactors = FALSE)

##############################################################
## Create parallel workers
##############################################################
fn.register.wk <- function(n.proc = NULL) {
  if (is.null(n.proc)) {
    n.proc = as.integer(Sys.getenv("NUMBER_OF_PROCESSORS"))
    if (is.na(n.proc)) {
      library(parallel)
      n.proc <-detectCores()
    }
  }
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (!exists(".pworkers", envir=baseenv()) || length(workers$.pworkers) == 0) {
    
    library(doSNOW)
    library(foreach)
    workers<-suppressWarnings(makeSOCKcluster(n.proc));
    suppressWarnings(registerDoSNOW(workers))
    clusterSetupRNG(workers, seed=5478557)
    assign(".pworkers", workers, envir=baseenv());
    cat("Workers start... ", "\n")
  }
  invisible(workers);
}

##############################################################
## Kill parallel workers
##############################################################
fn.kill.wk <- function() {
  library("doSNOW")
  library("foreach")
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (exists(".pworkers", envir=baseenv()) && length(workers$.pworkers) != 0) {
    stopCluster(workers$.pworkers);
    assign(".pworkers", NULL, envir=baseenv());
    cat("Workers finish. ", "\n")
  }
  invisible(workers);
}

fn.init.worker <- function(log = NULL) {
  setwd(path.wd)
  if (!is.null(log)) {
    date.str <- format(Sys.time(), format = "%Y-%m-%d_%H-%M-%S")
    output.file <- paste0(log,".log")
    output.file <- file(output.file, open = "wt")
    sink(output.file)
    sink(output.file, type = "message")
    cat("Start:", date.str, "\n")
  }
}

fn.clean.worker <- function() {
  gc()
  suppressWarnings(sink())
  suppressWarnings(sink(type = "message"))
}
