source("utils.R")

# read train dataset
data.train <- fread(paste0(data.path, train.file))

# read test dataset
data.test <- fread(paste0(data.path, test.file))

data.all <- rbindlist(list(data.train, data.test), use.names=TRUE, fill=TRUE)
rm(data.train, data.test)
gc()

flist <- setdiff(colnames(data.all), c("id", "target"))
data.stats <- fn.stats(data.all, flist, "target")

for (target.value in unique(data.all$target)) {
  if (is.na(target.value)) next
  ind <- substr(target.value, 7, nchar(target.value))
  data.all[, target_new := ifelse(target == target.value, 1, 0)]
  setnames(data.all, "target_new", paste0("Class_",ind))
}

save(data.all, file=paste0(output.r.path, "data.all.RData"))
