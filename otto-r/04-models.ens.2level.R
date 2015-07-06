pred.xgb.ens <- fread("../data/output-py/ens_1level/model.xgb.ens.csv")
setkey(pred.xgb.ens, id)
ids <- pred.xgb.ens$id
pred.xgb.ens[,id := NULL]
pred.xgb.ens <- as.matrix(pred.xgb.ens)

pred.nn.ens <- fread("../data/output-py/ens_1level/model.nn.ens.csv")
setkey(pred.nn.ens, id)
pred.nn.ens[,id := NULL]
pred.nn.ens <- as.matrix(pred.nn.ens)

pred.xgbx.ens <- fread("../data/output-py/ens_1level/model.xgbx.ens.csv")
setkey(pred.xgbx.ens, id)
pred.xgbx.ens[,id := NULL]
pred.xgbx.ens <- as.matrix(pred.xgbx.ens)

pred.ens <- pred.xgb.ens^(0.4) * pred.nn.ens^(0.3) * pred.xgbx.ens^(0.3)
pred.ens <- as.data.table(pred.ens)
pred.ens[, id := ids]
fn.write.batches.csv(pred.ens, "../data/submission/model.ens.csv", col.names=T, sep=",", nchunks=1)
