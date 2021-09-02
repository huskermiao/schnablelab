alpha = 0.001
pvSort = sort(pvalue[, 2])
pvOrder = order(pvalue[, 2])
compare = (pvSort < alpha * c(1 : dim(pvalue)[1]) / dim(pvalue)[1])
index = order(compare)[dim(result)[1]]
Out = pvalue[pvalue[, 2] <= pvSort[index], 1]
write.csv(Out, "SignificantBioR.csv", row.names = FALSE, col.names = FALSE)
