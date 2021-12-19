data <- read.csv("trainGTZAN.csv", check.names = FALSE)

source("evalAttr.R")

data <- data[,c(ncol(data),(2:ncol(data)-1))]
data$class <- as.factor(data$class)
#head(data)

featureScore <- sort(evalAttr(class ~ ., data, "ReliefF", reliefIters=1000, reliefK = 100), decreasing = T)

featureDf = data.frame(featureScore)

write.table(featureDf,file="featureImportance.csv",quote=F, sep=";", col.names=F)
