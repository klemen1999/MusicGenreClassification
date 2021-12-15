setwd("D:/Šola/FRI MAG 1.letnik/Strojno uèenje/MusicGenreClassification/")
data <- read.csv("trainGTZAN.csv")
data$X <- NULL
#head(data)

source("D:/Šola/FRI MAG 1.letnik/Strojno uèenje/vaje/evalAttr.R")

data <- data[,c(ncol(data),(2:ncol(data)-1))]
#head(data)
data$class <- as.factor(data$class)

featureScore <- sort(evalAttr(class ~ ., data, "ReliefF", reliefIters=1000, reliefK = 100), decreasing = T)

featureDf = data.frame(featureScore)

write.table(featureDf,file="featureImportance.csv",quote=F, sep=";", col.names=F)
