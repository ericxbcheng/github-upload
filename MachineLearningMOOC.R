library(caret)
library(e1071)
library(kernlab)
library(ggplot2)
data(spam)

head(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75,list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

dim(training)

set.seed(32343)
modelFit <- train(type~., data=training, method="glm")
modelFit$finalModel

predictions <- predict(modelFit, newdata=testing)
predictions

confusionMatrix(predictions, testing$type)

#k-fold validation
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
sapply(folds, length)

#resampling
folds <- createResample(y=spam$type, times=10, list=TRUE)

#time slices
tme <- 1:1000
folds <- createTimeSlices(y=tme, initialWindow=20, horizon=10)
names(folds)

args(train)
args(trainControl)

#plotting
library(ISLR)
data(Wage)
summary(Wage)

inTrain <- createDataPartition(y = Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training)

featurePlot(x=training[,c("age","education", "jobclass")], y = training$wage, plot = "pairs")

qq <- qplot(age, wage, color = education, data=training)
qq + geom_smooth(method = "lm", formula = y~x)

library(Hmisc)
cutWage <- cut2(training$wage, g=4)
table(cutWage)

p1 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot"))
p1

t1 <- table(cutWage, training$jobclass)
t1

prop.table(t1,1)

#preprocess

#covariate creation
inTrain <- createDataPartition(y = Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]

table(training$jobclass)
dummies <- dummyVars(wage~jobclass, data=training)
head(predict(dummies, newdata=training))

library(splines)
bsBasis <- bs(training$age, df=3)
lm1 <- lm(wage~bsBasis, data=training)
plot(training$age, training$wage)
points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)

predict(bsBasis, age=testing$age)

#pca
preProc <- preProcess(log10(spam[,-58]+1), method="pca", pcaComp=2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
typeColor <- ((spam$type=="spam")*1+1)
plot(spamPC[,1], spamPC[,2], col=typeColor)

#tree model
library(caret)
library(ggplot2)
data(iris)
inTrain <- createDataPartition(y = iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
modFit <- train(Species~., method="rpart", data=training)
print(modFit$finalModel)
plot(modFit$finalModel, uniform=TRUE)
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=0.8)

library(rattle)
fancyRpartPlot(modFit$finalModel)

#bagging
install.packages("ElemStatLearn")
library(ElemStatLearn)
data(ozone, package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)

ll <- matrix(NA, nrow=10, ncol=155)
for(i in 1:10){
  ss <- sample(1:nrow(ozone), replace=T)
  ozone0 <- ozone[ss,]
  ozone0 <- ozone0[order(ozone0$ozone),]
  loess0 <- loess(temperature~ozone, data=ozone0, span=0.2)
  ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))
  
}

plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){lines(1:155, ll[i,], col="grey", lwd=2)}
lines(1:155, apply(X = ll,MARGIN = 2, mean), col="red", lwd=2)

#bagging advanced
predictor <- data.frame(ozone=ozone$ozone)
temperature <- ozone$temperature
treebag <- bag(predictor, temperaure, B = 10, bagControl = 
                 bagControl(fit = ctreeBag$fit,
                            predict = ctreeBag$pred,
                            aggregate = ctreeBag$aggregate))

#random forest
modFit <- train(Species~., data=training, method="rf", prox=TRUE)
library(randomForest)
getTree(modFit$finalModel, k=2)

irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
pred <- predict(modFit, testing)
testing$predRight <- pred==testing$Species
qplot(Petal.Width, Petal.Length, color=predRight, data=testing)

#boosting
library(ISLR)
data(Wage)
inTrain <- createDataPartition(Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
modFit <- train(wage~., method="gbm", data=training,verbose=F)

#model based prediction
data(iris)
library(ggplot2)
inTrain <- createDataPartition(iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modlda <- train(Species~., data=training, method="lda")
modnb <- train(Species~., data=training, method="nb")
plda <- predict(modlda, testing)
pnb <- predict(modnb, testing)
table(plda, pnb)

equalPred <- plda==pnb
qplot(Petal.Width, Sepal.Width, color=equalPred, data=testing)

#quiz 3.1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
inTrain <- createDataPartition(y = segmentationOriginal$Case, p=0.7, list=F)
training <- segmentationOriginal[inTrain,]
testing <- segmentationOriginal[-inTrain,]
set.seed(125)
mod <- train(Class~., method="rpart", data=training)
fancyRpartPlot(mod$finalModel)

#quiz 3.3
library(pgmm)
data(olive)
olive = olive[,-1]
inTrain <- createDataPartition(olive$Area, p=0.7, list=F)
training <- olive[inTrain,]
testing <- olive[-inTrain,]

newdata <- as.data.frame(t(colMeans(olive)))

modFit <- train(Area~., method="rpart", data=training)
predict(modFit, newdata)

#quiz 3.4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

modFit <- train(chd~age+alcohol+obesity+tobacco+typea+ldl, method="glm",family = "binomial", data=trainSA)
pred <- predict(modFit, testSA)
pred_t <- predict(modFit, trainSA)
missClass <- function(values, prediction){
  sum(((prediction>0.5)*1) != values)/length(values)
}
missClass(testSA$chd, pred)
missClass(trainSA$chd, pred_t)

#quiz 3.5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
modFit <- randomForest(y~., data = vowel.train)
imp <- varImp(object = modFit)
ord <- order(imp$Overall, decreasing = T)
rownames(imp)[ord]

#forecasting
install.packages("quantmod")
library(quantmod)
from.dat <- as.Date("01/01/13", format = "%m/%d/%y")
to.dat <- as.Date("12/31/16", format = "%m/%d/%y")
getSymbols("GOOG", src="google", from=from.dat, to=to.dat)
GOOG <- subset(GOOG, select = -c(GOOG.Volume))

mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen, frequency=12)
plot(ts1, xlab="Years+1", ylab="GOOG")
plot(decompose(ts1))

ts1Train <- window(ts1, start=1, end=10)
ts1Test <-  window(ts1, start=1, end=(7-0.01))

library(forecast)
plot(ts1Train)
lines(ma(ts1Train, order=3), col="red")

ets1 <- ets(ts1Train, model="MMM")
fcast <- forecast(ets1)
plot(fcast)
lines(ts1Test, col="red")

accuracy(fcast, ts1Test)

#unsupervised
data(iris)
inTrain <- createDataPartition(iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

kmeans1 <- kmeans(subset(training, select=-c(Species)), centers=3)
training$clusters <-as.factor(kmeans1$cluster)
qplot(Petal.Width, Petal.Length, color=clusters, data=training)

#Quiz 4.1
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)

library(caret)
library(randomForest)
modRF <- train(y~., method = "rf",data = vowel.train, prox=TRUE)
modb <- train(y~., method="gbm", data=vowel.train, verbose=F)

test1 <- mean(predict(modRF, vowel.test)==vowel.test$y)
test2 <- mean(predict(modb, vowel.test)==vowel.test$y)
agree <- predict(modRF, vowel.test)==predict(modb,vowel.test)
mean(predict(modRF, vowel.test)[agree]==vowel.test$y[agree])

#quiz 4.2
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
modRF <- train(diagnosis~., method="rf", data=training)
modb <- train(diagnosis~., method="gbm", data=training, verbose=F)
modlda <- train(diagnosis~., method="lda", data=training)

install.packages("caretEnsemble")
library(caretEnsemble)
stackControl <- trainControl(classProbs = TRUE)
models <- caretList(diagnosis~., methodList = c("rf","gbm", "lda"), trControl=stackControl, data=training, verbose=F)
stack.rf <- caretStack(models, metric="Accuracy", method="rf")

#quiz 4.3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(233)
modlasso <- train(CompressiveStrength~., data=training, method="lasso")
pred <- predict(modlasso, testing)
library(elasticnet)
plot.enet(modlasso$finalModel, xvar="penalty", use.color = T)

#quiz 4.4
library(lubridate) # For year() function below
library(forecast)
dat = read.csv("C:/Users/xianbin2/Documents/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr, start=1, end=365)
modts <- bats(y = tstrain)
tstesting <- ts(testing$visitsTumblr)
Fcast <- forecast(modts)

#Quiz 4.5
set.seed(325)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
library(e1071)
modsvm <- svm(x=training[,-9], y=training[,9])
sqrt(sum((testing[,9]-predict(modsvm, testing[,-9]))^2)/length(testing[,9]))
                    