# source("rf.my.r");

# run customized random forest and return testing dataset confusion matrix

library(randomForest)
source("getKappa.r");

rf.my <- function(xtrain, ytrain, xtest, ytest) {

	yftrain <- factor(ytrain);
	yftext  <- factor(ytest );
	ntrees <- 500;
	# iris.rf2 <- randomForest(x=xtrain, y=yftrain,  xtest=xtest, ytest=yftext, importance=TRUE, replace=FALSE, ntree=ntrees, keep.forest=TRUE, keep.inbag=TRUE);
	iris.rf2 <- randomForest(x=xtrain, y=yftrain,  xtest=xtest, ytest=yftext, replace=FALSE, ntree=ntrees);

	classn <- length(iris.rf2$classes)
	conf <- mat.or.vec(classn+2,classn+2);
	
	## Notice here is the swith
	tempconf = iris.rf2$test$confusion;
	# tempconf = iris.rf2$confusion;
	
	colnames(conf) <- c(colnames(tempconf)[1:classn],"Total","User's accuracy");
	rownames(conf) <- c(rownames(tempconf)[1:classn],"Total","Producer's accuracy");
	conf[1:(classn),1:(classn)] <- tempconf[1:(classn),1:(classn)]; 
	conf[classn+1,1:classn] <- colSums(conf[1:classn,1:classn])
	conf[classn+2,1:classn] <- diag(conf[1:classn,1:classn])/conf[classn+1,1:classn];
	conf[1:classn,classn+1] <- rowSums(conf[1:classn,1:classn])
	conf[1:classn,classn+2] <- diag(conf[1:classn,1:classn])/conf[1:classn,classn+1];
	conf[classn+1,classn+1] <- sum(conf[1:classn,classn+1]);
	conf[classn+2,classn+2] <- sum(diag(conf[1:classn,1:classn]))/sum(conf[1:classn,classn+1]);
	
	##kappa
	qqq <- conf[classn+1,1:classn]/conf[classn+1,classn+1]
	ppp <- conf[1:classn,classn+1]/conf[classn+1,classn+1]
	expAccuracy <- sum(ppp*qqq)
	kappaa <- (conf[classn+2,classn+2] - expAccuracy)/(1 - expAccuracy);
	OA <- conf[classn+2,classn+2];
	conf[classn+1,classn+2] <- kappaa;
	
    conf <- conf*100;
	
	OA <- sum(iris.rf2$predicted==yftrain)/length(yftrain);
	# source("getKappa.r");
	# kappaa <- getKappa(iris.rf2$predicted, yftrain);
	print(paste("kappaa =", sprintf("%.4f", kappaa), "OA =", sprintf("%.2f", OA*100)  ));
	
	
	list("Kappa"=kappaa, "Accuracy"=OA, "Confusion"=conf);
}


