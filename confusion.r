## Hankui wrote it on Nov 1, 2016
## confusion matrix 
## source('confusion.r');

# source("getKappa.r");

confusion <- function(predicted, actual, filename="temp.csv") 
{
	c_m <- as.matrix(table(Actual=actual, Predicted=predicted)) # create the confusion matrix
	# c_m <- as.data.frame(table(Predicted = predicted, Actual = actual )) # create the confusion matrix
	uniq_p <- sort(unique(predicted));
	uniq_t <- sort(unique(actual   ));
	nr      <- nrow(c_m) # number of classes

	##***************************************************************
	## important for those classes never been classified (Mar 13 2018 Hank) 
	block_list <- list();
	zeros <- rowsums <- apply(c_m, 1, sum)*0; 
	empty_index <- vector();
	for (i in 1:length(uniq_t)) {
		nc <- ncol(c_m);
		is_find <- 0;
		uniq_t_i <- uniq_t[i];
		for (j in 1:length(uniq_p)) if (uniq_p[j]==uniq_t_i) {is_find <- j; break; }
		
		if(is_find==0) { 
			block_list[[i]] <- zeros;
			cat("\tclass", uniq_t_i, "is never predicted\n");
			empty_index <- c(empty_index, i);
		}  else {
			block_list[[i]] <- c_m[,is_find];
		}
	}
	c_m_new <- block_list[[1]];
	for (i in 2:length(uniq_t)) c_m_new <- cbind(c_m_new, block_list[[i]]);
	
	c_m <- c_m_new;
	colnames(c_m) <- rownames(c_m);

	
	##***************************************************************
	## accuracy 
	n <- sum(c_m) # number of instances
	diags   <- diag(c_m) # number of correctly classified instances per class 
	rowsums <- apply(c_m, 1, sum) # number of instances per class
	colsums <- apply(c_m, 2, sum) # number of predictions per class
	
	accuracy = sum(diags) / n; 
	# print(c_m)
	# print(accuracy)
	# precision = diags / colsums 
	# recall = diags / rowsums 
	uses     <- diags / colsums * 100;
	produces <- diags / rowsums * 100;
	c_m <- cbind(c_m, rowsums, produces);
	
	colsumss <- c(colsums, n, 0       );
	uses.s   <- c(uses   , 0, accuracy);
	c_m <- rbind(c_m, colsumss, uses.s);
	
	# colnames(c_m) <- c(colnames(c_m)[1:nr],"Total truth","Producer's accuracy");
	# rownames(c_m) <- c(rownames(c_m)[1:nr],"Total predicted","User's accuracy");
	rownames(c_m) <- c(as.numeric(colnames(c_m)[1:nr])-1,"Total truth","Producer's accuracy");
	colnames(c_m) <- c(as.numeric(rownames(c_m)[1:nr])  ,"Total predicted","User's accuracy");
	
	# print(precision)
	options(scipen=999);
	# print(format(c_m, digits=1));
	write.csv(c_m, file=filename);
	
	## *******************************************************************************************
	## Kappa
	ppp <- rowsums / n; # distribution of instances over the actual classes
	qqq <- colsums / n; # distribution of instances over the predicted classes
	expAccuracy = sum(ppp*qqq);
	Kappa = (accuracy - expAccuracy) / (1 - expAccuracy);
	cat("\tKappa = ");
	cat(format(Kappa, digits=4));
	cat("\tAccuracy = ");
	print(format(accuracy*100, digits=4)); 
	list("Kappa"=Kappa, "Accuracy"=accuracy, "Confusion"=c_m);
}
## *******************************************************************************************
## test data
# set.seed(0)
# actual = c('a','b','c')[runif(100, 1,4)] # actual labels
# predicted = actual # predicted labels
# predicted[runif(30,1,100)] = actual[runif(30,1,100)]  # introduce incorrect predictions
# predicted[runif(30,1,10)] = c('a','b','c')[runif(10, 1,4)]  # introduce incorrect predictions
# confusion(predicted, actual);

## *******************************************************************************************
## test data global random forest
# prop <- 67;
# prop <- 100;
# load(paste("./file.plot/conus.urbanFALSE.prop",prop,".tree500", sep=""));
# actual <- truth;
# confusion(predicted, actual, "global.confusion.csv");
