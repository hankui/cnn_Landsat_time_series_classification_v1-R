# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <-FALSE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")

# sampling rate = 0.1 for test
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 3200; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 3200; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 3200; source("Pro.training.metric.cdl.r")


# sampling rate = 0.1
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 32000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 32000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 32000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.1; con_layer <- 2; is_shallow=10   ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 32000; source("Pro.training.metric.cdl.r")

# sampling rate = 0.25
# rm(list=ls(all=TRUE)); samplerate <- 0.25; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 80000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.25; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 80000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.25; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 80000; source("Pro.training.metric.cdl.r")

# sampling rate = 0.5
# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.5; con_layer <- 2; is_shallow=10   ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 160000; source("Pro.training.metric.cdl.r")

# sampling rate = 0.75
# rm(list=ls(all=TRUE)); samplerate <- 0.75; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 240000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.75; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 240000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.75; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 240000; source("Pro.training.metric.cdl.r")

# sampling rate = 0.9
# rm(list=ls(all=TRUE)); samplerate <- 0.9; con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 300000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.9; con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 300000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.9; con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 300000; source("Pro.training.metric.cdl.r")
# rm(list=ls(all=TRUE)); samplerate <- 0.9; con_layer <- 2; is_shallow=10   ; learningrate <- 0.01; is_norm <- TRUE;  beta.coe<-1E-3;    iters <- 300000; source("Pro.training.metric.cdl.r")

# Hank on Dec 5, 2017
currentwd <- getwd();
setwd("../");
source("rf.my.r");
source("cnn.metric.batch.validation.r");
source("split.training.testing.r");
setwd(currentwd); 

##***************************************************************************************************
## parameters needed to be adjusted in your own computer
## input file name
linux_folder=".";
if (R.Version()$os == "linux-gnu") linux_folder="/gpfs/data2/workspace/zhangh/dump";
if (R.Version()$os == "linux-gnu") linux_folder="/gpfs/data1/hankui/zhangh/dump"; ## changed on Dec 30 2019 to reflect the broken hardware of /gpfs/data2 
if (R.Version()$os == "linux-gnu") linux_folder="./"; ## changed on Aug 29 2021 to reflect new 
filename <- paste(linux_folder, "/metric.ard.cdl.Mar01.18.40.txt", sep=""); 
## number of threads when running paralleling computation depending on the number of available cores 
MAX_THREADS <- 18L;
MAX_THREADS <- 36L;
##***************************************************************************************************

if (!exists("dat")) dat <- read.table(filename, header=TRUE);
ptm <- proc.time();
ismetric <- 1; 
permutation <- 0;
threshold <- 1000; ## throw those classes with samples smaller than this threshold
training_split_dir <- ".//random.numbers//";
if (! dir.exists(training_split_dir)) {print(paste("Creating dir", training_split_dir)); dir.create(training_split_dir); }
cat("\n", filename, " throwing class threshold =", threshold, "sample rate =", samplerate, "is metric =", ismetric, "\n");

# stop("I want to sop");
## **************************************************
## find classes 1 & 5
classes_nlcd <- read.csv("./generic_cdl_attributes.csv", header=TRUE);
uniqs <- sort(unique(dat$nlcd));
sub.index <- vector(mode="integer");
class_indexs <- vector(mode="integer");
class_names <- vector(mode="character");
class_ns <- vector(mode="integer");
for (i in uniqs) {
	index <- classes_nlcd$VALUE==i;
	
	if ( (i>=64&&i<=65) || (i>78&&i<204) ) next; 
	
	indexi <- which (dat$nlcd==i);
	sample_subs <- length(indexi);
	if (sample_subs> threshold) {
		class_indexs <- c(class_indexs, i);
		class_names <- c(class_names, toString(classes_nlcd$CLASS_NAME[index]));
		class_ns <- c(class_ns, sample_subs); 		# cat("\nindex: ", i,  toString(classes_nlcd$CLASS_NAME[index]), ":", sample_subs) ; 
		sub.index <- c(sub.index, indexi); 
	}
}

dat_corn_soy_all <- dat[sub.index, ];

nlcd_all <- dat_corn_soy_all$nlcd;
nlcdindex <- grep("nlcd", colnames(dat_corn_soy_all));

# hist(dat$nlcd, breaks=c(-1, sort(unique(dat$nlcd)), 1000)  )$counts;

## **************************************************
## input metric index
sta_metric_index <- grep("low25.green"  , colnames(dat_corn_soy_all));
end_metric_index <- grep("high75.ratio7", colnames(dat_corn_soy_all));
metric_index <- sta_metric_index:end_metric_index;

## **************************************************
## split the data into training & testing
classes <- sort(unique(nlcd_all));
classn <- length(classes);

sub.index <- vector(mode="integer");
steps <- c(1,1,1);
steps <- rep(1, times=classn);
for (i in 1:classn) {
	indexi <- which (nlcd_all==classes[i]);
	sample_subs <- length(indexi);
	sub.index <- c(sub.index, indexi[seq(from=1,to=sample_subs,by=steps[i])]); 
}

nlcd <- nlcd_all[sub.index];
dat_corn_soy <- dat_corn_soy_all[sub.index, ];

train_test_file=paste(training_split_dir, "train_test_file.", basename(filename), ".", samplerate,".n", classn, ".threshold", threshold, sep="");

split.samples <- split.training.testing(train_test_file, classes, classn, class_ns, nlcd, samplerate);
train.index <- split.samples$train.index;
test.index  <- split.samples$test.index;
n_in_each_class_train <- split.samples$n_in_each_class_train; 

## **************************************************
## start training & testing
temp.xtrain <- dat_corn_soy[train.index, metric_index];
temp.xtest  <- dat_corn_soy[ test.index, metric_index];

xtrain <- cbind(temp.xtrain[, 1:6], temp.xtrain[, 19:25], temp.xtrain[, c(1:6)+6], temp.xtrain[, c(19:25)+7], temp.xtrain[, c(1:6)+12], temp.xtrain[, c(19:25)+14]);
xtest  <- cbind(temp.xtest [, 1:6], temp.xtest [, 19:25], temp.xtest [, c(1:6)+6], temp.xtest [, c(19:25)+7], temp.xtest [, c(1:6)+12], temp.xtest [, c(19:25)+14]);

## ***********************************************************************************************************************************
## Permutation bands
# permutation <- sample(13); 
# print(permutation);
# xtrain1 <- cbind(xtrain[, permutation], xtrain[, permutation+13], xtrain[, permutation+26]);
# xtest1  <- cbind(xtest [, permutation], xtest [, permutation+13], xtest [, permutation+26]);
# xtrain <- xtrain1;
# xtest  <- xtest1 ;

## ***********************************************************************************************************************************
## Permutation bands finished
# as.vector(xtrain[1,1:13] );
ytrain <- dat_corn_soy[train.index,    nlcdindex];
ytest  <- dat_corn_soy[ test.index,    nlcdindex];


itern <- 2;
itern <- 11;
itern <-  1;
accuracies <- vector(mode="double", length=itern); 
kapps <- vector(mode="double", length=itern); 
users     <- matrix(, nrow=itern, ncol=classn); 
producers <- matrix(, nrow=itern, ncol=classn); 
train.time <- matrix(, nrow=itern, ncol=5); 
test.time  <- matrix(, nrow=itern, ncol=5); 
conf <- mat.or.vec(classn+2,classn+2);

## ********************************************************************************************************************************************************************
## random forest
ptm_sub <- proc.time();
print_classes(class_indexs, class_names, class_ns); 

if (is_shallow==10) {
	cat("...now Random forest....\n");
	for (i in 1:itern)  {
		ntrees <- 500;
		yftrain <- factor(ytrain);
		yftext  <- factor(ytest );
		
		ptm1 <- proc.time();
        ## only training
        ris.rf2 <- randomForest(x=xtrain, y=yftrain, replace=FALSE, ntree=ntrees);	
        train.time.i <- proc.time() - ptm1;
        cat(paste("\n", sprintf("%.3f",train.time.i[1]/3600),"\t", sprintf("%.3f",train.time.i[2]/3600),"\t", sprintf("%.3f",train.time.i[3]/3600),"hours\n"));
        # print(ris.rf2$test$confusion);
        
        ## only testing
        ptm2 <- proc.time();
        ris.predicted <- predict(object=ris.rf2, newdata=xtest); # ris.rf2 <- randomForest(x=xtrain, y=yftrain, replace=FALSE, ntree=ntrees);
        test.time.i <- proc.time() - ptm2;
        cat(paste("\n", sprintf("%.3f",test.time.i[1]/3600),"\t", sprintf("%.3f",test.time.i[2]/3600),"\t", sprintf("%.3f",test.time.i[3]/3600),"hours\n"));
        print(paste("Accuracy:", 100*sum(as.numeric(ris.predicted) == as.numeric(yftext))/length(ris.predicted) ) );
 
        ## training & testing & recording accuracy
        # ptm_sub <- proc.time();
        # result <- rf.my(xtrain, ytrain, xtest, ytest);
        result <- confusion(ris.predicted, ytest, filename=paste("random.forest.temp.confusion", ".csv", sep="") );
	
        ## add results to matrices
		accuracies[i] <- result$Accuracy; 
        kapps[i] <- result$Kappa; 
        conf <- conf+(result$Confusion);
		users[i,] <- result$Confusion[classn+2, 1:classn]; producers[i,] <- result$Confusion[1:classn, classn+2];
        
        train.time[i,] <- train.time.i/3600; test.time[i,] <- test.time.i/3600; 		   
    }
	print_sta(ptm_sub, conf, itern, accuracies, kapps, classn);
	# stop("I stopped here");
} else { 
    ## ********************************************************************************************************************************************************************
    ## deep learning
    cat("\n...now Deep learning....\n");
    # conf <- mat.or.vec(classn+2,classn+2);
    # ptm_sub <- proc.time();
    for (i in 1:itern) {
        cat("\n", i, "th iteration \n");
        result <- cnn.metric.batch.validation(xtrain, ytrain, xtest, ytest, n_in_each_class_train, 
            input_dim_x=13, input_dim_y=3, classes=classes, classn=classn, learningrate=learningrate, 
            iters=iters, beta.coe=beta.coe, con_layer=con_layer, is_shallow=is_shallow);
        
        ## add results to matrices
        accuracies[i] <- result$Accuracy;
        kapps[i] <- result$Kappa;
        conf <- conf + result$Confusion;
        users[i,] <- result$Confusion[classn+2, 1:classn]; producers[i,] <- result$Confusion[1:classn, classn+2];
        train.time[i,] <- as.numeric(result$train.time); test.time[i,] <- as.numeric(result$test.time); 		
    }
}

print_sta(ptm_sub, conf, itern, accuracies, kapps, classn);
save(samplerate, class_names, split.samples, accuracies, kapps, users, producers, train.time, test.time, 
	file=paste0(".//itern", itern, "samplerate", samplerate, "is_shallow",  is_shallow, "cdl") );
time.my <- proc.time() - ptm_sub;
cat(paste("\n", sprintf("%.3f",time.my[1]/3600),"\t", sprintf("%.3f",time.my[2]/3600),"\t", sprintf("%.3f",time.my[3]/3600),"hours\n"));
print(train.time)

