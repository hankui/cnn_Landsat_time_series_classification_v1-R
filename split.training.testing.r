#　By Hank in SDSU
# source("split.training.testing.r")

## **********************************
# train_test_file: store the split index result
# classes: all the classes 
# classn: total number of classes
# n_in_each_class: number in each class
# y: class labels
# trainrates: split rate for training

split.training.testing <- function(train_test_file, classes, classn, n_in_each_class, y, trainrates) { 
	if (file.exists(train_test_file)) { 
		cat("The train_test_file", train_test_file, "exists ...\n");
		load(train_test_file);
	} else {  
		# sample.ns <- vector(mode="integer",  length=classn); # n_in_each_class
		train.index <- vector(mode="integer");
		test.index <- vector(mode="integer");
		n_in_each_class_train <- vector(mode="integer");
		n_in_each_class_test <- vector(mode="integer");
		for (i in 1:classn) {
			indexi <- which (y==classes[i]);

			## randomly selecting 80% training & 20% testing
			n_in_each_class_train[i] <- as.integer(n_in_each_class[i]*trainrates); 
			n_in_each_class_test[i] <- n_in_each_class[i] - n_in_each_class_train[i];
			train.index.i <- sample.int(n_in_each_class[i], n_in_each_class_train[i]);
			test.index.i <- setdiff(1:n_in_each_class[i], train.index.i);	
			
			train.index <- c(train.index, indexi[train.index.i]);
			test.index <- c(test.index, indexi[test.index.i]);
		}
		class_ns_train <- n_in_each_class_train;
		class_ns_test  <- n_in_each_class_test;
		
		save(file=train_test_file, n_in_each_class, train.index, test.index, class_ns_train, class_ns_test, n_in_each_class_train, n_in_each_class_test);
	}
	# save(file=train_test_file, class_ns, train.index, test.index, class_ns_train, class_ns_test);
	list("train.index"=train.index, "test.index"=test.index, "n_in_each_class_train"=class_ns_train, "n_in_each_class_test"=class_ns_test)
}

