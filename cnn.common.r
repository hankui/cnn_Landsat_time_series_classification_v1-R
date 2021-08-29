# source("cnn.common.r");

library(tensorflow)
use_condaenv('tensorflow-1.3')

combine_xy_dat <- function(xtrain, ytrain, classes) {
	dat1 <- xtrain;
	## convert from class label to a number from 1:n 
	for (i in classes) {
		dat1 <- cbind(dat1, ytrain==i);
	}

	## MUST BE matrix
	as.matrix(dat1)
}


#************************************************************************************************************************
## doing batch normalization (BN)
## ndim is a 3 dimensional array in convolution layer 
##         a 1 dimensional array in fully connected layer 
## refer to https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
batch_norm_wrapper <- function(inputs, is_training=TRUE, ndim=shape(100), decay = 0.999) {
	# cat("\t\t now in the batch normalization... \n")
	
	# Small epsilon value for the BN transform
	epsilon = 1e-3;

	# DO NOT invoke another tf$InteractiveSession() during session to derive ndim
	scale <- tf$Variable(tf$ones (ndim));
	beta  <- tf$Variable(tf$zeros(ndim));
	pop_mean <- tf$Variable(tf$zeros(ndim), trainable=FALSE);
	pop_var  <- tf$Variable(tf$ones (ndim), trainable=FALSE);

	if (is_training==1) {
		mean_var <- tf$nn$moments(inputs, as.integer(0) ); 
		batch_mean <- mean_var[[1]];
		batch_var <- mean_var[[2]];
		train_mean <- tf$assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay));
		train_var <- tf$assign(pop_var, pop_var * decay + batch_var * (1 - decay));
		
		## this command is only used in Python
		# with tf.control_dependencies([train_mean, train_var]):
		
		## not work
		# if (class( tf$control_dependencies(list(train_mean, train_var)) )[[1]]=="tensorflow.python.framework.ops._ControlDependenciesController") {
			# result <- tf$nn$batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon); 
		# }
		
		## work for only one dim
		# result <- tf$cond( tf$logical_and(tf$equal(train_mean[1], 0), tf$equal(train_var[1], 0)), 
			# lambda<-function(inputs1=inputs, pop_mean1=batch_mean, pop_var1=batch_var, beta1=beta, scale1=scale, epsilon1=epsilon) {tf$nn$batch_normalization(inputs1, pop_mean1, pop_var1, beta1, scale1, epsilon1) }, 
			# lambda<-function(inputs1=inputs, pop_mean1=batch_mean, pop_var1=batch_var, beta1=beta, scale1=scale, epsilon1=epsilon) {tf$nn$batch_normalization(inputs1, pop_mean1, pop_var1, beta1, scale1, epsilon1) })
		
		## work for any dims
		## Note that this is tricky to invoke the above 2 lines in the NN graphs using a condition: tf$logical_and(tf$equal(train_mean_1, 0), tf$equal(train_var_1, 0))
		train_mean_1 <- tf$reshape(train_mean, shape(-1L) )[1];
		train_var_1  <- tf$reshape(train_var , shape(-1L) )[1];	
		
		result <- tf$cond( tf$logical_and(tf$equal(train_mean_1, 0), tf$equal(train_var_1, 0)), 
			lambda<-function(inputs1=inputs, pop_mean1=batch_mean, pop_var1=batch_var, beta1=beta, scale1=scale, epsilon1=epsilon) {tf$nn$batch_normalization(inputs1, pop_mean1, pop_var1, beta1, scale1, epsilon1) }, 
			lambda<-function(inputs1=inputs, pop_mean1=batch_mean, pop_var1=batch_var, beta1=beta, scale1=scale, epsilon1=epsilon) {tf$nn$batch_normalization(inputs1, pop_mean1, pop_var1, beta1, scale1, epsilon1) })

	} else {
		result <- tf$nn$batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon); 
	} 
	
	## return result 
	result
}


get_dim <- function(xxxx) {
	dimsn <- tf$shape(xxxx)$shape$dims[[1]]
	dims <- tf$Session()$run(tf$shape(xxxx)); ## convert from tensor to array
	
	cat(paste("\tdimsn = ", dimsn, "\tdims =", sep=""));
	cat(paste("\t", dims));
	cat("\n");
}
# get_dim(ten);
   
weight_variable <- function(shape, stddev=0.01) {
	initial <- tf$truncated_normal(shape, stddev=stddev)
	# initial <- tf$random_normal(shape, stddev=stddev)
	tf$Variable(initial)
}

bias_variable <- function(shape, constant=0) {
	initial <- tf$constant(constant, shape=shape)
	tf$Variable(initial)
}

conv2d <- function(x, W) {
	tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}


get_mean_coe <- function(sess, W_conv2) { 
	mat <- sess$run(W_conv2);
	cat("\tmean=", mean(mat), "std=", sd(mat), "min=", min(mat), "max=", max(mat), "\n"); 
}
# max_pool_2x2 <- function(x) {
	# tf$nn$max_pool(
		# x, 
		# ksize=c(1L, 2L, 2L, 1L),
		# strides=c(1L, 2L, 2L, 1L), 
		# padding='SAME')
# }

max_pool_2x2 <- function(x) {
	tf$nn$max_pool(
		x, 
		ksize=c(1L, 2L, 2L, 1L),
		strides=c(1L, 2L, 2L, 1L), 
		padding='SAME')
}

avg_pool_2x2 <- function(x) {
	tf$nn$avg_pool(
		x, 
		ksize=c(1L, 2L, 2L, 1L),
		strides=c(1L, 2L, 2L, 1L), 
		padding='SAME')
}

max_pool_1x2 <- function(x) {
	tf$nn$max_pool(
		x, 
		ksize=c(1L, 1L, 2L, 1L),
		strides=c(1L, 1L, 2L, 1L), 
		padding='SAME') ## SAME can fix when odd happens
}

conv2d_no_pad <- function(x, W, is_pad=FALSE) {
	if (is_pad) { 
		result <- tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME');
	} else result <- tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='VALID');
	
	result
}

conv3d_no_pad <- function(x, W) {
	tf$nn$conv3d(x, W, strides=c(1L, 1L, 1L, 1L, 1L), padding='VALID')
}

conv1d_no_pad <- function(x, W) {
	tf$nn$conv1d(x, W, stride=1L, padding='VALID')
}

next_batch <- function(dat, totaln, dimx=125, n=100, classn) {
	# index <- floor(runif(n, min=1, max=totaln));
	index <- sample.int(totaln, n); ## RANDOM
	listout <- list("x_batch"=dat[index, 1:dimx], "y_batch"=dat[index, (dimx+1):(dimx+classn)]);
	listout
}


next_batch_with_shuffle <- function(shuffle_index, dat, totaln, startx=1, dimx=125, n=100, classn) {
	# index <- floor(runif(n, min=1, max=totaln));
	# index <- sample.int(totaln, n); ## RANDOM
	index_shuf <- startx:(startx+n);
	extra <- startx+n - totaln; 
	if (extra>0) index_shuf <- c(startx:totaln, 1:(startx+n-totaln) );
	
	index <- shuffle_index[index_shuf]; 
	listout <- list("x_batch"=dat[index, 1:dimx], "y_batch"=dat[index, (dimx+1):(dimx+classn)]);
	listout
}

next_batch_perclass <- function(dat, totaln, dimx=layersn*lengthn, n=100, classn, class_ns_train, list_train_index_per_class) {
	# index <- floor(runif(n, min=1, max=totaln));
	index <- vector(mode="integer");
	for (i in 1:classn) {
		indexi <- sample.int(class_ns_train[i], max(as.integer(n/totaln*class_ns_train[i]), 1) ); ## RANDOM
		index <- c(index, list_train_index_per_class[[i]][indexi])
	}
	listout <- list("x_batch"=dat[index, 1:dimx], "y_batch"=dat[index, (dimx+1):(dimx+classn)]);
	listout
}

print_sta <- function(ptm_sub, conf, itern, accuracies, kapps, classn) { 
	time.my <- proc.time() - ptm_sub;
	# cat(paste("\n", sprintf("%.4f",time.my[3]/60),"hours\n"));
	print(Sys.time() );
	# cat(paste("\n", sprintf("%.3f",time.my[1]/3600),"\t", sprintf("%.3f",time.my[2]/3600),"\t", sprintf("%.3f",time.my[3]/3600),"hours\n"));
	conf <- conf/itern; 
	# cat("\t Producer accuracy:", sprintf("%.4f", conf[1:classn, classn+2]), "\n");
	cat("Prod_accuracy:", sprintf("%20.4f", conf[1:classn, classn+2]), "\n");
	cat("User_accuracy:", sprintf("%20.4f", conf[classn+2, 1:classn]), "\n");

	# cat("\tMean Kappa =    ", sprintf("%20.4f", mean(kapps)     *100), "\n");
	# cat("\tMean Accuracy = ", sprintf("%20.4f", mean(accuracies)*100), "\n");
	print (accuracies)
	print (kapps)
	cat( sprintf("mean kapps      %6.4f", mean(kapps)         ), "\n");
	cat( sprintf("mean accuracies %6.2f", mean(accuracies)*100), "\n");
}

print_classes <-function(class_indexs, class_names, class_ns) {
	for (i in c(1:length(class_indexs)) ) { 
		cat(sprintf("%20d", class_indexs[i]));
	} 
	cat("\n");
	# for (i in c(1:length(class_indexs)) ) { 
		# cat(sprintf("%20s", class_names[i]));
	# } 
	# cat("\n");
	for (i in c(1:length(class_indexs)) ) { 
		cat(sprintf("%20d", class_ns[i]));
	} 
	cat("\n");
} 

# next_batch_all_class <- function(dat, totaln, dimx=125, n=100, classn) {
	# index <- sample.int(totaln, n); ## RANDOM
	# listout <- list("x_batch"=dat[index, 1:dimx], "y_batch"=dat[index, (dimx+1):(dimx+classn)]);
	# listout
# }
