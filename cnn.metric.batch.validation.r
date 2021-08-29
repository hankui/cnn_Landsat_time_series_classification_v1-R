# (1) Do not invoke another tf$InteractiveSession() during session noted by Mar 25 2018
# (2) Transpose: x_image <- tf$transpose(x_image2, perm=as.integer(c(0, 2, 3, 1)) )
# (3) Dynamic learning rate is the key noted by Mar 30 2018 
# (4) Drop-out is not helping (He et al. 2016)
# 	We create a placeholder for the probability that a neuron’s output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. 
# 	keep_prob <- tf$placeholder(tf$float32)
# 	h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)
# (5) mini-batch can save memory & avoid local minimum, Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010 (pp. 177-186). Physica-Verlag HD.
# (6) batch normalization is referring to https://r2rt.com/implementing-batch-normalization-in-tensorflow.html

# source("cnn.common.r");
source("build_graph.r");
source("confusion.r");

# input_dim_x=13; input_dim_y=3;
is_greater <- function(accuracies.validation, inner_i_validation, plat_n, type=1) {
	first_half  <- sort(accuracies.validation[(inner_i_validation-plat_n+1):(inner_i_validation-plat_n/2)], decreasing=FALSE);
	second_half <- sort(accuracies.validation[(inner_i_validation-plat_n/2+1):(inner_i_validation)], decreasing=TRUE);
	# top_n <- as.integer(plat_n/2*33/100);
	# top_n <- 1;
	top_n <- ceiling(plat_n/2/2);
	if (type==1) { 
		resutls <- mean(first_half[1:(plat_n/2-top_n)]) > mean(second_half[1:(plat_n/2-top_n)]);
	} else { 
		top_n <- 1; 
		# resutls <- mean(first_half[1:(plat_n/2-top_n)]) > mean(second_half[1:(plat_n/2-top_n)]); 
		resutls <- max(first_half[1:(plat_n/2-top_n)]) > min(second_half[1:(plat_n/2-top_n)]); 
	} 
	
	resutls
}

# input_dim_x=13; input_dim_y=3;
cnn.metric.batch.validation <- function(xtrain, ytrain, xtest, ytest, n_in_each_class_train, 
	input_dim_x=13, input_dim_y=3, classes=classes, classn=classn, learningrate=0.1, iters=8000, 
	beta.coe=1E-4, con_layer=4, is_shallow=TRUE) 
{
	ptm_cnn <- proc.time();
	
	input_dim_stacked <- input_dim_x*input_dim_y;

	## option 4 taken 4% as validation
	train_test_file=paste(".//random.numbers//train_validation_file.", basename(filename), ".sample.rate", samplerate,".threshold", threshold, sep="");
	split.samples <- split.training.testing(train_test_file, classes, classn, n_in_each_class_train, ytrain, 0.96);
	train.index <- split.samples$train.index;
	validation.index  <- split.samples$test.index;
	
	meanx <- apply(xtrain, 2, mean);
	sdx   <- apply(xtrain, 2, sd);
	# x2 <- t((t(xtrain)-meanx)/sdx);
	# (xtrain[train.index,]-meanx)/sdx;
	dat.validation <- combine_xy_dat(  t((t(xtrain[validation.index,])-meanx)/sdx), ytrain[validation.index], classes);
	batch.validation <- next_batch(dat.validation, dim(dat.validation)[1], dimx=input_dim_stacked, n=dim(dat.validation)[1], classn);
	
	dat.train <- combine_xy_dat( t((t(xtrain[train.index,])-meanx)/sdx), ytrain[train.index], classes);
	dat.test  <- combine_xy_dat( t((t(xtest               )-meanx)/sdx), ytest , classes);	
	
	# stop("I have to stop here");
	##**************************************
	## start to CNN
	totaln <- dim(dat.train)[1];
	totaln_test <- dim(dat.test)[1];
	cat("training n =", sprintf("%7d", totaln), "testing n =", totaln_test, "\n");
	batchsize <- 128; 

	isdim <- FALSE;
	# isdim <- TRUE;

	##**************************************
	## start to build graph for train
	tf$reset_default_graph();
	x  <- tf$placeholder(tf$float32, shape(NULL, input_dim_stacked))
	y_ <- tf$placeholder(tf$float32, shape(NULL, classn))
	if (isdim) {
		index <- 1:100; ## RANDOM
		batch <- list("x_batch"=dat.train[index, 1:input_dim_stacked], "y_batch"=dat.train[index, (input_dim_stacked+1):(input_dim_stacked+classn)]);
		x <- tf$cast(batch[[1]], tf$float32); 
	}
	
	bg_build_graph <- build_graph(con_layer, is_training=TRUE, x, input_dim_x, input_dim_y, is_norm=is_norm, is_shallow=is_shallow); 

	y_conv_train <- bg_build_graph$y_conv;
 	
	cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv_train), reduction_indices=1L))
	
	## training option # 4
	momentum <- 0.9; 
	train_step_lists <- list();
	
	rater <- 10; 
	max_change_rate <- 4;
	max_change_rate <- ceiling(log10(learningrate/1E-6)) + 1;
	max_change_rate <- 5;
	# max_change_rate <- 1;
	cat("\tmax_change_rate =", max_change_rate, "\n");
	
	for (ti in 1:max_change_rate) train_step_lists[[ti]] <- tf$train$MomentumOptimizer(learningrate/rater^(ti-1), momentum)$minimize(tf$reduce_mean(cross_entropy + beta.coe * bg_build_graph$regularizer));

	# for (ti in 1:max_change_rate) train_step_lists[[ti]] <- tf$train$MomentumOptimizer(learningrate/rater^(ti-1), momentum)$minimize(tf$reduce_mean(cross_entropy) );

	
	##**************************************
	pred <- tf$argmax(y_conv_train, 1L);	
	correct_prediction <- tf$equal(tf$argmax(y_conv_train, 1L), tf$argmax(y_, 1L));	
	
	accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32));
	
	##********************************************************************************************************************************************************
	## start to train CNN
	
	session_conf <- tf$ConfigProto(intra_op_parallelism_threads=MAX_THREADS, inter_op_parallelism_threads=MAX_THREADS);	
	sess <- tf$InteractiveSession(config=session_conf);
	sess$run(tf$global_variables_initializer())
	saver <- tf$train$Saver();
	
	factor.batchsize <- 2;
	# factor.batchsize <- 4;
	# factor.batchsize <- 32*32;
	
	steps <- (iters/factor.batchsize);
	steps.validation <- ceiling(10/factor.batchsize*totaln/1590919)*100;
	if (totaln<50000) steps.validation <- 20; 
	# steps.validation <- 50/factor.batchsize;
	# steps.validation <- 1024/factor.batchsize;
	cat("\tbatch size =", batchsize*factor.batchsize, "learning rate =", learningrate, "iterations =", iters, "MAX_THREADS =", MAX_THREADS, "steps.validation =", steps.validation, "\n");
	
	## steps
	accuracies <- vector(,  length=ceiling(iters/steps)); 
	kappas     <- vector(,  length=ceiling(iters/steps)); 
	accuracies.validation <- vector(,  length=ceiling(iters/factor.batchsize/steps.validation)); 
	accuracies.training   <- vector(,  length=ceiling(iters/factor.batchsize/steps.validation)); 
	learning.rates        <- vector(,  length=ceiling(iters/factor.batchsize/steps.validation)); 
	inner_i_validation <- 0;	
	which_rate <- 1; 
	# plat_accuracy <- 0.01; 
	plat_n <- 12; last_change_i <- 0;
	# print(pop_mean)
	is_shuffle <- 0;
	batch_position <- totaln+1;
	
	for (i in 1:steps ) {
	# for (i in 1:(10) ) {
		## *********************************
		## next batch
		# batch <- next_batch(dat.train, totaln, dimx=input_dim_stacked, n=batchsize*factor.batchsize, classn);
		epoch <- ceiling (i*batchsize*factor.batchsize/totaln); 
		if (batch_position>totaln ) { 
			# if (epoch%%5==0) cat("\tData is shuffled:) at epoch", epoch, "\n"); 
			shuffle_index <- sample.int(totaln, totaln); batch_position = 1; 
		} 
		
		batch <- next_batch_with_shuffle (shuffle_index, dat.train, totaln, startx=batch_position, dimx=input_dim_stacked, n=batchsize*factor.batchsize, classn); 
		batch_position <- batch_position + batchsize*factor.batchsize; 
		
		## *********************************
		## determine whether to reduce the learning rate
		min_epoch <- 2; 
		if ( (learningrate/rater^(which_rate-1)) >=0.01) min_epoch=30; # on May 14 for is_shallow=TRUE is good
		# if ( (learningrate/rater^(which_rate-1)) >=0.01) min_epoch=60; # on May 18 for is_shallow=TRUE is just OK

		if (
			(i-last_change_i)*batchsize*factor.batchsize/totaln>min_epoch ## min_epoch epochs 
			&& which_rate<max_change_rate 
			&& i%%steps.validation==0 
			&& is_greater(accuracies.validation, inner_i_validation, plat_n) 
		) {
			last_change_i <- i;
			which_rate <- which_rate+1;
			if (which_rate<=max_change_rate) cat("\n1.Note that the learning rate is reduced to", learningrate/rater^(which_rate-1), "in", i, "iteration \n");
		} 
		# else if ( (i-last_change_i)*batchsize*factor.batchsize/totaln> (min_epoch+10) ## max_epoch epochs 
		# ) {
			# last_change_i <- i;
			# which_rate <- which_rate+1;
			# if (which_rate<=max_change_rate) cat("\n2.Note that the learning rate is reduced to", learningrate/rater^(which_rate-1), "in", i, "iteration \n");
		# }
		which_rate <- min(max_change_rate, which_rate);
		
		## *********************************
		## gradient descent 
		for (ti in 1:max_change_rate) if (ti==which_rate) train_step_lists[[ti]]$run(feed_dict = dict(x = batch[[1]], y_ = batch[[2]]));
		
		## *********************************
		## record/print training/validation output
		if (i%%steps.validation==0) {	
			inner_i_validation <- i/steps.validation;
			accuracies.training[inner_i_validation] <- accuracy$eval(feed_dict = dict(x = batch[[1]], y_ = batch[[2]]));
			accuracies.validation[inner_i_validation] <- accuracy$eval(feed_dict = dict(x = batch.validation[[1]], y_ = batch.validation[[2]]));
			learning.rates[inner_i_validation] <- learningrate/rater^(which_rate-1); 
			if(inner_i_validation%%20==0 || (inner_i_validation<20&&inner_i_validation%%10==0) ) cat(sprintf("  %7d, training accuracy %.4f, validation accuracy %.4f\n", i, accuracies.training[inner_i_validation], accuracies.validation[inner_i_validation])); 
			if (accuracies.validation[inner_i_validation]<0.1) { cat("\n!!!SHIT happen again on i =", i, "\n");  break; } 
		}
		
		## *********************************
		## record/print test output using the trained graph
		# if (i%%steps==0 || i==iters) {			
			# ptm_cnn_i <- proc.time();
			# predicted  <- pred$eval(feed_dict = dict(x = dat.test[,1:(input_dim_stacked)]));
			# predicted2 <- predicted;
			
			## convert from a number from 1:n to class label 
			# for ( ip in sort(unique(predicted)) ) predicted2[predicted==ip] <- classes[ip+1];
			# time.my <- proc.time() - ptm_cnn_i;
			# cat(paste("\n\tTest data time using train model", sprintf("%.3f",time.my[1]/3600),"\t", sprintf("%.3f",time.my[2]/3600),"\t", sprintf("%.3f",time.my[3]/3600),"hours\n"));
			
			# result <- confusion(predicted2, ytest);	
			# inner_i <- i/steps;
			# kappas[inner_i] <- result$Kappa;
			# accuracies[inner_i] <- result$Accuracy;
		# }
	}
	time.train <- proc.time() - ptm_cnn;
	cat(paste("\nTraining  data time", sprintf("%.3f",time.train[1]/3600),"\t", sprintf("%.3f",time.train[2]/3600),"\t", sprintf("%.3f",time.train[3]/3600),"hours\n"));
	
	# for (i in 1:depth) print(sess$run(weightlists[[i]]) ); 
	
	plot.dir <- "./plot.dir/"; 
	if (! dir.exists(plot.dir)) {print(paste("Creating dir", plot.dir)); dir.create(plot.dir); }
	unique.file.name <- paste("DNN_type", con_layer, ".is_shallow", is_shallow, ".sample", samplerate, 
		".learning", learningrate, ".is_norm", is_norm, 
		".perm", permutation[1], ".beta", beta.coe,
		# ".max_change_rate", max_change_rate, "change_rate_ratio", rater, 
		".iter", iters, ".max_change", max_change_rate, sep="");
	modir.dir <- paste0("./model.dir/",unique.file.name); 
	if (! dir.exists(modir.dir)) {print(paste("Creating dir", modir.dir)); dir.create(modir.dir); }
	save(meanx, sdx, file=paste0(modir.dir,"//mean_std.rfile.data"))
	model.file.name <- paste(modir.dir, "/model.", unique.file.name, sep=""); 
	saver$save(sess, model.file.name);
	sess$close();

	save(samplerate, weightlists, kappas, accuracies, learning.rates, accuracies.validation, accuracies.training, 
		file=paste(plot.dir, unique.file.name, sep="") );
	
	## ***********************************************************************************************
	## get prediction 1
	cat("\nStart up the testing model....\n");
	tf$reset_default_graph();
	x  <- tf$placeholder(tf$float32, shape(NULL, input_dim_stacked));
	bg_build_graph_test <- build_graph(con_layer, is_training=FALSE, x, input_dim_x, input_dim_y, is_norm=is_norm, is_shallow=is_shallow); 
	y_conv_test <- bg_build_graph_test$y_conv;
	pred <- tf$argmax(y_conv_test, 1L);	
	sess <- tf$InteractiveSession();
	sess$run(tf$global_variables_initializer())
	saver <- tf$train$Saver();
	saver$restore(sess, model.file.name);
	# cat("\tafter restore the saved model\n");
	ptm_cnn_i <- proc.time();
	predicted  <- pred$eval(feed_dict = dict(x = dat.test[,1:(input_dim_stacked)]));

	predicted2 <- predicted;
	for ( i in sort(unique(predicted)) ) predicted2[predicted==i] <- classes[i+1];	## convert from a number from 1:n to class label 
	time.test <- proc.time() - ptm_cnn_i;
	cat(paste("\nTest data time", sprintf("%.3f",time.test[1]/3600),"\t", sprintf("%.3f",time.test[2]/3600),"\t", sprintf("%.3f",time.test[3]/3600),"hours\n"));

	sess$close();
		

	## ***********************************************************************************************
	## get confusion
	result <- confusion(predicted2, ytest, filename=paste(modir.dir, "cnn.temp", unique.file.name, ".csv", sep="") );
	
	result$train.time <- time.train/3600; 
	result$test.time <- time.test/3600; 
	result
}
