# Hankui in SDSU
# source("build_graph.r");

# build the CNN structure 
## con_indicator (2 or 4) indicates the whether convolution is used or not
## is_shallow indicates the DNN structure
# input_dim_x=13; input_dim_y=3;

source("cnn.common.r");

#************************************************************************************************************************
## non-linear activation function
## type = 0: RELU
## type = 1: parametric RELU He et al. 2015
## type = 6: RELU6
non_linear <- function(x, ndim=shape(100), type=1) {
	alphas <- tf$Variable(tf$ones(1L)*0.25);	
	if (type==1) {
		pos <- tf$nn$relu(x);
		# neg <- alphas * (x - abs(x)) * 0.5; ## not work
		# neg <- (x - abs(x)) * 0.5; ## work
		# neg <- tf$scalar_mul(alphas, (x - abs(x)) * 0.5 ); ## not work
		neg <- tf$multiply(alphas, (x - abs(x)) * 0.5 ); ## work
		result <- pos + neg; 
	} else if (type==6) {
		result <- tf$nn$relu6(x);
	} else result <- tf$nn$relu(x);
	
	result
}


#************************************************************************************************************************
## a neural network unit including a linear transformation (convolution/fully connection) and a non-linear function
## leftwidth*leftlength*n1 representing the dimension of the convolution layer to be normalized if is_norm=TRUE and is_con=TRUE
##                      n1 representing the dimension of the fully connected layer to be normalized if is_norm=TRUE and is_con=FALSE
nn_unit <- function (x_image, W_conv1, b_conv1, leftwidth=1, leftlength=11, n1=100, is_norm=TRUE, is_training=TRUE, non.linear.type=0, is_con=TRUE, is_pad=FALSE, is_max_pool=FALSE) {

	# i=1
	# is_training = TRUE
	# x_image=h_convs[[i]]
	# W_conv1=weightlists[[i]]
	# b_conv1=baislists[[i]]
	# leftwidth=left_widths[i]
	# leftlength=left_lengths[i]
	# n1=n_nodes[i]
	# is_training=is_training; 
	# non.linear.type=non.linear.type;
	# is_con=TRUE
	# is_pad=FALSE

	
	if (is_norm) {
		if (is_con) { 
			h_cov_nopad_1 <- conv2d_no_pad(x_image, W_conv1, is_pad); 
			h_batch_1 <- batch_norm_wrapper(h_cov_nopad_1, is_training, ndim=shape(leftwidth,leftlength,n1) ); 
			h_conv1 <- non_linear(h_batch_1, type=non.linear.type);
		} else { 
			h_fc_last <- tf$matmul(x_image, W_conv1); 
			h_batch_last <- batch_norm_wrapper(h_fc_last, is_training, ndim=shape(n1) ); 
			h_conv1 <- non_linear(h_batch_last, type=non.linear.type);
		}
		# h_conv1 <- tf$nn$relu6(h_batch_1);
	} else {
		if (is_con) { 
			h_cov_nopad_1 <- conv2d_no_pad(x_image, W_conv1, is_pad) + b_conv1; 
			h_conv1 <- non_linear(h_cov_nopad_1, type=non.linear.type);
		} else {
			h_cov_nopad_1 <- tf$matmul(x_image, W_conv1) + b_conv1; 
			h_conv1 <- non_linear(h_cov_nopad_1, type=non.linear.type);
		}
	}
	if (is_max_pool==TRUE) {
		h_conv1_pull <- max_pool_2x2(h_conv1);
		h_conv1 <- h_conv1_pull;
	} else if (is_max_pool==2) {
		# print("Average pooling has been used \n");
		h_conv1_pull <- avg_pool_2x2(h_conv1);
		# h_conv1_pull <- max_pool_2x2(h_conv1);
		# h_conv1_pull <- max_pool_1x2(h_conv1);
		h_conv1 <- h_conv1_pull;
	}
	# h_pool1_in <- max_pool_1x2(h_conv1);
	h_conv1
}

#************************************************************************************************************************
## global variables for building graph
max_depth   <- 21; 
weightlists <<- vector("list", max_depth); 
baislists   <<- vector("list", max_depth); 

#************************************************************************************************************************
## build the graph for training or testing
## con_indicator (2 or 4) indicates the whether convolution is used or not
## is_shallow indicates the DNN structure
# is_training=TRUE;
# input_dim_x=13; input_dim_y=3;
build_graph <- function (con_indicator, is_training=TRUE, x, input_dim_x, input_dim_y, input_dim_z=1L, is_norm=TRUE, is_shallow=TRUE) {

	input_dim_stacked <- input_dim_x*input_dim_y*input_dim_z;
	non.linear.type <- 6;
	non.linear.type <- 1;
	non.linear.type <- 0;
	cat("\tbuild_graph with non.linear.type", non.linear.type, "\n");
	## ***************************************
	## option 2: 2d filter followed by a fully connected layer
	isdim <- FALSE;
	# isdim <- TRUE;
	n_nodes <- vector(mode="integer",  length=max_depth);
	depth <<- 4;
	conv_depth <- depth-2;
	# n1 <- 0L; n2 <- 0L; n3 <- 0L; n4 <- 0L; n5 <- 0L;  n6 <- 0L;  n7 <- 0L;
	if (con_indicator==2) {
		# left_lengths <- vector(mode="integer",  length=max_depth);
		## dimension left after convolution 
		left_lengths <- rep(1L, max_depth);
		left_widths <- left_lengths;
		
		con_dims  <- rep(1L, max_depth);
		con_dim2  <- rep(1L, max_depth);
		
		con_dims[1  ] <- 3;
		con_dim2[1:6] <- 3;
		
		n_nodes[1] <- 64L;
		n_nodes[2] <- 128L;
		n_nodes[3] <- 256L; 
		# n_nodes[1] <- 128L;
		# n_nodes[2] <- 256L;
		# n_nodes[3] <- 1024L; ## tested on May 23 2018 for simplified VGG accuracy
		left_lengths[1] <- 11L;
		left_lengths[2] <- 9L;
		# left_lengths[3] <- 7L; 		
		
		if (is_shallow==FALSE) {
			depth <<- 8;
			conv_depth <- depth-2;
			
			left_lengths[1] <- 11L;
			left_lengths[2] <- 9L;
			left_lengths[3] <- 7L; 
			left_lengths[4] <- 5L; 
			left_lengths[5] <- 3L; 
			left_lengths[6] <- 1L; 
			
			n_nodes[1] <- 128L;
			n_nodes[1] <- 64L; ## revised on Nov 1 2018
			n_nodes[2] <- 128L;
			n_nodes[3] <- 256L;
			n_nodes[4] <- 256L;
			n_nodes[5] <- 512L;
			n_nodes[6] <- 512L;
			n_nodes[7] <- 1024L; 
		} else if (is_shallow==5) {
			depth <<- 19;
			conv_depth <- depth-3;## note this has been changed from depth-2 to depth-3 on June 20, 2018
			left_widths[1:1 ] <- 3;
			
			con_dims[1:2 ] <- 3;
			con_dim2[1:17] <- 3;
				
			left_lengths[1 ] <- 13L;
			left_lengths[2 ] <- 11L;
			left_lengths[3 ] <- 11L; 
			left_lengths[4 ] <- 9L; 
			left_lengths[5 ] <- 9L; 
			left_lengths[6 ] <- 9L; 
			left_lengths[7 ] <- 9L; 
			left_lengths[8 ] <- 7L; 
			left_lengths[9 ] <- 7L; 
			left_lengths[10] <- 7L; 
			left_lengths[11] <- 5L; 
			left_lengths[12] <- 5L; 
			left_lengths[13] <- 5L; 
			left_lengths[14] <- 3L; 
			left_lengths[15] <- 3L; 
			# left_lengths[16] <- 3L; ## NOTE THAT THIS HAS BEEN CHANGED FROM 1 to 3
			left_lengths[16] <- 1L; ## NOTE THAT THIS HAS BEEN CHANGED FROM 3 to 1 on June 20, 2018
			left_lengths[17] <- 1L; 

			n0 <- 64  ; n_nodes[ 1] <- n0; n_nodes[ 2] <- n0; 
			n0 <- 128 ; n_nodes[ 3] <- n0; n_nodes[ 4] <- n0; 
			n0 <- 256 ; n_nodes[ 5] <- n0; n_nodes[ 6] <- n0; n_nodes[ 7] <- n0; n_nodes[ 8] <- n0; 
			n0 <- 512 ; n_nodes[ 9] <- n0; n_nodes[10] <- n0; n_nodes[11] <- n0; n_nodes[12] <- n0; n_nodes[13] <- n0; n_nodes[14] <- n0; n_nodes[15] <- n0; n_nodes[16] <- n0; 
			n0 <- 1024; n_nodes[17] <- n0; n_nodes[18] <- n0; 
		}
		
		# is_use_pooling = 2; 
		# is_use_pooling = TRUE; 
		is_use_pooling = FALSE; 
		if (is_use_pooling>=1) {
			con_dims[1:2] <- 3;
			con_dim2[1:6] <- 3;
			
			n_nodes[1] <- 64L;
			n_nodes[2] <- 128L;
			n_nodes[3] <- 256L; 
			left_lengths[1] <- 13L;
			left_lengths[2] <- 7L;
			left_lengths[3] <- 4L;
			left_widths[1] <- 3L;
			left_widths[2] <- 2L;
			left_widths[3] <- 1L;
			
			if (is_shallow==FALSE) {
				depth <<- 7;
				conv_depth <- depth-2;
				con_dims[1:3] <- 3;
				con_dim2[1:5] <- 3;
				
				left_lengths[1] <- 13L;
				left_lengths[2] <- 7L;
				left_lengths[3] <- 4L;
				left_lengths[4] <- 2L;
				left_lengths[5] <- 1L;
				
				left_widths[1] <- 3L;
				left_widths[2] <- 2L;
				left_widths[3] <- 1L;
				
				n_nodes[1] <- 64L;
				n_nodes[2] <- 128L;
				n_nodes[3] <- 256L;
				n_nodes[4] <- 256L;
				n_nodes[5] <- 512L;
				n_nodes[6] <- 1024L; 
			} else if (is_shallow==5) {
				depth <<- 16;
				conv_depth <- depth-3;## note this has been changed from depth-2 to depth-3 on June 20, 2018
				
				con_dims[1:7 ] <- 3;
				con_dim2[1:13] <- 3;
					
				left_lengths[1 ] <- 13L;
				left_lengths[2 ] <- 13L;
				left_lengths[3 ] <- 13L;
				left_lengths[4 ] <- 7L;
				left_lengths[5 ] <- 7L;
				left_lengths[6 ] <- 7L; 
				left_lengths[7 ] <- 4L; 
				left_lengths[8 ] <- 4L;
				left_lengths[9 ] <- 4L;
				left_lengths[10] <- 2L;
				left_lengths[11] <- 2L;
				left_lengths[12] <- 2L;
				left_lengths[13] <- 1L; 
				
				
				left_widths[1 ] <- 3L;
				left_widths[2 ] <- 3L;
				left_widths[3 ] <- 3L;
				left_widths[4 ] <- 2L;	
				left_widths[5 ] <- 2L;
				left_widths[6 ] <- 2L;
				left_widths[7 ] <- 1L;

				# left_lengt14hs[1 ] <- 13L;
				# left_lengths[2 ] <- 11L;
				# left_lengths[3 ] <- 11L; 
				# left_lengths[4 ] <- 9L; 
				# left_lengths[5 ] <- 9L; 
				# left_lengths[6 ] <- 9L; 
				# left_lengths[7 ] <- 9L; 
				# left_lengths[8 ] <- 7L; 
				# left_lengths[9 ] <- 7L; 
				# left_lengths[10] <- 7L; 
				# left_lengths[11] <- 5L; 
				# left_lengths[12] <- 5L; 
				# left_lengths[13] <- 5L; 
				# left_lengths[14] <- 3L; 
				# left_lengths[15] <- 3L; 
				# left_lengths[16] <- 1L; ## NOTE THAT THIS HAS BEEN CHANGED FROM 3 to 1 on June 20, 2018
				# left_lengths[17] <- 1L; 

				n0 <- 64  ; n_nodes[ 1] <- n0; n_nodes[ 2] <- n0; 
				n0 <- 128 ; n_nodes[ 3] <- n0; n_nodes[ 4] <- n0; 
				n0 <- 256 ; n_nodes[ 5] <- n0; n_nodes[ 6] <- n0; n_nodes[ 7] <- n0; 
				n0 <- 512 ; n_nodes[ 8] <- n0; n_nodes[ 9] <- n0; n_nodes[10] <- n0; n_nodes[11] <- n0; n_nodes[12] <- n0; n_nodes[13] <- n0; 
				n0 <- 1024; n_nodes[14] <- n0; n_nodes[15] <- n0;
			}
		} ## pooling is true
		
		## weights and bias
		for (i in 1:depth) {
			scale_factor <- 2;
			if (i==1) {
				# weightlists[[i]] <- weight_variable(shape(con_dims[i], con_dim2[i], 1, n_nodes[i]), stddev=sqrt(scale_factor/(con_dims[i]*con_dim2[i]*n_nodes[i])) );
				weightlists[[i]] <- weight_variable(shape(con_dims[i], con_dim2[i], 1, n_nodes[i]), stddev=sqrt(scale_factor/(con_dims[i]*con_dim2[i])) );
				baislists  [[i]] <- bias_variable(shape(1, left_lengths[i], n_nodes[i]) );
			} else if (i<=conv_depth) {
				weightlists[[i]] <- weight_variable(shape(con_dims[i], con_dim2[i], n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/(con_dims[i]*con_dim2[i]*n_nodes[i])) );
				baislists  [[i]] <- bias_variable(shape(1, left_lengths[i], n_nodes[i]) );
			} else if (i==(conv_depth+1) ) { ## last convolution layer
				concatenation <- left_widths[i-1] * left_lengths[i-1];
				if (is_use_pooling>=1) concatenation <- ceiling(left_widths[i-1]/2) * ceiling(left_lengths[i-1]/2); ## max pool is used 
				# weightlists[[i]] <- weight_variable(shape(1 * left_lengths[i-1] * n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/(1*1*n_nodes[i])) );
				weightlists[[i]] <- weight_variable(shape(concatenation * n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/(1*1*n_nodes[i])) );
				baislists  [[i]] <- bias_variable(shape(1, n_nodes[i]) );
			} else if (i<=(depth-1) ) { ## fully connected layers 
				# concatenation <- left_widths[i-1] * left_lengths[i-1];
				# if (is_use_pooling) concatenation <- ceiling(left_widths[i-1]/2) * ceiling(left_lengths[i-1]/2); ## max pool is used 
				# weightlists[[i]] <- weight_variable(shape(1 * left_lengths[i-1] * n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/(1*1*n_nodes[i])) );
				weightlists[[i]] <- weight_variable(shape(n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/(1*1*n_nodes[i])) );
				baislists  [[i]] <- bias_variable(shape(1, n_nodes[i]) );			
			} else {
				W_fc_n  <- weight_variable(shape(n_nodes[i-1], classn), stddev=sqrt(scale_factor/(1*1*n_nodes[i-1])) ); 
				b_fc_n  <- bias_variable(shape(classn) );
				weightlists[[i]] <- W_fc_n; 
				baislists  [[i]] <- b_fc_n; 
			}
		}
		
		## convolution
		h_conv0 <- tf$reshape(x, shape(-1L, input_dim_y, input_dim_x, input_dim_z));
		
		h_convs <- list(); 
		h_convs[[1]] <- h_conv0; 
		if (is_use_pooling==FALSE) { ## pooling is NOT used
			if (is_shallow<5) {
				for (i in 1:conv_depth)
					h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE); 
			} else {
				for (i in 1 :1 )
					h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=TRUE);
				for (i in 2:conv_depth)
					h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=(left_lengths[i-1]==left_lengths[i])  );
			}
		} else { ## pooling is used is_max_pool = if (left_widths[i-1]>1 && left_lengths[i-1]>1) { TRUE } else if (left_widths[i-1]==1 && left_lengths[i-1]>1) {2} else {FALSE}
				## revised on Aug 10 2019 for pooling is true 
			for (i in 1:conv_depth) { 
				# is_max_pool = if (left_widths[i-1]>1 && left_lengths[i-1]>1) { TRUE } else if (left_widths[i-1]==1 && left_lengths[i-1]>1) {2} else {FALSE};
				# is_max_pool = if (left_lengths[i]==left_lengths[i+1]) {FALSE} else { if (left_widths[i]>1 && left_lengths[i]>1) { TRUE } else if (left_widths[i]==1 && left_lengths[i]>1) {2} else {FALSE} };
				is_max_pool <- (left_lengths[i]!=left_lengths[i+1]);
				if (is_max_pool && is_use_pooling==2) is_max_pool <- 2;
				h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=TRUE, is_max_pool=is_max_pool); 
			}
		}
		## fully connected layer
		# h_pool2_flat <- tf$reshape(h_convs[[conv_depth+1]], shape(-1L, 1 * left_lengths[conv_depth] * n_nodes[conv_depth]));
		h_pool2_flat <- tf$reshape(h_convs[[conv_depth+1]], shape(-1L, concatenation*n_nodes[conv_depth]));
		h_convs[[conv_depth+1]] <- h_pool2_flat; 
		
		for (i in (conv_depth+1):(depth-1) ) 
			h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftlength=left_lengths[i], n1=n_nodes[i], is_norm=is_norm, is_training=is_training, non.linear.type=non.linear.type, is_con=FALSE); 

		
		if (isdim) {
			cat("W_fc_n: ") ; get_dim(W_fc_n);
			cat("b_fc_n: ") ; get_dim(b_fc_n);
			cat("y_conv: ") ; get_dim(y_conv);
			stop("Stop because of isdim CHECK");
		}
		
	}

	## ***************************************
	## option 4: only contain fully connected layers
	if (con_indicator==4) {
		conv_depth <- 0;
		## ***************************************
		## option 4: only one filter is applied
		if (is_shallow==TRUE) {
			n0 <- 256L;
			n1 <- 512L;
			nn <- 512L; ## S2: z=256; x=512; 39*z+ z* x + x* x + x* 15 (410880): ?
			n_nodes[1] <- n0;
			n_nodes[2] <- n1;
			n_nodes[3] <- n1;
		} else if (is_shallow==FALSE) {
			depth <<- 8; 
			n0 <- 256L;
			n1 <- 512L;
			nn <- 1024L; ## S2: z=256; x=512; y=1024L; 39*x + x* x + x* x + x* x + x* x + x* y + y*y + y*15 (2656768): 84.5174%
			##S3: z=256L; x=512L; y=1024L; 39*z + z* z + z* z + z* x + x* x + x* y + y*y + y*15 (2122496)

			n_nodes[1] <- n0;
			n_nodes[2] <- n0;
			n_nodes[3] <- n0;
			n_nodes[4] <- n1;
			n_nodes[5] <- n1;
			n_nodes[6] <- nn;
			n_nodes[7] <- nn; 
		} else {
			depth <<- 19;
			n0 <- 64  ; n_nodes[ 1] <- n0; n_nodes[ 2] <- n0; 
			n0 <- 128 ; n_nodes[ 3] <- n0; n_nodes[ 4] <- n0; 
			n0 <- 256 ; n_nodes[ 5] <- n0; n_nodes[ 6] <- n0; n_nodes[ 7] <- n0; n_nodes[ 8] <- n0; 
			n0 <- 512 ; n_nodes[ 9] <- n0; n_nodes[10] <- n0; n_nodes[11] <- n0; n_nodes[12] <- n0; n_nodes[13] <- n0; n_nodes[14] <- n0; n_nodes[15] <- n0; n_nodes[16] <- n0; 
			# n0 <- 4096; n_nodes[17] <- n0; n_nodes[18] <- n0; 
			n0 <- 1024; n_nodes[17] <- n0; n_nodes[18] <- n0; 
		}

		## weights and bias
		widths0 <- input_dim_stacked; 
		# weightlists <- list(); 
		# baislists <- list(); 
		for (i in 1:depth) {
			scale_factor <- 2;
			if (i==1) {
				weightlists[[i]] <<- weight_variable(shape(widths0, n_nodes[i]), stddev=sqrt(scale_factor/n_nodes[i]) );
				baislists  [[i]] <<- bias_variable(shape(n_nodes[i]) );
			} else if (i<=depth-1) {
				weightlists[[i]] <<- weight_variable(shape(n_nodes[i-1], n_nodes[i]), stddev=sqrt(scale_factor/n_nodes[i]) );
				baislists  [[i]] <<- bias_variable(shape(n_nodes[i]) );
			} else {
				W_fc_n  <- weight_variable(shape(n_nodes[i-1], classn), stddev=sqrt(scale_factor/n_nodes[i-1]) ); 
				b_fc_n  <- bias_variable(shape(classn) );
				weightlists[[i]] <<- W_fc_n; 
				baislists  [[i]] <<- b_fc_n; 
			}
		}

		## convolution / non-linear functions
		h_conv0 <- tf$cast(tf$reshape(x, shape(-1L, widths0)), tf$float32);
		
		h_convs <- list(); 
		h_convs[[1]] <- h_conv0; 
		for (i in 1:(depth-1) )
			h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftlength=left_lengths[i], n1=n_nodes[i], is_norm=is_norm, is_training=is_training, non.linear.type=non.linear.type, is_con=FALSE); 

		## softmax layer
		h_fc_n_drop <- h_convs[[depth]]; 
		
		## final layer
		# y_conv <- tf$nn$softmax(tf$matmul(h_fc_n_drop, W_fc_n) + b_fc_n);
		
		if (isdim) {
			cat("W_fc2 : ") ; get_dim(W_fc2 );
			cat("b_fc2 : ") ; get_dim(b_fc2 );
			cat("y_conv: ") ; get_dim(y_conv);
			stop("Stop because of isdim CHECK");
		}
	}
	
	# last_active <- tf$matmul(h_convs[[depth]], W_fc_n); 
	last_active <- tf$matmul(h_convs[[depth]], W_fc_n) + b_fc_n; 
	last_mean <- tf$reduce_mean(last_active, 1L);
	y_conv  <- tf$nn$softmax(last_active);
	
	## regularizer
	regularizer <- 0;
	# for (i in 1:(depth-1) )  {
		# if (i<=conv_depth) {
			# regularizer <- regularizer+tf$nn$l2_loss(weightlists[[i]]); 
		# } else regularizer <- regularizer+10*tf$nn$l2_loss(weightlists[[i]]); 
	# }
	# regularizer <- regularizer+10*tf$nn$l2_loss(W_fc_n); 

	for (i in 1:(depth-1) )  {
		regularizer <- regularizer+tf$nn$l2_loss(weightlists[[i]]); 
	}
	regularizer <- regularizer+tf$nn$l2_loss(W_fc_n); 
	
	## print information
	cat("\tcon_indicator =", con_indicator, "depth =", depth, "conv_depth =", conv_depth, "is_use_pooling =", is_use_pooling, "\n\t");
	for (i in 1:(depth-1) ) cat("width[", i, "] = ", n_nodes[i], "  ", sep="");
	cat("\n");
	# if (con_indicator==2) cat("\t\tn3 = ", n3, "n4 = ", n4, "n5 = ", n5, "n6 = ", n6, "\n"); 

	list("y_conv"=y_conv, "regularizer"=regularizer, "last_active"=last_active, last_mean="last_mean")
}

			# if (is_shallow<6) {
				# i = 1;
				# is_max_pool = TRUE; 
				# h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=TRUE, is_max_pool=is_max_pool); 
			# } else {
				# for (i in 1 :1 )
					# h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=TRUE);
				# for (i in 2:conv_depth)
					# h_convs[[i+1]] <- nn_unit(h_convs[[i]], weightlists[[i]], baislists[[i]], leftwidth=left_widths[i], leftlength=left_lengths[i], n1=n_nodes[i], is_training=is_training, is_con=TRUE, is_pad=(left_lengths[i-1]==left_lengths[i])  );
			# }
