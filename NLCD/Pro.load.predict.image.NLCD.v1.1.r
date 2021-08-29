# v1.1 on Jul 16 2021 to predict all the tiles

# rm(list=ls(all=TRUE)); 
# samplerate <- 0.9;  con_layer <- 2; is_shallow=5    ; learningrate <- 0.01; is_norm <- TRUE; beta.coe<-1E-3;   iters <- 1600000; 
# samplerate <- 0.9;  con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE; beta.coe<-1E-3;   iters <- 1600000; 

# 4-layer 0.1 sample rate
# rm(list=ls(all=TRUE)); samplerate <- 0.1;  con_layer <- 2; is_shallow=TRUE ; learningrate <- 0.01; is_norm <- TRUE; beta.coe<-1E-3;   iters <- 180000; source("Pro.load.predict.image.v1.1.r")

# 8-layer 0.9 sample rate
# rm(list=ls(all=TRUE)); samplerate <- 0.9;  con_layer <- 2; is_shallow=FALSE; learningrate <- 0.01; is_norm <- TRUE; beta.coe<-1E-3;   iters <- 1600000; source("Pro.load.predict.image.v1.1.r")
# source("Pro.load.predict.image.r")

# read hdf4 refer to  
# https://stackoverflow.com/questions/36772341/reading-hdf-files-into-r-and-converting-them-to-geotiff-rasters

ptm_cnn_start <- proc.time();

# Hank on Jul 11, 2021
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
filename <- paste(linux_folder, "/metric.ard.nlcd.Mar01.18.40.txt", sep=""); 
## number of threads when running paralleling computation depending on the number of available cores 
MAX_THREADS <- 24L;
# MAX_THREADS <- 36L;

##***************************************************************************************************
## load parameters 
# meanx <- apply(x_input, 2, mean);
# sdx   <- apply(x_input, 2, sd);
# x_input_norm <- t((t(xtrain[train.index,])-meanx)/sdx)
load(file="parameters.rfile.data")                              
classn = length (classes)
max_change_rate = 5
input_dim_x=13; input_dim_y=3
input_dim_stacked <- input_dim_x*input_dim_y;
permutation <- 0;
unique.file.name <- paste("DNN_type", con_layer, ".is_shallow", is_shallow, ".sample", samplerate, 
	".learning", learningrate, ".is_norm", is_norm, 
	".perm", permutation[1], ".beta", beta.coe,
	".iter", iters, ".max_change", max_change_rate, sep="");

modir.dir <- paste0("./model.dir/",unique.file.name); 
model.file.name <- paste(modir.dir, "/model.", unique.file.name, sep=""); 
load(file=paste0(modir.dir,"/mean_std.rfile.data"))

## get cnn model
cat("\nLoad the prediction model....\n");
tf$reset_default_graph();
x  <- tf$placeholder(tf$float32, shape(NULL, input_dim_stacked));
bg_build_graph_test <- build_graph(con_layer, is_training=FALSE, x, input_dim_x, input_dim_y, is_norm=is_norm, is_shallow=is_shallow); 
y_conv_test <- bg_build_graph_test$y_conv;
pred <- tf$argmax(y_conv_test, 1L);	
session_conf <- tf$ConfigProto(intra_op_parallelism_threads=MAX_THREADS, inter_op_parallelism_threads=MAX_THREADS);	
# sess <- tf$InteractiveSession();
sess <- tf$InteractiveSession(config=session_conf);
sess$run(tf$global_variables_initializer())
saver <- tf$train$Saver();
saver$restore(sess, model.file.name);

##***************************************************************************************************
## set directory 
require("raster")
DIR="/gpfs/data2/workspace/zhangh/ard.Mar01.2018/metric.of.daily/"
DIR_TIF="/gpfs/data2/workspace/zhangh/ard.Mar01.2018/metric.of.daily.tif/"
CNN_TIF=paste0("/gpfs/data2/workspace/zhangh/ard.Mar01.2018/cnn.samplerate",samplerate,".is_shallow",is_shallow,"/")
if (! dir.exists(DIR_TIF)) {print(paste("Creating dir", DIR_TIF)); dir.create(DIR_TIF); }
if (! dir.exists(CNN_TIF)) {print(paste("Creating dir", CNN_TIF)); dir.create(CNN_TIF); }
h= 4; v=11
h= 4; v= 9
# h= 4; v= 9

for (h in c(0:32)) {
for (v in c(0:21)) {
# for (h in c(12:13)) {
# for (v in c(12:13)) {
	##***************************************************************************************************
	## load image 
	hv_str = paste0("h", sprintf("%02d",h),"v",sprintf("%02d",v))
	bands_file = paste0(DIR,"daily.metric.",hv_str,".hdf")
	ratio_file = paste0(DIR,"daily.metric.",hv_str,".hdf.ratio.hdf")
	if (!(file.exists(bands_file) && file.exists(ratio_file) )) {
		cat("!!! file does not exists ", hv_str, "\n")
		next 
	}
	library(gdalUtils)

	# Provides detailed data on hdf4 files but takes ages
	gdalinfo(bands_file)
	# Tells me what subdatasets are within my hdf4 MODIS files and makes them into a list
	sds <- get_subdatasets(bands_file)
	sds[ 8:12] ## 25% percentile bands 2-7
	sds[14:18] ## 50% percentile bands 2-7
	sds[20:24] ## 75% percentile bands 2-7
	sds[38:40] ## 25%, 50% and 75% for NDVI
	IMAGE_DIM = 5000
	x_input =  matrix(,nrow=IMAGE_DIM*IMAGE_DIM,ncol= 39 ); 

	## 25% percentile bands 2-7
	for (i in c(1:5)) {
		print (i+7)
		band = i+1
		if (i>=5) band = i+2
		tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".25percent.band", band,".tif")
		print (tif_file_name)
		if (!(file.exists(tif_file_name))) gdal_translate(sds[i+7], dst_dataset = tif_file_name)
		rast <- raster(tif_file_name)
		x_input[,i] = as.vector(rast)
	}
	## 50% percentile bands 2-7
	for (i in c(1:5)) {
		print (i+13)
		band = i+1
		if (i>=5) band = i+2
		tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".50percent.band", band,".tif")
		print (tif_file_name)
		if (!(file.exists(tif_file_name))) gdal_translate(sds[i+13], dst_dataset = tif_file_name)
		rast <- raster(tif_file_name)
		x_input[,i+6] = as.vector(rast)
	}
	## 50% percentile band 1
	band = 1
	tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".50percent.band", band,".tif")
	print (tif_file_name)
	if (!(file.exists(tif_file_name))) gdal_translate(sds[13], dst_dataset = tif_file_name)

	## 75% percentile bands 2-7
	for (i in c(1:5)) {
		print (i+19)
		band = i+1
		if (i>=5) band = i+2
		tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".75percent.band", band,".tif")
		print (tif_file_name)
		if (!(file.exists(tif_file_name))) gdal_translate(sds[i+19], dst_dataset = tif_file_name)
		rast <- raster(tif_file_name)
		x_input[,i+12] = as.vector(rast)
	}

	## ndvi
	for (i in c(1:3)){
		# ii = 6*i 
		tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".",25*i,"percent.ndvi",".tif")
		print (tif_file_name)
		if (!(file.exists(tif_file_name))) gdal_translate(sds[37+i], dst_dataset = tif_file_name)
		rast <- raster(tif_file_name)
		x_input[,6*i ] = as.vector(rast)
	}

	# Provides detailed data on hdf4 files but takes ages
	gdalinfo(ratio_file)
	# Tells me what subdatasets are within my hdf4 MODIS files and makes them into a list
	sds <- get_subdatasets(ratio_file)
	# 35 = 5% * 7 bands in the order of % then band 
	xi = 18
	for (i in c(1:3)){
	for (b in c(1:7)){
		sdsi = (b-1)*5+i+1
		xi = xi+1
		print (sdsi)
		tif_file_name = paste0(DIR_TIF,"daily.metric.",hv_str,".",25*i,"percent.ratio",b,".tif")
		print (tif_file_name)
		if (!(file.exists(tif_file_name))) gdal_translate(sds[sdsi], dst_dataset = tif_file_name)
		rast <- raster(tif_file_name)
		x_input[,xi ] = as.vector(rast)
	}}

	## test 
	x_input[40*5000+41,]

	 
	## ***********************************************************************************************
	## get prediction 
	cat("\nStart to predict....for ", hv_str, "\n");
	temp.x = x_input
	PREDICTED <- vector(mode="integer", length=IMAGE_DIM*IMAGE_DIM);
	PREDICTED[] <- 0 
	predict_index <- x_input[,6]!= -32768 & x_input[,6]!= -32768 & x_input[,6]!= -32768

	x_input <- cbind(temp.x[, 1:6], temp.x[, 19:25], temp.x[, c(1:6)+6], temp.x[, c(19:25)+7], temp.x[, c(1:6)+12], temp.x[, c(19:25)+14]);
	x_input_norm <- t((t(x_input)-meanx)/sdx)
	# cat("\tafter restore the saved model\n");
	ptm_cnn_i <- proc.time();
	total_lines = 10
	each_sub_n = IMAGE_DIM*IMAGE_DIM/total_lines
	predicted <- vector(mode="integer", length=sum(predict_index));
	subi_start = 1
	subi_end = 0
	for (i in c(1:total_lines)) {
		sub_index = c(1:each_sub_n)+(i-1)*each_sub_n
		pre_n_i = sum(predict_index[sub_index])
		cat("\nprocess line ", i, " of ", total_lines, " start at ", sub_index[1])
		if (pre_n_i>=1) {
			subi_end = pre_n_i+subi_start-1
			predicted[subi_start:subi_end]  <- pred$eval(feed_dict = dict(x = x_input_norm[sub_index,][predict_index[sub_index],]));
			subi_start = subi_end+1
		}
		# predicted  <- pred$eval(feed_dict = dict(x = x_input_norm[predict_index,]));
		# if (i>2) break
	}

	predicted2 <- predicted;
	unique_pred <- sort(unique(predicted))
	for ( i in  unique_pred) predicted2[predicted==i] <- classes[i+1];	## convert from a number from 1:n to class label 
	PREDICTED[predict_index] <- predicted2
	time.test <- proc.time() - ptm_cnn_i;

	## ***********************************************************************************************
	## save prediction 
	current_crs <- crs(rast);
	current_extent <- extent(rast);
	output.raster <- raster(t(matrix(PREDICTED,nrow = IMAGE_DIM,ncol = IMAGE_DIM)) );
	crs(output.raster) <- current_crs;
	extent(output.raster) <- current_extent; 
	cnn_file = paste0(CNN_TIF,"cnn.samplerate",samplerate,".is_shallow",is_shallow,".",hv_str,".tif")

	writeRaster(output.raster, filename=cnn_file, format="GTiff", datatype='INT1U', overwrite=TRUE);
	# writeRaster(output.raster, filename=paste("./predicted.CNN.", ".tif", sep=""), format="GTiff", datatype='INT2S', overwrite=TRUE);


	time.total <- proc.time() - ptm_cnn_start 
	cat(paste("\nTest data time ", sprintf("%.3f",time.test[1]/3600),"\t", sprintf("%.3f",time.test[2]/3600),"\t", sprintf("%.3f",time.test[3]/3600),"hours\n"));
}}

sess$close();
cat(paste("\ntotal data time", sprintf("%.3f",time.total[1]/3600),"\t", sprintf("%.3f",time.total[2]/3600),"\t", sprintf("%.3f",time.total[3]/3600),"hours\n"));

# accuracy = sum(predicted2==y_truth[predict_index])/length(predict_index)
	






# rast2 = t(matrix(x_input[,i],nrow = IMAGE_DIM,ncol = IMAGE_DIM))
# I'm only interested in the first subdataset and I can use gdal_translate to convert it to a .tif
# gdal_translate(sds[8], dst_dataset = "NPP2000.tif")
# Load and plot the new .tif
# rast <- raster("NPP2000.tif")

##***************************************************************************************************
	# if (metric_tile.percent50_NDVI[n] ==  -32768 ||
		# metric_tile.percent25_NDVI[n] ==  -32768 ||
		# metric_tile.percent75_NDVI[n] ==  -32768   ) is_valid = 0;

