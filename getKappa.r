## Hankui wrote it on Aug 27 2016

# source("getKappa.r");
# a is predicted 
# b is ground truth

getKappa <- function (a, b) 
{
	# a <- iris.rf2$predicted[is.element.index]
	# b <- yftrain[is.element.index]
	samplesn <- length(a);
	classes <- union(unique(a),unique(b));
	lengthc <- length(classes);
	kappaa <- 0;
	if (lengthc<=1 || length(unique(b))<=1) {kappaa <- 1;return(NA);}
	conf1 <- vector(mode="double", length=lengthc);
	conf2 <- vector(mode="double", length=lengthc);
	for (i in 1:lengthc) {
		conf1[i] <- sum(a==classes[i])
		conf2[i] <- sum(b==classes[i])
	}
	qqq <- conf1/samplesn;
	ppp <- conf2/samplesn;
	expAccuracy <- sum(ppp*qqq);
	OA <- sum(a==b)/samplesn;
	kappaa = (OA - expAccuracy)/(1 - expAccuracy)
	return(kappaa);
}