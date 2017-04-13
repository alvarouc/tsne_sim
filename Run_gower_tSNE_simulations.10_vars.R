##Let's run gower + tSNE on the python simulations
#load libraries
library(cluster)
library(Rtsne)
library(dplyr)
library(ggplot2)

##Let's start with 10 var sims
#Read in data files
path<-"R:/personal/Anna/Projects/Phenotype_Cluster_Analysis/COPD/Simulation_Analysis/Python_Simulations/10_var_sims/data/"
out<-"R:/personal/Anna/Projects/Phenotype_Cluster_Analysis/COPD/Simulation_Analysis/Python_Simulations/10_var_sims/output_results/"
file.names<-dir(path, pattern="X_5000_")
for (i in 1:length(file.names)) {
  nam<-paste(file.names[i])
  assign(nam, read.table(paste(path,nam,sep=""), header=FALSE, sep=" "))
}
#Read in label files
label.names<-dir(path, pattern="Y_5000_")
for (x in 1:length(file.names)) {
  lab<-paste(label.names[x])
  assign(lab, read.table(paste(path,lab,sep=""), header=FALSE, sep=" "))
}
#Let's change the variable types for the necessary files
X_5000_0_10_0.class<-sapply(X_5000_0_10_0.npy, class)
X_5000_0_10_0.class[1:10]<-"factor"
X_5000_0_10_0.npy<- read.table(file=paste(path,"X_5000_0_10_0.npy",sep=""), sep=" ", header=F, colClasses = X_5000_0_10_0.class)
X_5000_0_2_8.class<-sapply(X_5000_0_2_8.npy, class)
X_5000_0_2_8.class[1:2]<-"factor"
X_5000_0_2_8.npy<- read.table(file=paste(path,"X_5000_0_2_8.npy",sep =""), sep=" ", header=F, colClasses = X_5000_0_2_8.class)
X_5000_0_5_5.class<-sapply(X_5000_0_5_5.npy, class)
X_5000_0_5_5.class[1:5]<-"factor"
X_5000_0_5_5.npy<- read.table(file=paste(path,"X_5000_0_5_5.npy", sep=""), sep=" ", header=F, colClasses = X_5000_0_5_5.class)
X_5000_0_8_2.class<-sapply(X_5000_0_8_2.npy, class)
X_5000_0_8_2.class[1:8]<-"factor"
X_5000_0_8_2.npy<- read.table(file=paste(path, "X_5000_0_8_2.npy", sep=""), sep=" ", header=F, colClasses=X_5000_0_8_2.class)
X_5000_1_1_8.class<-sapply(X_5000_1_1_8.npy, class)
X_5000_1_1_8.class[1]<-"factor"
X_5000_1_1_8.npy<- read.table(file=paste(path, "X_5000_1_1_8.npy", sep=""), sep=" ", header=F, colClasses = X_5000_1_1_8.class)
X_5000_3_2_5.class<-sapply(X_5000_3_2_5.npy, class)
X_5000_3_2_5.class[1:2]<-"factor"
X_5000_3_2_5.npy<- read.table(file=paste(path,"X_5000_3_2_5.npy", sep=""), sep=" ", header=F, colClasses = X_5000_3_2_5.class)
X_5000_4_4_2.class<-sapply(X_5000_4_4_2.npy, class)
X_5000_4_4_2.class[1:4]<-"factor"
X_5000_4_4_2.npy<- read.table(file=paste(path,"X_5000_4_4_2.npy",sep=""), sep=" ", header=F, colClasses = X_5000_4_4_2.class)
X_5000_5_5_0.class<-sapply(X_5000_5_5_0.npy, class)
X_5000_5_5_0.class[1:5]<-"factor"
X_5000_5_5_0.npy<- read.table(file=paste(path,"X_5000_5_5_0.npy", sep=""), sep=" ", header=F, colClasses= X_5000_5_5_0.class)

#Let' run Gower, t-SNE, and create our plots
for (i in 1:length(file.names)) 
  {
  nam<-paste(file.names[i])
  assign(paste(nam,"gower.dm", sep="."), daisy(get(nam), metric=c("gower")))
  set.seed(5)
  assign(paste("rtsne",nam, sep="."), Rtsne(get(paste(nam,"gower.dm", sep=".")), is_distance=TRUE, verbose=TRUE, pca=FALSE, max_iter=1000))
  for (x in 1:length(label.names))
    {
    lab<-paste(label.names[x])
    assign(paste("rtsne",nam,"df", sep="."), get(paste("rtsne",nam, sep="."))$Y %>%
             data.frame() %>%
             setNames(c("X","Y")) %>%
             mutate(cluster=round(get(lab)$V1,1)))
    }
  png(file=paste(out,nam,".png",sep=""))
  print(ggplot(get(paste("rtsne",nam,"df",sep=".")), aes(x=X, y=Y)) + geom_point(aes(color=as.factor(cluster))) + ggtitle(paste("10 Var Sims:",nam, sep="")))
  dev.off()
}














