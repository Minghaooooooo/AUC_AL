##Delicious##
Processed as 
X,Y=util.cardi_based_data_selection(X,Y,minC=1,maxC=20)
Y=util.selectLabel(Y,label_freq=0.025);

from sklearn.decomposition import PCA
#feature size too large, try reduce by PCA
pcaSolver=PCA(n_components=0.90)

X dim 223
y dim 156

##Delicious-no Dup##
remove the duplicate records(6833 left)

##Col5k##
Processed as 
X,Y=util.cardi_based_data_selection(X,Y,minC=1,maxC=16)
Y=util.selectLabel(Y,label_freq=0.005);
X dim 499
y dim 132

##Web-light##
Additional layer of PCA, reduce the features
X dim 1149
y dim 90

##eurLEX##
X,Y=fileIO.readData(readPath,'eurlex-dc-leaves-fold1-train.arff',5000,'float')
X,Y=util.cardi_based_data_selection(X,Y,minC=2,maxC=20)

Y=util.selectLabel(Y,label_freq=0.003);
#select frequent labels

#X,Y=util.cardi_based_data_selection(X,Y,minC=1,maxC=8)
from sklearn.decomposition import PCA
#feature size too large, try reduce by PCA
pcaSolver=PCA(n_components=0.90)
tem=normalize(X)
X=pcaSolver.fit_transform(tem)

X dim 999
y dim 169

##bookmark##
remove duplicate.(about 2000 data)
reduce feature space.






