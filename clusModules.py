

import pandas as pd
pd.set_option('precision',15)
import numpy as np


class uniq:
    def __init__(self,data):
        self.identifier='ID'
        self.target='gcCode'
        self.data=data
        self.uniq=data.groupby(data[self.identifier])#pandas groups

    def split(self): #buildings with one or more pga
        out1=self.uniq.filter(lambda x: len(x)==1)
        out2=self.data.drop(out1.index)
        return out1,out2
 
    def multOK(self,index=-1,): #get those with more pga but same gccode everywhere, and then those with more pga but different entries
        if len(np.shape(index))==0:
            if index==-1: index=self.split()[1].index    

        data=self.data.loc[index].copy()
        groups=data.groupby(data[self.identifier])
        
        ix=[]
        for g in groups:
            u=g[1][self.target].unique()
            if len(u)==1:
                if 'GC' in u[0]: 
                    #print u,g[1].index[0]
                    ix.extend(g[1].index.values)
        out1=data.loc[ix]
        return out1.drop_duplicates(),data.drop(ix)

    def doAllAndParse(self):
        splitted=self.split()
        multiGood=self.multOK(splitted[1].index)
        return pd.concat([splitted[0],multiGood[0]])





from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cross_validation import KFold#,train_test_split


class gcFill:
    def __init__(self):
        self.targetKey='gcCode'
        self.gc_zetting = {"GC1": 20,"GC2": 20,"GC13": 20,"GC15": 20,"GC17a": 20,"GC17b": 20,"GC17c": 20,"GC20": 20,\
            "GC24a": 20,"GC25a": 20,"GC3": 30,"GC5": 30,"GC7": 30,"GC9": 30,"GC14": 30,"GC16": 30,\
            "GC20a": 30,"GC20b": 30,"GC20c": 30,"GC20d": 30,"GC24b":30,"GC25b": 30,"GC26": 30,"GC6": 40,"GC8": 40,\
            "GC10": 40,"GC18": 40,"GC4": 60,"GC11": 60,"GC12": 60,"GC22": 60,"GC23": 60,"GC21a": 80,\
            "GC21b": 80,"GC21c": 80,"GC21d": 80,"GC21e": 80,"GC27a": 80,"GC27b": 80,"GC27c": 80,"GC27e": 80,\
            "GC19": 100}#guesstimated GC20d
        self.mapping={}
        for i,gc in enumerate(self.gc_zetting.keys()): self.mapping[gc]=i+1
        self.mapInverse={v: k for k, v in self.mapping.items()}
        self.toFillKey=-1

    def setN(self,n=3): self.n=n

    
    def mp(self,x):
        if 'GC' in x: return self.mapping[x]
        else: return -1

    def mapTarget(self,data):
        self.targetMapped=data[self.targetKey].apply(lambda x: self.mp(x))

        
    def featureCheck(self,data,columns,keys,trueFalse=True):
        if len(columns)!=len(keys):
            print "keys and columns don't match"
            return []
        else:
            subset=pd.Series([True]*len(data))
            if len(np.shape(trueFalse))==0:
                if trueFalse==True:
                    for c,k in zip(columns,keys): subset=subset & (data[c]==k)
                else:
                    for c,k in zip(columns,keys): subset=subset & (data[c]!=k)
            else:
                if len(trueFalse)==len(keys):
                    for c,k,b in zip(columns,keys,trueFalse):
                        if b==True: subset=subset & (data[c]==k)
                        else: subset=subset & (data[c]!=k)
                else:
                    print "keys and columns don't match"
                    return []   
            return subset


    def sortData(self,subset=-1):
        if len(np.shape(subset))==0:
            ok=self.targetMapped!=self.toFillKey
            fill=ok==False
        else:
            ok=(self.targetMapped!=self.toFillKey) & subset
            fill=(self.targetMapped==self.toFillKey) & subset
        return ok,fill

    
    def listFeatures(self,featuresExtra='none'):
        features=['x','y','bouwjaar']
        if len(np.shape(featuresExtra))==0:
            if featuresExtra!='none': features.append(featuresExtra)
        else:
            for f in featuresExtra:
                features.append(f)
        return features
    
    
    def train(self,data,subsetTrain=-1,featuresExtra='none'):
        ok,fill=self.sortData(subset=subsetTrain)
        features=self.listFeatures(featuresExtra=featuresExtra)
        self.setN()
                
        trainData=data[ok]
        trainData=trainData[features]
        trainTarget=self.targetMapped[ok]
        self.clf=knn(self.n,weights='uniform')
        self.clf.fit(trainData,trainTarget)#self,parameters
                
        return self.clf,features

    def spatialDistance(self,data,features,subset=-1,subsetTrain=-1,normalize=False):
        dummy,fill=self.sortData(subset=subset)
        ok,dummy=self.sortData(subset=subsetTrain)
        result=self.clf.kneighbors(data[fill][features],return_distance=False)
        ix=[i[self.n-1] for i in result]
        distance=data[fill][['x','y']]
        distance['nbIndex']=data[ok].iloc[ix].index.values
        distance['nbX']=data[ok].iloc[ix].x.values
        distance['nbY']=data[ok].iloc[ix].y.values
        distance['distance']=distance.apply(lambda x: np.sqrt(pow((x.x-x.nbX),2)+pow((x.y-x.nbY),2)),axis=1)
        return distance
    

    def spatialDensity(self,data,features,subset=-1,subsetTrain=-1,normalize=False):
        dummy,fill=self.sortData(subset=subset)
        ok,dummy=self.sortData(subset=subsetTrain)
        result=self.clf.kneighbors(data[fill][features],return_distance=False)
        ix=[i[self.n-1] for i in result]
        density=data[fill][['x','y']]
        density['nbIndex']=data[ok].iloc[ix].index.values
        density['nbX']=data[ok].iloc[ix].x.values
        density['nbY']=data[ok].iloc[ix].y.values
        density['density']=density.apply(lambda x: self.n/(np.pi*(pow((x.x-x.nbX),2)+pow((x.y-x.nbY),2))),axis=1)
        if normalize==True:
            dMax=density.density.max()
            density.density=density.density.apply(lambda x: x/dMax)
        return density
    
    
    def predict(self,data,features,subset=-1):
        ok,fill=self.sortData(subset=subset)
        return pd.DataFrame({'predicted':self.clf.predict(data[fill][features])},index=data[fill].index)

    def proba(self,data,features,subset=-1):
        ok,fill=self.sortData(subset=subset)
        result=pd.DataFrame(self.clf.predict_proba(data[fill][features]),index=data[fill].index)
        for i,c in enumerate(self.clf.classes_): result.rename(columns={i:c+100},inplace=True)
        return result.rename(columns=lambda x: x-100)

    
    def mapProba(self,predicted=-1,proba=-1,data=-1,features=-1,subset=-1):
        ok,fill=self.sortData(subset=subset)
        if len(np.shape(predicted))==0:
            if predicted==-1: predicted=self.predict(data,features,subset)
        if len(np.shape(proba))==0:
            if proba==-1: proba=self.proba(data,features,subset)
        mapP=[]
        for i,value in enumerate(predicted.predicted): mapP.append(proba.iloc[i][value])
        return pd.DataFrame({'probability':mapP},index=predicted.index)
    
    
    def crossValidate(self,data,subsetTrain=-1,featuresExtra='none'):
        ok,fill=self.sortData(subset=subsetTrain)
        features=self.listFeatures(featuresExtra=featuresExtra)
        
        kf = KFold(len(np.where(ok)[0]),n_folds=10,shuffle=True)
        
        accuracy=[]
        for n in self.n:
            print 'validating n=',n,'...'
            predicted=np.zeros_like(self.targetMapped[ok])
            for trainIdx, testIdx in kf:
                train=data[ok].iloc[trainIdx]
                test=data[ok].iloc[testIdx]
                clf=knn(n,weights='uniform')
                clf.fit(train[features],self.targetMapped[ok].iloc[trainIdx])
                predicted[testIdx]=clf.predict(test[features])
            accuracy.append(np.mean(predicted==self.targetMapped[ok]))
            print '-->accuracy: ',accuracy[-1]
            
        return accuracy,self.n
                            

    def mapResult(self,data): return pd.DataFrame({'stretch':[self.gc_zetting[self.mapInverse[d]] for d in data]},index=data.index)


   
