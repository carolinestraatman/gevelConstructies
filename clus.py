#kijk ook eens naar welke veranderen bij K=3 of K=4 en die zijn dan onzeker

import pandas as pd
pd.set_option('precision',15)
import numpy as np
import sys
sys.path.append('../../../py/')
from processes import status
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
rcParams.update({'font.size': 9})
from matplotlib.colors import LinearSegmentedColormap
from clusModules import uniq,gcFill




#to be used for clustering
data=pd.read_csv('tables/converted_v9_6_2016.csv',sep=';',index_col=0)
#data=data[:len(data)/5]
#data.materiaalBinnen=data.materiaalBinnen.apply(lambda x: (x*2).clip(-1))
#data.materiaalBuiten=data.materiaalBuiten.apply(lambda x: (x*2).clip(-1))
#data.bouwjaar=data.bouwjaar.apply(lambda x: (x*2).clip(-1))
dataGood=uniq(data).doAllAndParse()#dataset without buildings with multiple pga and all kinds of different descriptions




imputate=gcFill()
imputate.mapTarget(dataGood)
imputate.setN(4) 

status('using materiaal binnen & buiten')
subsetA=imputate.featureCheck(dataGood,['materiaalBinnen','materiaalBuiten'],[-1,-1],[False,False])
clfA,featuresA=imputate.train(dataGood,subsetA,['materiaalBinnen','materiaalBuiten'])
resultA=imputate.predict(dataGood,featuresA,subsetA)
probA=imputate.proba(dataGood,featuresA,subsetA)
pA=imputate.mapProba(predicted=resultA,proba=probA)
distanceA=imputate.spatialDistance(dataGood,featuresA,subsetA,subsetA)

status('using materiaal binnen')
subsetTrainB=imputate.featureCheck(dataGood,['materiaalBinnen'],[-1],[False])
clfB,featuresB=imputate.train(dataGood,subsetTrainB,'materiaalBinnen')
subsetB=imputate.featureCheck(dataGood,['materiaalBinnen','materiaalBuiten'],[-1,-1],[False,True])
resultB=imputate.predict(dataGood,featuresB,subsetB)
probB=imputate.proba(dataGood,featuresB,subsetB)
pB=imputate.mapProba(predicted=resultB,proba=probB)
distanceB=imputate.spatialDistance(dataGood,featuresB,subsetB,subsetTrainB)

status('using materiaal buiten')
subsetTrainC=imputate.featureCheck(dataGood,['materiaalBuiten'],[-1],[False])
clfC,featuresC=imputate.train(dataGood,subsetTrainC,'materiaalBuiten')
subsetC=imputate.featureCheck(dataGood,['materiaalBinnen','materiaalBuiten'],[-1,-1],[True,False])
resultC=imputate.predict(dataGood,featuresC,subsetC)
probC=imputate.proba(dataGood,featuresC,subsetC)
pC=imputate.mapProba(predicted=resultC,proba=probC)
distanceC=imputate.spatialDistance(dataGood,featuresC,subsetC,subsetTrainC)

status('using bouwjaar & xy only')
clfD,featuresD=imputate.train(dataGood)
subsetD=imputate.featureCheck(dataGood,['materiaalBinnen','materiaalBuiten'],[-1,-1],[True,True])
resultD=imputate.predict(dataGood,featuresD,subsetD)
probD=imputate.proba(dataGood,featuresD,subsetD)
pD=imputate.mapProba(predicted=resultD,proba=probD)
distanceD=imputate.spatialDistance(dataGood,featuresD,subsetD)




stretchA=imputate.mapResult(resultA.predicted)
stretchB=imputate.mapResult(resultB.predicted)
stretchC=imputate.mapResult(resultC.predicted)
stretchD=imputate.mapResult(resultD.predicted)


normFactor=distanceA.distance.median()
print normFactor
distanceA.distance=distanceA.distance.apply(lambda x: x/normFactor)
distanceB.distance=distanceB.distance.apply(lambda x: x/normFactor)
distanceC.distance=distanceC.distance.apply(lambda x: x/normFactor)
distanceD.distance=distanceD.distance.apply(lambda x: x/normFactor)

###values from results/accuracyABCD.csv
pScaledA=pA.copy()
pScaledB=pB.copy()
pScaledC=pC.copy()
pScaledD=pD.copy()
pScaledA.probability=pScaledA.probability.apply(lambda x: x*0.92)
pScaledB.probability=pScaledB.probability.apply(lambda x: x*0.91)
pScaledC.probability=pScaledC.probability.apply(lambda x: x*0.9)
pScaledD.probability=pScaledD.probability.apply(lambda x: x*0.89)


result=pd.concat([resultA,resultB,resultC,resultD])
p=pd.concat([pScaledA,pScaledB,pScaledC,pScaledD])
stretch=pd.concat([stretchA,stretchB,stretchC,stretchD])
distance=pd.concat([distanceA,distanceB,distanceC,distanceD])

output=pd.merge(result,p,left_index=True,right_index=True)
output=pd.merge(output,stretch,left_index=True,right_index=True)
#output=pd.merge(result,stretch,left_index=True,right_index=True)
output=pd.merge(output,distance.drop([u'x', u'y', u'nbIndex', u'nbX', u'nbY'],1),left_index=True,right_index=True)
output.predicted=output.predicted.apply(lambda x: imputate.mapInverse[x])
output=pd.merge(output,dataGood,left_index=True,right_index=True)
    
output.drop('bedrijfsRegels',axis=1).to_csv('results/newv6.k.4.csv',sep=';',index=True)                





imputate.setN(np.arange(6)+2)
accuracyA=imputate.crossValidate(dataGood,subsetA,['materiaalBinnen','materiaalBuiten'])
accuracyB=imputate.crossValidate(dataGood,subsetTrainB,'materiaalBinnen')
accuracyC=imputate.crossValidate(dataGood,subsetTrainC,'materiaalBuiten')
accuracyD=imputate.crossValidate(dataGood)

accuracyA=pd.DataFrame({'n':accuracyA[1],'accuracy':accuracyA[0]})
accuracyB=pd.DataFrame({'n':accuracyB[1],'accuracy':accuracyB[0]})
accuracyC=pd.DataFrame({'n':accuracyC[1],'accuracy':accuracyC[0]})
accuracyD=pd.DataFrame({'n':accuracyD[1],'accuracy':accuracyD[0]})

accuracyA.to_csv('results/accuracyA.csv',sep=';',index=False)
accuracyB.to_csv('results/accuracyB.csv',sep=';',index=False)
accuracyC.to_csv('results/accuracyC.csv',sep=';',index=False)
accuracyD.to_csv('results/accuracyD.csv',sep=';',index=False)

#accuracyA=pd.read_csv('results/accuracyA.csv',sep=';')
#accuracyB=pd.read_csv('results/accuracyB.csv',sep=';')
#accuracyC=pd.read_csv('results/accuracyC.csv',sep=';')
#accuracyD=pd.read_csv('results/accuracyD.csv',sep=';')

fig=plt.figure(figsize=(4,3))
plt.subplots_adjust(left=0.18,right=0.95,bottom=0.14,top=0.95)
plt.plot(accuracyA.n,accuracyA.accuracy,label='A: bouwjaar, x, y, materiaal binnen, materiaal buiten')
plt.plot(accuracyB.n,accuracyB.accuracy,label='B: bouwjaar, x, y, materiaal binnen')
plt.plot(accuracyC.n,accuracyC.accuracy,label='C: bouwjaar, x, y, materiaal buiten')
plt.plot(accuracyD.n,accuracyD.accuracy,label='D: bouwjaar, x, y')
plt.xlabel('K')
plt.ylabel('accuracy (x 100%)')
plt.ylim([0.8,1])
plt.xlim([0,7])
plt.legend(fontsize=6)
plt.savefig('figures/accuracy.pdf',dpi=300)
plt.close('all')


#multOK[1] result is buildings with multiple pga, but different descriptions. Apply clf.fit to these after training with dataGOOD (note dataGood contains "onbekend" --> this is for class gcFill)
#check jaar van aanleg=1800






