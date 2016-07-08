

import pandas as pd
from clusModules import uniq,gcFill
import numpy as np


gcf=gcFill()

output3=pd.read_csv('results/newv5.k.3.csv',sep=';',index_col=0)
#output4=pd.read_csv('results/newv6.k.4.csv',sep=';',index_col=0)
output=output3.copy()
#output.predicted=output.predicted.apply(lambda x: gcf.mapInverse[x])


output3.rename(columns={'predicted':'pred3'},inplace=True)
#output4.rename(columns={'predicted':'pred4'},inplace=True)

#predicted=pd.merge(pd.DataFrame(output3.pred3),pd.DataFrame(output4.pred4),left_index=True,right_index=True)
#diff=predicted.apply(lambda x: x[0]!=x[1],axis=1)



output['inspectieAdvies']='nee'
adviceCheck=((output.probability<=0.5) & (output.distance>=3)) | (output.probability<=0.35) | (output.distance>10)# | (output.diff==True)
output.inspectieAdvies[adviceCheck]='ja' # this was kind of random. first step to estimate of where it could be wrong

output.drop('probability',axis=1,inplace=True)
output.drop('distance',axis=1,inplace=True)


output.rename(columns={'predicted':'voorspelde GC code','stretch':'toegestane verzakking'},inplace=True)


orig=pd.read_csv('data/bag_pga_intersect_v200516.txt',sep=';')
orig.rename(columns={'MATERIAAL_AANSLUITL_BINNEN':'materiaalBinnen','MATERIAAL_AANSLUITL_BUITEN':'materiaalBuiten','BOUWJAAR':'bouwjaar','CODE_GEVELCONSTRUCTIE':'gcCode'},inplace=True)

output.materiaalBinnen=orig.materiaalBinnen.loc[output.index]
output.materiaalBuiten=orig.materiaalBuiten.loc[output.index]
output.x=orig.x.loc[output.index]
output.y=orig.y.loc[output.index]
output.bouwjaar=orig.bouwjaar.loc[output.index]


onc=pd.read_csv("../ONC_GAS.csv",sep=';',index_col=1)
def assignONC(x):
    try: index=np.int(x)
    except ValueError: return 'onbekend'
    try: return onc[u'ONC Gebied Gas'].loc[index]#,onc.loc[x].values
    except KeyError: return 'onbekend'

def assignRegio(x):
    try: index=np.int(x)
    except ValueError: return 'onbekend'
    try: return onc[u'ONC Gebied Gas'].loc[index].split('-')[2]#,onc.loc[x].values
    except KeyError: return 'onbekend'

    
output['ONC']=output.POSTCODE.str[:4].apply(lambda x: assignONC(x))
output['regio']=output.POSTCODE.str[:4].apply(lambda x: assignRegio(x))


#output.drop('bedrijfsRegels',axis=1).to_csv('results/newv7.csv',sep=';',index=False)                
output.to_csv('results/newv8.csv',sep=';',index=False)          


loc=output.ONC!='onbekend'
key=output[loc].ONC.apply(lambda x: x.split('-')[-2])
key=key.combine_first(output.ONC)
output=output.join(output3[['distance','probability']])
groups=output.groupby(key)


import matplotlib.pyplot as plt

distanceConversion=0.00290350885073*185000.#handwavy. Automate export from divFact in convert.py and normFactor from clus.py. 185 is average x and y scalings
count=1
fig=plt.figure()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.3,hspace=0.5)
for g in groups:
    fig.add_subplot(5,2,count)
    plt.title(g[0])
    #plt.hist(g[1].distance*distanceConversion,range=[0,1200],bins=30)
    plt.hist(g[1].distance,range=[0,3],bins=30)
    count+=1
    fig.add_subplot(5,2,count)
    plt.hist(g[1].probability,range=[0,1],bins=20)
    count+=1

plt.show()
