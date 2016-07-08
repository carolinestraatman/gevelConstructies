
from booletc import mkBoolFromName
import pandas as pd
pd.set_option('precision',15)
import numpy as np
from datetime import datetime
from processes import status


def scaleStringInt(string,uniq): return np.where(uniq==string)[0][0]


#fileIn='tables/extracted_v28_4_2016.csv'
#fileIn='data/bag_pga_intersect_v030516.csv'
fileIn='data/bag_pga_intersect_v200516.txt'
#fileOut=fileIn.replace('extracted','converted')
today=str(datetime.now().day)+'_'+str(datetime.now().month)+'_'+str(datetime.now().year)
fileOut='tables/converted_v'+today+'.csv'

status('reading data')
data=pd.read_csv(fileIn,sep=';')
status('end')

status('string conversions')
data.rename(columns={'MATERIAAL_AANSLUITL_BINNEN':'materiaalBinnen','MATERIAAL_AANSLUITL_BUITEN':'materiaalBuiten','BOUWJAAR':'bouwjaar','CODE_GEVELCONSTRUCTIE':'gcCode'},inplace=True)
data.materiaalBinnen=data.materiaalBinnen.apply(lambda x: x.lower())
data.materiaalBuiten=data.materiaalBuiten.apply(lambda x: x.lower())
status('end')

status('assign GC14/GC16')
toFill=(data.gcCode.str[:2]!="GC") & (mkBoolFromName(data.gcCode.str.lower(),"buitenkast")==False)
copper=(mkBoolFromName(data.gcCode.str.lower(),"koper") | (mkBoolFromName(data.materiaalBuiten.str.lower(),"koper"))) & toFill
steel=(mkBoolFromName(data.gcCode.str.lower(),"staal") | (mkBoolFromName(data.materiaalBuiten.str.lower(),"staal"))) & toFill
data.gcCode.loc[copper]='GC14'
data.gcCode.loc[steel]='GC16'
data['bedrijfsRegels']='-'
data.bedrijfsRegels.loc[copper]='GC14'
data.bedrijfsRegels.loc[steel]='GC16'
status('end')

#ix=np.where(data.bedrijfsRegels!='-')[0]
#data[['x','y','gcCode']].iloc[ix].to_csv('data/bag_pga_intersect_v200516_with_replacements.txt',sep=';',index=False)


status('generating cropped frame')
nvt=mkBoolFromName(data.materiaalBinnen,'n.v.t.') | mkBoolFromName(data.materiaalBuiten,'n.v.t.')
data=data[nvt==False]

#dataML=data[[u'BRON_ID', u'gcCode', u'HUISNUMMER', u'materiaalBinnen',
#       u'materiaalBuiten', u'PLAATS', u'POSTCODE', u'STRAATNAAM',
#       u'DATUM_AANLEG_GEVELCONSTRUCTIE', u'OBJECT_ID', u'ID', u'bouwjaar',
#       u'x', u'y']].copy()
#dataAncil=data[[]].copy()
dataML=data.copy()
status('end')

status('scaling x coordinates...')
minFact=dataML.x.min()
divFact=dataML.x.max()-minFact
print divFact
dataML.x=dataML.x.apply(lambda x: (np.float(x)-minFact)/divFact)
status('end')
status('scaling y coordinates...')
minFact=dataML.y.min()
divFact=dataML.y.max()-minFact
print divFact###actually we should use the same divFact for x- and y- coo...-->needs fix
dataML.y=dataML.y.apply(lambda x: (np.float(x)-minFact)/divFact)
status('end')

status('scaling building year parameter...')
#dataML.bouwjaar.astype(float,inplace=True)
minFact=dataML.bouwjaar.min()
divFact=dataML.bouwjaar.max()-minFact
dataML.bouwjaar=dataML.bouwjaar.apply(lambda x: (np.float(x)-minFact)/divFact)
status('end')

status('scaling material inside parameter...')
materialInsideKnown=(mkBoolFromName(data.materiaalBinnen,'onbekend')==False)# & (data.materiaalBinnen!='verschillend')

refStrings=dataML.materiaalBinnen.loc[materialInsideKnown].unique()
dataML.materiaalBinnen.loc[materialInsideKnown]=dataML.materiaalBinnen.loc[materialInsideKnown].apply(lambda x: scaleStringInt(x,refStrings))
divFact=dataML.materiaalBinnen.loc[materialInsideKnown].max()
dataML.materiaalBinnen.loc[materialInsideKnown]=dataML.materiaalBinnen.loc[materialInsideKnown].apply(lambda x: np.float(x)/divFact)

dataML.materiaalBinnen.loc[materialInsideKnown==False]=-1
status('end')

status('scaling material outside parameter')
materialOutsideKnown=(mkBoolFromName(data.materiaalBuiten,'onbekend')==False)# & (data.materiaalBuiten!='verschillend')

refStrings=dataML.materiaalBuiten.loc[materialOutsideKnown].unique()
dataML.materiaalBuiten.loc[materialOutsideKnown]=dataML.materiaalBuiten.loc[materialOutsideKnown].apply(lambda x: scaleStringInt(x,refStrings))
divFact=dataML.materiaalBuiten.loc[materialOutsideKnown].max()
dataML.materiaalBuiten.loc[materialOutsideKnown]=dataML.materiaalBuiten.loc[materialOutsideKnown].apply(lambda x: np.float(x)/divFact)

dataML.materiaalBuiten.loc[materialOutsideKnown==False]=-1
status('end')

status('writing to file '+fileOut+'...')
#dataML.to_csv(fileOut,index=False,sep=';')
dataML.to_csv(fileOut,sep=';')
status('end')
