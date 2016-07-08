

import numpy as np

def mkBoolFromName(x,name):
    names=[]
    #for n in x.unique():
    for n in np.unique(x):
        if name in n: names.append(n)

    if len(names)==0: boolY=x==names
    else:
        boolY=x==names[0]
        if len(names)>1:
            for n in names:
                boolY=((x==n) | boolY)

    return boolY
