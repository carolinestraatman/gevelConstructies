
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import os


# # Import Data

# In[2]:

datadir = "~/Data/Stedin/GCFiles/data"
data = pd.read_csv(os.path.join(datadir, "bag_pga_intersect_v200516.txt"), ';')


# In[3]:

data = data[['CODE_GEVELCONSTRUCTIE','MATERIAAL_AANSLUITL_BINNEN','MATERIAAL_AANSLUITL_BUITEN','PLAATS','POSTCODE','DATUM_AANLEG_GEVELCONSTRUCTIE','BOUWJAAR','x','y']]
data.head(1)


# Vervang je target class, in dit geval CODE_GEVELCONSTRUCTIE met de naam 'class'

# In[5]:

data.rename(columns={'CODE_GEVELCONSTRUCTIE': 'class'}, inplace=True)


# In[7]:

for cat in ['class', 'MATERIAAL_AANSLUITL_BINNEN', 'MATERIAAL_AANSLUITL_BUITEN', 'PLAATS']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, data[cat].unique().size))


# ## GC code

# In[8]:

data['class'] = data['class'].str.lower()


# ### Filter onbruikbare door onbekende data

# In[9]:

data = data[data['class']!='onbekend']


# ### Unificeer code
# Staal = GC16
# Koper = GC14

# In[10]:

data['class'] = data['class'].replace('staal','gc16').replace('koper','gc14')



# ## Vervang ontbrekende waardes met -999 placeholder

# In[12]:

data = data.fillna(-999)

# ## Datum aanleg

# Vervang datum aanleg met enkel het jaartal

# In[13]:

aanleg_datum = data['DATUM_AANLEG_GEVELCONSTRUCTIE'].astype(str)
converted_date = []
for date_time in aanleg_datum:
    date = date_time.split(' ')[0]
    units = date.split('-')
    try:
        year_found = False
        for unit in units:
            if len(unit) == 4: 
                year = unit
                converted_date.append(year)
                year_found = True
        if year_found == False:
            converted_date.append(-999)
    except:
        print(date_time)
        print(units)
        print("WENT WRONG")
        raise
            


data['DATUM_AANLEG_GEVELCONSTRUCTIE'] = converted_date


# ## binarise features

# In[14]:

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
materiaal_binnen_trans = mlb.fit_transform([{str(val)} for val in data['MATERIAAL_AANSLUITL_BINNEN'].values])
materiaal_buiten_trans = mlb.fit_transform([{str(val)} for val in data['MATERIAAL_AANSLUITL_BUITEN'].values])

input_data = data.drop(['MATERIAAL_AANSLUITL_BINNEN', 'MATERIAAL_AANSLUITL_BUITEN', 'PLAATS', 'POSTCODE','class'], 1)
input_data['DATUM_AANLEG_GEVELCONSTRUCTIE'] = pd.to_numeric(input_data['DATUM_AANLEG_GEVELCONSTRUCTIE'])


# Stack de nieuwe binary featuers bij de bestaande

# In[16]:

input_data = np.hstack((input_data.values,materiaal_binnen_trans))
input_data = np.hstack((input_data,materiaal_buiten_trans))

# Check de feature space

# # Create data sets

# In[18]:

from sklearn.cross_validation import train_test_split

output_data = data['class']
output_data = output_data.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, train_size=0.75, test_size=0.25)

# # Find ideal model

# In[ ]:

from tpot import TPOT

tpot = TPOT(generations=5, verbosity=2)
tpot.fit(X_train, y_train)


# # Score model

# In[ ]:

print(tpot.score(X_test, y_test))