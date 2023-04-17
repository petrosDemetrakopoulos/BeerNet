import pandas as pd
import numpy as np
from ctgan import CTGAN
from sklearn.impute import SimpleImputer

df = pd.read_json('./recipes_processed.json')

print(df.head())

#some more preprocessing
df['yeast'] = df['yeast'].apply(lambda x: x[0] if x else '-')

#drop unneeded columns
df = df.drop(['id', 'name','style','other'], axis=1)
# keep only all grain recipes
df = df[df['method'] == 'All Grain']

df['hop counts'] = df['hops'].apply(lambda x: len(x))
df['fermentables_count'] = df['fermentables'].apply(lambda x: len(x))

df = df.sort_values(by=['hop counts'], ascending=False)
df = df[df['hop counts'] <= 5]
df = df[df['fermentables_count'] <= 5]

df = df.drop(['hop counts', 'fermentables_count'], axis=1)

def fermentable_weight(x, ferm_index):
    if x:
        if len(x) >= ferm_index + 1:
            if len(x[ferm_index]) >= 1:
                return x[ferm_index][0]
    return 0

def fermentable_name(x, ferm_index):
    if x:
        if len(x) >= ferm_index + 1:
            if len(x[ferm_index]) >= 2:
                return x[ferm_index][1]
    return '-'


def hop_weight(x, hop_index):
    if x:
        if len(x) >= hop_index + 1:
            if len(x[hop_index]) >= 1:
                return x[hop_index][0]
    return 0

def hop_name(x, hop_index):
    if x:
        if len(x) >= hop_index+1:
            if len(x[hop_index]) >= 2:
                return x[hop_index][1]
    return '-'

def hop_form(x, hop_index):
    if x:
        if len(x) >= hop_index+1:
            if len(x[hop_index]) >= 3:
                return x[hop_index][2]
    return '-'

def hop_phase(x, hop_index):
    if x:
        if len(x) >= hop_index+1:
            if len(x[hop_index]) >= 4:
                return x[hop_index][4]
    return '-'

def hop_time(x, hop_index):
    if x:
        if len(x) >= hop_index+1:
            if len(x[hop_index]) >= 5:
                return x[hop_index][5]
    return '-'

df['hop 1 name'] = df['hops'].apply(lambda x: hop_name(x,0))
df['hop 1 weight'] = df['hops'].apply(lambda x: hop_weight(x,0))
df['hop 1 form'] = df['hops'].apply(lambda x: hop_form(x, 0))
df['hop 1 phase'] = df['hops'].apply(lambda x: hop_phase(x, 0))
df['hop 1 time'] = df['hops'].apply(lambda x: hop_time(x, 0))

df['hop 2 name'] = df['hops'].apply(lambda x: hop_name(x,1))
df['hop 2 weight'] = df['hops'].apply(lambda x: hop_weight(x,1))
df['hop 2 form'] = df['hops'].apply(lambda x: hop_form(x, 1))
df['hop 2 phase'] = df['hops'].apply(lambda x: hop_phase(x, 1))
df['hop 2 time'] = df['hops'].apply(lambda x: hop_time(x, 1))

df['hop 3 name'] = df['hops'].apply(lambda x: hop_name(x,2))
df['hop 3 weight'] = df['hops'].apply(lambda x: hop_weight(x,2))
df['hop 3 form'] = df['hops'].apply(lambda x: hop_form(x, 2))
df['hop 3 phase'] = df['hops'].apply(lambda x: hop_phase(x, 2))
df['hop 3 time'] = df['hops'].apply(lambda x: hop_time(x, 2))

df['hop 4 name'] = df['hops'].apply(lambda x: hop_name(x,3))
df['hop 4 weight'] = df['hops'].apply(lambda x: hop_weight(x,3))
df['hop 4 form'] = df['hops'].apply(lambda x: hop_form(x, 3))
df['hop 4 phase'] = df['hops'].apply(lambda x: hop_phase(x, 3))
df['hop 4 time'] = df['hops'].apply(lambda x: hop_time(x, 3))

df.loc[df['hop 1 phase'].str.startswith('Whirlpool'), 'hop 1 phase'] = 'Whirlpool'
df.loc[df['hop 2 phase'].str.startswith('Whirlpool'), 'hop 2 phase'] = 'Whirlpool'
df.loc[df['hop 3 phase'].str.startswith('Whirlpool'), 'hop 3 phase'] = 'Whirlpool'
df.loc[df['hop 4 phase'].str.startswith('Whirlpool'), 'hop 4 phase'] = 'Whirlpool'

df.loc[df['hop 1 phase'].str.startswith('Boil'), 'hop 1 phase'] = 'Boil'
df.loc[df['hop 2 phase'].str.startswith('Boil'), 'hop 2 phase'] = 'Boil'
df.loc[df['hop 3 phase'].str.startswith('Boil'), 'hop 3 phase'] = 'Boil'
df.loc[df['hop 4 phase'].str.startswith('Boil'), 'hop 4 phase'] = 'Boil'

df.loc[df['hop 1 phase'].str.startswith('First Wort'), 'hop 1 phase'] = 'First Wort'
df.loc[df['hop 2 phase'].str.startswith('First Wort'), 'hop 2 phase'] = 'First Wort'
df.loc[df['hop 3 phase'].str.startswith('First Wort'), 'hop 3 phase'] = 'First Wort'
df.loc[df['hop 4 phase'].str.startswith('First Wort'), 'hop 4 phase'] = 'First Wort'

df.loc[df['hop 1 phase'].str.startswith('Dry Hop'), 'hop 1 phase'] = 'Dry Hop'
df.loc[df['hop 2 phase'].str.startswith('Dry Hop'), 'hop 2 phase'] = 'Dry Hop'
df.loc[df['hop 3 phase'].str.startswith('Dry Hop'), 'hop 3 phase'] = 'Dry Hop'
df.loc[df['hop 4 phase'].str.startswith('Dry Hop'), 'hop 4 phase'] = 'Dry Hop'

df.loc[df['hop 1 phase'].str.startswith('Hopback'), 'hop 1 phase'] = 'Hopback'
df.loc[df['hop 2 phase'].str.startswith('Hopback'), 'hop 2 phase'] = 'Hopback'
df.loc[df['hop 3 phase'].str.startswith('Hopback'), 'hop 3 phase'] = 'Hopback'
df.loc[df['hop 4 phase'].str.startswith('Hopback'), 'hop 4 phase'] = 'Hopback'

df['hop 1 time'] = df['hop 1 time'].apply(lambda x: x[:-4].strip())
df['hop 2 time'] = df['hop 2 time'].apply(lambda x: x[:-4].strip())
df['hop 3 time'] = df['hop 3 time'].apply(lambda x: x[:-4].strip())
df['hop 4 time'] = df['hop 4 time'].apply(lambda x: x[:-4].strip())

df['hop 1 weight'] = df['hop 1 weight'].apply(lambda x: x[:-3].strip() if isinstance(x, str) and x.endswith('ml') else x)
df['hop 2 weight'] = df['hop 2 weight'].apply(lambda x: x[:-3].strip() if isinstance(x, str) and x.endswith('ml') else x)
df['hop 3 weight'] = df['hop 3 weight'].apply(lambda x: x[:-3].strip() if isinstance(x, str) and x.endswith('ml') else x)
df['hop 4 weight'] = df['hop 4 weight'].apply(lambda x: x[:-3].strip() if isinstance(x, str) and x.endswith('ml') else x)

df = df[df['hop 1 time'].str.isdigit() & df['hop 2 time'].str.isdigit() & df['hop 3 time'].str.isdigit() & df['hop 4 time'].str.isdigit()]

df['fermentable 1 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 0))
df['fermentable 1 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 0))

df['fermentable 2 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 1))
df['fermentable 2 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 1))

df['fermentable 3 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 2))
df['fermentable 3 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 2))

df['fermentable 4 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 3))
df['fermentable 4 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 3))

df['fermentable 5 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 4))
df['fermentable 5 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 4))

df = df.drop(['hops','fermentables','method'], axis = 1)

discrete_columns = [
    'hop 1 form',
    'hop 2 form',
    'hop 3 form',
    'hop 4 form',
    'hop 1 phase',
    'hop 2 phase',
    'hop 3 phase',
    'hop 4 phase',
    'hop 1 name',
    'hop 2 name',
    'hop 3 name',
    'hop 4 name',
    'fermentable 1 name',
    'fermentable 2 name',
    'fermentable 3 name',
    'fermentable 4 name',
    'fermentable 5 name',
    'yeast'
]
df['batch'] = df['batch'].astype(float)

imp = SimpleImputer(strategy='most_frequent', missing_values='-', keep_empty_features=True)
cols = df.columns

imputed = imp.fit_transform(df)

df = pd.DataFrame(imputed, columns=cols)
df['batch'] = df['batch'].astype(float)

df['fermentable 1 weight'] = df['fermentable 1 weight'].astype(float)
df['fermentable 2 weight'] = df['fermentable 2 weight'].astype(float)
df['fermentable 3 weight'] = df['fermentable 3 weight'].astype(float)
df['fermentable 4 weight'] = df['fermentable 4 weight'].astype(float)
df['fermentable 5 weight'] = df['fermentable 5 weight'].astype(float)

df['hop 1 weight'] = df['hop 1 weight'].astype(float)
df['hop 2 weight'] = df['hop 2 weight'].astype(float)
df['hop 3 weight'] = df['hop 3 weight'].astype(float)
df['hop 4 weight'] = df['hop 4 weight'].astype(float)

df['hop 1 time'] = df['hop 1 time'].astype(float)
df['hop 2 time'] = df['hop 2 time'].astype(float)
df['hop 3 time'] = df['hop 3 time'].astype(float)
df['hop 4 time'] = df['hop 4 time'].astype(float)

df['yeast'] = df['yeast'].astype(str)

ctgan = CTGAN(epochs=25, cuda=True, verbose=True)
ctgan.fit(df, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(20)
synthetic_data.to_csv('./synthetic_recipes_25_epochs.csv')
print(synthetic_data)