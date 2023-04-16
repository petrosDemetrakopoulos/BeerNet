import pandas as pd
import category_encoders as ce
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras import backend as K
from keras import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

df['fermentable 5 weight'] = df['fermentables'].apply(lambda x: fermentable_weight(x, 5))
df['fermentable 5 name'] = df['fermentables'].apply(lambda x: fermentable_name(x, 5))

df = df.drop(['hops','fermentables','method'], axis = 1)

encoder = ce.CountEncoder()
df['hop 1 name_transformed'] = encoder.fit_transform(df['hop 1 name'])
df['hop 2 name_transformed'] = encoder.fit_transform(df['hop 2 name'])
df['hop 3 name_transformed'] = encoder.fit_transform(df['hop 3 name'])
df['hop 4 name_transformed'] = encoder.fit_transform(df['hop 4 name'])

df['hop 1 phase_transformed'] = encoder.fit_transform(df['hop 1 phase'])
df['hop 2 phase_transformed'] = encoder.fit_transform(df['hop 2 phase'])
df['hop 3 phase_transformed'] = encoder.fit_transform(df['hop 3 phase'])
df['hop 4 phase_transformed'] = encoder.fit_transform(df['hop 4 phase'])

df['hop 1 form_transformed'] = encoder.fit_transform(df['hop 1 form'])
df['hop 2 form_transformed'] = encoder.fit_transform(df['hop 2 form'])
df['hop 3 form_transformed'] = encoder.fit_transform(df['hop 3 form'])
df['hop 4 form_transformed'] = encoder.fit_transform(df['hop 4 form'])

df['fermentable 1 name_transformed'] = encoder.fit_transform(df['fermentable 1 name'])
df['fermentable 2 name_transformed'] = encoder.fit_transform(df['fermentable 2 name'])
df['fermentable 3 name_transformed'] = encoder.fit_transform(df['fermentable 3 name'])
df['fermentable 4 name_transformed'] = encoder.fit_transform(df['fermentable 4 name'])
df['fermentable 5 name_transformed'] = encoder.fit_transform(df['fermentable 5 name'])

df['yeast_transformed'] = encoder.fit_transform(df['yeast'])

yeast_trans = df[['yeast_transformed', 'yeast']].drop_duplicates()

hop_trans_1 = df[['hop 1 name_transformed', 'hop 1 name']].drop_duplicates()
hop_trans_2 = df[['hop 2 name_transformed', 'hop 2 name']].drop_duplicates()
hop_trans_3 = df[['hop 3 name_transformed', 'hop 3 name']].drop_duplicates()
hop_trans_4 = df[['hop 4 name_transformed', 'hop 4 name']].drop_duplicates()

hop_trans_1_phase = df[['hop 1 phase_transformed', 'hop 1 phase']].drop_duplicates()
hop_trans_2_phase = df[['hop 2 phase_transformed', 'hop 2 phase']].drop_duplicates()
hop_trans_3_phase = df[['hop 3 phase_transformed', 'hop 3 phase']].drop_duplicates()
hop_trans_4_phase = df[['hop 4 phase_transformed', 'hop 4 phase']].drop_duplicates()

hop_trans_1_form = df[['hop 1 form_transformed', 'hop 1 form']].drop_duplicates()
hop_trans_2_form = df[['hop 2 form_transformed', 'hop 2 form']].drop_duplicates()
hop_trans_3_form = df[['hop 3 form_transformed', 'hop 3 form']].drop_duplicates()
hop_trans_4_form = df[['hop 4 form_transformed', 'hop 4 form']].drop_duplicates()

fermentable_trans_1 = df[['fermentable 1 name_transformed', 'fermentable 1 name']].drop_duplicates()
fermentable_trans_2 = df[['fermentable 2 name_transformed', 'fermentable 2 name']].drop_duplicates()
fermentable_trans_3 = df[['fermentable 3 name_transformed', 'fermentable 3 name']].drop_duplicates()
fermentable_trans_4 = df[['fermentable 4 name_transformed', 'fermentable 4 name']].drop_duplicates()
fermentable_trans_5 = df[['fermentable 5 name_transformed', 'fermentable 5 name']].drop_duplicates()

df['fermentable 1 name'] = df['fermentable 1 name_transformed']
df['fermentable 2 name'] = df['fermentable 2 name_transformed']
df['fermentable 3 name'] = df['fermentable 3 name_transformed']
df['fermentable 4 name'] = df['fermentable 4 name_transformed']
df['fermentable 5 name'] = df['fermentable 5 name_transformed']

df['hop 1 name'] = df['hop 1 name_transformed']
df['hop 2 name'] = df['hop 2 name_transformed']
df['hop 3 name'] = df['hop 3 name_transformed']
df['hop 4 name'] = df['hop 4 name_transformed']

df['hop 1 phase'] = df['hop 1 phase_transformed']
df['hop 2 phase'] = df['hop 2 phase_transformed']
df['hop 3 phase'] = df['hop 3 phase_transformed']
df['hop 4 phase'] = df['hop 4 phase_transformed']

df['hop 1 form'] = df['hop 1 form_transformed']
df['hop 2 form'] = df['hop 2 form_transformed']
df['hop 3 form'] = df['hop 3 form_transformed']
df['hop 4 form'] = df['hop 4 form_transformed']

df['yeast'] = df['yeast_transformed']

df = df.drop(['yeast_transformed', 'fermentable 1 name_transformed', 'fermentable 2 name_transformed', 'fermentable 3 name_transformed', 'fermentable 4 name_transformed', 'fermentable 5 name_transformed',
'hop 1 name_transformed','hop 2 name_transformed', 'hop 3 name_transformed','hop 4 name_transformed', 'hop 1 phase_transformed','hop 2 phase_transformed', 'hop 3 phase_transformed','hop 4 phase_transformed',
'hop 1 form_transformed', 'hop 2 form_transformed', 'hop 3 form_transformed', 'hop 4 form_transformed'], axis=1)

print(df.columns)

print(df.head())
print(df.shape)

latent_dimension = 4
batch_size = 20
hidden_nodes = 16

X = np.asarray(df).astype(np.float32)

# Encoder
input_encoder = Input(shape=(32,), name="Input_Encoder")
x = BatchNormalization()(input_encoder)
x = Dense(hidden_nodes, activation="relu", name="Hidden_Encoding")(x)
x = BatchNormalization()(x)
z = Dense(latent_dimension, name="Mean")(x)
encoder = Model(input_encoder, z)

# Decoder
input_decoder = Input(shape=(latent_dimension,), name="Input_Decoder")
x = BatchNormalization()(input_decoder)
x = Dense(hidden_nodes, activation="relu", name="Hidden_Decoding")(x)
x = BatchNormalization()(x)
decoder_output = Dense(32, activation="linear", name="Decoded")(x)
decoder = Model(input_decoder, decoder_output, name="Decoder")

encoder_decoder = decoder(encoder(input_encoder))
vae = Model(input_encoder, encoder_decoder)

vae.compile(loss="mean_squared_error", optimizer="adam")
history = vae.fit(
    X, X, shuffle=True, epochs=500, batch_size=20, validation_split=0.2, verbose=2
).history
vae.save('./trained_vae.h5')

def plot_loss(history):
    train_loss = history["loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(16, 12))
    plt.title("Mean squared error", fontname="Times New Roman Bold")
    plt.plot(train_loss, label="Train", linewidth=3)
    plt.plot(val_loss, label='Test', linewidth=3)
    plt.xlabel("Epochs")

    plt.legend()
    plt.savefig("Loss.png", dpi=300)
    plt.show()
    print(f"Training MSE = {np.sqrt(train_loss[-1])}")
    print(f"Validation MSE = {np.sqrt(val_loss[-1])}")


plot_loss(history)