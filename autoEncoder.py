# Package imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,Flatten,Reshape, LeakyReLU
from tensorflow.keras.models import Sequential

# Auto-encoding function
def autoEncode(data, catColumns = [], numColumns = [], 
               nFeatures = 5, minSize = 1, nEpochs=100, 
               deleteOld = True, verbose = False) :
    """
    data: Dataset that dimension reduction ought to be performed on
    catColumns: List of categorical column names that are to be autoencoded
    numColumns: List of numerical column names that are to be autoencoded
    nFeatures: Number of features that the selected category columns ought to be reduced to
    minSize: Minimum size of a category not to be lumped into the "Other" category (increase this to reduce running time at the cost of performance)
    nEpochs: Number of epochs that the autoencoder performs (increase this to get better performance but also longer running time)
    deleteOld: Delete columns for which data reduction has been performed (True = Delete, False = Do not delete)
    """
    
    # Create copy of data to prevent unintentional overwrites
    data = data.copy()
    
    # Set less frequent categories to "other" to reduce running time (if <minSize> is high enough)
    for category in catColumns :
        counts = data[category].value_counts()
        data.loc[data[category].isin(counts[counts < minSize].index), category] = "Other"
        
    # One-hot encoding
    OHenc = OneHotEncoder() 
    dataCategorical = OHenc.fit_transform(data[catColumns]).toarray()
    dataNumerical = data[numColumns].to_numpy()
    reducableData = np.concatenate((dataCategorical, dataNumerical), axis=1)
    
    # Encoder
    encoder = Sequential()
    encoder.add(Flatten(input_shape=[reducableData.shape[1]]))
    encoder.add(Dense(512,activation=LeakyReLU()))
    encoder.add(Dense(256,activation=LeakyReLU()))
    encoder.add(Dense(128,activation=LeakyReLU()))
    encoder.add(Dense(64,activation=LeakyReLU()))
    encoder.add(Dense(nFeatures,activation=LeakyReLU()))
     
    # Decoder
    decoder = Sequential()
    decoder.add(Dense(64,input_shape=[nFeatures],activation=LeakyReLU()))
    decoder.add(Dense(128,activation=LeakyReLU()))
    decoder.add(Dense(256,activation=LeakyReLU()))
    decoder.add(Dense(512,activation=LeakyReLU()))
    decoder.add(Dense(reducableData.shape[1], activation=LeakyReLU()))
    decoder.add(Reshape([reducableData.shape[1]]))
     
    # Autoencoder
    autoencoder = Sequential([encoder,decoder])
    autoencoder.compile(loss="mse")
    callback = EarlyStopping(monitor='loss', patience=75, min_delta=0.0001)
    fit = autoencoder.fit(reducableData, reducableData,
                          epochs=nEpochs, callbacks=[callback], verbose=verbose)
    mse = np.min(fit.history['loss'])
    encoded_nFeatures = encoder.predict(reducableData)
    
    # Putting everything ino a new dataframe & then adding it back to the original data
    reducedColumnsFrame = pd.DataFrame(encoded_nFeatures, columns = ["cat"+str(i) for i in range(1,nFeatures+1)])
    if deleteOld == True :
        data.drop(columns=catColumns+numColumns, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = pd.concat([data, reducedColumnsFrame], axis=1)
    
    # Return result
    return data, mse