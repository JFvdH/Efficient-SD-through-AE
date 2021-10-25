# Package imports
import pandas as pd

# Function used to import a specified dataset
def getData(dataset) :
    
    # Initializations
    dictio = {}

    # Ionosphere dataset
    if dataset == "Ionosphere" :
        for i in range(34):
            dictio[i]= 'attribute' + str(i)
        dictio[34] = 'target'
        df =  pd.read_csv('data/ionosphere.data', sep=",", header = None).rename(columns=dictio)
        df = df.replace({'g': 1, 'b':0})
        cat = []
        num =  [i for i in df.columns if i != 'target']
    
    # Mushroom dataset
    elif dataset == "Mushroom" :
        for i in range(1,23):
            dictio[i]= 'attribute' + str(i)
        dictio[0] = 'target'
        df =  pd.read_csv('data/Mushroom.data', sep=",", header = None).rename(columns=dictio)
        df['target'] = [1 if x == 'p' else 0 for x in df['target']]
        cat = [i for i in df.columns if i != 'target']
        num =  []
    
    # Adult dataset
    elif dataset == "Adult" :
        for i in range(14):
            dictio[i]= 'attribute' + str(i)
        dictio[14] = 'target'
        df =  pd.read_csv('data/adult.data', sep=",", header = None).rename(columns=dictio)
        df2 =  pd.read_csv('data/adult.test', sep=",", header = None, skiprows=1).rename(columns=dictio)
        df = pd.concat([df, df2])
        df = df.reset_index().drop(columns=['index'])
        df = df.replace({' <=50K': 0, ' >50K':1})
        num = ['attribute{}'.format(i) for i in [0,2,4,10,11,12]]
        cat = [i for i in df.columns if i != 'target' and i not in num]
    
    #Soybean dataset
    elif dataset == "Soybean" :
        df =  pd.read_csv('data/soybean-large.data', sep=",", header = None, na_values = ['?'])
        df = df.replace({1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 0: '0'})
        df = df.replace({'brown-spot': 1, 'frog-eye-leaf-spot': 1, 'alternarialeaf-spot': 1, 'phyllosticta-leaf-spot': 1})
        df['target'] = [i if i == 1 else 0 for i in df[0]]
        df = df.drop([0], axis=1)
        dictio = {}
        for i in range(1, 36):
              dictio[i]= 'attribute' + str(i)
        df = df.rename(columns=dictio)
        df = df.dropna()
        df = df.reset_index().drop(columns=['index'])
        cat = [i for i in df.columns if i != 'target']
        num =  []
    
    # Arrhythmia dataset
    elif dataset == "Arrhythmia" :
        lijst = {}
        count = 0
        for i in range (15, 160):
            if count >= 6:
                lijst['attribute{}'.format(i)] = 'str'    
            count+= 1
            if count == 12:
                count = 0
        lijst['attribute1'] = 'str'  
        dictio = {}
        for i in range(279):
              dictio[i]= 'attribute' + str(i)
        dictio[279] = 'target'
        df =  pd.read_csv('data/arrhythmia.data', sep=",", header = None, na_values = ['?']).rename(columns=dictio)
        df['target'] = [0 if x == 1 else 1 for x in df['target']]
        df = df.drop(columns=['attribute{}'.format(x) for x in [13, 19, 67, 139, 151, 164, 204, 264, 274]])
        df = df.astype(lijst)
        df = df.dropna()
        df = df.reset_index().drop(columns=['index'])
        cat = list(lijst.keys())
        num =  [i for i in df.columns if (i != 'target') and i not in lijst.keys()]
        
    # Indoor dataset
    elif dataset == "Indoor" :
        df = pd.read_csv('data/Indoor1.csv', sep=",", na_values = ['na'])
        df2 = pd.read_csv('data/Indoor2.csv', sep=",", na_values = ['na'])
        df = pd.concat([df, df2])
        df['target'] = [1 if x == 2 else 0 for x in df['BUILDINGID']]
        df = df.drop(['LONGITUDE','LATITUDE','FLOOR', 'SPACEID', 'RELATIVEPOSITION', 'TIMESTAMP', 'BUILDINGID'], axis=1)
        num = [i for i in df.columns if (i not in ['target','BUILDINGID', 'USERID','PHONEID'])]
        cat = ['USERID', 'PHONEID']
        df["USERID"]= df["USERID"].astype(str)
        df["PHONEID"]= df["PHONEID"].astype(str)
        df = df.reset_index()
        df = df.drop(columns=['index'])
    
    # Return the dataset <df>, a list of categorical features <cat>,
    # a list of numerical features <num> and a total set of features
    features = num+cat
    return df, cat, num, features