# Function to standardize numerical variables
# Used for the arrhythmia and adult datasets
def standardize(df, num) :
    """
    df: DataFrame of the dataset
    num: List containing numerical column names
    """
    df_std = df.copy()
    for col in num :
        if col != 'target' :
            mean = df_std[col].mean()
            std = df_std[col].std()
            df_std[col] = (df_std[col]-mean)/std
    return df_std