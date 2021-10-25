# Package imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from beamSearch import as_string

# Function used to evaluate and summarize BeamSearch outcomes
def calc_result_bs(df_1, df_2, subgroups_1, subgroups_2):  
    # df_1/subgroups_1 should refer to the auto-encoded case (dataset and selectors respectively)
    # df_2/subgroups_2 should refer to the nonauto-encoded case (dataset and selectors respectively)
    
    # Count number of times each entry occurs in a subgroup for the (non)auto-encoded case
    # and retrieve the WRAcc
    df_1['subgroups_1'] = 0
    df_2['subgroups_2'] = 0
    wracc_g1, wracc_g2 = [], []
    for i in subgroups_1 :
        wracc_g1.append(i[0])
        occurence = df_1.eval(as_string(i[1]))
        df_1['subgroups_1'] += occurence
    for i in subgroups_2 :
        wracc_g2.append(i[0])
        occurence = df_2.eval(as_string(i[1]))
        df_2['subgroups_2'] += occurence    
    
    # Calculate the number of times an entry is included in the auto-encoded/nonauto-encoded case, 
    # when it is not included in the nonautoencoded/auto-encoded case
    add = len(df_1[(df_1['subgroups_1'] > 0) & (df_2['subgroups_2'] == 0)])
    delete = len(df_1[(df_1['subgroups_1'] == 0) & (df_2['subgroups_2'] > 0)])
    
    # Print statements
    print('coverage autoencoding: {}, ({})'.format(len(df_1[df_1['subgroups_1']>0]), len(df_1[df_1['subgroups_1']>0])/len(df_1)))
    print('coverage no auto encoding: {}, ({})'.format(len(df_2[df_2['subgroups_2']>0]), len(df_2[df_2['subgroups_2']>0])/len(df_2)))
    print('# rows added in subgroups: {} ({})'.format(add, add/len(df_1)))
    print('# rows no longer in subgroups: {}, ({})'.format(delete, delete/len(df_1)))
    print('average subgroup size auto encoded: {}'.format(df_1['subgroups_1'].sum()/len(subgroups_1)))
    print('average subgroup size no auto encoding: {}'.format(df_2['subgroups_2'].sum()/len(subgroups_2)))
    print('WRACC auto encoding: Max: {}, Mean: {}'.format(subgroups_1[0][0], np.mean(wracc_g1)))
    print('WRACC no auto encoding: Max: {}, Mean: {}'.format(subgroups_2[0][0], np.mean(wracc_g2)))
    
    df_1['subgroups_1'].hist();
    plt.title("Auto-encoding")
    plt.show()
    df_2['subgroups_2'].hist();
    plt.title("No auto-encoding")
    plt.show()
    
# Function used to evaluate and summarize the PySubgroup algorithm outcomes
def calc_result_ps(df_1, df_2, results_df_1, results_df_2):
    # df_1/results_df_1 should refer to the auto-encoded case (dataset and selectors respectively)
    # df_2/results_df_2 should refer to the nonauto-encoded case (dataset and selectors respectively)
    
    # Count number of times each entry occurs in a subgroup for the (non)auto-encoded case
    # and retrieve the WRAcc
    df_1['subgroups_1'] = 0
    df_2['subgroups_2'] = 0
    for i in range(len(results_df_1)) :
        oper = results_df_1["subgroup"][i]
        oper = oper.replace("AND", "&")
        if oper.find(":") >= 0 :
            newOpers = []
            splitOper = oper.split(" & ")
            for j in range(len(splitOper)-1, -1, -1) :
                dpIndex = splitOper[j].find(":")
                if dpIndex >= 0 :
                    attr = splitOper[j][:dpIndex]
                    brIndex = splitOper[j].find("[")
                    dpIndex2 = splitOper[j].find(":", dpIndex+1)
                    lb = splitOper[j][brIndex+1:dpIndex2]
                    ub = splitOper[j][dpIndex2+1:-1]
                    newOpers.append(attr+">="+lb)
                    newOpers.append(attr+"<="+ub)
                    del splitOper[j]
            splitOper += newOpers
            oper = " & ".join(splitOper)
        df_1['subgroups_1'] += df_1.eval(oper)
    s1_wracc_max = results_df_1['quality'].max()
    s1_wracc_mean = results_df_1['quality'].mean()  
    for i in range(len(results_df_2)) :
        oper = results_df_2["subgroup"][i]
        oper = oper.replace("AND", "&")
        if oper.find(":") >= 0 :
            newOpers = []
            splitOper = oper.split(" & ")
            for j in range(len(splitOper)-1, -1, -1) :
                dpIndex = splitOper[j].find(":")
                if dpIndex >= 0 :
                    attr = splitOper[j][:dpIndex]
                    brIndex = splitOper[j].find("[")
                    dpIndex2 = splitOper[j].find(":", dpIndex+1)
                    lb = splitOper[j][brIndex+1:dpIndex2]
                    ub = splitOper[j][dpIndex2+1:-1]
                    newOpers.append(attr+">="+lb)
                    newOpers.append(attr+"<="+ub)
                    del splitOper[j]
            splitOper += newOpers
            oper = " & ".join(splitOper)
        df_2['subgroups_2'] += df_2.eval(oper)
    s2_wracc_max = results_df_2['quality'].max()
    s2_wracc_mean = results_df_2['quality'].mean()

    # Calculate the number of times an entry is included in the auto-encoded/nonauto-encoded case, 
    # when it is not included in the nonautoencoded/auto-encoded case
    add = len(df_1[(df_1['subgroups_1'] > 0) & (df_2['subgroups_2'] == 0)])
    delete = len(df_1[(df_1['subgroups_1'] == 0) & (df_2['subgroups_2'] > 0)])

    # Print statements
    print('coverage auto-encoding: {}, ({})'.format(len(df_1[df_1['subgroups_1']>0]), len(df_1[df_1['subgroups_1']>0])/len(df_1)))
    print('coverage no auto-encoding: {}, ({})'.format(len(df_2[df_2['subgroups_2']>0]), len(df_2[df_2['subgroups_2']>0])/len(df_2)))
    print('# rows added in subgroups: {} ({})'.format(add, add/len(df_1)))
    print('# rows no longer in subgroups: {}, ({})'.format(delete, delete/len(df_1)))
    print('average subgroup size auto encoded: {}'.format(df_1['subgroups_1'].sum()/len(results_df_1)))
    print('average subgroup size no auto encoding: {}'.format(df_2['subgroups_2'].sum()/len(results_df_2)))
    
    print('WRACC auto-encoding: Max: {}, Mean: {}'.format(s1_wracc_max, s1_wracc_mean))
    print('WRACC no auto-encoding: Max: {}, Mean: {}'.format(s2_wracc_max, s2_wracc_mean))

    # Plot histograms for the number of times entries occur in a subgroup for auto-encoding/nonauto-encoding respectively
    df_1['subgroups_1'].hist();
    plt.title("Auto-encoding")
    plt.show()
    df_2['subgroups_2'].hist();
    plt.title("No auto-encoding")
    plt.show()