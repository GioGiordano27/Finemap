import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
from itertools import combinations
import matplotlib.pyplot as plt

# When running, it will take 5 minutes in order to generate results for 3 SNPS . All the figures will be shown and
# generated as well. I believe you might need to move figure 2, because it is hiding figure 1. As soon as you cloe the
# figures,it will stop the program. It will also generate the data into the COMP565_A2_SNP_pip.csv as instructed.  :)

### -------------------------------------------------Part 0-------------------------------------------------------------
#Reading z scores and LD matrix

f_z_scores = "/Users/user/Downloads/zscore.csv"
f_LD = "/Users/user/Downloads/LD.csv"

df_z = pd.read_csv(f_z_scores)
df_LD = pd.read_csv(f_LD)

print("--------------------------Below, you will find the dataframes of the z scores and the LDs----------------------")
print(df_z)
print(df_LD)

df_LD.rename(columns={'Unnamed: 0': 'SNP_name'}, inplace=True)
df_z.rename(columns={'Unnamed: 0': 'SNP_name'}, inplace=True)

print("----------------------Below, you will find the renamed dataframes of the z scores and the LDs------------------")
print(df_z)
print(df_LD)

####-------------------------------------------------Parts 1-3----------------------------------------------------------
###---------------------------------------------------Part 1------------------------------------------------------------
##Implementation of the efficient Bayes factor

snps = df_z["SNP_name"].tolist()

# 1 SNP
comb1 = combinations(snps, 1)
colnames1 = ["SNP1"]

comb1_list = []
for i in list(comb1):
    comb1_list.append(i)

df_comb1 = pd.DataFrame(comb1_list, columns=colnames1)

print("--------------------------Below, you will find the dataframe of the combinations of 1 SNP----------------------")
print(df_comb1)

# Steps used are nearly identical for 1-3 SNPs, this should make it more readable
for i, r in df_comb1.iterrows(): # i = index , r = rows
    oneSNP = [r.SNP1]
    oneSNP_copy1 = oneSNP.copy()
    oneSNP_copy1.insert(0, "SNP_name")

    df_LD1 = df_LD[oneSNP_copy1]

    df_LD1 = df_LD1[df_LD1['SNP_name'].isin(oneSNP)]
    df_z1 = df_z[df_z['SNP_name'].isin(oneSNP)]

    # Dataframes -> Numpy arrays
    z_c1 = df_z1["V1"].to_numpy()
    R_cc1 = df_LD1.to_numpy()[:, 1:]

    # Variable initialization
    cov_CC1 = 2.49 * np.identity(1) # Given in the question
    cov_num1 = R_cc1 + np.matmul(np.matmul(R_cc1, cov_CC1), R_cc1)
    num1 = multivariate_normal.pdf(z_c1, mean=np.zeros(1), cov=cov_num1)
    denom1 = multivariate_normal.pdf(z_c1, mean=np.zeros(1), cov=R_cc1)

    BF1 = num1 / denom1
    df_comb1.loc[i, "BF3"] = BF1
    df_comb1.loc[i, "SNP2"] = "-----"
    df_comb1.loc[i, "SNP3"] = "-----"

df_comb1 = df_comb1.reindex(columns=['SNP1', 'SNP2', 'SNP3', 'BF3'])

print("---------------------Below, you will find the updated dataframe of the combinations of 1 SNP-------------------")
print(df_comb1)

# 2 SNPs
comb2 = combinations(snps, 2)
colnames2 = ["SNP1", "SNP2"]

comb2_list = []
for i in list(comb2):
    comb2_list.append(i)

df_comb2 = pd.DataFrame(comb2_list, columns=colnames2)
print("--------------------------Below, you will find the dataframe of the combinations of 2 SNP----------------------")
print(df_comb2)

for i, r in df_comb2.iterrows():

    twoSNPs = [r.SNP1, r.SNP2]
    one_SNP_copy2 = twoSNPs.copy()
    one_SNP_copy2.insert(0, "SNP_name")

    df_LD2 = df_LD[one_SNP_copy2]

    df_LD2 = df_LD2[df_LD2['SNP_name'].isin(twoSNPs)]
    df_z2 = df_z[df_z['SNP_name'].isin(twoSNPs)]

    z_c2 = df_z2["V1"].to_numpy()
    R_cc2 = df_LD2.to_numpy()[:, 1:]

    cov_CC2 = 2.49 * np.identity(2) # diag matrix of 2.49
    # (0,0) -> mean of the multivariate normal

    cov_num2 = R_cc2 + np.matmul(np.matmul(R_cc2, cov_CC2), R_cc2)
    cov_num2 = cov_num2.astype('float64')

    if np.linalg.det(cov_num2) != 0:
        num2 = multivariate_normal.pdf(z_c2, mean=np.zeros(2), cov=cov_num2, allow_singular=True)
        denom2 = multivariate_normal.pdf(z_c2, mean=np.zeros(2), cov=R_cc2, allow_singular=True)

        BF2 = num2 / denom2
        df_comb2.loc[i, "BF3"] = BF2
        df_comb2.loc[i, "SNP3"] = "----"
    else:
        df_comb2.loc[i, "BF3"] = 99999999999999999999

df_comb2 = df_comb2.reindex(columns=['SNP1', 'SNP2', 'SNP3', 'BF3'])
df_comb2 = df_comb2[df_comb2.BF3 != 99999999999999999999]

print("---------------------Below, you will find the updated dataframe of the combinations of 2 SNP-------------------")
print(df_comb2)

# 3 SNPs
comb3 = combinations(snps, 3)
colnames3 = ["SNP1", "SNP2", "SNP3"]

comb3_list = []
for i in list(comb3):
    comb3_list.append(i)

df_comb3 = pd.DataFrame(comb3_list, columns=colnames3)
print("--------------------------Below, you will find the dataframe of the combinations of 3 SNP----------------------")
print(df_comb3)

for i, r in df_comb3.iterrows():

    threeSNPs = [r.SNP1, r.SNP2, r.SNP3]
    one_SNP_copy3 = threeSNPs.copy()
    one_SNP_copy3.insert(0, "SNP_name")

    df_LD3 = df_LD[one_SNP_copy3]

    df_LD3 = df_LD3[df_LD3['SNP_name'].isin(threeSNPs)]
    df_z3 = df_z[df_z['SNP_name'].isin(threeSNPs)]

    z_c3 = df_z3["V1"].to_numpy()
    R_cc3 = df_LD3.to_numpy()[:, 1:]

    cov_CC3 = 2.49 * np.identity(3) 

    cov_num3 = R_cc3 + np.matmul(np.matmul(R_cc3, cov_CC3), R_cc3)
    cov_num3 = cov_num3.astype('float64')

    if np.linalg.det(cov_num3) > 10 ** (-9) or np.linalg.det(cov_num3) != 0: # this fixed a lot of bugs
        num3 = multivariate_normal.pdf(z_c3, mean=np.zeros(3), cov=cov_num3, allow_singular=True)
        denom3 = multivariate_normal.pdf(z_c3, mean=np.zeros(3), cov=R_cc3, allow_singular=True)
        BF3 = num3 / denom3
        df_comb3.loc[i, "BF3"] = BF3
    else:
        df_comb3.loc[i, "BF3"] = 99999999999999999999

print("---------------------Below, you will find the updated dataframe of the combinations of 3 SNP-------------------")
print(df_comb3)

### -------------------------------------------------Part 2-------------------------------------------------------------
#Implementation of the prior calculation
num_SNPs = 100

print("-----------------------------Below, you will find the priors of 1-3 SNP combinations---------------------------")
df_comb1["prior"] = (1 / num_SNPs) * ((num_SNPs - 1) / num_SNPs) ** (num_SNPs - 1)
print(df_comb1["prior"])

df_comb2["prior"] = ((1 / num_SNPs) ** 2) * ((num_SNPs - 1) / num_SNPs) ** (num_SNPs - 2)
print(df_comb2["prior"])

df_comb3["prior"] = ((1 / num_SNPs) ** 3) * ((num_SNPs - 1) / num_SNPs) ** (num_SNPs - 3)
print(df_comb3["prior"])

df_comb1['posterior'] = df_comb1['BF3'] * df_comb1['prior']
print("----------------------------Below, you will find the posteriors of 1-3 SNP combinations------------------------")
print(df_comb1["posterior"])

df_comb2['posterior'] = df_comb2['BF3'] * df_comb2['prior']
print(df_comb2["posterior"])

df_comb3['posterior'] = df_comb3['BF3'] * df_comb3['prior']
print(df_comb3["posterior"])

comb_frames = [df_comb1, df_comb2, df_comb3]
df_merged = pd.concat(comb_frames)

### -------------------------------------------------Part 3-------------------------------------------------------------
# Implementation of posterior inference

df_merged["normalized_posterior"] = df_merged["posterior"] / df_merged["posterior"].sum()
df_merged = df_merged.sort_values(by=['normalized_posterior'], ascending=False)
df_merged['sorted_configs'] = df_merged.index
df_merged.plot.scatter(x="sorted_configs", y="normalized_posterior", alpha=0.6) # Generated at the end

### -------------------------------------------------Part 4-------------------------------------------------------------
# Pip implementation and visualization

total_scores_sum = df_merged["posterior"].sum()
snp_pip = {}

for snp in snps:
    df_filtered = df_merged[(df_merged == snp).any(axis=1)] # Extract the rows containing the SNP
    snp_scores_sum = df_filtered["posterior"].sum() # Sum the scores of the snp
    pip = snp_scores_sum / total_scores_sum
    snp_pip[snp] = [pip]

df_result = pd.DataFrame(snp_pip)
df_result = df_result.T.reset_index()

df_result.columns = ['SNP_name', 'pip']
df_result.to_csv("COMP565_A2_SNP_pip.csv.gz")

# Calculate -log10p value from z-scores
df_z["pValue"] = scipy.stats.norm.sf(abs(df_z["V1"]))
df_z["-log10p"] = - np.log10(df_z["pValue"])

df_merged_p_pip = pd.merge(df_z, df_result, on=['SNP_name']) # To get the plot like in the example
fig, axes = plt.subplots(2, 1)

df_merged_p_pip.plot.scatter(x="SNP_name", y="-log10p", alpha=0.6, ax=axes[0])
df_merged_p_pip.plot.scatter(x="SNP_name", y="pip", alpha=0.6, ax=axes[1])
plt.show()

# ------------------Relative differences between the calculated inferred pips and the ones provided---------------------
