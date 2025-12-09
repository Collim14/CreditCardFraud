import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib as plt

#df = pd.read_csv("hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv")
#df.to_csv("fraudTrain.csv", index=False)
#df = pd.read_csv("fraudTrain.csv")


#df.to_csv("fraudTrain.csv", index=False)
df = pd.read_csv("Data/creditcard.csv")
print(df.head())
df2 = pd.read_csv("Data/fraudTrain.csv")
print(df2.head())
# print('First one done')
# df2 = pd.read_csv("Data/IEEE-Data/train_identity.csv")
# print(df2.head())
# print('Second one done')
# new = df.join(df2.set_index('TransactionID'), on='TransactionID', how='left') 
# print('Joined')
# print(new.head())
# new.to_csv("Data/IEEE-Data/train_Combined.csv", index=False)
# categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# for col in categorical_columns:
#     print(f"\nColumn: '{col}'")
    
#     # 3. Get and print the value counts, including NaNs
#     print("Value Counts (including missing values):")
#     print(df[col].value_counts(dropna=False))
#     print("-" * 30) # Separator


# print("\nCalculating correlation matrix...")
# correlation_matrix = df.corr()

# print("\nCorrelation Matrix:")
# print(correlation_matrix)


# # 3. Create the Correlation Heatmap
# g = sns.clustermap(
#     correlation_matrix,
#     cmap='coolwarm',
#     figsize=(18, 18),
#     vmin=-1, vmax=1,
#     xticklabels=False,  # Hide x-axis labels for cleanliness
#     yticklabels=False   # Hide y-axis labels for cleanliness
# )

# g.fig.suptitle('Clustered Correlation Heatmap (434x434 would be denser)', fontsize=16)
# plt.show()