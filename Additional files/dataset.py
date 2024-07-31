# # will load dataset here & preprocess
# # will be made available to all algorithms

# import pandas as pd
# from sklearn import preprocessing

# combinedData = pd.read_csv("./InSDN/mergedCSV.csv", low_memory=False)

# minMaxScaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# scaledData = minMaxScaler.fit_transform(combinedData)
# scaledDataFrame = pd.DataFrame(scaledData, columns=combinedData.columns)

# scaledDataFrame.to_csv('./InSDN/normalisedData.csv', index=False)

# batch_size = 10
# num_batches_printed = 0


# # Iterate over the DataFrame in batches
# for i in range(0, len(scaledDataFrame), batch_size):
#     if num_batches_printed >= 10:
#         break  # Stop the loop after printing 10 batches
    
#     batch = scaledDataFrame.iloc[i:i+batch_size]
#     print(f"Batch {num_batches_printed + 1}:")
#     print(batch)
#     print()  # Adds an empty line between batches
    
#     # Increment the counter
#     num_batches_printed += 1
