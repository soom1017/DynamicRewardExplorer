import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('/Users/lee/Downloads/wandb_export_2024-10-21T22_00_53.427+09_00.csv', encoding='cp949')
df2 = pd.read_csv('/Users/lee/Downloads/wandb_export_2024-10-21T22_00_59.042+09_00.csv', encoding='cp949')

diff = df1["dynamic - VLMEpRet"] - df2["dynamic - EpRet"]

plt.figure(figsize=(10, 6))
plt.plot(df1["Step"], diff)
plt.title("Difference between VLMEpRet and EpRet")
plt.grid(True)
plt.show()
