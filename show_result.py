"""
File to show result of learning from the csv file.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

df = pd.read_csv("log_self_train.csv")
sns.lineplot(data=df, x="episode_id", y="score")
plt.savefig('result.png')
