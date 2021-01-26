import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

path_results = os.path.join(os.getcwd(), "results")

df_face_counts = pd.read_csv(os.path.join(path_results, "results_face_detect.csv"))

sns.barplot(data=df_face_counts, x="modes", y="val", color="C0")
plt.ylabel("images with correctly detected number of faces")
plt.tight_layout()
plt.show()