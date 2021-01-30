import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

plt.style.use("seaborn-whitegrid")
matplotlib.rcParams.update({'font.size': 14})

path_results = os.path.join(os.getcwd(), "results")

df_face_counts = pd.read_csv(os.path.join(path_results, "results_face_detect.csv"))

fig1, ax1 = plt.subplots()
sns.barplot(data=df_face_counts, x="modes", y="val", color="C0", ax=ax1)
for p in ax1.patches:
    ax1.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()), ha='center', va='bottom', color='black')

plt.ylabel("images with correctly detected number of faces")
plt.tight_layout()
plt.show()