import os
import pandas as pd

path_meta = os.path.join(os.getcwd(), "data", "Fam4a_used", "Fam4a_used_meta.csv")
df_meta = pd.read_csv(path_meta)
#df_col = df_meta.dropna(subset=["genders_0", "genders_1", "genders_2", "genders_3"])

for col in [f"genders_{i}" for i in range(4)]:
    df_col = df_meta.dropna(subset=[col])
    count_correct = 0
    count_sum = 0
    for i, row in df_col.iterrows():
        count_sum += len(row["genders"])
        for correct, classified in zip(row["genders"], row[col]):
            if correct == classified:
                count_correct += 1
    print(count_correct, count_sum, "{:.2f}".format(count_correct/count_sum))