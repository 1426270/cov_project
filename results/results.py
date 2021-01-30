import os
import pandas as pd

path_images = os.path.join(os.getcwd(), "data", "Fam4a_used")
path_meta = os.path.join(path_images, "Fam4a_used_meta.csv")

df_meta = pd.read_csv(path_meta)

for mode in range(4):
    correctgendersmode = 0
    wronggendersmode = 0
    for i, row in df_meta.iterrows():
        path_output_img = df_meta[f'path_{mode}'][i]
        if "correct_number_of_faces" in path_output_img:
            genders = row["genders"]
            genders_mode = row[f'genders_{mode}']

            for i, char in enumerate(genders):
                if genders_mode[i] == genders[i]:
                    correctgendersmode += 1
                else:
                    wronggendersmode += 1

    print("Correct genders mode", mode, correctgendersmode, 100*correctgendersmode/(correctgendersmode+wronggendersmode), " %")
    print("Wrong genders   mode", mode, wronggendersmode)