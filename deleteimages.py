import os

path = os.path.join(os.getcwd(), "projectB_data", "images")

for img in os.listdir(path):
    if "segmentat" in img:
        os.remove(os.path.join(path, img))

