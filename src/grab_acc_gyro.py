import pandas as pd
import glob

for file in glob.glob("../original_data/*_norm_labelled*"):
    # print(file)
    f = pd.read_csv(file,header=None)
    # print(f.head())
    # print(f.columns)
    f = f.drop(columns=range(6,12))
    # print(f.head())
    # print(f.columns)

    name = file.split('/')[-1]
    print(name)

    f.to_csv("../original_data_accgyro/"+name, index=False, header=False)
