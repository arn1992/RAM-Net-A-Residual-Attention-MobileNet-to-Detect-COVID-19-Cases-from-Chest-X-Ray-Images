import pandas as pd
import  numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
df = pd.read_csv('D:/polynomial/covid19/new_version/test_covid19.csv')
#print(df)
df=shuffle(df)
print(df.head())

reshaped_image = df["image_address"].map(lambda x: np.asarray(Image.open(x).resize((224,224), resample=Image.LANCZOS).\
                                                          convert("RGB")))

out_vec = np.stack(reshaped_image, 0)

print(out_vec.shape)

out_vec = out_vec.astype("float32")
print(out_vec.max())
out_vec /= 255

labels = df["dis_cat"].values
print(labels)
'''
X_train, X_val, y_train, y_val = train_test_split(out_vec, labels, test_size=0.10,  stratify=labels)

np.save("new_224_224_val.npy", X_val)
np.save("new_val_labels.npy", y_val)

np.save("new_224_224_train.npy", X_train)
np.save("new_train_labels.npy", y_train)

'''
np.save("D:/polynomial/covid19/new_version/new_covid19_test.npy", out_vec)
np.save("D:/polynomial/covid19/new_version/new_covid19_test_labels.npy", labels)

