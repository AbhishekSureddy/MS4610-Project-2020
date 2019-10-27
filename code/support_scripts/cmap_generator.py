import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate cmaps and save to drive
train_zero = pd.read_csv("/home/nishant/Desktop/IDA Project/mod_data/train_zero.csv")
app_keys = train_zero.application_key.values
features = train_zero.drop('default_ind', axis=1).values

for i in range(81000, len(features)):
    print("Now processing %d of %d" % (i, len(features)))
    plt.figure()
    plt.axis('off')
    plt.imshow(features[i].reshape((8, 5)), cmap='nipy_spectral')
    plt.savefig('/home/nishant/Desktop/IDA Project/cmaps/' + str(app_keys[i]), format='png', bbox_inches='tight')