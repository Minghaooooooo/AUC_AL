import numpy as np
import pandas as pd

filename = ['Delicious_X', 'Delicious_Y', 'bibTex_X', 'bibTex_Y']
for fn in filename:
    # Load the .npy file
    data = np.load(fn+'.npy')

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save as .csv
    df.to_csv(fn+'.csv', index=False)
