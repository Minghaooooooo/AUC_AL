import numpy as np
import pandas as pd

filename = ['Delicious_X', 'bibTex_X', 'Bookmark_X', 'Col5k_X', 'eron_X']
for fn in filename:
    # Load the .npy file
    data = np.load(fn+'.npy')

    print(f"Length of {fn}: {round(len(data) * 0.02)}")
    # # Convert to DataFrame
    # df = pd.DataFrame(data)
    #
    # # Save as .csv
    # df.to_csv(fn+'.csv', index=False)
