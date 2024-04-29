import pandas as pd
import matplotlib.pyplot as plt

from config import get_args
from data import read_dataset
from matplotlib.backends.backend_pdf import PdfPages

args = get_args()
features, _ = read_dataset(args.data_name)
number_of_whole_samples = int(features.shape[0])

# Read the first CSV file
df1 = pd.read_csv("bibTex,multimargin,20-30-30-10.csv", skip_blank_lines=True)
labels = df1.columns.tolist()
print(labels)
# Read the second CSV file
df2 = pd.read_csv("bibTex,random,20-30-30-10.csv", skip_blank_lines=True)

# Transpose both DataFrames
df1 = df1.T
df2 = df2.T
print(df1)
# Extract the metrics and algorithms from the first DataFrame
metrics = df1.index.tolist()
number_samples = ['{} samples'.format(args.train*number_of_whole_samples + i * args.active_instances) for i in range(0, len(df1.columns) )]

# Use the first row as names for the figures from the first DataFrame
pdf_filename = "results_bibtex.pdf"
with PdfPages(pdf_filename) as pdf:
    # Plot each metric from the first DataFrame and compare with the second DataFrame
    for metric, fig_name in zip(metrics, labels):
        plt.figure(figsize=(10, 6))
        plt.plot(number_samples, df1.loc[metric], marker='o', label='multimargin')

        # Add data from the second DataFrame
        plt.plot(number_samples, df2.loc[metric], marker='x', label='random')

        plt.title(fig_name)
        plt.xlabel('Number of samples')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()  # Show legend for better comparison
        plt.tight_layout()
        # Save the current plot to the PDF
        pdf.savefig()
        plt.close()  # Close the current plot to avoid memory issues
print("Plots saved to:", pdf_filename)
