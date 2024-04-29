import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the first CSV file
df1 = pd.read_csv("Multimargin,r10,i120,e30,p30.csv", skip_blank_lines=True)
labels = df1.columns.tolist()
print(labels)
# Read the second CSV file
df2 = pd.read_csv("Random,r10,i120,e30,p30.csv", skip_blank_lines=True)
df3 = pd.read_csv("Entropy, r10,i120,e30,p30.csv", skip_blank_lines=True)
df4 = pd.read_csv("MarginSampling,r10,i120,e30,p30.csv", skip_blank_lines=True)


# Transpose both DataFrames
df1 = df1.T
df2 = df2.T
df3 = df3.T
df4 = df4.T
print(df1)
# Extract the metrics and algorithms from the first DataFrame
metrics = df1.index.tolist()
number_samples = ['{} samples'.format(82 + i * 120) for i in range(0, len(df1.columns) )]

# Create a PDF file to save the plots
pdf_filename = "results_delicious.pdf"
with PdfPages(pdf_filename) as pdf:
    for metric, fig_name in zip(metrics, labels):
        plt.figure(figsize=(10, 6))
        plt.plot(number_samples, df1.loc[metric], marker='o', label='Multimargin')

        # Add data from the second DataFrame
        plt.plot(number_samples, df2.loc[metric], marker='x', label='Random')
        plt.plot(number_samples, df3.loc[metric], marker='x', label='Entropy')
        plt.plot(number_samples, df4.loc[metric], marker='x', label='MarginSampling')

        plt.title(fig_name)
        plt.xlabel('Number of samples')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()  # Show legend for better comparison
        plt.tight_layout()

        # Save the current plot to the PDF
        pdf.savefig()
        plt.close()  # Close the current plot to avoid memory issues

# Show a message indicating the PDF file has been saved
print("Plots saved to:", pdf_filename)




