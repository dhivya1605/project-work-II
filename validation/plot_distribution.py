import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/lhs_dataset.csv")

crop = "rice"

rice = df[df["CROPS"] == crop]

plt.hist(rice["TEMPERATURE"], bins=25)

plt.title("Rice Temperature Distribution")

plt.xlabel("Temperature")

plt.ylabel("Frequency")

plt.show()