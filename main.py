from config import INPUT_FILE
from utils.excel_reader import load_dataset
from generators.generate_lhs import generate_lhs_dataset

# Load the original dataset
df = load_dataset(INPUT_FILE)

# Generate the LHS synthetic dataset
lhs_df = generate_lhs_dataset(df)

print("\nSynthetic Dataset Generated Successfully")

print("Rows :", len(lhs_df))
print("Columns :", len(lhs_df.columns))

# Save the dataset
output_file = "output/lhs_dataset2.csv"

lhs_df.to_csv(
    output_file,
    index=False
)

print("\nDataset Saved Successfully")
print("Location :", output_file)


# from config import INPUT_FILE
# from utils.excel_reader import load_dataset
# from generators.generate_sobol import generate_sobol_dataset

# # Load original dataset
# df = load_dataset(INPUT_FILE)

# # Generate Sobol synthetic dataset
# sobol_df = generate_sobol_dataset(df)

# print("\nSynthetic Dataset Generated Successfully")

# print("Rows :", len(sobol_df))
# print("Columns :", len(sobol_df.columns))

# # Save Dataset
# output_file = "output/sobol_dataset2.csv"

# sobol_df.to_csv(
#     output_file,
#     index=False
# )

# print("\nDataset Saved Successfully")
# print("Location :", output_file)

# from config import INPUT_FILE
# from utils.excel_reader import load_dataset
# from generators.generate_truncated_normal import generate_truncated_normal_dataset

# # Load original dataset
# df = load_dataset(INPUT_FILE)

# # Generate Truncated Normal synthetic dataset
# truncated_df = generate_truncated_normal_dataset(df)

# print("\nSynthetic Dataset Generated Successfully")

# print("Rows :", len(truncated_df))
# print("Columns :", len(truncated_df.columns))

# # Save Dataset
# output_file = "output/truncated_normal_dataset2.csv"

# truncated_df.to_csv(
#     output_file,
#     index=False
# )

# print("\nDataset Saved Successfully")
# print("Location :", output_file)


# from config import INPUT_FILE
# from utils.excel_reader import load_dataset
# from generators.generate_beta import generate_beta_dataset

# # Load original dataset
# df = load_dataset(INPUT_FILE)

# # Generate Beta Distribution dataset
# beta_df = generate_beta_dataset(df)

# print("\nSynthetic Dataset Generated Successfully")

# print("Rows :", len(beta_df))
# print("Columns :", len(beta_df.columns))

# # Save Dataset
# output_file = "output/beta_dataset2.csv"

# beta_df.to_csv(
#     output_file,
#     index=False
# )

# print("\nDataset Saved Successfully")
# print("Location :", output_file)

# from config import INPUT_FILE
# from utils.excel_reader import load_dataset
# from generators.generate_gaussian_copula import generate_gaussian_copula_dataset

# # Load the original dataset
# df = load_dataset(INPUT_FILE)

# # Generate Gaussian Copula synthetic dataset
# gaussian_df = generate_gaussian_copula_dataset(df)

# print("\nSynthetic Dataset Generated Successfully")

# print("Rows :", len(gaussian_df))
# print("Columns :", len(gaussian_df.columns))

# # Save the dataset
# output_file = "output/gaussian_copula_dataset2.csv"

# gaussian_df.to_csv(
#     output_file,
#     index=False
# )

# print("\nDataset Saved Successfully")
# print("Location :", output_file)