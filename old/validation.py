# import pandas as pd

# # ==========================
# # LOAD DATASETS
# # ==========================
# original_df = pd.read_excel("crop-dataset.xlsx")
# synthetic_df = pd.read_csv("synthetic_crop_datasets1_100000.csv")

# # ==========================
# # VALIDATION
# # ==========================
# all_passed = True

# for _, crop_row in original_df.iterrows():

#     crop = crop_row["CROPS"]

#     crop_data = synthetic_df[
#         synthetic_df["CROPS"] == crop
#     ]

#     print("\n" + "=" * 70)
#     print(f"Crop: {crop}")
#     print("=" * 70)

#     validations = [
#         ("SOIL_PH", "SOIL_PH_LOW", "SOIL_PH_HIGH"),
#         ("TEMPERATURE", "MIN_TEMP", "MAX_TEMP"),
#         ("CROP_DURATION", "CROPDURATION_MIN", "CROPDURATION_MAX"),
#         ("WATER_REQUIRED", "WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
#         ("RELATIVE_HUMIDITY", "RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
#         ("N", "N_MIN", "N_MAX"),
#         ("P", "P_MIN", "P_MAX"),
#         ("K", "K_MIN", "K_MAX")
#     ]

#     crop_passed = True

#     for feature, min_col, max_col in validations:

#         original_min = crop_row[min_col]
#         original_max = crop_row[max_col]

#         generated_min = crop_data[feature].min()
#         generated_max = crop_data[feature].max()

#         status = (
#             "PASS"
#             if generated_min >= original_min
#             and generated_max <= original_max
#             else "FAIL"
#         )

#         if status == "FAIL":
#             all_passed = False
#             crop_passed = False

#         print(f"\n{feature}")
#         print(f"Original Range : {original_min} - {original_max}")
#         print(f"Generated Range: {generated_min} - {generated_max}")
#         print(f"Status         : {status}")

#     # Crop Summary
#     print("\n" + "-" * 70)

#     if crop_passed:
#         print(f"✅ {crop}: All generated values are within the crop-specific ranges.")
#     else:
#         print(f"❌ {crop}: Some generated values are outside the crop-specific ranges.")

#     print("-" * 70)

# # ==========================
# # FINAL SUMMARY
# # ==========================
# print("\n" + "=" * 70)
# print("FINAL VALIDATION REPORT")
# print("=" * 70)

# if all_passed:
#     print("✅ VALIDATION SUCCESSFUL")
#     print("✅ All crops passed validation.")
#     print("✅ All generated records satisfy their crop-specific ranges.")
#     print("✅ Soil pH, Temperature, Crop Duration, Water Requirement,")
#     print("   Relative Humidity, N, P and K values are valid.")
# else:
#     print("❌ VALIDATION FAILED")
#     print("❌ One or more crops contain values outside their original ranges.")

# print("=" * 70)



import pandas as pd

# Load datasets
original_df = pd.read_excel("crop-dataset.xlsx")
# synthetic_df = pd.read_csv("synthetic_crop_datasets1_600.csv")
# synthetic_df = pd.read_csv("synthetic_crop_datasets1_1200.csv")
synthetic_df = pd.read_csv("synthetic_crop_datasets1_2000.csv")


all_passed = True

for _, crop_row in original_df.iterrows():

    crop = crop_row["CROPS"]

    crop_data = synthetic_df[
        synthetic_df["CROPS"] == crop
    ]

    crop_passed = True

    validations = [
        ("SOIL_PH", "SOIL_PH_LOW", "SOIL_PH_HIGH"),
        ("TEMPERATURE", "MIN_TEMP", "MAX_TEMP"),
        ("CROP_DURATION", "CROPDURATION_MIN", "CROPDURATION_MAX"),
        ("WATER_REQUIRED", "WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
        ("RELATIVE_HUMIDITY", "RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
        ("N", "N_MIN", "N_MAX"),
        ("P", "P_MIN", "P_MAX"),
        ("K", "K_MIN", "K_MAX")
    ]

    for feature, min_col, max_col in validations:

        generated_min = crop_data[feature].min()
        generated_max = crop_data[feature].max()

        if (
            generated_min < crop_row[min_col]
            or generated_max > crop_row[max_col]
        ):
            crop_passed = False
            all_passed = False
            break

    if crop_passed:
        print(f"✅ {crop} : All generated values are within the specified crop range.")
    else:
        print(f"❌ {crop} : Validation failed.")

print("\n" + "=" * 60)

if all_passed:
    print("✅ VALIDATION SUCCESSFUL")
    print("✅ All crops satisfy their original crop-specific ranges.")
else:
    print("❌ VALIDATION FAILED")
    print("❌ Some crops contain values outside the specified ranges.")

print("=" * 60)


# KS TEST IMPLEMENTATION

# import pandas as pd
# import numpy as np
# from scipy.stats import ks_2samp

# # ==========================
# # LOAD DATASETS
# # ==========================
# original_df = pd.read_excel("crop-dataset.xlsx")
# synthetic_df = pd.read_csv("synthetic_crop_datasets1_100000.csv")

# # ==========================
# # FEATURES TO TEST
# # ==========================
# features = [
#     ("SOIL_PH", "SOIL_PH_LOW", "SOIL_PH_HIGH"),
#     ("TEMPERATURE", "MIN_TEMP", "MAX_TEMP"),
#     ("CROP_DURATION", "CROPDURATION_MIN", "CROPDURATION_MAX"),
#     ("WATER_REQUIRED", "WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
#     ("RELATIVE_HUMIDITY", "RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
#     ("N", "N_MIN", "N_MAX"),
#     ("P", "P_MIN", "P_MAX"),
#     ("K", "K_MIN", "K_MAX")
# ]

# print("\nKS TEST RESULTS")
# print("=" * 70)

# all_passed = True

# for feature, low_col, high_col in features:

#     original_values = []

#     for _, row in original_df.iterrows():

#         low = row[low_col]
#         high = row[high_col]

#         # Skip missing values
#         if pd.isna(low) or pd.isna(high):
#             continue

#         low = float(low)
#         high = float(high)

#         # Safe handling
#         if low == high:
#             vals = np.full(100, low)

#         elif low > high:
#             print(
#                 f"⚠ Invalid range in {feature} "
#                 f"for crop {row['CROPS']} "
#                 f"({low} > {high})"
#             )
#             continue

#         else:
#             mode = (low + high) / 2

#             vals = np.random.triangular(
#                 low,
#                 mode,
#                 high,
#                 100
#             )

#         original_values.extend(vals)

#     # Synthetic values
#     synthetic_values = synthetic_df[feature].dropna()

#     # KS Test
#     statistic, p_value = ks_2samp(
#         original_values,
#         synthetic_values
#     )

#     result = "PASS" if p_value > 0.05 else "FAIL"

#     if result == "FAIL":
#         all_passed = False

#     print(
#         f"{feature:<20} "
#         f"KS={statistic:.4f}   "
#         f"P-value={p_value:.4f}   "
#         f"{result}"
#     )

# print("\n" + "=" * 70)

# if all_passed:
#     print("✅ KS TEST PASSED")
#     print("✅ Synthetic data distribution is similar to the original data.")
# else:
#     print("❌ KS TEST FAILED")
#     print("❌ Some features differ significantly from the original data.")

# print("=" * 70)

# import pandas as pd

# original_df = pd.read_excel("crop-dataset.xlsx")
# synthetic_df = pd.read_csv("synthetic_crop_datasets1_100000.csv")

# total_checks = 0
# valid_checks = 0

# validations = [
#     ("SOIL_PH", "SOIL_PH_LOW", "SOIL_PH_HIGH"),
#     ("TEMPERATURE", "MIN_TEMP", "MAX_TEMP"),
#     ("CROP_DURATION", "CROPDURATION_MIN", "CROPDURATION_MAX"),
#     ("WATER_REQUIRED", "WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
#     ("RELATIVE_HUMIDITY", "RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
#     ("N", "N_MIN", "N_MAX"),
#     ("P", "P_MIN", "P_MAX"),
#     ("K", "K_MIN", "K_MAX")
# ]

# for _, row in synthetic_df.iterrows():

#     crop = row["CROPS"]

#     crop_info = original_df[
#         original_df["CROPS"] == crop
#     ].iloc[0]

#     for feature, min_col, max_col in validations:

#         total_checks += 1

#         if (
#             crop_info[min_col]
#             <= row[feature]
#             <= crop_info[max_col]
#         ):
#             valid_checks += 1

# accuracy = (valid_checks / total_checks) * 100

# print(f"Total Checks : {total_checks}")
# print(f"Valid Checks : {valid_checks}")
# print(f"Validity Score : {accuracy:.2f}%")

# if accuracy == 100:
#     print("✅ All generated values satisfy crop-specific constraints.")
# else:
#     print("❌ Some generated values violate crop-specific constraints.")