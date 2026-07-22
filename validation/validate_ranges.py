import pandas as pd

# Load datasets
original = pd.read_excel("data/change_dataset.xlsx")
synthetic = pd.read_csv("output/lhs_dataset.csv")

errors = []

for _, crop in original.iterrows():

    crop_name = crop["CROPS"]

    generated = synthetic[synthetic["CROPS"] == crop_name]

    checks = [

        ("SOIL_PH", "SOIL_PH_LOW", "SOIL_PH_HIGH"),
        ("CROPDURATION", "CROPDURATION_MIN", "CROPDURATION_MAX"),
        ("TEMPERATURE", "MIN_TEMP", "MAX_TEMP"),
        ("WATERREQUIRED", "WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
        ("RELATIVE_HUMIDITY", "RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
        ("N", "N_MIN", "N_MAX"),
        ("P", "P_MIN", "P_MAX"),
        ("K", "K_MIN", "K_MAX")

    ]

    for feature, low, high in checks:

        invalid = generated[
            (generated[feature] < crop[low]) |
            (generated[feature] > crop[high])
        ]

        if len(invalid) > 0:
            errors.append(
                f"{crop_name} -> {feature} : {len(invalid)} invalid values"
            )

if len(errors) == 0:

    print("\nAll generated values are within the specified ranges.")

else:

    print("\nValidation Errors:\n")

    for e in errors:
        print(e)