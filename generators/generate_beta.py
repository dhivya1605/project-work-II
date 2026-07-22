import random
import pandas as pd

from config import SAMPLES_PER_CROP, RANDOM_SEED
from samplers.beta_sampler import beta_sample
from samplers.lhs import lhs_sample
from utils.categorical_utils import choose_random

random.seed(RANDOM_SEED)

def generate_beta_dataset(df):

    synthetic_data = []

    for _, row in df.iterrows():

        print(f"Generating data for {row['CROPS']}")

        # Generate numerical values
        soil_ph = beta_sample(
            row["SOIL_PH_LOW"],
            row["SOIL_PH_HIGH"],
            SAMPLES_PER_CROP
        )

        duration = beta_sample(
            row["CROPDURATION_MIN"],
            row["CROPDURATION_MAX"],
            SAMPLES_PER_CROP
        )

        temperature = beta_sample(
            row["MIN_TEMP"],
            row["MAX_TEMP"],
            SAMPLES_PER_CROP
        )

        water = beta_sample(
            row["WATERREQUIRED_MIN"],
            row["WATERREQUIRED_MAX"],
            SAMPLES_PER_CROP
        )

        humidity = beta_sample(
            row["RELATIVE_HUMIDITY_MIN"],
            row["RELATIVE_HUMIDITY_MAX"],
            SAMPLES_PER_CROP
        )

        nitrogen = beta_sample(
            row["N_MIN"],
            row["N_MAX"],
            SAMPLES_PER_CROP
        )

        phosphorus = beta_sample(
            row["P_MIN"],
            row["P_MAX"],
            SAMPLES_PER_CROP
        )

        potassium = beta_sample(
            row["K_MIN"],
            row["K_MAX"],
            SAMPLES_PER_CROP
        )

        # Create 600 rows
        for i in range(SAMPLES_PER_CROP):

            synthetic_data.append({

                "CROPS": row["CROPS"],

                "TYPE_OF_CROP": row["TYPE_OF_CROP"],

                "SOIL": choose_random(row["SOIL"]),

                "SEASON": row["SEASON"],

                "SOWN": choose_random(row["SOWN"]),

                "HARVESTED": choose_random(row["HARVESTED"]),

                "WATER_SOURCE": choose_random(row["WATER_SOURCE"]),

                "SOIL_PH": round(soil_ph[i], 2),

                "CROPDURATION": round(duration[i], 2),

                "TEMPERATURE": round(temperature[i], 2),

                "WATERREQUIRED": round(water[i], 2),

                "RELATIVE_HUMIDITY": round(humidity[i], 2),

                "N": round(nitrogen[i], 2),

                "P": round(phosphorus[i], 2),

                "K": round(potassium[i], 2)

            })

    return pd.DataFrame(synthetic_data)