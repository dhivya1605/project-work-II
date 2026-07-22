import random
import pandas as pd

from config import SAMPLES_PER_CROP, RANDOM_SEED
from samplers.gaussian_copula import gaussian_copula_sample
from utils.categorical_utils import choose_random

random.seed(RANDOM_SEED)


def generate_gaussian_copula_dataset(df):

    synthetic_data = []

    for _, row in df.iterrows():

        print(f"Generating data for {row['CROPS']}")

        # Generate correlated numerical features
        soil_ph, duration, temperature, water, humidity, N, P, K = gaussian_copula_sample(
            row["SOIL_PH_LOW"], row["SOIL_PH_HIGH"],
            row["CROPDURATION_MIN"], row["CROPDURATION_MAX"],
            row["MIN_TEMP"], row["MAX_TEMP"],
            row["WATERREQUIRED_MIN"], row["WATERREQUIRED_MAX"],
            row["RELATIVE_HUMIDITY_MIN"], row["RELATIVE_HUMIDITY_MAX"],
            row["N_MIN"], row["N_MAX"],
            row["P_MIN"], row["P_MAX"],
            row["K_MIN"], row["K_MAX"],
            SAMPLES_PER_CROP
        )

        # Generate synthetic rows
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

                "N": round(N[i], 2),

                "P": round(P[i], 2),

                "K": round(K[i], 2)

            })

    return pd.DataFrame(synthetic_data)