import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


"""
This class `Data_cleaning` for preprocessing real estate data. 
The class provides methods for reading a dataset, performing label encoding, and saving the processed data.

The script performs the following main tasks:
1. **Reading Data**:
   - Reads a CSV file containing real estate data.
   - Removes duplicate rows to ensure data quality.

2. **Encoding Features**:
   - Encodes categorical columns such as 'Locality', 'Type', 'SubType', 'Muniplicity', and 'Region' into numeric formats for machine learning.
   - Uses `LabelEncoder` for the 'Locality' column.
   - Maps specific values of the 'Type' column into binary encoding.
   - Uses `pandas.factorize` for encoding 'SubType', 'Muniplicity', and 'Region'.
   - Drops original categorical columns after encoding.

3. **Dropping Features**:
   - Removes irrelevant or unnecessary features like 'Is_Furnished', 'Terrace_Area', 'Garden_Area', etc., to simplify the dataset.

4. **Saving Processed Data**:
   - Saves the cleaned and encoded dataset as a CSV file.

Classes:
--------
Data_cleaning:
    A class for cleaning and encoding real estate data.

    Methods:
    --------
    __init__(link: str):
        Initializes the class with the file path to the dataset.

    read_data() -> pd.DataFrame:
        Reads the dataset from the specified file path and removes duplicate rows.

    encoding() -> pd.DataFrame:
        Performs label encoding and feature removal on the dataset. 
        Encodes categorical columns into numeric format and drops unwanted columns.

    save1() -> pd.DataFrame:
        Saves the processed dataset to a CSV file and returns the encoded DataFrame.

Usage:
------
1. Create an instance of the `Data_cleaning` class by providing the file path to the dataset.
2. Call the `save1()` method to preprocess the data and save it to a new CSV file.

Example:
--------
link = "/path/to/your/dataset.csv"
Program = Data_cleaning(link)
Encoded = Program.save1()
"""


class Data_cleaning:

    def __init__(self, link) -> None:
        self.link = link
        pass

    def read_data(self):
        link = self.link
        Data = pd.read_csv(link, index_col=0)
        Data.drop_duplicates(inplace=True)
        self.Data = Data
        return Data

    def encoding(self):

        df1 = self.read_data()
        # Initialize the label encoder(each locality to integer)
        label_encoder = LabelEncoder()

        # Encode the 'Locality' column
        df1["Locality_encoded"] = label_encoder.fit_transform(df1["Locality"])

        condition_mapping = {
            "Good": 1,
            "Not Known": 2,
            "As new": 3,
            "To renovate": 4,
            "To be done up": 5,
            "Just renovated": 6,
            "To restore": 7,
        }
        # Alerady done so

        # Binary encoding for Type
        df1["Type_encoded"] = df1["Type"].map({"Apartment": 0, "House": 1})

        # Label encoding 'SubType' using pandas' factorize method
        df1["SubType_encoded"] = pd.factorize(df1["SubType"])[0]
        # Label encoding 'Muniplcitiy' using pandas' factorize method
        df1["Prov_encoded"] = pd.factorize(df1["Muniplicity"])[0]

        # Label encoding 'Region' using pandas' factorize method
        df1["Region_encoded"] = pd.factorize(df1["Region"])[0]
        df1.drop(
            ["Locality", "Type", "SubType", "Muniplicity", "Region"],
            axis=1,
            inplace=True,
        )
        # List of features to discard
        features_to_discard = [
            "Is_Furnished",
            "Terrace_Area",
            "Garden_Area",
            "X",
            "Y",
            "Land_Surface",
            "Surface_total",
            "Is_Open_Fire",
            "Swim_pool",
        ]

        # Drop the features from the DataFrame
        df1 = df1.drop(columns=features_to_discard)
        self.Encoded_Data = df1
        return df1

    def save1(self):
        self.Encoded_Data = self.encoding()
        self.Encoded_Data.to_csv("Data_Engineering_pre.csv")
        return self.Encoded_Data


# Feature Engineering


class Feature_Engineering:
    def __init__(self, DF) -> None:
        self.DF = DF
        pass

    ## Is the property in a Locality on the coast?
    def is_locality_on_Coast(self):
        DF = self.DF
        # reading dataset that contains name of cities that are on coast
        Data1 = pd.read_csv(
            "Final_cleaned_Data.csv"
        )
        coastal_municipalities = [
            "De Panne",
            "Koksijde",
            "Nieuwpoort",
            "Middelkerke",
            "Oostende",
            "Bredene",
            "De Haan",
            "Blankenberge",
            "Zeebrugge",
            "Knokke-Heist",
        ]

        # Add a new column 'Is_On_Coast' based on the municipality
        Data1["Is_On_Coast"] = Data1["Locality"].apply(
            lambda x: 1 if x in coastal_municipalities else 0
        )
        # Merge the 'Is_On_Coast' feature from Data1 into DF based on 'Id'
        DF = DF.merge(Data1[["Id", "Is_On_Coast"]], on="Id", how="left")
        self.DF = DF
        return self.DF

    ## Adding GDP for each province
    def GDP(self):

        DF = self.is_locality_on_Coast()

        df = pd.DataFrame(
            {
                "Province": [
                    "Antwerpen",
                    "Brussel",
                    "Oost-Vlaanderen",
                    "West-Vlaanderen",
                    "Vlaams-Brabant",
                    "Henegouwen",
                    "Luik",
                    "Limburg",
                    "Waals-Brabant",
                    "Namen",
                    "Luxemburg",
                ],
                "GDP": [
                    98_189,
                    90_459,
                    62_123,
                    52_323,
                    51_731,
                    36_940,
                    34_715,
                    31_766,
                    21_155,
                    14_697,
                    7_887,
                ],
            }
        )
        # Convert GDP values to integers
        df["GDP"] = df["GDP"].astype(int)

        Data1 = pd.read_csv(
            "Final_cleaned_Data.csv"
        )
        Data1 = Data1.rename(columns={"Muniplicity": "Province"})
        Data1 = pd.merge(Data1, df, on="Province", how="left")

        # Merge the 'Is_On_Coast' feature from Data1 into DF based on 'Id'
        DF = DF.merge(Data1[["Id", "GDP"]], on="Id", how="left")
        self.DF = DF

        return self.DF

    ## Adding Avg rent Price and avg sqd m price (not from internal data) per Prov
    def Avg(self):
        DF = self.GDP()
        # add the avg rent per province
        data = {
            "Province": [
                "Brussel",
                "Vlaams-Brabant",
                "Waals-Brabant",
                "Antwerpen",
                "Oost-Vlaanderen",
                "West-Vlaanderen",
                "Henegouwen",
                "Luik",
                "Luxemburg",
                "Namen",
            ],
            "Avg_rent": [1205, 1013, 1013, 1000, 950, 950, 759, 759, 759, 759],
        }
        df2 = pd.DataFrame(data)
        Data1 = pd.read_csv(
            "Final_cleaned_Data.csv"
        )
        Data1 = Data1.rename(columns={"Muniplicity": "Province"})
        Data1 = pd.merge(Data1, df2, on="Province", how="left")

        # add the pricer per sqd meter
        provinces = [
            "Antwerpen",
            "Brussel",
            "Oost-Vlaanderen",
            "Vlaams-Brabant",
            "Henegouwen",
            "Luik",
            "Limburg",
            "Luxemburg",
            "Namen",
            "Waals-Brabant",
            "West-Vlaanderen",
        ]
        average_prices = [
            2577,
            3323,
            2546,
            2841,
            1618,
            1949,
            2193,
            1985,
            2084,
            2729,
            2888,
        ]  # Create the DataFrame
        df = pd.DataFrame({"Province": provinces, "Avg price": average_prices})
        df2 = df
        Data1 = pd.merge(Data1, df2, on="Province", how="left")
        # Merge the 'Is_On_Coast' feature from Data1 into DF based on 'Id'
        DF = DF.merge(Data1[["Id", "Avg_rent", "Avg price"]], on="Id", how="left")

        self.DF = DF

        return self.DF

    # Adding bedroms per area
    def bedrooms_per_area(self):
        DF = self.Avg()

        DF["Bedrooms_per_area"] = DF.Bedrooms / DF.Living_Area
        self.DF = DF
        return self.DF

    # Save
    def Save(self):
        DF = self.bedrooms_per_area()
        DF.to_csv("ED.csv")
        return


link = (
    "Final_cleaned_Data.csv"
)
Program = Data_cleaning(link)
Encoded = Program.save1()

DE = Feature_Engineering(Encoded)
DE.Save()
