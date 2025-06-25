import pandas as pd
from langchain_core.tools import tool

#Load the laptop product pricing CSV into a Pandas dataframe.
product_pricing_df = pd.read_csv("data/Laptop pricing.csv")
# print(product_pricing_df)

@tool
def get_laptop_price(laptop_name:str) -> int :
    """
    This function returns the price of a laptop, given its name as input.
    It performs a substring match between the input name and the laptop name.
    Consider previous agent responses to be the source of truth for the laptop names.
    If a match is found, it returns the price of the laptop.
    If there is NO match found, return to ask user to retry with a different name.
    """

    #Filter Dataframe for matching names
    match_records_df = product_pricing_df[
                        product_pricing_df["Name"].str.contains(
                                                "^" + laptop_name, case=False)
                        ]
    #Check if a record was found, if not return -1
    if len(match_records_df) == 0 :
        return -1
    else:
        return match_records_df["Price"].iloc[0]

