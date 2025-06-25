from langchain_core.tools import tool

import pandas as pd

#Load the laptop product orders CSV into a Pandas dataframe.
product_orders_df = pd.read_csv("data/Laptop Orders.csv")

@tool
def get_order_details(order_id:str) -> str :
    """
    This function is used by the Orders Agent to retrieve order details
    Check the input order number against the available orders ids in data storage and provide 100% correct information.
    If a match is found, return the details in customer satisfactory way with necessary details (
    Order ID,
    Product Name,
    Quantity Ordered,
    Delivery Date
    )
    Check user input carefully, either search for the number in the order ID or the full order ID.
    If no match is found, return invalid order ID -1.
    """
    #Filter Dataframe for order ID
    match_order_df = product_orders_df[
                        product_orders_df["Order ID"] == order_id ]

    #Check if a record was found, if not return -1
    if len(match_order_df) == 0 :
        return -1
    else:
        return match_order_df.iloc[0].to_dict()