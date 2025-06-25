import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from tools.productPrice_tool import get_laptop_price


print(get_laptop_price.invoke("alpha"))
print(get_laptop_price.invoke("testing"))