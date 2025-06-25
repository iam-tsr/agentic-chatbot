import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from tools.orderDetails_tool import get_order_details


print(get_order_details.invoke("ORD-6948"))
print(get_order_details.invoke("ORD-9999"))