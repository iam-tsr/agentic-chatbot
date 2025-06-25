import sys
sys.path.append('/mnt/Linux/Projects/Ecommerce-Agentic-Chatbot')

from tools.quantityUpdate_tool import update_quantity


print(update_quantity.invoke({"order_id": "ORD-9999", "new_quantity": 2}))
print(update_quantity.invoke({"order_id": "ORD-6948", "new_quantity": 1}))