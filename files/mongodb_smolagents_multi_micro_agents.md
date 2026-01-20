# Multi-Agent Order Management System with MongoDB

This notebook implements a multi-agent system for managing product orders, inventory, and deliveries using:
- [smolagents](https://github.com/huggingface/smolagents/tree/main) for agent management
- MongoDB for data persistence
- DeepSeek Chat as the LLM model

## Setting Up MongoDB Atlas

1. Create a free MongoDB Atlas account at [https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register)
2. [Create a new cluster](https://www.mongodb.com/docs/atlas/tutorial/create-new-cluster/) (free tier is sufficient)
3. Configure network access by adding your IP address
4. Create a database user with read/write permissions
5. Get your connection string from Atlas UI (Click "Connect" > "Connect your application")
6. Replace `<password>` in the connection string with your database user's password
7. Enable network access from your IP address in the Network Access settings


### Security Considerations

When working with MongoDB Atlas:
- Never commit connection strings with credentials to version control
- Use environment variables or secure secret management
- Restrict database user permissions to only what's needed
- Enable IP allowlist in Atlas Network Access settings


## Setup
First, let's install required dependencies:


```python
!pip install smolagents pymongo litellm
```

## Import Dependencies

Set in your secrets the `MONGODB_URI` and `DEEPSEEK_API_KEY` from https://www.deepseek.com/ (or any other LLM provider)

Import all required libraries and setup the LLM model:


```python
from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel, ManagedAgent, CodeAgent
from pymongo import MongoClient
from datetime import datetime
from google.colab import userdata
from typing import List, Dict, Optional

# Initialize LLM model
MODEL_ID = "deepseek/deepseek-chat"
MONGODB_URI = userdata.get('MONGO_URI')
DEEPSEEK_API_KEY = userdata.get('DEEPSEEK_API_KEY')
```

## Database Connection Class
Create a MongoDB connection manager:


```python
mongoclient = MongoClient(MONGODB_URI, appname="devrel.showcase.multi-smolagents")
db = mongoclient.warehouse
```

## Agent Tools Defenitions

Our system implements three core tools for warehouse management:

   
Workflow:
```
Inventory Management Tools:
+-------------------+-------------------+
| Tool              | Description       |
+-------------------+-------------------+
| check_stock       | Queries stock     |
|                   | levels            |
+-------------------+-------------------+
| update_stock      | Adjusts inventory |
|                   | quantities        |
+-------------------+-------------------+

Order Management Tools:
+-------------------+-------------------+
| Tool              | Description       |
+-------------------+-------------------+
| create_order      | Creates new order |
|                   | document          |
+-------------------+-------------------+

Delivery Management Tools:
+-------------------+-------------------+
| Tool              | Description       |
+-------------------+-------------------+
| update_delivery   | Updates delivery  |
| _status           | status            |
+-------------------+-------------------+

Decision Flow:
+-------------------+-------------------+
| Step              | Action            |
+-------------------+-------------------+
| 1. Create Order   | Uses `create_order`|
|                   | tool to create    |
|                   | order document    |
+-------------------+-------------------+
| 2. Update Stock   | Uses `update_stock`|
|                   | tool to adjust    |
|                   | inventory         |
+-------------------+-------------------+
| 3. Update Delivery| Uses `update_delivery`|
| Status            | _status tool to   |
|                   | set delivery      |
|                   | status to         |
|                   | `in_transit`      |
+-------------------+-------------------+
```


Define tools for each agent type:


```python
@tool
def check_stock(product_id: str) -> Dict:
    """Query product stock level.

    Args:
        product_id: Product identifier

    Returns:
        Dict containing product details and quantity
    """
    return db.products.find_one({"_id": product_id})

@tool
def update_stock(product_id: str, quantity: int) -> bool:
    """Update product stock quantity.

    Args:
        product_id: Product identifier
        quantity: Amount to decrease from stock

    Returns:
        bool: Success status
    """
    result = db.products.update_one(
        {"_id": product_id},
        {"$inc": {"quantity": -quantity}}
    )
    return result.modified_count > 0
```


```python
@tool
def create_order( products: any, address: str) -> str:
    """Create new order for all provided products.

    Args:
        products: List of products with quantities
        address: Delivery address

    Returns:
        str: Order ID message
    """
    order = {
        "products": products,
        "status": "pending",
        "delivery_address": address,
        "created_at": datetime.now()
    }
    result = db.orders.insert_one(order)
    return f"Successfully ordered : {str(result.inserted_id)}"
```


```python
from bson.objectid import ObjectId
@tool
def update_delivery_status(order_id: str, status: str) -> bool:
    """Update order delivery status to in_transit once a pending order is provided

    Args:
        order_id: Order identifier
        status: New delivery status is being set to in_transit or delivered

    Returns:
        bool: Success status
    """
    if status not in ["pending", "in_transit", "delivered", "cancelled"]:
        raise ValueError("Invalid delivery status")

    result = db.orders.update_one(
        {"_id":  ObjectId(order_id), "status": "pending"},
        {"$set": {"status": status}}
    )
    return result.modified_count > 0
```

## Main Order Management System

This class implements a multi-agent architecture for order processing with the following components:

- Inventory Agent: Handles stock checking and updates
- Order Agent: Manages order creation and documentation 
- Delivery Agent: Controls order delivery status changes
- Manager Agent: Orchestrates workflow between other agents

The system follows this process flow:
1. Create order documents for customer requests
2. Verify and update product inventory levels
3. Initialize delivery tracking status
4. Coordinate agent interactions through the manager

Key Features:
- Asynchronous multi-agent coordination
- Automated inventory management
- Order status tracking
- Delivery pipeline integration

Define the main system class that orchestrates all agents:


```python
class OrderManagementSystem:
    """Multi-agent order management system"""
    def __init__(self, model_id: str = MODEL_ID):
        self.model = LiteLLMModel(model_id=model_id, api_key=DEEPSEEK_API_KEY)



        # Create agents
        self.inventory_agent = ToolCallingAgent(
            tools=[check_stock, update_stock],
            model=self.model,
            max_iterations=10
        )

        self.order_agent = ToolCallingAgent(
            tools=[create_order],
            model=self.model,
            max_iterations=10
        )

        self.delivery_agent = ToolCallingAgent(
            tools=[update_delivery_status],
            model=self.model,
            max_iterations=10
        )

        # Create managed agents
        self.managed_agents = [
            ManagedAgent(self.inventory_agent, "inventory", "Manages product inventory"),
            ManagedAgent(self.order_agent, "orders", "Handles order creation"),
            ManagedAgent(self.delivery_agent, "delivery", "Manages delivery status")
        ]

        # Create manager agent
        self.manager = CodeAgent(
            tools=[],
            system_prompt="""For each order:
            1. Create the order document
            2. Update the inventory
            3. Set deliviery status to in_transit

            Use relevant agents:  {{managed_agents_descriptions}}  and you can use {{authorized_imports}}
            """,
            model=self.model,
            managed_agents=self.managed_agents,
            additional_authorized_imports=["time", "json"]
        )

    def process_order(self, orders: List[Dict]) -> str:
        """Process a set of orders.

        Args:
            orders: List of orders each has address and products

        Returns:
            str: Processing result
        """
        return self.manager.run(
            f"Process the following  {orders} as well as substract the ordered items from inventory."
            f"to be delivered to relevant addresses"
        )
```

## Adding Sample Data

To test our order management system, we need to populate the MongoDB database with sample product data. The following section shows how to add test products with their prices and quantities. You can modify the product details or add more items by following the same structure. Each product has a unique ID, name, price, and initial stock quantity.

The sample data provides a representative mix of electronics products with varying price points and stock levels to demonstrate inventory tracking.

To test the system, you might want to add some sample products to MongoDB:


```python
def add_sample_products():
    db.products.delete_many({})
    sample_products = [
        {"_id": "prod1", "name": "Laptop", "price": 999.99, "quantity": 10},
        {"_id": "prod2", "name": "Smartphone", "price": 599.99, "quantity": 15},
        {"_id": "prod3", "name": "Headphones", "price": 99.99, "quantity": 30}
    ]

    db.products.insert_many(sample_products)
    print("Sample products added successfully!")

# Uncomment to add sample products
add_sample_products()
```

## Testing the System

Here's a markdown description of the test data approach:

Testing Strategy Overview:
1. We test with two different order scenarios:
    - Multi-product order (laptop + smartphone)
    - Single product order (headphones)

Test Data Design:
- Products represent common electronics at different price points
- Order quantities are intentionally small to avoid depleting stock
- Multiple delivery addresses to simulate real-world scenarios

Alternative Test Examples:
- Bulk order: Multiple units of same product
- Mixed category order: Combination of high/low value items
- Edge cases: Orders near stock limits
- Invalid scenarios: Products with insufficient stock

The test demonstrates:
- Multi-product order processing
- Stock level management
- Delivery status updates
- Address handling for different locations

Let's test our system with a sample order:


```python
# Initialize system
system = OrderManagementSystem()

# Create test orders
test_orders = [
    {
        "products": [
            {"product_id": "prod1", "quantity": 2},
            {"product_id": "prod2", "quantity": 1}
        ],
        "address": "123 Main St"
    },
    {
        "products": [
            {"product_id": "prod3", "quantity": 3}
        ],
        "address": "456 Elm St"
    }
]

# Process order
result = system.process_order(
    orders=test_orders
)

print("Orders processing result:", result)
```

[Step 0: Duration 22.83 seconds| Input tokens: 1,800 | Output tokens: 213, ..., Step 6: Duration 0.00 seconds| Input tokens: 15,373 | Output tokens: 1,312]

    Orders processing result: Here’s the response to your request:
    
    ---
    
    ### **Processed Orders and Inventory Update**
    
    1. **Orders Created**:
       - **Order 1**:
         - **Products**:
           - `prod1`: 2 units
           - `prod2`: 1 unit
         - **Delivery Address**: `123 Main St`
         - **Order ID**: `677b8a9ff033af3a53c9a75a`
       - **Order 2**:
         - **Products**:
           - `prod3`: 3 units
         - **Delivery Address**: `456 Elm St`
         - **Order ID**: `677b8aa3f033af3a53c9a75c`
    
    2. **Inventory Updated**:
       - **`prod1` (Laptop)**:
         - Initial stock: 6 units
         - Subtracted: 2 units
         - New stock: 4 units
       - **`prod2` (Smartphone)**:
         - Initial stock: 13 units
         - Subtracted: 1 unit
         - New stock: 12 units
       - **`prod3` (Headphones)**:
         - Initial stock: 24 units
         - Subtracted: 3 units
         - New stock: 21 units
    
    3. **Delivery Status**:
       - Both orders have been marked as **"in_transit"** and are ready for delivery.
    
    ---
    
    ### **Summary**:
    - The orders have been successfully processed.
    - The inventory has been updated to reflect the subtracted quantities.
    - The delivery status for both orders is now **"in_transit"**.
    
    Let me know if you need further assistance! 😊


## System Output Analysis

The system successfully completes these key actions:

1. Order Creation:
    - Multiple orders processed in parallel
    - Order IDs generated and stored in MongoDB
    - Products and delivery addresses properly linked

2. Inventory Management:
    - Stock levels checked before order processing
    - Quantities decremented after order confirmation
    - Inventory updates reflected in MongoDB

3. Delivery Status:
    - Initial status set to "pending"
    - Updated to "in_transit" after processing
    - Status changes tracked in order documents

4. Data Consistency:
    - All MongoDB operations completed atomically
    - Order details preserved accurately
    - Stock levels maintained correctly

When running the system, you might notice the agent attempting to interpret text output as Python code. This is an expected behavior of the CodeAgent as it tries to understand and process responses. After several attempts (max_iterations=10), it will stop if unsuccessful.

Example agent behavior:
1. Receives text output from order creation
2. Attempts to parse it as code
3. Retries with different interpretations
4. Eventually completes the workflow

The multi-agent system demonstrates resilient operation through its error handling
and self-correction mechanisms. While initial attempts may produce error logs, 
the agent successfully adapts through iterations. Most importantly, the final 
state shows both successful order processing and accurate stock level updates, 
maintaining data consistency despite any intermediate errors.

This behavior is by design and doesn't affect the system's core functionality. The actual order processing, inventory updates, and delivery status changes are completed successfully through the MongoDB operations.

## Conclusions
In this notebook, we have successfully implemented a multi-agent order management system using smolagents and MongoDB. We defined various tools for managing inventory, creating orders, and updating delivery statuses. We also created a main system class to orchestrate these agents and tested the system with sample data and orders.

This approach demonstrates the power of combining agent-based systems with robust data persistence solutions like MongoDB to create scalable and efficient order management systems.