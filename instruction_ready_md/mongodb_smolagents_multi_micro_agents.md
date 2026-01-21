# Multi-Agent Order Management System with MongoDB

This guide implements a multi-agent system for managing product orders, inventory, and deliveries using the `smolagents` framework and MongoDB for data persistence.

## Prerequisites

Before you begin, ensure you have:
1. A **MongoDB Atlas** account with a deployed cluster. You can create a free account at [https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register).
2. An **API key** from [DeepSeek](https://www.deepseek.com/) (or another supported LLM provider).

### Security Setup for MongoDB Atlas
1. In the Atlas UI, configure **Network Access** to allow connections from your IP address.
2. Create a **Database User** with read/write permissions.
3. Obtain your **Connection String** from the Atlas UI (Click "Connect" > "Connect your application").
4. Replace the `<password>` placeholder in the connection string with your database user's password.
5. Store your connection string and API key securely using environment variables or a secrets manager. **Never commit them directly to version control.**

## Step 1: Install Dependencies

First, install the required Python libraries.

```bash
pip install smolagents pymongo litellm
```

## Step 2: Import Libraries and Configure Secrets

Import the necessary modules and configure your API keys and database URI. This example assumes you are storing secrets in environment variables.

```python
from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel, ManagedAgent, CodeAgent
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from typing import List, Dict
import os

# Initialize LLM model and database connection using environment variables
MODEL_ID = "deepseek/deepseek-chat"
MONGODB_URI = os.getenv('MONGO_URI')  # e.g., "mongodb+srv://<user>:<password>@cluster.mongodb.net/"
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
```

## Step 3: Establish the Database Connection

Create a connection to your MongoDB database.

```python
mongoclient = MongoClient(MONGODB_URI, appname="devrel.showcase.multi-smolagents")
db = mongoclient.warehouse  # Uses the 'warehouse' database
```

## Step 4: Define the Agent Tools

The system is built around three core agents, each with specialized tools. We'll define these tools using the `@tool` decorator.

### 4.1 Inventory Management Tools

These tools allow an agent to check and update product stock levels.

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

### 4.2 Order Management Tool

This tool creates a new order document in the database.

```python
@tool
def create_order(products: any, address: str) -> str:
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

### 4.3 Delivery Management Tool

This tool updates the delivery status of an existing order.

```python
@tool
def update_delivery_status(order_id: str, status: str) -> bool:
    """Update order delivery status.

    Args:
        order_id: Order identifier
        status: New delivery status ('pending', 'in_transit', 'delivered', 'cancelled')

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

## Step 5: Build the Multi-Agent Orchestrator

Now, we create the main `OrderManagementSystem` class. This class instantiates three specialized agents and a manager agent that coordinates the workflow between them.

```python
class OrderManagementSystem:
    """Multi-agent order management system"""
    def __init__(self, model_id: str = MODEL_ID):
        # Initialize the LLM model
        self.model = LiteLLMModel(model_id=model_id, api_key=DEEPSEEK_API_KEY)

        # Create the three specialized agents
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

        # Wrap the agents in ManagedAgent containers for the manager
        self.managed_agents = [
            ManagedAgent(self.inventory_agent, "inventory", "Manages product inventory"),
            ManagedAgent(self.order_agent, "orders", "Handles order creation"),
            ManagedAgent(self.delivery_agent, "delivery", "Manages delivery status")
        ]

        # Create the manager agent that orchestrates the workflow
        self.manager = CodeAgent(
            tools=[],
            system_prompt="""For each order:
            1. Create the order document
            2. Update the inventory
            3. Set delivery status to in_transit

            Use relevant agents:  {{managed_agents_descriptions}}  and you can use {{authorized_imports}}
            """,
            model=self.model,
            managed_agents=self.managed_agents,
            additional_authorized_imports=["time", "json"]
        )

    def process_order(self, orders: List[Dict]) -> str:
        """Process a set of orders.

        Args:
            orders: List of orders, each containing 'products' and 'address'

        Returns:
            str: Processing result summary
        """
        return self.manager.run(
            f"Process the following  {orders} as well as subtract the ordered items from inventory."
            f"to be delivered to relevant addresses"
        )
```

## Step 6: Populate the Database with Sample Data

Before testing, let's add some sample products to the database.

```python
def add_sample_products():
    # Clear existing products (optional, for a clean test)
    db.products.delete_many({})

    sample_products = [
        {"_id": "prod1", "name": "Laptop", "price": 999.99, "quantity": 10},
        {"_id": "prod2", "name": "Smartphone", "price": 599.99, "quantity": 15},
        {"_id": "prod3", "name": "Headphones", "price": 99.99, "quantity": 30}
    ]

    db.products.insert_many(sample_products)
    print("Sample products added successfully!")

# Execute the function to add products
add_sample_products()
```

## Step 7: Test the Complete System

Finally, we can initialize the system and process a test batch of orders.

```python
# 1. Initialize the multi-agent system
system = OrderManagementSystem()

# 2. Define test orders
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

# 3. Process the orders
result = system.process_order(orders=test_orders)

# 4. Print the result
print("Orders processing result:", result)
```

When you run this code, the manager agent will orchestrate the following workflow for each order:
1. The **Order Agent** creates an order document in MongoDB, generating a unique Order ID.
2. The **Inventory Agent** checks stock levels and decrements the ordered quantities.
3. The **Delivery Agent** updates the order status from `"pending"` to `"in_transit"`.

The final output will be a summary confirming the orders were created, inventory was updated, and delivery status was set.

## Conclusion

You have successfully built a multi-agent order management system. This architecture demonstrates how to decompose a complex business workflow (order processing) into specialized agents, coordinate them with a manager, and maintain persistent state in MongoDB. You can extend this system by adding more agents (e.g., for customer support or payment processing) or more sophisticated tools.