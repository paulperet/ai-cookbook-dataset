# Data Extraction and Transformation in ELT Workflows using GPT-4o as an OCR Alternative

Enterprise data is often locked in unstructured formats like PDFs, PPTs, and images, making it difficult to utilize for analysis and product development. Traditional Optical Character Recognition (OCR) struggles with complex layouts and multilingual content. GPT-4o's multimodal capabilities offer a powerful alternative, enabling adaptable extraction and intelligent transformation of data from diverse documents.

This guide walks through a complete ELT (Extract, Load, Transform) workflow using GPT-4o to process multilingual hotel invoices. You will:
1.  **Extract** data from PDF invoices into JSON using GPT-4o's vision capabilities.
2.  **Transform** the unstructured JSON into a clean, consistent schema.
3.  **Load** the structured data into a relational database (SQLite) for querying.

> **Note:** For production-scale processing, consider using the [Batch API](https://platform.openai.com/docs/guides/batch) to reduce costs.

## Prerequisites

Ensure you have the necessary Python libraries installed and your OpenAI API key set as an environment variable.

```bash
pip install openai pymupdf pillow
```

```python
import os
from openai import OpenAI
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import json
import sqlite3

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
```

## Part 1: Extract Data from PDFs

GPT-4o processes images, not PDFs directly. The first step is to convert each page of a PDF into a base64-encoded image.

### Step 1.1: Convert PDF Pages to Base64 Images

The following function handles multi-page PDFs, converting each page to an image and encoding it.

```python
def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_base64_images(pdf_path):
    """
    Converts each page of a PDF to a base64-encoded PNG image.
    Returns a list of base64 strings.
    """
    pdf_document = fitz.open(pdf_path)
    base64_images = []
    temp_image_paths = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # Save page as a temporary image file
        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path, format="PNG")
        temp_image_paths.append(temp_image_path)
        
        # Encode the image
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)

    # Clean up temporary files
    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)

    return base64_images
```

### Step 1.2: Define the Data Extraction Prompt

With the images ready, you can call GPT-4o to extract the invoice data. The system prompt instructs the model to act as an OCR tool, preserving the original language and structure, including tables and null values.

```python
def extract_invoice_data(base64_image):
    """
    Sends a base64-encoded invoice image to GPT-4o for data extraction.
    Returns the extracted data as a JSON string.
    """
    system_prompt = """
    You are an OCR-like data extraction tool that extracts hotel invoice data from PDFs.
   
    1. Please extract the data in this hotel invoice, grouping data according to theme/sub groups, and then output into JSON.
    2. Please keep the keys and values of the JSON in the original language. 
    3. The type of data you might encounter in the invoice includes but is not limited to: hotel information, guest information, invoice information, room charges, taxes, and total charges etc. 
    4. If the page contains no charge data, please output an empty JSON object and don't make up any data.
    5. If there are blank data fields in the invoice, please include them as "null" values in the JSON object.
    6. If there are tables in the invoice, capture all of the rows and columns in the JSON object. Even if a column is blank, include it as a key in the JSON object with a null value.
    7. If a row is blank denote missing fields with "null" values. 
    8. Don't interpolate or make up data.
    9. Please maintain the table structure of the charges, i.e. capture all of the rows and columns in the JSON object.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "extract the data in this hotel invoice and output into JSON "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        temperature=0.0, # Use zero temperature for deterministic, reproducible output
    )
    return response.choices[0].message.content
```

### Step 1.3: Process Multi-Page PDFs and Save Results

Invoices often span multiple pages. This function processes all pages of a PDF, aggregates the extracted JSON objects into a list, and saves the final result to a file.

```python
def extract_from_multiple_pages(base64_images, original_filename, output_directory):
    """
    Extracts data from all pages of an invoice and saves the combined result as a JSON file.
    """
    entire_invoice = []

    for base64_image in base64_images:
        invoice_json = extract_invoice_data(base64_image)
        invoice_data = json.loads(invoice_json)
        entire_invoice.append(invoice_data)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the output file path
    output_filename = os.path.join(output_directory, original_filename.replace('.pdf', '_extracted.json'))
    
    # Save the entire_invoice list as a JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(entire_invoice, f, ensure_ascii=False, indent=4)
    return output_filename
```

### Step 1.4: Run the Extraction Pipeline

Finally, create a main function to process all PDFs in a specified directory.

```python
def main_extract(read_path, write_path):
    """Processes all PDF files in a directory, extracting data to JSON."""
    for filename in os.listdir(read_path):
        file_path = os.path.join(read_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.pdf'):
            print(f"Processing: {filename}")
            base64_images = pdf_to_base64_images(file_path)
            extract_from_multiple_pages(base64_images, filename, write_path)

# Define your input and output paths
read_path = "./data/hotel_invoices/receipts_2019_de_hotel"
write_path = "./data/hotel_invoices/extracted_invoice_json"

# Run the extraction
main_extract(read_path, write_path)
```

**Result:** You now have a folder containing JSON files for each invoice. The data is unstructured but accurately reflects the source document, including German text and preserved null values. This raw JSON is suitable for storage in a data lake.

## Part 2: Transform Data According to a Schema

The extracted JSON files have varying structures. The next step is to transform this data into a consistent schema suitable for database ingestion. We'll also translate all text to English.

### Step 2.1: Define the Target Schema

Create a JSON file (`invoice_schema.json`) that defines the structure for the transformed data. This schema specifies field names, data types, and formats (like `YYYY-MM-DD` for dates).

```json
[
    {
        "hotel_information": {
            "name": "string",
            "address": {
                "street": "string",
                "city": "string",
                "country": "string",
                "postal_code": "string"
            },
            "contact": {
                "phone": "string",
                "fax": "string",
                "email": "string",
                "website": "string"
            }
        },
        "guest_information": {
            "company": "string",
            "address": "string",
            "guest_name": "string"
        },
        "invoice_information": {
            "invoice_number": "string",
            "reservation_number": "string",
            "date": "YYYY-MM-DD",
            "room_number": "string",
            "check_in_date": "YYYY-MM-DD",
            "check_out_date": "YYYY-MM-DD"
        },
        "charges": [
            {
                "date": "YYYY-MM-DD",
                "description": "string",
                "charge": "number",
                "credit": "number"
            }
        ],
        "totals_summary": {
            "currency": "string",
            "total_net": "number",
            "total_tax": "number",
            "total_gross": "number",
            "total_charge": "number",
            "total_credit": "number",
            "balance_due": "number"
        },
        "taxes": [
            {
                "tax_type": "string",
                "tax_rate": "string",
                "net_amount": "number",
                "tax_amount": "number",
                "gross_amount": "number"
            }
        ]
    }
]
```

### Step 2.2: Create the Transformation Function

This function uses GPT-4o to map the raw, unstructured JSON to the target schema. It instructs the model to translate data to English, apply correct formatting, and handle missing values appropriately.

```python
def transform_invoice_data(json_raw, json_schema):
    """
    Transforms raw invoice JSON to match a target schema using GPT-4o.
    Returns the transformed data as a Python dictionary.
    """
    system_prompt = f"""
    You are a data transformation tool that takes in JSON data and a reference JSON schema, and outputs JSON data according to the schema.
    Not all of the data in the input JSON will fit the schema, so you may need to omit some data or add null values to the output JSON.
    Translate all data into English if not already in English.
    Ensure values are formatted as specified in the schema (e.g. dates as YYYY-MM-DD).
    Here is the schema:
    {json_schema}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Transform the following raw JSON data according to the provided schema. Ensure all data is in English and formatted as specified by values in the schema. Here is the raw JSON: {json_raw}"}
                ]
            }
        ],
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)
```

### Step 2.3: Run the Transformation Pipeline

Now, process all extracted JSON files, transforming each one and saving the results.

```python
def main_transform(extracted_invoice_json_path, json_schema_path, save_path):
    """
    Loads a schema, transforms all JSON files in a directory, and saves the results.
    """
    # Load the JSON schema
    with open(json_schema_path, 'r', encoding='utf-8') as f:
        json_schema = json.load(f)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Process each JSON file in the extracted invoices directory
    for filename in os.listdir(extracted_invoice_json_path):
        if filename.endswith(".json"):
            file_path = os.path.join(extracted_invoice_json_path, filename)

            # Load the extracted JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_raw = json.load(f)

            # Transform the JSON data
            transformed_json = transform_invoice_data(json_raw, json_schema)

            # Save the transformed JSON
            transformed_filename = f"transformed_{filename}"
            transformed_file_path = os.path.join(save_path, transformed_filename)
            with open(transformed_file_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_json, f, ensure_ascii=False, indent=2)
            print(f"Transformed: {filename}")

# Define your paths
extracted_invoice_json_path = "./data/hotel_invoices/extracted_invoice_json"
json_schema_path = "./data/hotel_invoices/invoice_schema.json"
save_path = "./data/hotel_invoices/transformed_invoice_json"

# Run the transformation
main_transform(extracted_invoice_json_path, json_schema_path, save_path)
```

**Result:** You now have a second folder with `transformed_*.json` files. The data within is now structured, consistent, and in English, ready for loading into a database.

## Part 3: Load Transformed Data into a Database

The final step is to segment the schematized data into relational tables and ingest it into a database. We'll create four tables: `Hotels`, `Invoices`, `Charges`, and `Taxes`.

### Step 3.1: Create the Database Schema

Connect to a SQLite database (or your preferred RDBMS) and execute `CREATE TABLE` statements.

```python
def create_tables(db_path):
    """Creates the necessary tables in the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Hotels Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Hotels (
            hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            street TEXT,
            city TEXT,
            country TEXT,
            postal_code TEXT,
            phone TEXT,
            fax TEXT,
            email TEXT,
            website TEXT
        )
    ''')

    # Invoices Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Invoices (
            invoice_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hotel_id INTEGER,
            invoice_number TEXT UNIQUE,
            reservation_number TEXT,
            date DATE,
            room_number TEXT,
            check_in_date DATE,
            check_out_date DATE,
            company TEXT,
            guest_address TEXT,
            guest_name TEXT,
            currency TEXT,
            total_net REAL,
            total_tax REAL,
            total_gross REAL,
            total_charge REAL,
            total_credit REAL,
            balance_due REAL,
            FOREIGN KEY (hotel_id) REFERENCES Hotels(hotel_id)
        )
    ''')

    # Charges Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Charges (
            charge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id INTEGER,
            date DATE,
            description TEXT,
            charge REAL,
            credit REAL,
            FOREIGN KEY (invoice_id) REFERENCES Invoices(invoice_id)
        )
    ''')

    # Taxes Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Taxes (
            tax_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id INTEGER,
            tax_type TEXT,
            tax_rate TEXT,
            net_amount REAL,
            tax_amount REAL,
            gross_amount REAL,
            FOREIGN KEY (invoice_id) REFERENCES Invoices(invoice_id)
        )
    ''')

    conn.commit()
    conn.close()
```

### Step 3.2: Ingest Transformed JSON Data

This function reads the transformed JSON files, parses the nested structure, and inserts the data into the corresponding tables, handling relationships via foreign keys.

```python
def ingest_transformed_jsons(json_folder_path, db_path):
    """
    Reads transformed JSON files and inserts the data into the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for filename in os.listdir(json_folder_path):
        if filename.startswith('transformed_') and filename.endswith('.json'):
            file_path = os.path.join(json_folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Assuming each file contains a list with one invoice object
            for invoice in data:
                # 1. Insert Hotel Information
                hotel_info = invoice.get('hotel_information', {})
                address = hotel_info.get('address', {})
                contact = hotel_info.get('contact', {})
                
                cursor.execute('''
                    INSERT OR IGNORE INTO Hotels 
                    (name, street, city, country, postal_code, phone, fax, email, website)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hotel_info.get('name'),
                    address.get('street'),
                    address.get('city'),
                    address.get('country'),
                    address.get('postal_code'),
                    contact.get('phone'),
                    contact.get('fax'),
                    contact.get('email'),
                    contact.get('website')
                ))
                hotel_id = cursor.lastrowid

                # 2. Insert Invoice Information
                invoice_info = invoice.get('invoice_information', {})
                guest_info = invoice.get('guest_information', {})
                totals = invoice.get('totals_summary', {})
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Invoices 
                    (hotel_id, invoice_number, reservation_number, date, room_number, 
                     check_in_date, check_out_date, company, guest_address, guest_name,
                     currency, total_net, total_tax, total_gross, total_charge, total_credit, balance_due)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hotel_id,
                    invoice_info.get('invoice_number'),
                    invoice_info.get('reservation_number'),
                    invoice_info.get('date'),
                    invoice_info.get('room_number'),
                    invoice_info.get('check_in_date'),
                    invoice_info.get('check_out_date'),
                    guest_info.get('company'),
                    guest_info.get('address'),
                    guest_info.get('guest_name'),
                    totals.get('currency'),
                    totals.get('total_net'),
                    totals.get('total_tax'),
                    totals.get('total_gross'),
                    totals.get('total_charge'),
                    totals.get('total_credit'),
                    totals.get('balance_due