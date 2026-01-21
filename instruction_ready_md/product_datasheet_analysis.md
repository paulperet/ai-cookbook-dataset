# Automated Product Datasheet Analysis with Mistral AI Document AI

## Overview

This guide demonstrates how to automate the analysis of technical product datasheets using **Mistral AI's Document AI**. We'll focus on a practical procurement use case: validating lithium-ion battery specifications against your design requirements.

### Use Case: Battery Procurement & Vendor Validation

When sourcing components like lithium-ion batteries, vendors provide PDF datasheets containing hundreds of specifications. Manually comparing each parameter against your design requirements is time-consuming and prone to error.

This tutorial automates the entire process:
1. **Extract structured data** from PDF datasheets using Mistral OCR with Document Annotations
2. **Compare specifications** against your design requirements
3. **Generate detailed technical reports** with comprehensive analysis for each parameter

### Prerequisites

- Python 3.8+
- A Mistral AI API key (set as `MISTRAL_API_KEY` environment variable)
- Required input files:
  - `lithium_iron_cell_datasheet.pdf` - Vendor-provided battery specification document
  - `battery_requirements.txt` - Your project's design criteria with acceptable ranges

### Technology Stack

- **Mistral OCR** (`mistral-ocr-latest`) - PDF parsing with document annotations
- **Mistral Medium** (`mistral-medium-latest`) - Technical report generation
- **Pydantic** - Data validation and schema definition

---

## Step 1: Setup and Installation

First, install the required Python package and import necessary modules.

```bash
pip install mistralai
```

```python
import base64
import os
import json
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from typing import List, Optional

print("✓ All imports successful")
```

Initialize the Mistral client with your API key:

```python
# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")

client = Mistral(api_key=api_key)
print("✓ Mistral client initialized")
```

## Step 2: Define Data Schemas

We'll create comprehensive Pydantic schemas to structure the battery specifications we want to extract. This ensures consistent data validation and makes the extracted information easy to work with.

```python
# Schema for lithium battery specifications
class CapacitySpec(BaseModel):
    """Battery capacity specifications."""
    normal_capacity: float = Field(..., description="Normal capacity in mAh")
    minimum_capacity: float = Field(..., description="Minimum capacity in mAh")
    unit: str = Field("mAh", description="Capacity unit")

class VoltageSpec(BaseModel):
    """Voltage specifications."""
    nominal_voltage: float = Field(..., description="Nominal voltage in Volts")
    charge_voltage: float = Field(..., description="Charge voltage in Volts")
    discharge_cutoff_voltage: float = Field(..., description="Discharge cut-off voltage in Volts")

class CurrentSpec(BaseModel):
    """Current specifications."""
    standard_charge_current: float = Field(..., description="Standard charge current in mA")
    maximum_charge_current: float = Field(..., description="Maximum charge current in mA")
    standard_discharge_current: float = Field(..., description="Standard discharge current in mA")
    maximum_discharge_current: float = Field(..., description="Maximum discharge current in mA")
    max_instantaneous_discharge: float = Field(..., description="Maximum instantaneous discharge current in mA")

class TemperatureRange(BaseModel):
    """Temperature range specifications."""
    min_temp: float = Field(..., description="Minimum temperature in °C")
    max_temp: float = Field(..., description="Maximum temperature in °C")
    condition: str = Field(..., description="Condition (e.g., 'Charge', 'Discharge', 'Storage')")

class DimensionsSpec(BaseModel):
    """Physical dimensions specifications."""
    height: float = Field(..., description="Cell height in mm")
    diameter: float = Field(..., description="Diameter in mm")
    weight: float = Field(..., description="Weight in grams")

class PerformanceSpec(BaseModel):
    """Performance test results."""
    test_name: str = Field(..., description="Name of the performance test")
    criteria: str = Field(..., description="Performance criteria/requirement")
    result: str = Field(..., description="Test result status")

class LithiumBatterySpec(BaseModel):
    """Complete specification for a lithium battery cell."""
    model_name: str = Field(..., description="Model name or number")
    product_type: str = Field(..., description="Product type (e.g., 'Lithium-ion Cell Battery')")
    capacity: CapacitySpec = Field(..., description="Capacity specifications")
    voltage: VoltageSpec = Field(..., description="Voltage specifications")
    current: CurrentSpec = Field(..., description="Current specifications")
    internal_impedance: str = Field(..., description="Internal impedance specification")
    dimensions: DimensionsSpec = Field(..., description="Physical dimensions")
    cycle_life: int = Field(..., description="Cycle life (number of cycles)")
    operating_temperatures: List[TemperatureRange] = Field(..., description="Operating temperature ranges")
    storage_temperatures: List[TemperatureRange] = Field(..., description="Storage temperature ranges")
    performance_tests: List[PerformanceSpec] = Field(default=[], description="Performance test results")
    certifications: List[str] = Field(default=[], description="Certifications and standards")
    manufacturer: str = Field(..., description="Manufacturer company name")
    distributor: str = Field(..., description="Distributor/vendor information")
    warnings: List[str] = Field(default=[], description="Key safety warnings and precautions")

class LithiumBatterySchema(BaseModel):
    """Wrapper for extracted lithium battery specifications."""
    specs: List[LithiumBatterySpec] = Field(
        ..., description="List of extracted lithium battery specifications"
    )

print("✓ Pydantic schemas for lithium battery defined")
```

## Step 3: Create Helper Functions

We need a helper function to encode PDF files for processing by the Mistral API.

```python
def encode_pdf(pdf_path: str) -> str:
    """Encode PDF file to base64 string.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Base64 encoded string of the PDF
    """
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error encoding PDF: {str(e)}")

print("✓ Helper functions defined")
```

## Step 4: Verify Input Files

Before proceeding, let's verify that our required input files exist.

```python
# Define file paths
PDF_PATH = "lithium_iron_cell_datasheet.pdf"
REQUIREMENTS_PATH = "battery_requirements.txt"

# Verify files exist
if os.path.exists(PDF_PATH):
    print(f"✓ Found PDF: {PDF_PATH}")
else:
    raise FileNotFoundError(f"❌ PDF not found: {PDF_PATH}")
    
if os.path.exists(REQUIREMENTS_PATH):
    print(f"✓ Found requirements: {REQUIREMENTS_PATH}")
else:
    raise FileNotFoundError(f"❌ Requirements not found: {REQUIREMENTS_PATH}")
```

## Step 5: Extract Structured Data with Document Annotations

This is the core feature of this tutorial. We'll use Mistral OCR's `document_annotation_format` parameter to extract structured battery specifications directly from the PDF in a **single API call**.

**How it works:**
1. The PDF is encoded to base64
2. Mistral OCR processes the document
3. The `document_annotation_format` parameter tells the OCR to extract data matching our comprehensive battery schema
4. We get back structured data including capacity, voltage, current, temperatures, dimensions, and safety specs

**Benefits:**
- Single API call (no separate LLM call needed)
- Direct schema extraction during OCR
- More accurate (extraction happens with full document context)
- Captures complex nested specifications
- Safety-critical validation

**Note:** Document annotations are limited to **8 pages**. For larger documents, split them into chunks.

```python
print("📄 Extracting structured data from battery datasheet...")
print(f"   Processing: {PDF_PATH}")

# Encode PDF to base64
base64_pdf = encode_pdf(PDF_PATH)
print("   ✓ PDF encoded to base64")

# Extract structured data using Mistral OCR with document annotations
print("   🔍 Running Mistral OCR with document annotations...")
annotations_response = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),  # Document Annotations limited to 8 pages
    document={
        "type": "document_url",
        "document_url": f""
    },
    document_annotation_format=response_format_from_pydantic_model(LithiumBatterySchema),
    include_image_base64=True
)

print(f"   ✓ OCR completed - {len(annotations_response.pages)} pages processed")
print("   ✓ Structured data extracted successfully")
```

Now let's parse the extracted data into our Pydantic model:

```python
# Parse the extracted data into our Pydantic model
extracted_data = LithiumBatterySchema(**json.loads(annotations_response.document_annotation))

print("\n" + "="*60)
print("🔋 EXTRACTED BATTERY SPECIFICATIONS")
print("="*60)
print(json.dumps(extracted_data.model_dump(), indent=2))
```

You should see structured output containing all the battery specifications extracted from the PDF, including model details, electrical specifications, physical dimensions, temperature ranges, performance tests, certifications, and safety warnings.

## Step 6: Load Design Requirements

Now that we have structured battery data, we need to load our design requirements for comparison.

```python
# Load design requirements
print("📋 Loading battery design requirements...")
with open(REQUIREMENTS_PATH, 'r') as f:
    requirements = f.read()
print(f"   ✓ Requirements loaded from {REQUIREMENTS_PATH}")
print("\nDesign Requirements:")
print(requirements)
```

The requirements file should contain your project's specification criteria, such as:
- Minimum capacity requirements
- Voltage ranges
- Current limits
- Physical dimension constraints
- Temperature operating ranges
- Safety certifications needed

## Next Steps

With both the extracted battery specifications and your design requirements loaded, you can now:

1. **Compare specifications** programmatically by writing validation logic against each parameter
2. **Generate a compliance report** using Mistral LLM to analyze how well the battery meets your requirements
3. **Create a procurement recommendation** based on the analysis results
4. **Generate safety documentation** highlighting any potential concerns or warnings

The structured data extraction using Document Annotations provides a solid foundation for automated technical analysis, reducing manual review time from hours to minutes while improving accuracy and consistency.

---

### Key Takeaways

- **Document Annotations** allow for single-call structured data extraction from PDFs
- **Pydantic schemas** ensure data consistency and validation
- **Automated comparison** eliminates manual specification checking errors
- **Scalable approach** that can be adapted for other technical document types

This workflow demonstrates how AI-powered document processing can transform procurement and technical validation processes, making them faster, more accurate, and more consistent.