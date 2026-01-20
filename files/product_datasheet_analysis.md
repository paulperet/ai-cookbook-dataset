# Product Datasheet Analysis using Document AI

## Overview

This cookbook demonstrates automated product datasheet analysis using **Mistral AI's Document AI**.

### Use Case: Battery Procurement & Vendor Validation

You're sourcing lithium-ion batteries for a portable device. Vendors send PDF datasheets with hundreds of specifications. Manually comparing each against your design requirements is time-consuming and error-prone.

**This cookbook automates the process:**
1. **Extract structured data** from lithium battery PDF datasheets using Document AI - Mistral OCR with Document Annotations
2. **Compare specifications** against design requirements
3. **Generate detailed technical reports** with comprehensive analysis for each parameter

---

### Input Files Required:

1. **📄 Product Datasheet PDF** (`lithium_iron_cell_datasheet.pdf`)
   - Vendor-provided specification document containing technical specs, safety info, and performance data

2. **📋 Design Requirements** (`battery_requirements.txt`)  
   - Your project's specification criteria defining acceptable ranges for capacity, voltage, temperature, safety, etc.

---

### Technology Stack:

- ✅ **Mistral OCR** (`mistral-ocr-latest`) - PDF parsing with document annotations
- ✅ **Mistral Medium** (`mistral-medium-latest`) - Technical report generation

### Key Features:

- Document AI for OCR + structured extraction
- Direct Pydantic schema extraction
- Comprehensive battery specification coverage
- Safety-focused validation
- Professional technical report generation

**Benefits:** Fast, accurate, and generates professional documentation for procurement decisions.

## 1. Setup and Imports

```python
# Install required packages (uncomment if needed)
# !pip install mistralai
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

    ✓ All imports successful

```python
# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")

client = Mistral(api_key=api_key)
print("✓ Mistral client initialized")
```

    ✓ Mistral client initialized

## 2. Define Data Schemas

We define comprehensive Pydantic schemas for lithium battery specifications including capacity, voltage, current, temperature, dimensions, and safety features.

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

    ✓ Pydantic schemas for lithium battery defined

## 3. Helper Functions

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

    ✓ Helper functions defined

## 4. File Setup

Verify that the required files exist.

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

    ✓ Found PDF: lithium_iron_cell_datasheet.pdf
    ✓ Found requirements: battery_requirements.txt

## 5. Extract Structured Data with Document Annotations

This is the **key feature** of this cookbook. We use Mistral OCR's `document_annotation_format` parameter to extract structured battery specifications directly from the PDF in a **single API call**.

### How it works:

1. The PDF is encoded to base64
2. Mistral OCR processes the document
3. The `document_annotation_format` parameter tells the OCR to extract data matching our comprehensive battery schema
4. We get back structured data including capacity, voltage, current, temperatures, dimensions, and safety specs

### Benefits:

- ✅ Single API call (no separate LLM call needed)
- ✅ Direct schema extraction during OCR
- ✅ More accurate (extraction happens with full document context)
- ✅ Captures complex nested specifications
- ✅ Safety-critical validation

### Note:

Document annotations are limited to **8 pages**. For larger documents, split them into chunks.

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

    📄 Extracting structured data from battery datasheet...
       Processing: lithium_iron_cell_datasheet.pdf
       ✓ PDF encoded to base64
       🔍 Running Mistral OCR with document annotations...
       ✓ OCR completed - 8 pages processed
       ✓ Structured data extracted successfully

```python
# Parse the extracted data into our Pydantic model
extracted_data = LithiumBatterySchema(**json.loads(annotations_response.document_annotation))

print("\n" + "="*60)
print("🔋 EXTRACTED BATTERY SPECIFICATIONS")
print("="*60)
print(json.dumps(extracted_data.model_dump(), indent=2))
```

    
    ============================================================
    🔋 EXTRACTED BATTERY SPECIFICATIONS
    ============================================================
    {
      "specs": [
        {
          "model_name": "INR18650-2500A",
          "product_type": "Lithium-ion Cell Battery",
          "capacity": {
            "normal_capacity": 2500.0,
            "minimum_capacity": 2450.0,
            "unit": "mAh"
          },
          "voltage": {
            "nominal_voltage": 3.7,
            "charge_voltage": 4.2,
            "discharge_cutoff_voltage": 2.75
          },
          "current": {
            "standard_charge_current": 500.0,
            "maximum_charge_current": 1250.0,
            "standard_discharge_current": 500.0,
            "maximum_discharge_current": 2500.0,
            "max_instantaneous_discharge": 5000.0
          },
          "internal_impedance": "<=60 m\u03a9 (with PCB)",
          "dimensions": {
            "height": 65.0,
            "diameter": 18.1,
            "weight": 50.0
          },
          "cycle_life": 300,
          "operating_temperatures": [
            {
              "min_temp": 0.0,
              "max_temp": 45.0,
              "condition": "Charge"
            },
            {
              "min_temp": -10.0,
              "max_temp": 60.0,
              "condition": "Discharge"
            }
          ],
          "storage_temperatures": [
            {
              "min_temp": -5.0,
              "max_temp": 45.0,
              "condition": "Storage (\u2264 1 month)"
            },
            {
              "min_temp": 0.0,
              "max_temp": 45.0,
              "condition": "Storage (\u2264 6 months)"
            }
          ],
          "performance_tests": [
            {
              "test_name": "Discharge Performance",
              "criteria": "\u2265294min.",
              "result": "After the battery is standardly charged, set it aside for 0.5h - 1h at an ambient temperature of 23\u2103\u00b12\u2103, and then discharge to the cut-off voltage with a current of 0.2C."
            },
            {
              "test_name": "High Temperature Performance",
              "criteria": "\u22655h",
              "result": "After the battery is standardly charged and stored in an ambient temperature of 55\u2103\u00b12\u2103 for 2h and then discharge to the cut-off voltage with a current of 0.2C."
            },
            {
              "test_name": "Low Temperature Performance",
              "criteria": "\u22653h",
              "result": "After the battery is standardly charged and stored in an ambient temperature of -10\u2103\u00b12\u2103 for 5h and then discharge to the cut-off voltage with a current of 0.2C."
            },
            {
              "test_name": "Capacity Retention",
              "criteria": "Retention: 85%C-Ah",
              "result": "After the battery is standardly charged and stored in an ambient temperature of 23\u2103\u00b12\u2103 for 28 days, discharge to the cut-off voltage with a current of 0.2C."
            },
            {
              "test_name": "Cycle Life (25\u2103)",
              "criteria": "Cycle life \u2265 300 cycles",
              "result": "First charge with a constant current of 0.5C to 4.20V and a constant voltage of 4.20V until the charge current is less than or equal to 0.01C. Leave it aside for 10 minutes. Then discharge to 2.75V with a current of 0.5C - leave it aside for another 10 minutes. Repeat above steps until the discharge capacity is higher than 80% of the initial capacities of the cells."
            }
          ],
          "certifications": [
            "ECCN: EAR99",
            "USHTS: 8507600020"
          ],
          "manufacturer": "Shenzhen Hondark Electronics Co., Ltd.",
          "distributor": "TinyCircuits",
          "warnings": [
            "Do not immerse the pack in water, seawater, or other liquids.",
            "Do not use, or leave the battery near heat sources such as a fire or heater.",
            "Do not use or store the battery where it is exposed to an extremely hot environment, such as in a car under direct sunlight or on a hot day.",
            "Do not place the battery in a microwave oven or pressurized container.",
            "Do not use the battery in a location where static electricity or magnetic fields are great, otherwise, the safety devices in the pack may be damaged, which may cause unsafe risks.",
            "Keep the batteries out of the reach of young children. If a child somehow swallows a battery, seek medical attention immediately.",
            "Use the battery only under the environmental conditions mentioned in this document.",
            "The flexible packaging encasing the battery cells is vulnerable to sharp objects that could puncture or damage the integrity of the casing.",
            "Do not transport or store the battery together with metal objects such as keys, necklaces etc.",
            "Do not strike at pack with any sharp objects.",
            "Do not strike the battery with any sharp-edged parts.",
            "Trim nails or wear gloves before handling batteries.",
            "Clean worktable where battery is used to avoid any sharp objects.",
            "If the pack leaks and gets into the eyes, do not rub eyes. Instead, rinse the eyes with clean running water, and immediately seek medical attention.",
            "If the battery leaks and gets on your skin or clothing, immediately rinse the affected area with clean running water.",
            "Pay attention to the use of insulation structures between the battery core, as well as between the battery core and electrical appliances.",
            "If the battery leaks or emits an odor, immediately remove it from the proximity of any exposed flame.",
            "Do not use the battery in combination with batteries of different capacity, type, or brand.",
            "Do not attempt to disassemble or modify the battery in any way.",
            "Do not use any chargers other than those recommended for Lithium-ion Polymer batteries.",
            "Do not reverse the positive (+) and negative (-) terminals.",
            "Do not connect the pack to an electrical outlet, such as wall outlets or car cigarette-lighter sockets.",
            "Do not directly solder the pack or battery terminals.",
            "If the pack emits an odor, generates heat, becomes discolored or deformed, or any abnormal phenomenon occurs during charging, recharging or storage, immediately remove the battery from the charger or device, and stop use.",
            "If the case pack terminals are dirty, clean the terminals with a dry cloth before use.",
            "Be aware that discharged battery may cause fire or smoke; tape the terminals with insulating paper to insulate them.",
            "For directions on battery installation and removal, read the instruction manual that accompanies the equipment in which the battery will be used.",
            "If a device is not used for an extended period, the battery should be removed and stored in a cool, dry place."
          ]
        }
      ]
    }

## 6. Generate Comparison Report

Now that we have structured battery data, we use Mistral LLM to compare it against design requirements and generate a detailed safety and performance report.

```python
# Load design requirements
print("📋 Loading battery design requirements...")
with open(REQUIREMENTS_PATH, 'r') as f:
    requirements = f.read()
print(f"   ✓ Requirements loaded from {REQUIREMENTS_PATH}")
print("\nDesign Requirements:")
print(requirements)
```

    📋 Loading battery design requirements...
       ✓ Requirements loaded from battery_requirements.txt
    
    Design Requirements:
    Lithium-Ion Battery Design Requirements:
    
    Capacity:
    - Normal Capacity: ≥ 2400 mAh
    - Minimum Capacity: ≥ 2350 mAh
    
    Voltage:
    - Nominal Voltage: 3.6V - 3.7V
    - Charge Voltage: ≤ 4.2V
    - Discharge Cut-off Voltage: ≥ 2.75V
    
    Current:
    - Standard Charge Current: ≤ 1250 mA (0.5C)
    - Maximum Charge Current: ≤ 2500 mA (1C)
    - Standard Discharge Current: ≥ 500 mA
    - Maximum Continuous Discharge: ≥ 5000 mA (2C)
    
    Physical:
    - Diameter: 18mm ± 0.5mm (18650 standard)
   