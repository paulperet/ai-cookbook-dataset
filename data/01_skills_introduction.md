# Claude Skills Tutorial: Create Professional Documents with AI

Learn how to use Claude's Skills feature to generate Excel spreadsheets, PowerPoint presentations, and PDF documents programmatically. This guide walks you through the setup and core workflows to automate business document creation.

## Prerequisites

Before you begin, ensure you have:
*   Python 3.8 or higher installed.
*   An Anthropic API key from the [Anthropic Console](https://console.anthropic.com/).

## 1. Environment Setup

First, set up your Python environment and install the required dependencies.

### Step 1: Create and Activate a Virtual Environment

Navigate to the project directory and create a virtual environment.

```bash
# Navigate to the skills directory
cd /path/to/claude-cookbooks/skills

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### Step 2: Install Dependencies

With the virtual environment activated, install the required packages.

```bash
pip install -r requirements.txt
```

**Important:** Ensure you have `anthropic>=0.71.0`. If needed, upgrade and restart your kernel or Python session.
```bash
pip install anthropic>=0.71.0
```

### Step 3: Configure Your API Key

Copy the example environment file and add your API key.

```bash
# Copy the example file
cp .env.example .env
```

Edit the `.env` file and add your key:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## 2. Initialize the API Client

Now, let's configure the Python client to connect to the Anthropic API and enable the Skills beta features.

```python
import os
import sys
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path.cwd().parent))

# Load environment variables from parent directory
load_dotenv(Path.cwd().parent / ".env")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found. Copy ../.env.example to ../.env and add your API key."
    )

# Initialize client
# Note: We'll add beta headers per-request when using Skills
client = Anthropic(api_key=API_KEY)

# Create outputs directory if it doesn't exist
OUTPUT_DIR = Path.cwd().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ API key loaded")
print(f"✓ Using model: {MODEL}")
print(f"✓ Output directory: {OUTPUT_DIR}")
```

### Test the API Connection

Run a simple test to verify your setup is working correctly.

```python
# Simple test to verify API connection
test_response = client.messages.create(
    model=MODEL,
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Say 'Connection successful!' if you can read this.",
        }
    ],
)

print("API Test Response:")
print(test_response.content[0].text)
print(
    f"\n✓ Token usage: {test_response.usage.input_tokens} in, {test_response.usage.output_tokens} out"
)
```

## 3. Understanding Claude Skills

Skills are packaged units of expertise that Claude can load dynamically. They combine instructions, executable code, and resources to perform specific tasks—like generating formatted Excel files—without you having to write the complex logic yourself.

**Key Benefits:**
*   **Efficiency:** Skills use a "progressive disclosure" model. Claude initially sees only minimal metadata (name & description). The full instructions and helper scripts are loaded only when the skill is needed, saving tokens.
*   **Reliability:** Skills contain pre-tested, working code, leading to more consistent and professional results.
*   **Composability:** Multiple skills can work together in a single request for complex workflows.

**Required Beta Features:** To use Skills, you must call the beta API endpoint and enable specific features via the `betas` parameter:
*   `skills-2025-10-02`: Enables the Skills feature.
*   `code-execution-2025-08-25`: Enables the code execution tool, which Skills require.
*   `files-api-2025-04-14`: Required for downloading the files Claude creates.

## 4. Discovering Available Skills

Let's list the Anthropic-managed skills available to you.

```python
# List all available Anthropic skills
client_with_skills_beta = Anthropic(
    api_key=API_KEY, default_headers={"anthropic-beta": "skills-2025-10-02"}
)

skills_response = client_with_skills_beta.beta.skills.list(source="anthropic")

print("Available Anthropic-Managed Skills:")
print("=" * 80)

for skill in skills_response.data:
    print(f"\n📦 Skill ID: {skill.id}")
    print(f"   Title: {skill.display_title}")
    print(f"   Latest Version: {skill.latest_version}")

    # Get version details
    try:
        version_info = client_with_skills_beta.beta.skills.versions.retrieve(
            skill_id=skill.id, version=skill.latest_version
        )
        print(f"   Name: {version_info.name}")
        print(f"   Description: {version_info.description}")
    except Exception as e:
        print(f"   (Unable to fetch version details: {e})")

print(f"\n\n✓ Found {len(skills_response.data)} Anthropic-managed skills")
```

You should see skills like `xlsx`, `pptx`, and `pdf` in the list.

## 5. Quick Start: Creating an Excel File

Now, let's use the `xlsx` skill to create a monthly budget spreadsheet. **Please be patient**, as document generation with code execution typically takes 1-2 minutes.

### Step 1: Make the API Request

We'll ask Claude to create a budget with income, expenses, formulas, and a chart.

```python
# Create an Excel budget spreadsheet
excel_response = client.beta.messages.create(
    model=MODEL,
    max_tokens=4096,
    container={"skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]},
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    messages=[
        {
            "role": "user",
            "content": """Create a monthly budget Excel spreadsheet with the following:

Income:
- Salary: $5,000
- Freelance: $1,200
- Investments: $300

Expenses:
- Rent: $1,500
- Utilities: $200
- Groceries: $600
- Transportation: $300
- Entertainment: $400
- Savings: $1,000

Include:
1. Formulas to calculate total income and total expenses
2. A formula for net savings (income - expenses)
3. Format currency values properly
4. Add a simple column chart showing income vs expenses
5. Use professional formatting with headers
""",
        }
    ],
    betas=["code-execution-2025-08-25", "files-api-2025-04-14", "skills-2025-10-02"],
)

print("Excel Response:")
print("=" * 80)
for content in excel_response.content:
    if content.type == "text":
        print(content.text)
```

### Step 2: Download the Generated File

The response contains a `file_id` for the generated Excel file. We need helper functions to extract this ID and download the file. First, ensure you have the `file_utils` module (it should be in the project). Then, run the download.

```python
from file_utils import (
    download_all_files,
    extract_file_ids,
    get_file_info,
    print_download_summary,
)

# Extract file IDs from the response
file_ids = extract_file_ids(excel_response)

if file_ids:
    print(f"✓ Found {len(file_ids)} file(s)\n")
    # Download all files
    results = download_all_files(
        client, excel_response, output_dir=str(OUTPUT_DIR), prefix="budget_"
    )
    # Print summary
    print_download_summary(results)
else:
    print("❌ No files found in response")
```

**What happened?**
1.  Claude used the `xlsx` skill's expertise to generate Python code (`openpyxl`).
2.  The code was executed in a secure environment, creating the spreadsheet.
3.  The resulting file was uploaded, and a `file_id` was returned in the response.
4.  You downloaded the file using the Files API. Check your `outputs/` directory for `budget_[timestamp].xlsx`.

## 6. Quick Start: Creating a PowerPoint Presentation

Next, let's use the `pptx` skill to create a simple presentation. Generation time is typically 1-2 minutes.

### Step 1: Make the API Request

We'll request a basic two-slide revenue presentation.

```python
# Create a PowerPoint presentation
pptx_response = client.beta.messages.create(
    model=MODEL,
    max_tokens=4096,
    container={"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]},
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    messages=[
        {
            "role": "user",
            "content": """Create a 2-slide PowerPoint presentation about Q3 revenue.

Slide 1: Title Slide
- Title: Q3 Revenue Report
- Subtitle: Executive Summary
- Your Name

Slide 2: Revenue Breakdown
- Title: Quarterly Performance
- Bullet points:
  *  Total Revenue: $1.2M
  *  Growth: 15% QoQ
  *  Top Product: Product X ($450K)
- Add a simple pie chart showing revenue by product line (Product X, Y, Z)
- Use a clean, professional design.
""",
        }
    ],
    betas=["code-execution-2025-08-25", "files-api-2025-04-14", "skills-2025-10-02"],
)

print("PowerPoint Response:")
print("=" * 80)
for content in pptx_response.content:
    if content.type == "text":
        print(content.text)
```

### Step 2: Download the Presentation

Use the same utility functions to download the generated `.pptx` file.

```python
# Extract file IDs from the response
file_ids = extract_file_ids(pptx_response)

if file_ids:
    print(f"✓ Found {len(file_ids)} file(s)\n")
    results = download_all_files(
        client, pptx_response, output_dir=str(OUTPUT_DIR), prefix="revenue_presentation_"
    )
    print_download_summary(results)
```

## 7. Quick Start: Creating a PDF Document

Finally, let's generate a PDF using the `pdf` skill. This usually takes 40-60 seconds.

### Step 1: Make the API Request

Ask Claude to create a simple project proposal PDF.

```python
# Create a PDF document
pdf_response = client.beta.messages.create(
    model=MODEL,
    max_tokens=4096,
    container={"skills": [{"type": "anthropic", "skill_id": "pdf", "version": "latest"}]},
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    messages=[
        {
            "role": "user",
            "content": """Create a one-page PDF project proposal document.

Include:
- Title: "Website Redesign Project Proposal"
- Client: Acme Corp
- Prepared by: Your Consulting Team
- Date: Current date

Sections:
1. Project Overview: A brief description of redesigning their public website.
2. Objectives: Improve user experience, increase mobile traffic, refresh brand identity.
3. Key Deliverables: Wireframes, responsive design, CMS integration, SEO audit.
4. Timeline: 12 weeks total.
5. Next Steps: Schedule a kickoff meeting.

Use a formal, business-appropriate style with clear headings.
""",
        }
    ],
    betas=["code-execution-2025-08-25", "files-api-2025-04-14", "skills-2025-10-02"],
)

print("PDF Response:")
print("=" * 80)
for content in pdf_response.content:
    if content.type == "text":
        print(content.text)
```

### Step 2: Download the PDF

Download the generated PDF file to your `outputs/` folder.

```python
# Extract file IDs from the response
file_ids = extract_file_ids(pdf_response)

if file_ids:
    print(f"✓ Found {len(file_ids)} file(s)\n")
    results = download_all_files(
        client, pdf_response, output_dir=str(OUTPUT_DIR), prefix="proposal_"
    )
    print_download_summary(results)
```

## Summary and Next Steps

You've successfully used Claude Skills to generate three different types of professional documents. The core pattern is always the same:

1.  **Structure your request:** Use `client.beta.messages.create()` with the `container` parameter to specify the skill(s).
2.  **Enable required tools and betas:** Always include the `code_execution` tool and the necessary beta features in the `betas` list.
3.  **Make a natural language request:** Describe the document you want Claude to create.
4.  **Download the result:** Extract the `file_id` from the response and use the Files API to download the generated file.

**To build more complex workflows:**
*   **Combine Skills:** Load multiple skills (e.g., `xlsx` and `pptx`) in a single `container` list to create multi-format reports.
*   **Provide Context:** Upload existing files (using the `messages` parameter) and ask Claude to analyze or modify them using a skill.
*   **Create Custom Skills:** Package your own company workflows and helper scripts as custom skills for consistent, branded output.

Remember to account for the code execution time (typically 40 seconds to 2 minutes) in your application logic. For production use, consider pinning skill versions (e.g., `"version": "20250101"`) instead of using `"latest"` for stability.