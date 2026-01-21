# Building Custom Skills for Claude: A Practical Guide

Learn how to create, deploy, and manage custom skills to extend Claude's capabilities with your organization's specialized knowledge and workflows.

## Table of Contents

1. [Introduction & Setup](#introduction)
2. [Understanding Custom Skills Architecture](#architecture)
3. [Example 1: Financial Ratio Calculator](#financial-ratio)
4. [Example 2: Company Brand Guidelines](#brand-guidelines)
5. [Example 3: Financial Modeling Suite](#financial-modeling)
6. [Skill Management & Versioning](#management)
7. [Best Practices & Production Tips](#best-practices)
8. [Troubleshooting](#troubleshooting)

## 1. Introduction & Setup

### What are Custom Skills?

Custom skills are specialized expertise packages you create to teach Claude your organization's unique workflows, domain knowledge, and best practices. Unlike Anthropic's pre-built skills (Excel, PowerPoint, PDF), custom skills allow you to:

- **Codify organizational knowledge**: Capture your team's specific methodologies
- **Ensure consistency**: Apply the same standards across all interactions
- **Automate complex workflows**: Chain together multi-step processes
- **Maintain intellectual property**: Keep proprietary methods secure

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Expertise at Scale** | Deploy specialized knowledge to every Claude interaction |
| **Version Control** | Track changes and roll back if needed |
| **Composability** | Combine multiple skills for complex tasks |
| **Privacy** | Your skills remain private to your organization |

### Prerequisites

Before starting, ensure you have:
- Completed the Introduction to Skills tutorial
- An Anthropic API key with Skills beta access
- Python environment with the local SDK installed

### Environment Setup

Let's set up our environment and import necessary libraries:

```python
import os
import sys
from pathlib import Path
from typing import Any

# Add parent directory for imports
sys.path.insert(0, str(Path.cwd().parent))

from anthropic import Anthropic
from anthropic.lib import files_from_dir
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.cwd().parent / ".env")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found. Copy ../.env.example to ../.env and add your API key."
    )

# Initialize client with Skills beta
client = Anthropic(api_key=API_KEY, default_headers={"anthropic-beta": "skills-2025-10-02"})

# Setup directories
SKILLS_DIR = Path.cwd().parent / "custom_skills"
OUTPUT_DIR = Path.cwd().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ API key loaded")
print(f"✓ Using model: {MODEL}")
print(f"✓ Custom skills directory: {SKILLS_DIR}")
print(f"✓ Output directory: {OUTPUT_DIR}")
print("\n📝 Skills beta header configured for skill management")
```

## 2. Understanding Custom Skills Architecture

### Skill Structure

Every custom skill follows this directory structure:

```
skill_name/
├── SKILL.md           # REQUIRED: Instructions with YAML frontmatter
├── *.md               # Optional: Any additional .md files (documentation, guides)
├── scripts/           # Optional: Executable code
│   ├── process.py
│   └── utils.js
└── resources/         # Optional: Templates, data files
    └── template.xlsx
```

**Important Notes:**
- **SKILL.md is the ONLY required file** - everything else is optional
- **Multiple .md files allowed** - You can have any number of markdown files in the top-level folder
- **All .md files are loaded** - Not just SKILL.md and REFERENCE.md, but any .md file you include
- **Organize as needed** - Use multiple .md files to structure complex documentation

### SKILL.md Requirements

The `SKILL.md` file must include:

1. **YAML Frontmatter** (name: 64 chars, description: 1024 chars)
   - `name`: Lowercase alphanumeric with hyphens (required)
   - `description`: Brief description of what the skill does (required)

2. **Instructions** (markdown format)
   - Clear guidance for Claude
   - Examples of usage
   - Any constraints or rules
   - Recommended: Keep under 5,000 tokens

### Additional Documentation Files

You can include multiple markdown files for better organization:

```
skill_name/
├── SKILL.md           # Main instructions (required)
├── REFERENCE.md       # API reference (optional)
├── EXAMPLES.md        # Usage examples (optional)
├── TROUBLESHOOTING.md # Common issues (optional)
└── CHANGELOG.md       # Version history (optional)
```

All `.md` files in the root directory will be available to Claude when the skill is loaded.

### Progressive Disclosure

Skills load in three stages to optimize token usage:

| Stage | Content | Token Cost | When Loaded |
|-------|---------|------------|-------------|
| **1. Metadata** | Name & description | name: 64 chars, description: 1024 chars | Always visible |
| **2. Instructions** | All .md files | <5,000 tokens recommended | When relevant |
| **3. Resources** | Scripts & files | As needed | During execution |

### Create Skill Utility Functions

Now, let's create helper functions for skill management. These utilities will handle creation, listing, deletion, and testing of custom skills.

```python
def create_skill(client: Anthropic, skill_path: str, display_title: str) -> dict[str, Any]:
    """
    Create a new custom skill from a directory.

    Args:
        client: Anthropic client instance
        skill_path: Path to skill directory
        display_title: Human-readable skill name

    Returns:
        Dictionary with skill_id, version, and metadata
    """
    try:
        # Create skill using files_from_dir
        skill = client.beta.skills.create(
            display_title=display_title, files=files_from_dir(skill_path)
        )

        return {
            "success": True,
            "skill_id": skill.id,
            "display_title": skill.display_title,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
            "source": skill.source,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_custom_skills(client: Anthropic) -> list[dict[str, Any]]:
    """
    List all custom skills in the workspace.

    Returns:
        List of skill dictionaries
    """
    try:
        skills_response = client.beta.skills.list(source="custom")

        skills = []
        for skill in skills_response.data:
            skills.append(
                {
                    "skill_id": skill.id,
                    "display_title": skill.display_title,
                    "latest_version": skill.latest_version,
                    "created_at": skill.created_at,
                    "updated_at": skill.updated_at,
                }
            )

        return skills
    except Exception as e:
        print(f"Error listing skills: {e}")
        return []


def delete_skill(client: Anthropic, skill_id: str) -> bool:
    """
    Delete a custom skill and all its versions.

    Args:
        client: Anthropic client
        skill_id: ID of skill to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        # First delete all versions
        versions = client.beta.skills.versions.list(skill_id=skill_id)

        for version in versions.data:
            client.beta.skills.versions.delete(skill_id=skill_id, version=version.version)

        # Then delete the skill itself
        client.beta.skills.delete(skill_id)
        return True

    except Exception as e:
        print(f"Error deleting skill: {e}")
        return False


def test_skill(
    client: Anthropic,
    skill_id: str,
    test_prompt: str,
    model: str = "claude-sonnet-4-5",
) -> Any:
    """
    Test a custom skill with a prompt.

    Args:
        client: Anthropic client
        skill_id: ID of skill to test
        test_prompt: Prompt to test the skill
        model: Model to use for testing

    Returns:
        Response from Claude
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        container={"skills": [{"type": "custom", "skill_id": skill_id, "version": "latest"}]},
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        messages=[{"role": "user", "content": test_prompt}],
        betas=[
            "code-execution-2025-08-25",
            "files-api-2025-04-14",
            "skills-2025-10-02",
        ],
    )

    return response


print("✓ Skill utility functions defined")
print("  - create_skill()")
print("  - list_custom_skills()")
print("  - delete_skill()")
print("  - test_skill()")
```

### Check Existing Custom Skills

Before creating new skills, let's check if any custom skills already exist in your workspace. This is important because skills cannot have duplicate display titles.

```python
# Check for existing skills that might conflict
existing_skills = list_custom_skills(client)
skill_titles_to_create = [
    "Financial Ratio Analyzer",
    "Corporate Brand Guidelines",
    "Financial Modeling Suite",
]
conflicting_skills = []

if existing_skills:
    print(f"Found {len(existing_skills)} existing custom skill(s):")
    for skill in existing_skills:
        print(f"  - {skill['display_title']} (ID: {skill['skill_id']})")
        if skill["display_title"] in skill_titles_to_create:
            conflicting_skills.append(skill)

    if conflicting_skills:
        print(
            f"\n⚠️ Found {len(conflicting_skills)} skill(s) that will conflict with this notebook:"
        )
        for skill in conflicting_skills:
            print(f"  - {skill['display_title']} (ID: {skill['skill_id']})")

        print("\n" + "=" * 70)
        print("To clean up these skills and start fresh, uncomment and run:")
        print("=" * 70)
        print("\n# UNCOMMENT THE LINES BELOW TO DELETE CONFLICTING SKILLS:")
        print("# for skill in conflicting_skills:")
        print("#     if delete_skill(client, skill['skill_id']):")
        print("#         print(f\"✅ Deleted: {skill['display_title']}\")")
        print("#     else:")
        print("#         print(f\"❌ Failed to delete: {skill['display_title']}\")")
    else:
        print("\n✅ No conflicting skills found. Ready to proceed!")
else:
    print("✅ No existing custom skills found. Ready to create new ones!")
```

## 3. Example 1: Financial Ratio Calculator

Let's create our first custom skill - a financial ratio calculator that can analyze company financial health.

### Skill Overview

The **Financial Ratio Calculator** skill will:
- Calculate key financial ratios (ROE, P/E, Current Ratio, etc.)
- Interpret ratios with industry context
- Generate formatted reports
- Work with various data formats (CSV, JSON, text)

### Upload the Financial Analyzer Skill

Now let's upload our financial analyzer skill to Claude:

```python
# Upload the Financial Analyzer skill
financial_skill_path = SKILLS_DIR / "analyzing-financial-statements"

if financial_skill_path.exists():
    print("Uploading Financial Analyzer skill...")
    result = create_skill(client, str(financial_skill_path), "Financial Ratio Analyzer")

    if result["success"]:
        financial_skill_id = result["skill_id"]
        print("✅ Skill uploaded successfully!")
        print(f"   Skill ID: {financial_skill_id}")
        print(f"   Version: {result['latest_version']}")
        print(f"   Created: {result['created_at']}")
    else:
        print(f"❌ Upload failed: {result['error']}")
        if "cannot reuse an existing display_title" in str(result["error"]):
            print("\n💡 Solution: A skill with this name already exists.")
            print("   Run the 'Clean Up Existing Skills' cell above to delete it first,")
            print("   or change the display_title to something unique.")
else:
    print(f"⚠️ Skill directory not found: {financial_skill_path}")
    print(
        "Please ensure the custom_skills directory contains the analyzing-financial-statements folder."
    )
```

### Test the Financial Analyzer Skill

Let's test the skill with sample financial data to see it in action:

```python
# Test the Financial Analyzer skill
if "financial_skill_id" in locals():
    test_prompt = """
    Calculate financial ratios for this company:

    Income Statement:
    - Revenue: $1,000M
    - EBITDA: $200M
    - Net Income: $120M

    Balance Sheet:
    - Total Assets: $2,000M
    - Current Assets: $500M
    - Current Liabilities: $300M
    - Total Debt: $400M
    - Shareholders Equity: $1,200M

    Market Data:
    - Share Price: $50
    - Shares Outstanding: 100M

    Please calculate key ratios and provide analysis.
    """

    print("Testing Financial Analyzer skill...")
    response = test_skill(client, financial_skill_id, test_prompt)

    # Print response
    for content in response.content:
        if content.type == "text":
            print(content.text)
else:
    print("⚠️ Please upload the Financial Analyzer skill first (run the previous cell)")
```

## 4. Example 2: Company Brand Guidelines

Now let's create a skill that ensures all documents follow corporate brand standards.

### Skill Overview

The **Brand Guidelines** skill will:
- Apply consistent colors, fonts, and layouts
- Ensure logo placement and usage
- Maintain professional tone and messaging
- Work across all document types (Excel, PowerPoint, PDF)

### Upload the Brand Guidelines Skill

```python
# Upload the Brand Guidelines skill
brand_skill_path = SKILLS_DIR / "applying-brand-guidelines"

if brand_skill_path.exists():
    print("Uploading Brand Guidelines skill...")
    result = create_skill(client, str(brand_skill_path), "Corporate Brand Guidelines")

    if result["success"]:
        brand_skill_id = result["skill_id"]
        print("✅ Skill uploaded successfully!")
        print(f"   Skill ID: {brand_skill_id}")
        print(f"   Version: {result['latest_version']}")
    else:
        print(f"❌ Upload failed: {result['error']}")
        if "cannot reuse an existing display_title" in str(result["error"]):
            print("\n💡 Solution: A skill with this name already exists.")
            print("   Run the 'Clean Up Existing Skills' cell above to delete it first,")
            print("   or change the display_title to something unique.")
else:
    print(f"⚠️ Skill directory not found: {brand_skill_path}")
```

### Test Brand Guidelines with Document Creation

Let's test the brand skill by creating a branded PowerPoint presentation. Notice how we can combine custom skills with Anthropic's built-in skills:

```python
# Test Brand Guidelines skill with PowerPoint creation
if "brand_skill_id" in locals():
    # Combine brand skill with Anthropic's pptx skill
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        container={
            "skills": [
                {"type": "custom", "skill_id": brand_skill_id, "version": "latest"},
                {"type": "anthropic", "skill_id": "pptx", "version": "latest"}
            ]
        },
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        messages=[{
            "role": "user", 
            "content": "Create a 3-slide PowerPoint presentation introducing our new product line. Include our brand colors, logo, and follow our corporate messaging guidelines."
        }],
        betas=[
            "code-execution-2025-08-25",
            "files-api-2025-04-14",
            "skills-2025-10-02",
        ],
    )

    # Print the response
    for content in response.content:
        if content.type == "text":
            print(content.text)
else:
    print("⚠️ Please upload the Brand Guidelines skill first")
```

## 5. Example 3: Financial Modeling Suite

For our third example, let's create a more complex skill that bundles multiple capabilities for financial modeling.

### Skill Overview

The **Financial Modeling Suite** skill will:
- Build discounted cash flow (DCF) models
- Perform sensitivity analysis
- Generate investment memos
- Create financial projections

### Upload the Financial Modeling Skill

```python
# Upload the Financial Modeling skill
modeling_skill_path = SKILLS_DIR / "building-financial-models"

if modeling_skill_path.exists():
    print("Uploading Financial Modeling skill...")
    result = create_skill(client, str(modeling_skill_path), "Financial Modeling Suite")

    if result["success"]:
        modeling_skill_id = result["skill_id"]
        print("✅ Skill uploaded successfully!")
        print(f"   Skill ID: {modeling_skill_id}")
        print(f"   Version: {result['latest_version']}")
    else:
        print(f"❌ Upload failed: {result['error']}")
        if "cannot reuse an existing display_title" in str(result["error"]):
            print("\n💡 Solution: A skill with this name already exists.")
            print("   Run the 'Clean Up Existing Skills' cell above to delete it first,")
            print("   or change the display_title to something unique.")
else:
    print(f"⚠️ Skill directory not found: {modeling_skill_path}")
```

### Test Combined Skills

Now let's test a more complex workflow that combines multiple skills. This demonstrates how you can chain skills together for sophisticated tasks:

```python
# Test combined skills for a comprehensive financial analysis
if all(var in locals() for var in ["financial_skill_id", "modeling_skill_id"]):
    test_prompt = """
    I need a comprehensive investment analysis for Company XYZ.

    Historical Data (last 3 years):
    - Revenue: $800