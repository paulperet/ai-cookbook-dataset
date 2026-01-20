# Claude Skills for Financial Applications

Build real-world financial dashboards, portfolio analytics, and automated reporting workflows using Claude's Excel, PowerPoint, and PDF skills.

> **💡 Real-world Impact:** These are the same Skills that power **[Claude Creates Files](https://www.anthropic.com/news/create-files)**, enabling Claude to create professional financial documents directly in the interface.

**What you'll learn:**
- Create comprehensive financial models in Excel with formulas and charts
- Generate executive presentations from financial data
- Build portfolio analysis tools with risk metrics
- Automate multi-format reporting pipelines

## Table of Contents

1. [Setup & Data Loading](#setup)
2. [Use Case 1: Financial Dashboard Creation](#financial-dashboard)
   - [Excel Financial Model](#excel-model)
   - [Executive PowerPoint](#executive-ppt)
   - [PDF Financial Report](#pdf-report)
3. [Use Case 2: Portfolio Analysis Workflow](#portfolio-analysis)
   - [Portfolio Analytics Excel](#portfolio-excel)
   - [Investment Committee Deck](#investment-deck)
4. [Use Case 3: Automated Reporting Pipeline](#reporting-pipeline)

## Prerequisites

This notebook assumes you've completed **Notebook 1: Introduction to Skills**.

If you haven't:
1. Complete the setup in Notebook 1 first
2. Verify your environment with the test cells
3. Ensure you can create and download files

**Required:**
- Anthropic API key configured
- SDK version 0.69.0 installed from whl
- Virtual environment activated

## 1. Setup & Data Loading {#setup}

Let's start by importing our dependencies and loading the financial data we'll work with throughout this notebook.


```python
# Standard imports
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path.cwd().parent))

# Anthropic SDK
from anthropic import Anthropic
from dotenv import load_dotenv

# Our utilities
from file_utils import (
    download_all_files,
    print_download_summary,
)

# Load environment
load_dotenv(Path.cwd().parent / ".env")

# Configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-5"

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found. Please configure your .env file.")

# Initialize client
client = Anthropic(api_key=API_KEY)

# Setup directories
OUTPUT_DIR = Path.cwd().parent / "outputs" / "financial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.cwd().parent / "sample_data"

print("✓ Environment configured")
print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ Data directory: {DATA_DIR}")
```

### Load Financial Data

We have four datasets representing different aspects of a company's financial position:


```python
# Load financial statements
financial_statements = pd.read_csv(DATA_DIR / "financial_statements.csv")
print("📊 Financial Statements Overview:")
print(f"   Shape: {financial_statements.shape}")
print(f"   Categories: {len(financial_statements['Category'].unique())} financial metrics")
print(f"   Quarters: {list(financial_statements.columns[1:5])}")
print()

# Show sample data
print("Sample data (first 5 rows):")
financial_statements.head()
```


```python
# Load portfolio holdings
with open(DATA_DIR / "portfolio_holdings.json") as f:
    portfolio_data = json.load(f)

print("💼 Portfolio Overview:")
print(f"   Portfolio: {portfolio_data['portfolio_name']}")
print(f"   Total Value: ${portfolio_data['total_value']:,.2f}")
print(f"   Holdings: {len(portfolio_data['holdings'])} stocks")
print(f"   Cash Position: ${portfolio_data['cash_position']['amount']:,.2f}")
print(f"   Total Return: {portfolio_data['performance_metrics']['total_return_percent']:.1f}%")
print()

# Convert holdings to DataFrame for easier manipulation
portfolio_df = pd.DataFrame(portfolio_data["holdings"])
print("Top 5 holdings by value:")
portfolio_df.nlargest(5, "market_value")[["ticker", "name", "market_value", "unrealized_gain"]]
```


```python
# Load quarterly metrics
with open(DATA_DIR / "quarterly_metrics.json") as f:
    quarterly_metrics = json.load(f)

print("📈 Quarterly Metrics Overview:")
print(f"   Quarters available: {len(quarterly_metrics['quarters'])}")
print(f"   Metrics per quarter: {len(quarterly_metrics['quarters'][0])} KPIs")
print()

# Show latest quarter metrics
latest_quarter = quarterly_metrics["quarters"][-1]
print(f"Latest Quarter ({latest_quarter['quarter']}):")
for key, value in latest_quarter.items():
    if key != "quarter" and isinstance(value, int | float):
        if "revenue" in key.lower() or "cost" in key.lower():
            print(f"   {key.replace('_', ' ').title()}: ${value:,.0f}")
        elif "percent" in key.lower() or "margin" in key.lower() or "rate" in key.lower():
            print(f"   {key.replace('_', ' ').title()}: {value:.1f}%")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value:,.0f}")
```

### Helper Functions

Let's define some helper functions for this notebook:


```python
def create_skills_message(client, prompt, skills, prefix="", show_token_usage=True):
    """
    Helper function to create messages with Skills.

    Args:
        client: Anthropic client
        prompt: User prompt
        skills: List of skill dicts [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]
        prefix: Prefix for downloaded files
        show_token_usage: Whether to print token usage

    Returns:
        Tuple of (response, download_results)
    """
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        container={"skills": skills},
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        messages=[{"role": "user", "content": prompt}],
        betas=[
            "code-execution-2025-08-25",
            "files-api-2025-04-14",
            "skills-2025-10-02",
        ],
    )

    if show_token_usage:
        print(
            f"\n📊 Token Usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out"
        )

    # Download files
    results = download_all_files(client, response, output_dir=str(OUTPUT_DIR), prefix=prefix)

    return response, results


def format_financial_value(value, is_currency=True, decimals=0):
    """Format financial values for display."""
    if is_currency:
        return f"${value:,.{decimals}f}"
    else:
        return f"{value:,.{decimals}f}"


print("✓ Helper functions defined")
```

## 2. Use Case 1: Financial Dashboard Creation {#financial-dashboard}

Now that we have our data loaded and helper functions defined, let's dive into our first practical use case: creating comprehensive financial dashboards. We'll start by generating multi-sheet Excel workbooks that automatically include formulas, formatting, and charts.

### 2.1 Excel Financial Model {#excel-model}

We'll create a financial dashboard that includes:
- Profit & Loss statements with year-over-year comparisons
- Balance sheet analysis
- Cash flow tracking
- KPI dashboards with visualizations

This demonstrates how Claude's Skills can handle complex Excel generation tasks that would typically require hours of manual work.


```python
# Create Financial Dashboard Excel
print("Creating financial dashboard Excel file...")
print("This creates a 2-sheet dashboard optimized for the Skills API.")
print("\n⏱️ Generation time: 1-2 minutes\n")

# Prepare the financial data
fs_data = financial_statements.to_dict("records")
quarters_2024 = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]

# Extract key financial metrics
revenue_by_quarter = {
    "Q1 2024": financial_statements[financial_statements["Category"] == "Revenue"][
        "Q1_2024"
    ].values[0],
    "Q2 2024": financial_statements[financial_statements["Category"] == "Revenue"][
        "Q2_2024"
    ].values[0],
    "Q3 2024": financial_statements[financial_statements["Category"] == "Revenue"][
        "Q3_2024"
    ].values[0],
    "Q4 2024": financial_statements[financial_statements["Category"] == "Revenue"][
        "Q4_2024"
    ].values[0],
}

financial_dashboard_prompt = f"""
Create a financial dashboard Excel workbook with 2 sheets:

Sheet 1 - "P&L Summary":
Create a Profit & Loss summary table for 2024 quarters with these rows:
- Revenue: {", ".join([f"Q{i + 1}: ${v / 1000000:.1f}M" for i, v in enumerate(revenue_by_quarter.values())])}
- Gross Profit: Use values from the data
- Operating Income: Use values from the data
- Net Income: Use values from the data
- Add a Total column with SUM formulas
- Add a row showing profit margins (Net Income / Revenue)
- Apply currency formatting and bold headers
- Add a simple bar chart showing quarterly revenue

Sheet 2 - "Key Metrics":
Create a metrics dashboard with:
- Total Revenue 2024: SUM of all quarters
- Average Quarterly Revenue: AVERAGE formula
- Q4 vs Q1 Growth: Percentage increase
- Best Quarter: MAX formula to identify
- Operating Margin Q4: Calculate from data
- Year-over-year growth vs 2023

Apply professional formatting with borders, bold headers, and currency formats.
"""

# Create the Excel financial dashboard
excel_response, excel_results = create_skills_message(
    client,
    financial_dashboard_prompt,
    [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}],
    prefix="financial_dashboard_",
)

print("\n" + "=" * 60)
print_download_summary(excel_results)

if len(excel_results) > 0 and excel_results[0]["success"]:
    print("\n✅ Financial dashboard Excel created successfully!")
```

### 💡 Best Practices for Excel Generation

Based on our testing, here are the optimal approaches for creating Excel files with Skills:

**Recommended Approach:**
- **2-3 sheets per workbook** works reliably and generates quickly
- **Focus each sheet** on a specific purpose (e.g., P&L, metrics, charts)
- **Add complexity incrementally** - start simple, then enhance

**For Complex Dashboards:**
1. **Create multiple focused files** instead of one complex file
   - Example: `financial_pnl.xlsx`, `balance_sheet.xlsx`, `kpi_dashboard.xlsx`
2. **Use the pipeline pattern** to create and enhance files sequentially
3. **Combine files programmatically** using pandas or openpyxl if needed

**Performance Tips:**
- Simple 2-sheet dashboards: ~1-2 minutes
- PowerPoint and PDF generation: Very reliable for complex content
- Token usage: Structured data (JSON/CSV) is more efficient than prose

### 2.2 Executive PowerPoint {#executive-ppt}

With our financial data now organized in Excel, let's create an executive presentation that summarizes the key insights. This demonstrates how Skills can generate professional PowerPoint presentations with charts, formatted text, and multiple slides - perfect for board meetings or investor updates.

The presentation will include:
- Q4 2024 performance highlights
- Financial metrics with year-over-year comparisons
- Profitability trends with visualizations
- Key takeaways and outlook


```python
print("Creating executive presentation from financial metrics...")
print("\n⏱️ Generation time: 1-2 minutes\n")

# Calculate some key metrics for the presentation
q4_2024_revenue = 14500000
q4_2023_revenue = 12300000
yoy_growth = (q4_2024_revenue - q4_2023_revenue) / q4_2023_revenue * 100

q4_2024_net_income = 1878750
q4_2023_net_income = 1209000
net_income_growth = (q4_2024_net_income - q4_2023_net_income) / q4_2023_net_income * 100

executive_ppt_prompt = f"""
Create a 4-slide executive presentation for Q4 2024 financial results:

Slide 1 - Title:
- Title: "Q4 2024 Financial Results"
- Subtitle: "Executive Summary - Acme Corporation"
- Date: January 2025

Slide 2 - Financial Highlights:
- Title: "Q4 2024 Performance Highlights"
- Create a two-column layout:
  Left side - Key Metrics:
  • Revenue: $14.5M (+{yoy_growth:.1f}% YoY)
  • Net Income: $1.88M (+{net_income_growth:.1f}% YoY)
  • Operating Margin: 17.9% (up 2.9pp)
  • Operating Cash Flow: $2.85M

  Right side - Column chart showing quarterly revenue:
  Q1 2024: $12.5M
  Q2 2024: $13.2M
  Q3 2024: $13.8M
  Q4 2024: $14.5M

Slide 3 - Profitability Trends:
- Title: "Margin Expansion & Profitability"
- Add a line chart showing net margin % by quarter:
  Q1 2024: 11.4%
  Q2 2024: 11.8%
  Q3 2024: 12.4%
  Q4 2024: 13.0%
- Add bullet points below:
  • Consistent margin expansion throughout 2024
  • Operating leverage driving profitability
  • Cost optimization initiatives delivering results

Slide 4 - Key Takeaways:
- Title: "Key Takeaways & Outlook"
- Bullet points:
  ✓ Record Q4 revenue of $14.5M
  ✓ 17.9% YoY revenue growth
  ✓ 55% increase in net income YoY
  ✓ Strong cash generation: $2.85M operating cash flow
  ✓ Well-positioned for continued growth in 2025

Use professional corporate design:
- Dark blue (#003366) for headers
- Clean, modern layout
- Data-driven visualizations
"""

# Create the executive presentation
ppt_response, ppt_results = create_skills_message(
    client,
    executive_ppt_prompt,
    [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}],
    prefix="executive_summary_",
)

print("\n" + "=" * 60)
print_download_summary(ppt_results)

if len(ppt_results) > 0 and ppt_results[0]["success"]:
    print("\n✅ Executive presentation created successfully!")
```

## 3. Use Case 2: Portfolio Analysis Workflow {#portfolio-analysis}

Now let's shift our focus from company financials to investment portfolio analysis. In this section, we'll demonstrate how to create comprehensive portfolio analytics and investment committee presentations using the portfolio data we loaded earlier.

This workflow showcases:
- Detailed portfolio performance analysis in Excel
- Risk metrics and sector allocation visualization
- Professional investment committee presentations
- Data-driven rebalancing recommendations

We'll start by creating an Excel workbook with portfolio analytics, then generate an investment committee presentation that summarizes our findings.

### First, let's create a comprehensive portfolio analysis Excel workbook

Before we create the investment committee presentation, we need to analyze our portfolio data in detail. This Excel workbook will serve as the foundation for our investment recommendations.


```python
print("Creating portfolio analysis Excel workbook...")
print("This creates a focused 2-sheet portfolio analysis optimized for the Skills API.")
print("\n⏱️ Generation time: 1-2 minutes\n")

# Prepare portfolio data for the prompt
top_holdings = portfolio_df.nlargest(5, "market_value")
sector_allocation = portfolio_data["sector_allocation"]

portfolio_excel_prompt = f"""
Create a portfolio analysis Excel workbook with 2 sheets:

Sheet 1 - "Portfolio Overview":
Create a comprehensive holdings and performance table:

Section 1 - Holdings (top of sheet):
{portfolio_df[["ticker", "name", "shares", "current_price", "market_value", "unrealized_gain", "allocation_percent"]].head(10).to_string()}

Section 2 - Portfolio Summary:
- Total portfolio value: ${portfolio_data["total_value"]:,.2f}
- Total unrealized gain: ${portfolio_df["unrealized_gain"].sum():,.2f}
- Total Return: {portfolio_data["performance_metrics"]["total_return_percent"]:.1f}%
- YTD Return: {portfolio_data["performance_metrics"]["year_to_date_return"]:.1f}%
- Sharpe Ratio: {portfolio_data["performance_metrics"]["sharpe_ratio"]:.2f}
- Portfolio Beta: {portfolio_data["performance_metrics"]["beta"]:.2f}

Apply conditional formatting: green for gains, red for losses.
Add a bar chart showing top 5 holdings by value.

Sheet 2 - "Sector Analysis & Risk":
Create sector allocation and risk metrics:

Section 1 - Sector Allocation:
{json.dumps(sector_allocation, indent=2)}
Include a pie chart of sector allocation.

Section 2 - Key Risk Metrics:
- Portfolio Beta: {portfolio_data["performance_metrics"]["beta"]:.2f}
- Standard Deviation: {portfolio_data["performance_metrics"]["standard_deviation"]:.1f}%
- Value at Risk (95%): $62,500
- Maximum Drawdown: -12.3%
- Sharpe Ratio: {portfolio_data["performance_metrics"]["sharpe_ratio"]:.2f}

Section 3 - Rebalancing Recommendations:
- Reduce Technology from 20% to 18%
- Increase Healthcare from 8.7% to 10%
- Maintain current diversification

Apply professional formatting with clear