# Multi-Agent Earnings Call Analysis System (MAECAS): A Step-by-Step Guide

This guide walks you through building a Multi-Agent Earnings Call Analysis System (MAECAS). This system transforms lengthy, dense earnings call transcripts into structured, actionable insights using a coordinated team of specialized AI agents.

## Overview

Quarterly earnings calls are critical for understanding a company's performance and strategy, but manually analyzing 20+ page transcripts is time-consuming and inconsistent. MAECAS automates this by deploying specialized agents—each focused on a specific analytical dimension—that work together under a central orchestrator.

### Key Capabilities
- **Extracts Insights**: Pulls financial metrics, strategic initiatives, sentiment, risks, and competitive intelligence from transcripts.
- **Identifies Trends**: Analyzes patterns across multiple quarters.
- **Answers Queries**: Responds to specific questions using the stored knowledge base.
- **Generates Reports**: Produces structured, comprehensive analysis reports.

### Specialized Analysis Agents
The system uses six agent types:
1.  **Financial Agent**: Extracts revenue, profit margins, growth metrics.
2.  **Strategic Agent**: Identifies product roadmaps, market expansions, partnerships.
3.  **Sentiment Agent**: Evaluates management's confidence and tone.
4.  **Risk Agent**: Detects supply chain, market, and regulatory challenges.
5.  **Competitor Agent**: Tracks competitive positioning and market share discussions.
6.  **Temporal Agent**: Analyzes trends and evolving priorities across quarters.

## Prerequisites & Setup

### 1. Install Required Library
You need the `mistralai` Python package to interact with Mistral AI's models.

```bash
pip install mistralai
```

### 2. Import Libraries and Set Up API Key

```python
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional, Union
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from mistralai import Mistral
from IPython.display import display, Markdown

# Set your Mistral AI API key
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRALAI API KEY>'  # Get from https://console.mistral.ai/api-keys/
api_key = os.environ.get('MISTRAL_API_KEY')

# Initialize the Mistral client
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
```

### 3. Download Sample Data
For this tutorial, we'll use NVIDIA's quarterly earnings call transcripts from 2025.

```python
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q1.pdf" -O "nvidia_earnings_2025_Q1.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q2.pdf" -O "nvidia_earnings_2025_Q2.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q3.pdf" -O "nvidia_earnings_2025_Q3.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q4.pdf" -O "nvidia_earnings_2025_Q4.pdf"
```

### 4. Define Model Constants
We'll use different Mistral AI models optimized for specific tasks.

```python
DEFAULT_MODEL = "mistral-small-latest"   # For general analysis
STRUCTURED_MODEL = "mistral-large-latest" # For structured JSON output
OCR_MODEL = "mistral-ocr-latest"         # For parsing PDF transcripts
```

## Step 1: Define Data Models with Pydantic

We use Pydantic models to enforce a consistent schema for the insights extracted by each agent. This ensures reliable storage, retrieval, and comparison of data.

### Financial Insights Model
Captures quantitative performance metrics like revenue and profit margins.

```python
class FinancialInsight(BaseModel):
    """Financial insights extracted from transcript"""
    metric_name: str = Field(description="Name of the financial metric")
    value: Optional[str] = Field(description="Numerical or textual value of the metric")
    context: str = Field(description="Surrounding context for the metric")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to")
    confidence: float = Field(description="Confidence score for the insight with limits ge=0.0, le=1.0")

class FinancialInsightsResponse(BaseModel):
    """Wrapper for list of financial insights"""
    insights: List[FinancialInsight] = Field(description="Collection of financial insights")
```

### Strategic Insights Model
Tracks qualitative business initiatives and long-term plans.

```python
class StrategicInsight(BaseModel):
    """Strategic insights about business direction"""
    initiative: str = Field(description="Name of the strategic initiative")
    description: str = Field(description="Details about the strategic initiative")
    timeframe: Optional[str] = Field(description="Expected timeline for implementation")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to")
    importance: int = Field(description="Importance rating with limits ge=1, le=5")

class StrategicInsightsResponse(BaseModel):
    """Wrapper for list of strategic insights"""
    insights: List[StrategicInsight] = Field(description="Collection of strategic insights")
```

### Sentiment Insights Model
Analyzes the tone and confidence expressed by management.

```python
class SentimentInsight(BaseModel):
    """Insights about management sentiment"""
    topic: str = Field(description="Subject matter being discussed")
    sentiment: Literal["very negative", "negative", "neutral", "positive", "very positive"] = Field(description="Tone expressed by management")
    evidence: str = Field(description="Quote or context supporting the sentiment analysis")
    speaker: str = Field(description="Person who expressed the sentiment")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to")

class SentimentInsightsResponse(BaseModel):
    """Wrapper for list of sentiment insights"""
    insights: List[SentimentInsight] = Field(description="Collection of sentiment insights")
```

### Risk Insights Model
Identifies potential challenges and their severity.

```python
class RiskInsight(BaseModel):
    """Identified risks or challenges"""
    risk_factor: str = Field(description="Name or type of risk identified")
    description: str = Field(description="Details about the risk")
    potential_impact: str = Field(description="Possible consequences of the risk")
    mitigation_mentioned: Optional[str] = Field(description="Strategies to address the risk")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to")
    severity: int = Field(description="Severity rating with limits ge=1, le=5")

class RiskInsightsResponse(BaseModel):
    """Wrapper for list of risk insights"""
    insights: List[RiskInsight] = Field(description="Collection of risk insights")
```

### Competitor Insights Model
Tracks discussions about market competition.

```python
class CompetitorInsight(BaseModel):
    """Insights about competitive positioning"""
    competitor: Optional[str] = Field(description="Name of the competitor company")
    market_segment: str = Field(description="Specific market area being discussed")
    positioning: str = Field(description="Competitive stance or market position")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to")
    mentioned_by: str = Field(description="Person who mentioned the competitive information")

class CompetitorInsightsResponse(BaseModel):
    """Wrapper for list of competitor insights"""
    insights: List[CompetitorInsight] = Field(description="Collection of competitor insights")
```

### Temporal Insights Model
Identifies patterns and trends across multiple quarters.

```python
class TemporalInsight(BaseModel):
    """Insights about trends across quarters"""
    trend_type: Literal["growth", "decline", "stable", "volatile", "emerging", "fading"] = Field(description="Direction or pattern of the trend")
    topic: str = Field(description="Subject matter of the trend")
    description: str = Field(description="Explanation of the trend's significance")
    quarters_observed: List[Literal["Q1", "Q2", "Q3", "Q4"]] = Field(description="Quarters where the trend appears")
    supporting_evidence: str = Field(description="Data or quotes supporting the trend identification")

class TemporalInsightsResponse(BaseModel):
    """Wrapper for list of temporal insights"""
    insights: List[TemporalInsight] = Field(description="Collection of temporal insights")
```

### Query Analysis Model
Analyzes a user's question to determine which agents and quarters are needed to answer it.

```python
class QueryAnalysis(BaseModel):
    """Analysis of user query to determine required components"""
    quarters: List[str] = Field(description="List of quarters to analyze")
    agent_types: List[str] = Field(description="List of agent types to use")
    temporal_analysis_required: bool = Field(description="Whether temporal analysis across quarters is needed")
    query_intent: str = Field(description="Brief description of user's intent")
```

### Report Section Model
Structures the content for the final generated report.

```python
class ReportSection(BaseModel):
    """Section of the final report"""
    title: str = Field(description="Heading for the report section")
    content: str = Field(description="Main text content of the section")
    subsections: Optional[List["ReportSection"]] = Field(description="Nested sections within this section.")
```

## Step 2: Build the PDF Parser with Caching

The `PDFParser` class uses Mistral's OCR model to extract text from PDF transcripts. It implements a file-based cache to avoid re-processing the same document.

```python
class PDFParser:
    """Parse a transcript PDF file and extract text from all pages using Mistral OCR."""

    CACHE_DIR = Path("transcript_cache")

    @staticmethod
    def _ensure_cache_dir():
        """Make sure cache directory exists"""
        PDFParser.CACHE_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _get_cache_path(file_path: str) -> Path:
        """Get the path for a cached transcript file"""
        # Create a hash of the file path to use as the cache filename
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return PDFParser.CACHE_DIR / f"{file_hash}.txt"

    @staticmethod
    def read_transcript(file_path: str, mistral_client: Mistral) -> str:
        """Extract text from PDF transcript using Mistral OCR"""
        print(f"Processing PDF file: {file_path}")

        uploaded_pdf = mistral_client.files.upload(
            file={
                "file_name": file_path,
                "content": open(file_path, "rb"),
            },
            purpose="ocr"
        )

        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_response = mistral_client.ocr.process(
            model=OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )

        text = "\n".join([x.markdown for x in (ocr_response.pages)])
        return text

    @staticmethod
    def get_transcript_by_quarter(company: str, quarter: str, year: str, mistral_client: Mistral) -> str:
        """Get the transcript for a specific quarter"""
        company_lower = company.lower()
        file_path = f"{company_lower}_earnings_{year}_{quarter}.pdf"

        PDFParser._ensure_cache_dir()
        cache_path = PDFParser._get_cache_path(file_path)

        # Check if transcript is in cache
        if cache_path.exists():
            print(f"Using cached transcript for {company} {year} {quarter}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            try:
                print(f"Parsing transcript for {company} {year} {quarter}")
                transcript = PDFParser.read_transcript(file_path, mistral_client)

                # Store in cache for future use
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

                print(f"Cached transcript for {company} {year} {quarter}")
                return transcript
            except Exception as e:
                print(f"Error processing transcript: {str(e)}")
                raise
```

## Step 3: Create the Centralized Insights Store

The `InsightsStore` class acts as a persistent JSON database for all insights extracted by the agents. It organizes data by type and quarter for efficient retrieval.

```python
class InsightsStore:
    """Centralized storage for insights across all quarters and analysis types"""

    def __init__(self, company: str, year: str):
        self.company = company.lower()
        self.year = year
        self.db_path = Path(f"{self.company}_{self.year}_insights.json")
        self.insights = self._load_insights()

    def _load_insights(self) -> Dict:
        """Load insights from database file or initialize if not exists"""
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "financial": {},
                "strategic": {},
                "sentiment": {},
                "risk": {},
                "competitor": {},
                "temporal": {}
            }

    def save_insights(self):
        """Save insights to database file"""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.insights, f, indent=2)

    def add_insights(self, insight_type: str, quarter: str, insights: List):
        """Add insights for a specific type and quarter"""
        if quarter not in self.insights[insight_type]:
            self.insights[insight_type][quarter] = []

        # Convert insights to dictionaries for storage
        insight_dicts = []
        for insight in insights:
            if hasattr(insight, "dict"):
                insight_dicts.append(insight.dict())
            elif isinstance(insight, dict):
                insight_dicts.append(insight)
            else:
                insight_dicts.append(str(insight))

        self.insights[insight_type][quarter].extend(insight_dicts)
        self.save_insights()

    def get_insights(self, insight_type: str, quarter: Optional[str] = None) -> List:
        """Retrieve insights by type and optionally by quarter"""
        if quarter:
            return self.insights.get(insight_type, {}).get(quarter, [])
        else:
            # Return all insights for this type across all quarters
            all_insights = []
            for quarter_insights in self.insights.get(insight_type, {}).values():
                all_insights.extend(quarter_insights)
            return all_insights
```

## Next Steps

You have now set up the foundational components of the MAECAS:
1.  **Data Models**: Structured schemas for six types of insights.
2.  **PDF Parser**: A tool to extract and cache text from transcript PDFs.
3.  **Insights Store**: A persistent database to hold extracted insights.

In the next part of this guide, you will implement the specialized analysis agents, the central orchestrator that manages them, and the query interface that brings the entire system together.