# Multi-Agent Earnings Call Analysis System (MAECAS)

Companies earnings calls provide critical insights into a company's performance, strategy, and future outlook. However, these transcripts are lengthy, dense, and cover diverse topics - making it challenging to extract targeted insights efficiently.

## The Problem

Quarterly earnings calls provide critical insights into company performance, strategies, and outlook, but extracting meaningful analysis presents significant challenges:

- Earnings call transcripts are lengthy and dense, often spanning 20+ pages of complex financial discussions
Key insights are scattered throughout the text without clear organization.
- Different stakeholders need different types of information (financial metrics, strategic initiatives, risk factors).
- Cross-quarter analysis requires manually tracking evolving narratives across multiple calls.
- Traditional manual analysis is time-consuming, inconsistent, and prone to missing important details.

## Why This Matters

For investors, analysts, and business leaders, comprehensive earnings call analysis delivers significant value:

- Time Efficiency: Reduce analysis time from days to minutes
- Decision Support: Provide structured insights for investment and strategic decisions
- Comprehensive Coverage: Ensure no important insights are missed
- Consistent Analysis: Apply the same analytical rigor to every transcript
- Trend Detection: Identify patterns across quarters that might otherwise go unnoticed

## Our Solution

The Earnings Call Analysis Orchestrator transforms how earnings calls are processed through a multi-agent workflow that:

1. Extracts insights from quarterly transcripts using specialized analysis agents
2. Delivers both comprehensive reports and targeted query responses
3. Identifies trends and patterns across quarters
4. Maintains a structured knowledge base of earnings insights

## Specialized Analysis Agents

Our system employs specialized agents working in coordination to deliver comprehensive analysis:

**Financial Agent:** Extracts revenue figures, profit margins, growth metrics, and other quantifiable performance indicators.

**Strategic Agent:** Identifies product roadmaps, market expansions, partnerships, and long-term vision initiatives.

**Sentiment Agent:** Evaluates management's confidence, tone, and enthusiasm across different business segments.

**Risk Agent:** Detects supply chain, market, regulatory challenges, and assesses their severity and mitigation plans.

**Competitor Agent:** Tracks competitive positioning, market share discussions, and differentiation strategies.

**Temporal Agent:** Analyzes trends across quarters to identify business trajectory and evolving priorities.

## Workflow Orchestration

The orchestrator serves as the central coordinator that:

- Efficiently processes and caches transcript text using advanced OCR
- Activates specialized agents based on analysis needs
Stores structured insights in a centralized knowledge base
- Generates comprehensive reports with executive summaries, sectional analyses, and outlook
- Answers specific queries by leveraging relevant insights across quarters

## Dataset

For demonstration purposes, we use NVIDIA's quarterly earnings call transcripts from 2025:

- Q1 2025 Earnings Call Transcript
- Q2 2025 Earnings Call Transcript
- Q3 2025 Earnings Call Transcript
- Q4 2025 Earnings Call Transcript

These transcripts contain discussions of financial results, strategic initiatives, market conditions, and forward-looking statements by NVIDIA's management team and their interactions with financial analysts.

## Mistral AI Models

For our implementation, we use Mistral AI's LLMs:

`mistral-small-latest`: Used for general analysis and response generation.

`mistral-large-latest`: Used for structured output generation.

`mistral-ocr-latest`: Used for PDF transcript extraction and processing.

This modular approach enables both in-depth report generation and targeted question answering while maintaining efficiency through selective agent activation and insights reuse.

## Solution Architecture

### Installation

We need `mistralai` for LLM usage.


```python
!pip install mistralai
```

    [Collecting mistralai, ..., Successfully installed eval-type-backport-0.2.2 mistralai-1.6.0 typing-inspection-0.4.0]


## Imports


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
```

### Setup API Keys

Here we setup MistralAI API key.


```python
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRALAI API KEY>'  # Get your API key from https://console.mistral.ai/api-keys/
api_key = os.environ.get('MISTRAL_API_KEY')
```

### Initialize Mistral client

Here we initialise Mistral client.


```python
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
```

### Download Data

We will use NVIDIA's quarterly earnings call transcripts from 2025:

- Q1 2025 Earnings Call Transcript
- Q2 2025 Earnings Call Transcript
- Q3 2025 Earnings Call Transcript
- Q4 2025 Earnings Call Transcript

These transcripts contain discussions of financial results, strategic initiatives, market conditions, and forward-looking statements by NVIDIA's management team and their interactions with financial analysts.




```python
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q1.pdf" -O "nvidia_earnings_2025_Q1.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q2.pdf" -O "nvidia_earnings_2025_Q2.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q3.pdf" -O "nvidia_earnings_2025_Q3.pdf"
!wget "https://github.com/mistralai/cookbook/blob/main/mistral/agents/non_framework/earnings_calls/data/nvidia_earnings_2025_Q4.pdf" -O "nvidia_earnings_2025_Q4.pdf"
```

    [--2025-04-10 19:13:26--  https://github.com/mistralai/cookbook/blob/main/mistral/agents/earnings_calls/data/nvidia_earnings_2025_Q1.pdf, ..., 2025-04-10 19:13:30 (1.51 MB/s) - ‘nvidia_earnings_2025_Q4.pdf’ saved [213563]]


### Initiate Models

1. DEFAULT_MODEL - For General Analysis

2. STRUCTURED_MODEL - For Structured Outputs

3. OCR_MODEL - For parsing the earnings call document.


```python
DEFAULT_MODEL = "mistral-small-latest"
STRUCTURED_MODEL = "mistral-large-latest"
OCR_MODEL = "mistral-ocr-latest"
```

### Data Models

The solution uses specialized Pydantic models to structure and extract insights:

#### Core Analysis Models
- **FinancialInsight**: Captures metrics, values, and confidence scores for financial performance
- **StrategicInsight**: Represents initiatives, descriptions, timeframes, and importance ratings
- **SentimentInsight**: Tracks topic sentiment, evidence, and speaker attributions
- **RiskInsight**: Documents risks, impacts, mitigations, and severity scores
- **CompetitorInsight**: Records market segments, positioning, and competitive dynamics
- **TemporalInsight**: Identifies trends, patterns, and supporting evidence across quarters

#### Workflow Models
- **QueryAnalysis**: Determines required quarters, agent types, and analysis dimensions from user queries
- **ReportSection**: Structures report content with title, body, and optional subsections

#### Response Wrappers
- Each analysis model has a corresponding response wrapper (e.g., FinancialInsightsResponse) that packages insights into structured formats compatible with the Mistral API parsing capabilities


The models use Python's Literal types for categorized fields (such as sentiment levels or trend types) to enforce strict validation and ensure consistent terminology, enabling reliable cross-quarter comparisons while providing consistent knowledge extraction, storage, and retrieval across multiple analysis dimensions for both comprehensive reports and targeted queries.

#### Financial Insight


```python
class FinancialInsight(BaseModel):
    """Financial insights extracted from transcript"""
    metric_name: str = Field(description="Name of the financial metric")
    value: Optional[str] = Field(description="Numerical or textual value of the metric")
    context: str = Field(description="Surrounding context for the metric")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to (e.g., Q1, Q2)")
    confidence: float = Field(description="Confidence score for the insight with limits ge=0.0, le=1.0")

class FinancialInsightsResponse(BaseModel):
    """Wrapper for list of financial insights"""
    insights: List[FinancialInsight] = Field(description="Collection of financial insights")
```

#### Strategic Insight


```python
class StrategicInsight(BaseModel):
    """Strategic insights about business direction"""
    initiative: str = Field(description="Name of the strategic initiative")
    description: str = Field(description="Details about the strategic initiative")
    timeframe: Optional[str] = Field(description="Expected timeline for implementation")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to (e.g., Q1, Q2, Q3, Q4)")
    importance: int = Field(description="Importance rating with limits ge=1, le=5")

class StrategicInsightsResponse(BaseModel):
    """Wrapper for list of strategic insights"""
    insights: List[StrategicInsight] = Field(description="Collection of strategic insights")
```

#### Sentiment Insight


```python
class SentimentInsight(BaseModel):
    """Insights about management sentiment"""
    topic: str = Field(description="Subject matter being discussed")
    sentiment: Literal["very negative", "negative", "neutral", "positive", "very positive"] = Field(description="Tone expressed by management")
    evidence: str = Field(description="Quote or context supporting the sentiment analysis")
    speaker: str = Field(description="Person who expressed the sentiment")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to (e.g., Q1, Q2, Q3, Q4)")

class SentimentInsightsResponse(BaseModel):
    """Wrapper for list of sentiment insights"""
    insights: List[SentimentInsight] = Field(description="Collection of sentiment insights")
```

#### Risk Insight


```python
class RiskInsight(BaseModel):
    """Identified risks or challenges"""
    risk_factor: str = Field(description="Name or type of risk identified")
    description: str = Field(description="Details about the risk")
    potential_impact: str = Field(description="Possible consequences of the risk")
    mitigation_mentioned: Optional[str] = Field(description="Strategies to address the risk")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to (e.g., Q1, Q2, Q3, Q4)")
    severity: int = Field(description="Severity rating with limits ge=1, le=5")

class RiskInsightsResponse(BaseModel):
    """Wrapper for list of risk insights"""
    insights: List[RiskInsight] = Field(description="Collection of risk insights")
```

#### Competitor Insight


```python
class CompetitorInsight(BaseModel):
    """Insights about competitive positioning"""
    competitor: Optional[str] = Field(description="Name of the competitor company")
    market_segment: str = Field(description="Specific market area being discussed")
    positioning: str = Field(description="Competitive stance or market position")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter the insight relates to (e.g., Q1, Q2, Q3, Q4)")
    mentioned_by: str = Field(description="Person who mentioned the competitive information")

class CompetitorInsightsResponse(BaseModel):
    """Wrapper for list of competitor insights"""
    insights: List[CompetitorInsight] = Field(description="Collection of competitor insights")
```

#### Temporal Insight


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

#### Query Analysis


```python
class QueryAnalysis(BaseModel):
    """Analysis of user query to determine required components"""
    quarters: List[str] = Field(description="List of quarters to analyze")
    agent_types: List[str] = Field(description="List of agent types to use")
    temporal_analysis_required: bool = Field(description="Whether temporal analysis across quarters is needed")
    query_intent: str = Field(description="Brief description of user's intent")
```

#### Report Section


```python
class ReportSection(BaseModel):
    """Section of the final report"""
    title: str = Field(description="Heading for the report section")
    content: str = Field(description="Main text content of the section")
    subsections: Optional[List["ReportSection"]] = Field(description="Nested sections within this section.")
```

### PDF Parser

Our PDF parser uses Mistral's OCR capabilities to extract high-quality text from earnings call transcripts while implementing a file-based caching system to improve performance. This approach enables accurate text extraction with minimal processing overhead for repeated analyses.


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

### Insights Storage
The system includes a centralized InsightsStore component that:

- Maintains a persistent JSON database of all extracted insights
- Organizes insights by type (financial, strategic, etc.) and quarter
- Provides efficient retrieval for both report generation and query answering
- Eliminates redundant processing by caching analysis results


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
                insight_dicts