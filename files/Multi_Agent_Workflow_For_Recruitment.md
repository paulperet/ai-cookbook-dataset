# Multi Agent Workflow For Recruitment

## Introduction

The Multi Agent Workflow For Recruitment is an automated system designed to help streamline the hiring process through specialized AI agents working in harmony to improve candidate evaluation, save time and resources, and improve overall hiring outcomes.

## The Problem

Today's recruitment landscape faces three critical challenges:

1. **Overwhelming Volume**: Recruiters struggle to efficiently process large numbers of applications, often missing qualified candidates.

2. **Manual Inefficiency**: Traditional resume screening is time-consuming, inconsistent, and vulnerable to bias.

3. **Poor Candidate Experience**: Slow response times and fragmented communication damage employer brand and lose top talent.

## Why This Matters

Ineffective recruitment directly impacts business outcomes through:

- **Reduced Performance**: Missing qualified candidates leads to suboptimal hires and team performance
- **Business Delays**: Extended hiring cycles postpone critical projects and initiatives
- **Higher Costs**: Inefficient processes and prolonged vacancies increase recruitment costs

## Our Solution

The Multi Agent Workflow For Recruitment addresses these challenges through a coordinated system of specialized AI agents:

1. **DocumentAgent**: Intelligently extracts and processes text from resumes and job descriptions using advanced Mistral's OCR
  
2. **JobAnalysisAgent**: Analyzes job descriptions to identify required skills, experience, and qualifications

3. **ResumeAnalysisAgent**: Parses resumes to create structured candidate profiles with key capabilities

4. **MatchingAgent**: Evaluates candidates against job requirements with nuanced understanding beyond keyword matching

5. **EmailCommunicationAgent**: Generates personalized email communications and schedules interviews with qualified candidates

6. **CoordinatorAgent**: Orchestrates the entire workflow between agents for seamless operation.

The solution uses Mistral LLM for language understanding, structured output mechanisms for consistent data extraction, and Mistral OCR for document parsing.

### Example: Data Scientist Hiring

To illustrate how the Multi Agent Workflow For Recruitment operates in practice, consider a realistic example:

HireFive needs to hire a Senior Data Scientist with machine learning expertise. The job description specifies requirements including 3+ years of experience, proficiency in Python and deep learning frameworks, and a Master's degree in a quantitative field. From a pool of candidate resumes, the workflow automatically:

- Extracts structured requirements from the job description, identifying critical skills
- Parses all the resumes, creating standardized profiles with skills, experience, and education
- Evaluates each candidate, assigning scores like "Technical Skills: 32/40" and "Experience: 25/30"
- Identifies candidates scoring above the 70-point threshold
- Automatically sends personalized interview invitations with scheduling links to these candidates

The entire process completes in minutes, providing HireFive's hiring manager with a ranked list of qualified candidates while eliminating hours of manual resume screening.

### Solution Architecture

### Installation

```python
!pip install mistralai
```

[Collecting mistralai, ..., Successfully installed eval-type-backport-0.2.2 mistralai-1.6.0]

### Imports

```python
import os
import time
import json
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from mistralai import Mistral

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
```

### Setup API Keys

```python
os.environ['MISTRAL_API_KEY'] = "YOUR MISTRALAI API KEY" # Get it from https://console.mistral.ai/api-keys
```

### Initialize Mistral API Client

```python
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

### Download Data

Here, we download the necessary data for the demonstration.

1. Job Descrition.
2. Candidate Resumes.

##### Helper functions to download Job description and candidate resumes

```python
def download_job_description(url, output_path = "job_description.pdf"):
    """
    Download job description from a given URL.
    """
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {output_path}")

def download_resumes(url, local_dir="resumes"):
    """
    Download resumes from the given URL.
    """

    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to retrieve folder contents:", response.text)
        return
    
    data = response.json()
    os.makedirs(local_dir, exist_ok=True)

    print(f"{len(data)} files available for download:")
    for file in data:
        file_name = file["name"]
        download_url = file["download_url"]

        r = requests.get(download_url)
        with open(os.path.join(local_dir, file_name), "wb") as f:
            f.write(r.content)
        print(f"Downloaded {file_name}")
```

#### Download Job Description

```python
url = "https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/non_framework/recruitment_agent/job_description.pdf"
output_path = "job_description.pdf"

download_job_description(url, output_path)
```

    Downloaded job_description.pdf

#### Download Candidate Resumes

```python
download_resumes(
    url = "https://api.github.com/repos/mistralai/cookbook/contents/mistral/agents/non_framework/recruitment_agent/resumes",
    local_dir="resumes"
)
```

    13 files available for download:
    Downloaded Resume 10_ Carlos Mendez.pdf
    Downloaded Resume 11_ Alex Patel.pdf
    Downloaded Resume 12_ Taylor Williams.pdf
    Downloaded Resume 13_ Jordan Smith.pdf
    Downloaded Resume 1_ Sarah Chen.pdf
    Downloaded Resume 2_ Michael Rodriguez.pdf
    Downloaded Resume 3_ Jennifer Park.pdf
    Downloaded Resume 4_ David Wilson.pdf
    Downloaded Resume 5_ Priya Sharma.pdf
    Downloaded Resume 6_ James Lee.pdf
    Downloaded Resume 7_ Emily Johnson.pdf
    Downloaded Resume 8_ Robert Thompson.pdf
    Downloaded Resume 9_ Lisa Wang.pdf

### Define Pydantic Models

Pydantic models provide structured data validation between agents, ensuring consistent formats for candidate profiles, job requirements, and evaluation scores while enabling seamless integration with Mistral LLM's parsing capabilities. Following are the different pydantic models we use for

- **Skill**: Represents a candidate's technical or soft skill with its proficiency level and years of experience.

- **Education**: Captures educational qualifications including degree, field of study, institution, and performance metrics.

- **Experience**: Tracks professional experience with role details, duration, utilized skills, and key accomplishments.

- **ContactDetails**: Stores candidate contact information including name, email, and optional communication channels.

- **JobRequirements**: Defines position requirements including mandatory and preferred skills, experience level, and educational qualifications.

- **CandidateProfile**: Consolidates a candidate's complete professional profile including contact details, skills, education, and work history.

- **SkillMatch**: Evaluates individual skill alignment between job requirements and candidate capabilities with confidence scores.

- **CandidateScore**: Provides comprehensive scoring across key evaluation areas with total score calculation and identified strengths/gaps.

- **CandidateResult**: Connects file information with extracted candidate data and evaluation scores for final ranking and selection.

#### Pydantic Models for structured extraction.

```python
class Skill(BaseModel):
    name: str = Field(description="Name of the skill or technology")
    level: Optional[str] = Field(description="Proficiency level (beginner, intermediate, advanced)")
    years: Optional[float] = Field(description="Years of experience with this skill")

class Education(BaseModel):
    degree: str = Field(description="Type of degree or certification obtained")
    field: str = Field(description="Field of study or specialization")
    institution: str = Field(description="Name of educational institution")
    year_completed: Optional[int] = Field(description="Year when degree was completed")
    gpa: Optional[float] = Field(description="Grade Point Average, typically on 4.0 scale")

class Experience(BaseModel):
    title: str = Field(description="Job title or position held")
    company: str = Field(description="Name of employer or organization")
    duration_years: float = Field(description="Duration of employment in years")
    skills_used: List[str] = Field(description="Skills utilized in this role")
    achievements: List[str] = Field(description="Key accomplishments or responsibilities")
    relevance_score: Optional[float] = Field(description="Relevance to current job opening (0-10 scale)")

class ContactDetails(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Primary email address for contact")
    phone: Optional[str] = Field(description="Phone number with country code if applicable")
    location: Optional[str] = Field(description="Current city and country/state")
    linkedin: Optional[str] = Field(description="LinkedIn profile URL")
    website: Optional[str] = Field(description="Personal or portfolio website URL")

class JobRequirements(BaseModel):
    required_skills: List[Skill] = Field(description="Skills that are mandatory for the position")
    preferred_skills: List[Skill] = Field(description="Skills that are desired but not required")
    min_experience_years: float = Field(description="Minimum years of experience required")
    required_education: List[Education] = Field(description="Mandatory educational qualifications")
    preferred_domains: List[str] = Field(description="Industry domains or sectors preferred for experience")

class CandidateProfile(BaseModel):
    contact_details: ContactDetails = Field(description="Candidate's personal and contact information")
    skills: List[Skill] = Field(description="Technical and soft skills possessed by the candidate")
    education: List[Education] = Field(description="Educational background and qualifications")
    experience: List[Experience] = Field(description="Professional work history and experience")

class SkillMatch(BaseModel):
    skill_name: str = Field(description="Name of the skill being evaluated")
    present: bool = Field(description="Whether the candidate possesses this skill")
    match_level: float = Field(description="How well the candidate's skill matches the requirement (0-10 scale)")
    confidence: float = Field(description="Confidence in the skill evaluation (0-1 scale)")
    notes: str = Field(description="Additional context about the skill match assessment")

class CandidateScore(BaseModel):
    technical_skills_score: float = Field(description="Assessment of technical capabilities (0-40 points)")
    experience_score: float = Field(description="Evaluation of relevant work experience (0-30 points)")
    education_score: float = Field(description="Rating of educational qualifications (0-15 points)")
    additional_score: float = Field(description="Score for other relevant factors (0-15 points)")
    total_score: float = Field(description="Aggregate candidate evaluation score (0-100)")
    key_strengths: List[str] = Field(description="Primary candidate advantages for this role")
    key_gaps: List[str] = Field(description="Areas where the candidate lacks desired qualifications")
    confidence: float = Field(description="Overall confidence in the evaluation accuracy (0-1 scale)")
    notes: str = Field(description="Supplementary observations about the candidate fit")

class CandidateResult(BaseModel):
    file_name: str = Field(description="Name of the source resume file")
    contact_details: ContactDetails = Field(description="Candidate's contact information")
    candidate_profile: CandidateProfile = Field(description="Complete extracted candidate profile")
    score: CandidateScore = Field(description="Detailed evaluation scores and assessment")
```

### Base Agent Class

The `Agent` class serves as the foundation for all specialized agents, providing a standardized interface for processing and communicating between agents in the recruitment workflow.

Each agent implements the common `process()` method while inheriting identity management and communication capabilities.

```python
class Agent:
    def __init__(self, name: str, client: Mistral):
        self.name = name
        self.client = client

    def process(self, message):
        """Base process method - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement process method")

    def communicate(self, recipient_agent, message):
        """Send message to another agent"""
        return recipient_agent.process(message)
```

### DocumentAgent: Handles document extraction and OCR

The `DocumentAgent` handles document processing by extracting structured text from various files using Mistral's OCR capabilities. It transforms complex resume PDFs and job descriptions into text, serving as the initial data gateway for the entire recruitment workflow.

```python
class DocumentAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("DocumentAgent", client)

    def process(self, file_info):
        """Process document extraction request"""
        file_path, file_name = file_info
        return self.extract_text_from_file(file_path, file_name)

    def extract_text_from_file(self, file_path: str, file_name: str) -> str:
        """Extract text from a file using Mistral OCR"""
        try:
            # Upload the file
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": file_name,
                    "content": open(file_path, "rb"),
                },
                purpose="ocr"
            )

            # Get signed URL
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)

            # Process with OCR
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )

            # Extract and return the text
            extracted_text = ""
            for page in ocr_response.pages:
                extracted_text += page.markdown + "\n\n"

            return extracted_text

        except Exception as e:
            print(f"Error extracting text from {file_name}: {str(e)}")
            return ""
```

### JobAnalysisAgent: Handles job requirement extraction and analysis

The JobAnalysisAgent extracts structured job requirements from plain text job descriptions using Mistral LLM. It transforms unstructured job postings into organized data models capturing required skills, experience levels, and educational qualifications needed for candidate matching.

```python
class JobAnalysisAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("JobAnalysisAgent", client)

    def process(self, jd_text):
        """Process job description text"""
        return self.extract_job_requirements(jd_text)

    def extract_job_requirements(self, jd_text: str) -> JobRequirements:
        """Extract structured job requirements from a job description"""
        prompt = f"""
        Extract the key job requirements from the following job description.
        Focus on required skills, preferred skills, experience requirements, and education requirements.

        Job Description:
        {jd_text}
        """

        response = self.client.chat.parse(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "Extract structured job requirements from the job description."},
                {"role": "user", "content": prompt}
            ],
            response_format=JobRequirements,
            temperature=0
        )

        return json.loads(response.choices[0].message.content)
```

### ResumeAnalysisAgent: Handles resume parsing and profile extraction

The ResumeAnalysisAgent transforms raw resume text into structured candidate profiles using Mistral LLM's parsing capabilities. It extracts and organizes key information including contact details, skills, education history, and professional experience into standardized data structures for consistent evaluation.

```python
class ResumeAnalysisAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("ResumeAnalysisAgent", client)

    def process(self, resume_text):
        """Process resume text"""
        return self.extract_candidate_profile(resume_text)

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        """Extract structured candidate information from resume text"""
        prompt = f"""
        Extract the candidate's contact details, skills, education, and experience from the following resume.
        Be thorough and include all relevant information.

        Resume:
        {resume_text}
        """

        response = self.client.chat.parse(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "Extract structured candidate information from the resume."},
                {"role": "user", "content": prompt}
            ],
            response_format=CandidateProfile,
            temperature=0
        )

        return json.loads(response.choices[0].message.content)
```

### MatchingAgent: Evaluates candidate fit against job requirements

The `MatchingAgent` evaluates candidate profiles against job requirements to generate comprehensive scoring across technical skills, experience, education and additional qualifications. It employs Mistral LLM to assess the quality and relevance of candidate attributes beyond simple keyword matching, producing a detailed evaluation with confidence metrics and identified strengths and gaps.

```python
class MatchingAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("MatchingAgent", client)

    def process(self, data):
        """Process job requirements and candidate profile to generate score"""
        job_requirements, candidate_profile, resume_text = data
        return self.evaluate_candidate(job_requirements, candidate_profile, resume_text)

    def evaluate_candidate(self, job_requirements: JobRequirements, candidate_profile: CandidateProfile, resume_text: str) -> CandidateScore:
        """Evaluate how well a candidate matches the job requirements"""
        # Convert to JSON for inclusion in the prompt
        job_req_json = json.dumps(job_requirements, indent=2)
        candidate_json = json.dumps(candidate_profile, indent=2)

        prompt = f"""
        Evaluate how well the candidate matches the job requirements.

        Job Requirements:
        {job_req_json}

        Candidate Profile:
        {candidate_json}

        Provide a detailed scoring breakdown, highlighting strengths and gaps.
        Assess the quality and relevance of the candidate's experience, not just keyword matches.
        Include confidence levels for your assessment.

        Technical skills should be scored out of 40 points.
        Experience should be scored out of 30 points.
        Education should be scored out of 15 points.
        Additional qualifications should be scored out of 15 points.
        The total score should be out of 100 points.
        """

        response = self.client.chat.parse(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "Evaluate the candidate's match to the job requirements with detailed scoring."},
                {"role": "user", "content": prompt}
            ],
            response_format=CandidateScore,
            temperature=0.2  # Slight randomness for nuanced evaluation
        )

        return json.loads(response.choices[0].message.content)
```

# EmailCommunicationAgent: Handles email generation and sending

The `EmailCommunicationAgent` generates personalized email communications to candidates and sends them through SMTP integration. It crafts contextually relevant messages based on candidate qualifications and scheduling information, managing the critical final step of candidate engagement in the recruitment workflow.

```python
class EmailCommunicationAgent(Agent):
    def