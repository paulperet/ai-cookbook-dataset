# Multi-Agent Recruitment Workflow: A Step-by-Step Guide

## Introduction

This guide walks you through building an automated recruitment system using specialized AI agents. The system streamlines hiring by coordinating multiple agents to extract, analyze, match, and communicate with candidates—all powered by Mistral AI.

You will learn how to:
1.  Set up the environment and download sample data.
2.  Define structured data models for consistency.
3.  Build and orchestrate six specialized agents.
4.  Run a complete workflow from resume parsing to candidate outreach.

## Prerequisites & Setup

Before you begin, ensure you have a Mistral AI API key. You can get one from the [Mistral AI console](https://console.mistral.ai/api-keys).

### 1. Install Required Libraries

Install the necessary Python package using pip.

```bash
pip install mistralai
```

### 2. Import Dependencies

Import all required libraries at the start of your script.

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

### 3. Configure API Access

Set your API key as an environment variable and initialize the Mistral client.

```python
# Replace with your actual API key
os.environ['MISTRAL_API_KEY'] = "YOUR_MISTRALAI_API_KEY"

# Initialize the client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

## Step 1: Download Sample Data

We'll use a sample job description and a set of candidate resumes to demonstrate the workflow.

### Helper Functions for Downloading

First, define helper functions to download PDF files.

```python
def download_job_description(url, output_path="job_description.pdf"):
    """Download a job description PDF from a URL."""
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {output_path}")

def download_resumes(url, local_dir="resumes"):
    """Download multiple resume PDFs from a GitHub directory."""
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

### Execute the Downloads

Now, run the functions to fetch the sample data.

```python
# Download the job description
job_desc_url = "https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/non_framework/recruitment_agent/job_description.pdf"
download_job_description(job_desc_url, "job_description.pdf")

# Download the candidate resumes
resumes_url = "https://api.github.com/repos/mistralai/cookbook/contents/mistral/agents/non_framework/recruitment_agent/resumes"
download_resumes(resumes_url, "resumes")
```

## Step 2: Define Structured Data Models

To ensure consistent data flow between agents, we define Pydantic models. These models validate and structure information like candidate profiles, job requirements, and evaluation scores.

Add the following class definitions to your script.

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

## Step 3: Build the Foundation - The Base Agent Class

All specialized agents will inherit from a common `Agent` base class. This provides a standard interface for processing data and communicating between agents.

```python
class Agent:
    def __init__(self, name: str, client: Mistral):
        self.name = name
        self.client = client

    def process(self, message):
        """Base process method - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement process method")

    def communicate(self, recipient_agent, message):
        """Send a message to another agent"""
        return recipient_agent.process(message)
```

## Step 4: Implement the Specialized Agents

Now, we'll build each of the six agents that form the core of the recruitment workflow.

### Agent 1: DocumentAgent

The `DocumentAgent` is the entry point. It uses Mistral's OCR to extract raw text from PDF resumes and job descriptions.

```python
class DocumentAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("DocumentAgent", client)

    def process(self, file_info):
        """Process a document extraction request. `file_info` should be a tuple (file_path, file_name)."""
        file_path, file_name = file_info
        return self.extract_text_from_file(file_path, file_name)

    def extract_text_from_file(self, file_path: str, file_name: str) -> str:
        """Extract text from a file using Mistral OCR."""
        try:
            # 1. Upload the file
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": file_name,
                    "content": open(file_path, "rb"),
                },
                purpose="ocr"
            )

            # 2. Get a signed URL for processing
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)

            # 3. Process the document with OCR
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )

            # 4. Concatenate the extracted text from all pages
            extracted_text = ""
            for page in ocr_response.pages:
                extracted_text += page.markdown + "\n\n"

            return extracted_text

        except Exception as e:
            print(f"Error extracting text from {file_name}: {str(e)}")
            return ""
```

### Agent 2: JobAnalysisAgent

This agent takes the raw text of a job description and uses the Mistral LLM to extract structured requirements.

```python
class JobAnalysisAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("JobAnalysisAgent", client)

    def process(self, jd_text):
        """Process job description text and return structured requirements."""
        return self.extract_job_requirements(jd_text)

    def extract_job_requirements(self, jd_text: str) -> JobRequirements:
        """Extract structured job requirements from a job description."""
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
        # Parse the LLM's structured response into our Pydantic model
        return json.loads(response.choices[0].message.content)
```

### Agent 3: ResumeAnalysisAgent

Similar to the JobAnalysisAgent, this agent parses raw resume text into a structured `CandidateProfile`.

```python
class ResumeAnalysisAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("ResumeAnalysisAgent", client)

    def process(self, resume_text):
        """Process resume text and return a structured candidate profile."""
        return self.extract_candidate_profile(resume_text)

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        """Extract structured candidate information from resume text."""
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

### Agent 4: MatchingAgent

The `MatchingAgent` is the core evaluator. It compares a `CandidateProfile` against `JobRequirements` to generate a detailed score.

```python
class MatchingAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("MatchingAgent", client)

    def process(self, data):
        """Process job requirements and candidate profile to generate a score.
        `data` should be a tuple: (job_requirements, candidate_profile, original_resume_text).
        """
        job_requirements, candidate_profile, resume_text = data
        return self.evaluate_candidate(job_requirements, candidate_profile, resume_text)

    def evaluate_candidate(self, job_requirements: JobRequirements, candidate_profile: CandidateProfile, resume_text: str) -> CandidateScore:
        """Evaluate how well a candidate matches the job requirements."""
        # Convert models to JSON for the prompt
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

        Scoring Guidelines:
        - Technical skills: 0-40 points.
        - Experience: 0-30 points.
        - Education: 0-15 points.
        - Additional qualifications: 0-15 points.
        - Total score: 0-100 points.
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

### Agent 5: EmailCommunicationAgent

This agent handles generating and sending personalized emails to candidates. **Note:** You must configure your SMTP server details (Gmail, SendGrid, etc.) for this to work.

```python
class EmailCommunicationAgent(Agent):
    def __init__(self, client: Mistral, smtp_config: Dict[str, str]):
        super().__init__("EmailCommunicationAgent", client)
        self.smtp_config = smtp_config  # Should contain: server, port, username, password

    def process(self, candidate_data):
        """Process candidate data to send an email.
        `candidate_data` should be a dict with: name, email, score, and other relevant info.
        """
        email_content = self.generate_email(candidate_data)
        self.send_email(candidate_data['email'], email_content)
        return f"Email sent to {candidate_data['name']}"

    def generate_email(self, candidate_data: Dict[str, Any]) -> str:
        """Generate a personalized email body using the LLM."""
        prompt = f"""
        Write a professional and encouraging email to a candidate named {candidate_data['name']}.
        They have applied for a Data Scientist position and have scored {candidate_data['score']}/100 in our initial screening.
        Invite them to schedule an interview using this link: https://calendly.com/company/interview.
        Keep the tone positive and concise.
        """
        response = self.client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def send_email(self, to_email: str, body: str):
        """Send an email using SMTP."""
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config['username']
        msg['To'] = to_email
        msg['Subject'] = "Interview Invitation - Data Scientist Role"

        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            print(f"Email successfully sent to {to_email}")
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
```

### Agent 6: CoordinatorAgent

The `CoordinatorAgent` orchestrates the entire workflow, passing data from one agent to the next in the correct sequence.

```python
class CoordinatorAgent(Agent):
    def __init__(self, client: Mistral):
        super().__init__("CoordinatorAgent", client)
        # Instantiate all the other agents
        self.document_agent = DocumentAgent(client)
        self.job_analysis_agent = JobAnalysisAgent(client)
        self.resume_analysis_agent = ResumeAnalysisAgent(client)
        self.matching_agent = MatchingAgent(client)
        # Initialize Email Agent with your SMTP config
        self.smtp_config = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password'  # Use an app-specific password
        }
        self.email_agent = EmailCommunicationAgent(client, self.smtp_config)

    def process(self, job_description_path: str, resume_directory: str):
        """Orchestrate the full recruitment workflow for one job and multiple resumes."""
        print("Starting recruitment workflow...")

        # 1. Process Job Description
        print("\n1. Analyzing Job Description...")
        jd_text = self.document_agent.process((job_description_path, "job_description.pdf"))
        job