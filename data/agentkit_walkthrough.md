# Build, Deploy, and Optimize Agentic Workflows with AgentKit

## Introduction

AgentKit is a comprehensive suite of tools designed to help developers and enterprises build, deploy, and optimize AI agents. It consists of three interconnected components:

*   **Agent Builder:** A visual canvas for building and iterating on agent workflows.
*   **ChatKit:** Tools for easily embedding chat-based workflows into your applications.
*   **Evals:** A framework for evaluating and improving the performance of LLM-powered applications.

This guide provides an end-to-end tutorial on using AgentKit. You will build a multi-agent application, deploy it as a chat interface, and then optimize its performance using automated evaluation tools.

**What You'll Build:** A career accelerator app. Users can upload their resume and describe their dream job. The app will analyze their skills, identify gaps, and recommend relevant online courses.

**Steps:**
1.  **Build** a multi-agent workflow in Agent Builder.
2.  **Deploy** a front-end chat app using the ChatKit starter template.
3.  **Optimize** the workflow's performance using Evals for prompt optimization and trace grading.

---

## Step 1: Build the Multi-Agent Workflow with Agent Builder

Agent Builder provides a visual interface to design agentic workflows by connecting nodes. We will create a workflow with three sequential agents.

### Prerequisites & Setup

Ensure you have access to the OpenAI Platform and Agent Builder.

### 1.1 Create the Resume Extraction Agent

This agent parses an uploaded resume to extract skills and professional experiences.

1.  **Add an Agent Node:** Drag an "Agent" node onto the canvas.
2.  **Configure the Agent:**
    *   **Model:** Select `gpt-5`.
    *   **Reasoning Effort:** Set to `minimal`.
    *   **Output Format:** Select `JSON` and provide a schema to enforce a structured response. (You can find an example schema [here](https://cdn.openai.com/cookbook/agent_walkthrough/Skills_schema.json)).
3.  **Write the System Prompt:**
    ```text
    Extract and summarize information from the input resume, organizing your output by category and providing context where available.
    - Analyze the provided input to identify skills and professional experiences.
    - For each skill or experience, extract the supporting context or evidence from the text (e.g., for the skill of Python, context might be “used Python in data analysis for three years at [Company]”).
    - Continue reviewing the text until all skills and experiences are extracted.
    ```

### 1.2 Create the Career Analysis Agent

This agent compares the user's current skills (from Agent 1) against their career goal to identify skill gaps.

1.  **Add a Second Agent Node:** Connect it to receive output from the first agent.
2.  **Configure the Agent:**
    *   **Model:** Select `gpt-5`.
    *   **Reasoning Effort:** Set to `low`.
3.  **Write the System Prompt:** Use context variables (enclosed in `{{ }}`) to pull data from previous nodes.
    ```text
    Your role is to analyze skill and knowledge gaps for an individual to progress to a desired professional or career goal.

    You will receive a list of the already-obtained skills and experiences of an individual, as well as a description of the goal. First, understand the goal and analyze the critical skills or knowledge areas required for achieving the goal. Then, compare the requirements to what the individual already possesses.

    Return a list of the top 3-5 skills that the individual does not possess, but are important for their professional goal. Along with each skill, include a brief description.

    Individual's expressed goal:
    {{workflow.input_as_text}}

    Already-obtained skills and experiences:
    {{input.output_text}}
    ```

### 1.3 Create the Course Recommendation Agent

This agent uses web search to find online courses that address the identified skill gaps.

1.  **Add a Third Agent Node:** Connect it to receive output from the second agent.
2.  **Configure the Agent:**
    *   **Model:** Select `gpt-5`.
    *   **Reasoning Effort:** Set to `minimal`.
    *   **Tools:** Enable **Web Search**.
3.  **Write the System Prompt:**
    ```text
    Your job is to identify and recommend online training courses that help develop one or more of the skills identified. Given the list of required skills and descriptions below, return a list of 3-5 online courses along with course details.

    Skills: {{input.output_text}}
    ```

### 1.4 Test and Publish the Workflow

1.  **Preview:** Use the "Preview" panel in Agent Builder. Upload a sample resume, enter a dream job description (e.g., "school superintendent"), and submit. Watch the workflow execute step-by-step.
2.  **Publish:** Once satisfied, click "Publish". This creates a versioned copy of your workflow and generates a unique **Workflow ID**. Save this ID for the next step.
3.  **Optional - View Code:** You can view the equivalent Agents SDK code (Python/JS) in the "Agents SDK" tab if you wish to run the workflow in a custom environment.

---

## Step 2: Deploy the Chat App with ChatKit

We'll use the ChatKit starter template to create a front-end chat interface that connects to your published workflow.

### 2.1 Set Up the Starter App

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/openai/openai-chatkit-starter-app
    cd openai-chatkit-starter-app
    ```
2.  **Install Dependencies:**
    ```bash
    npm install
    ```
3.  **Configure Environment:** Create a `.env.local` file in the project root and add your Workflow ID.
    ```bash
    NEXT_PUBLIC_CHATKIT_WORKFLOW_ID=your_workflow_id_here
    ```
4.  **Run the Development Server:**
    ```bash
    npm run dev
    ```
5.  **Test:** Open `http://localhost:3000` in your browser. Your workflow is now embedded in a functional chat interface.

---

## Step 3: Iterate and Improve the User Experience

AgentKit allows for rapid iteration. Let's enhance the app's look, feel, and functionality.

### 3.1 Customize the Chat Theme

1.  Visit [ChatKit Studio Playground](https://chatkit.studio/playground).
2.  Visually customize the theme (colors, fonts, etc.).
3.  Click the `</>` icon to copy the generated `theme` configuration object.
4.  In your starter app, replace the default `theme` object in `lib/config.ts` with your custom one.
5.  **Update Text Copy:** In the same `lib/config.ts` file, modify the greeting and placeholder to match your app's purpose.
    ```typescript
    export const GREETING = "Upload your resume, and tell me the job you're looking to get!";
    export const PLACEHOLDER_INPUT = "Describe your dream job, and don't forget to attach your resume!";
    ```

### 3.2 Design a Custom Output Widget

To display course recommendations better, we'll create a custom widget.

1.  Visit the [ChatKit Widget Builder](https://widgets.chatkit.studio/).
2.  Describe the desired output (e.g., "a clean list of courses with name, provider, description, and URL, followed by a summary").
3.  The Builder will generate starter widget code. Refine the design using the live preview.
4.  Download the finalized `.widget` file.

### 3.3 Update the Workflow to Use the Widget

1.  **Back in Agent Builder,** edit the **Course Recommendation** agent.
2.  **Change Output Format:** Select `Widget` and upload your downloaded `.widget` file.
3.  **Refine the Prompt:** Update the agent's system prompt to instruct the model to output the specific data fields required by the widget.
    ```text
    Your job is to identify and recommend online training courses that help develop one or more of the skills identified. Given the list of required skills, return a list of 3-5 online courses along with course details including course name, provider (school or program), recommendation reason (a brief sentence on why you're recommending the course), course format, and URL. In addition to the list of courses, share a few-sentence summary of the recommendations you're making.
    ```
4.  **Add a Guardrail (Optional):** To enhance privacy, insert a **PII Redaction** node between the Resume Extraction and Career Analysis agents. This prevents personally identifiable information from propagating through the rest of the workflow.
5.  **Republish** your workflow to save all changes.

---

## Step 4: Optimize Performance with Evals

After deployment, use real user interactions to improve your agents. We'll optimize a single agent and then the entire workflow.

### 4.1 Single Agent Optimization (Prompt Optimization)

Let's improve the Course Recommendation agent using example data and feedback.

1.  **Gather Data:** Collect sample prompts and outputs from your deployed app (e.g., from the API Logs). For this tutorial, you can use [this sample dataset](https://cdn.openai.com/cookbook/agent_walkthrough/course_recommendations_dataset.csv).
2.  **Start Evaluation:** In Agent Builder, select the **Course Recommendation** agent and click **"Evaluate"**. This opens the Evals **Datasets** feature with your agent's configuration pre-loaded.
3.  **Upload Dataset:** Upload your CSV file with sample prompts.
4.  **Generate Outputs:** Click **"Generate output"** to have the current agent process all samples.
5.  **Add Graders:**
    *   **Human Annotations:** Add columns for `Rating` (thumbs up/down) and `Feedback` (text). Manually review and score some outputs.
    *   **Model Graders:** Add automated graders to evaluate outputs against specific criteria (e.g., Relevance, Coverage, Summary Quality). Example grader prompts:
        ```text
        [Relevance] You are evaluating whether a list of recommended courses is relevant to the skills described. Return a pass if all courses are relevant to at least one skill, and fail otherwise.

        [Coverage] You are evaluating whether a list of recommended courses covers all of the skills described. Return a pass if all of the skills are covered by at least one course, and fail otherwise.

        [Summary] You are evaluating whether the summary recommendation provided is relevant, thoughtful, and related to the recommended courses proposed. Evaluate the summary recommendation on a scale of 0 to 1, with 1 being the highest quality.
        ```
6.  **Run Grading:** Select **Grade > All graders**. The system will run your human and model graders against all examples.
7.  **Optimize the Prompt:** Click the **"Optimize"** button. Evals will analyze the grader feedback and **automatically rewrite your agent's prompt** to address the identified issues. The new prompt will likely include clearer requirements and output formatting instructions.
8.  **Apply Changes:** Click **"Update"** to automatically push the optimized prompt back into your workflow in Agent Builder.

### 4.2 Entire Workflow Optimization (Trace Grading)

Now, evaluate the performance of the entire multi-agent sequence.

1.  **Access Traces:** In Agent Builder, click the **"Evaluate"** button at the top of the workflow canvas. This shows you complete execution traces.
2.  **Create Workflow-Level Graders:** Define evaluation criteria that span multiple agents. For example, create a grader to check if the *final course summary* is relevant to the *user's original dream job description* (inputs from Agent 1 and Agent 3).
3.  **Run Trace Grading:** Apply these graders across a dataset of full workflow traces. The system will automatically identify which steps in a trace pass or fail your end-to-end criteria.
4.  **Analyze and Iterate:** Use the results to pinpoint failure modes. You might discover issues with how information flows between agents or where additional guardrails are needed. Update your workflow design or individual agent prompts accordingly.

## Conclusion

You have successfully built, deployed, and optimized an agentic application using AgentKit. You learned how to:

*   Visually construct a multi-agent workflow in **Agent Builder**.
*   Rapidly deploy a chat interface using the **ChatKit** starter template.
*   Iterate on the user experience with custom themes and widgets.
*   Systematically improve performance using **Evals** for both single-agent prompt optimization and full workflow trace grading.

This integrated toolset enables a fast, feedback-driven development cycle for bringing reliable and effective AI agents into production.