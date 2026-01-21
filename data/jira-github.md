# Automate Jira ↔ GitHub with `codex-cli`: A Step-by-Step Guide

## Overview

This guide provides a practical, step-by-step approach to automating the workflow between Jira and GitHub. By labeling a Jira issue, you trigger an end-to-end process that creates a **GitHub pull request**, keeps both systems updated, and streamlines code review, all with minimal manual effort. The automation is powered by the [`codex-cli`](https://github.com/openai/openai-codex) agent running inside a GitHub Action.

The automated flow is:
1. Label a Jira issue.
2. Jira Automation calls the GitHub Action.
3. The action spins up `codex-cli` to implement the change.
4. A PR is opened.
5. Jira is transitioned & annotated, creating a neat, zero-click loop.

## Prerequisites

Before you begin, ensure you have:

*   **Jira:** Project admin rights and the ability to create automation rules.
*   **GitHub:** Write access to the target repository, permission to add repository secrets, and a protected `main` branch.
*   **API Keys & Secrets:** The following must be added as secrets in your GitHub repository:
    *   `OPENAI_API_KEY` – Your OpenAI key for `codex-cli`.
    *   `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` – For REST API calls from the GitHub Action.
*   **Local Setup (Optional):** `codex-cli` installed locally (`pnpm add -g @openai/codex`) for ad-hoc testing.
*   **Repository Structure:** A repository containing a `.github/workflows/` folder.

---

## Step 1: Create the Jira Automation Rule

The automation rule listens for label changes on an issue and triggers the GitHub workflow.

1.  Navigate to your Jira project settings and open **Project settings > Automation**.
2.  Click **Create rule**.
3.  **Set the trigger:** Select **Issue field value changed** and configure it to listen for changes to the `Labels` field.
4.  **Add a condition:** Configure a condition to check if the updated labels contain a specific keyword (e.g., `aswe`). This ensures only explicitly tagged issues trigger the automation.
5.  **Define the action:** Select **Send webhook**.
6.  Configure the webhook to send a `POST` request to GitHub's `workflow_dispatch` endpoint. The URL format is:
    ```
    https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file_name}/dispatches
    ```
7.  In the webhook payload, pass the Jira issue context. Use Jira's smart values to include the issue key, summary, and a cleaned description (escaping quotes and newlines for proper JSON/YAML parsing). For example:
    ```json
    {
      "ref": "main",
      "inputs": {
        "issue_key": "{{issue.key}}",
        "issue_summary": "{{issue.summary}}",
        "issue_description": "{{issue.description}}"
      }
    }
    ```
8.  Save and enable the rule.

This setup allows tight control over which issues trigger automation and provides GitHub with clean, structured metadata.

## Step 2: Add the GitHub Action

Create a new file in your repository at `.github/workflows/codex-auto-pr.yml`.

Copy the following YAML configuration into the file. This workflow defines the entire automation process triggered by the Jira webhook.

```yaml
name: Codex Automated PR

on:
  workflow_dispatch:
    inputs:
      issue_key:
        description: 'JIRA issue key (e.g., PROJ-123)'
        required: true
      issue_summary:
        description: 'Brief summary of the issue'
        required: true
      issue_description:
        description: 'Detailed issue description'
        required: true

permissions:
  contents: write      # Allow the action to push code & open the PR
  pull-requests: write # Allow the action to create and update PRs

jobs:
  codex_auto_pr:
    runs-on: ubuntu-latest

    steps:
      # Step 1 – Checkout repository
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Full history for git operations

      # Step 2 – Set up Node.js and Codex
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: pnpm add -g @openai/codex

      # Step 3 – Export and clean input variables
      - id: vars
        run: |
          echo "ISSUE_KEY=${{ github.event.inputs.issue_key }}" >> $GITHUB_ENV
          echo "TITLE=${{ github.event.inputs.issue_summary }}" >> $GITHUB_ENV
          echo "RAW_DESC=${{ github.event.inputs.issue_description }}" >> $GITHUB_ENV
          DESC_CLEANED=$(echo "${{ github.event.inputs.issue_description }}" | tr '\n' ' ' | sed 's/"/'\''/g')
          echo "DESC=$DESC_CLEANED" >> $GITHUB_ENV
          echo "BRANCH=codex/${{ github.event.inputs.issue_key }}" >> $GITHUB_ENV

      # Step 4 – Transition Jira issue to "In Progress"
      - name: Jira – Transition to In Progress
        env:
          ISSUE_KEY: ${{ env.ISSUE_KEY }}
          JIRA_BASE_URL: ${{ secrets.JIRA_BASE_URL }}
          JIRA_EMAIL: ${{ secrets.JIRA_EMAIL }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
        run: |
          curl -sS -X POST \
            --url "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/transitions" \
            --user "$JIRA_EMAIL:$JIRA_API_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{"transition":{"id":"21"}}'
          # Transition ID '21' moves the ticket to "In Progress". Find your IDs via the Jira API.

      # Step 5 – Configure Git author for CI commits
      - run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"

      # Step 6 – Let Codex implement the changes and commit
      - name: Codex implement & commit
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CODEX_QUIET_MODE: "1"
        run: |
          set -e
          codex --approval-mode full-auto --no-terminal --quiet \
                "Implement JIRA ticket $ISSUE_KEY: $TITLE. $DESC"
          git add -A
          git commit -m "feat($ISSUE_KEY): $TITLE"

      # Step 7 – Create the Pull Request
      - id: cpr
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          base: main
          branch: ${{ env.BRANCH }}
          title: "${{ env.TITLE }} (${{ env.ISSUE_KEY }})"
          body: |
            Auto-generated by Codex for JIRA **${{ env.ISSUE_KEY }}**.
            ---
            ${{ env.DESC }}

      # Step 8 – Update Jira: Transition to "In Review" and comment with PR link
      - name: Jira – Transition to In Review & Comment PR link
        env:
          ISSUE_KEY: ${{ env.ISSUE_KEY }}
          JIRA_BASE_URL: ${{ secrets.JIRA_BASE_URL }}
          JIRA_EMAIL: ${{ secrets.JIRA_EMAIL }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
          PR_URL: ${{ steps.cpr.outputs.pull-request-url }}
        run: |
          # Status transition to "In Review"
          curl -sS -X POST \
            --url "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/transitions" \
            --user "$JIRA_EMAIL:$JIRA_API_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{"transition":{"id":"31"}}'
          # Transition ID '31' moves the ticket to "In Review".

          # Add a comment with the PR link
          curl -sS -X POST \
            --url "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/comment" \
            --user "$JIRA_EMAIL:$JIRA_API_TOKEN" \
            --header 'Content-Type: application/json' \
            --data "{\"body\":{\"type\":\"doc\",\"version\":1,\"content\":[{\"type\":\"paragraph\",\"content\":[{\"type\":\"text\",\"text\":\"PR created: $PR_URL\"}]}]}}"
```

**Key Workflow Steps Explained:**

1.  **Codex Implementation & Commit (Step 6):** The `codex-cli` tool uses the OpenAI API to interpret the Jira ticket description and implement the required changes directly in the codebase. It then commits these changes with a standardized message.
2.  **Create Pull Request (Step 7):** The `peter-evans/create-pull-request` action pushes the new branch and opens a PR against `main`. The PR title and body are populated with the Jira ticket information.
3.  **Jira Updates (Step 8):** The workflow makes two final API calls to Jira: one to transition the ticket's status to "In Review" and another to post a comment containing the link to the newly created PR.

## Step 3: Trigger the Automation

To start the automated workflow, you simply need to add the designated label (e.g., `aswe`) to a Jira issue.

1.  **For a new issue:** Add the label in the **Labels** field during creation.
2.  **For an existing issue:** Hover over the label area, click the pencil icon, and type the label name (e.g., `aswe`).

## Step 4: Review and Merge the PR

Once the automation completes:

1.  Open the PR link posted in the Jira ticket comment.
2.  Review the changes made by `codex-cli`.
3.  If the changes are correct, merge the pull request following your team's standard process.

If you have branch protection and Jira Smart Commits integration enabled, merging the PR can automatically transition the Jira ticket to a "Done" or "Closed" status.

## Conclusion

This automation creates a seamless integration between Jira and GitHub, offering several key benefits:

*   **Automatic Status Tracking:** Tickets progress through your workflow (In Progress → In Review) without manual updates.
*   **Improved Developer Experience:** Engineers can focus on reviewing code quality instead of writing initial boilerplate or setup code.
*   **Reduced Handoff Friction:** A PR is generated and ready for review as soon as a ticket is labeled, accelerating the development cycle.

The `codex-cli` tool serves as a powerful AI coding assistant that automates repetitive programming tasks. You can explore its capabilities further in the [official repository](https://github.com/openai/codex/).