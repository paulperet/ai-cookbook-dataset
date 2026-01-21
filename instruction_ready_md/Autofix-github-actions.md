# Guide: Automatically Fix CI Failures on GitHub with Codex CLI

## Overview

This guide demonstrates how to embed the OpenAI Codex CLI into your CI/CD pipeline. When a build or test fails, Codex will automatically analyze the failure, generate a targeted fix, and propose the changes via a pull request. The example uses a Node.js project with GitHub Actions.

## Prerequisites

Before you begin, ensure you have:

1.  **A GitHub Repository** with GitHub Actions enabled.
2.  **An OpenAI API Key.** Create a secret named `OPENAI_API_KEY` in your repository's settings (`Settings > Secrets and variables > Actions`). You can also set this at the organization level for broader access.
3.  **Python.** The Codex CLI requires Python for its `codex login` command.
4.  **Repository Permissions.** Ensure your GitHub Actions workflow has permission to create pull requests. You may need to adjust your repository or organization settings to allow this.

## Implementation Steps

### Step 1: Add the Auto-Fix GitHub Action

Create a new workflow file in your repository at `.github/workflows/codex-autofix.yml`. This workflow will be triggered when your primary CI workflow fails.

Copy the following YAML configuration into the file. Replace `"CI"` in the `workflows` list with the exact name of the workflow you want to monitor for failures.

```yaml
name: Codex Auto-Fix on Failure

on:
  workflow_run:
    # Trigger this job after any run of the primary CI workflow completes
    workflows: ["CI"]
    types: [completed]

permissions:
  contents: write
  pull-requests: write

jobs:
  auto-fix:
    # Only run when the referenced workflow concluded with a failure
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      FAILED_WORKFLOW_NAME: ${{ github.event.workflow_run.name }}
      FAILED_RUN_URL: ${{ github.event.workflow_run.html_url }}
      FAILED_HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
      FAILED_HEAD_SHA: ${{ github.event.workflow_run.head_sha }}
    steps:
      - name: Check OpenAI API Key Set
        run: |
          if [ -z "$OPENAI_API_KEY" ]; then
            echo "OPENAI_API_KEY secret is not set. Skipping auto-fix." >&2
            exit 1
          fi
      - name: Checkout Failing Ref
        uses: actions/checkout@v4
        with:
          ref: ${{ env.FAILED_HEAD_SHA }}
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: |
          if [ -f package-lock.json ]; then npm ci; else npm i; fi
      - name: Run Codex
        uses: openai/codex-action@main
        id: codex
        with:
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          prompt: "You are working in a Node.js monorepo with Jest tests and GitHub Actions. Read the repository, run the test suite, identify the minimal change needed to make all tests pass, implement only that change, and stop. Do not refactor unrelated code or files. Keep changes small and surgical."
          codex_args: '["--config","sandbox_mode=\"workspace-write\""]'

      - name: Verify tests
        run: npm test --silent

      - name: Create pull request with fixes
        if: success()
        uses: peter-evans/create-pull-request@v6
        with:
          commit-message: "fix(ci): auto-fix failing tests via Codex"
          branch: codex/auto-fix-${{ github.event.workflow_run.run_id }}
          base: ${{ env.FAILED_HEAD_BRANCH }}
          title: "Auto-fix failing CI via Codex"
          body: |
            Codex automatically generated this PR in response to a CI failure on workflow `${{ env.FAILED_WORKFLOW_NAME }}`.
            Failed run: ${{ env.FAILED_RUN_URL }}
            Head branch: `${{ env.FAILED_HEAD_BRANCH }}`
            This PR contains minimal changes intended solely to make the CI pass.
```

**How it works:**
*   **Trigger:** The workflow listens for the `completed` event of your primary CI workflow.
*   **Condition:** It only executes if that primary workflow's conclusion was `failure`.
*   **Environment:** It checks out the exact commit that failed, sets up Node.js, and installs dependencies.
*   **Codex Execution:** The `openai/codex-action` runs Codex with a specific prompt instructing it to make minimal, surgical changes to fix failing tests.
*   **Verification:** After Codex runs, the workflow executes the test suite again to verify the fix.
*   **Pull Request:** If verification passes, it automatically creates a pull request with the proposed changes.

### Step 2: Monitor the Workflow Execution

1.  Navigate to the **Actions** tab in your GitHub repository.
2.  Your primary CI workflow will run as usual. If it fails, you will see a new run appear for the **"Codex Auto-Fix on Failure"** workflow.
3.  Click into this new run to monitor its progress as it checks out the code, runs Codex, and creates a pull request.

### Step 3: Review and Merge the Proposed Fix

1.  Once the Codex workflow completes successfully, navigate to the **Pull requests** tab in your repository.
2.  You should see a new PR with a title like **"Auto-fix failing CI via Codex"**, opened from a branch named `codex/auto-fix-{run_id}`.
3.  Review the changes proposed by Codex. The PR description will include context about the original CI failure.
4.  If the changes look correct, you can merge the pull request to apply the fix to the original branch.

## Conclusion

You have successfully integrated the OpenAI Codex CLI with GitHub Actions to create an automated feedback loop for CI failures. This automation reduces manual debugging time, helps maintain a healthy main branch, and allows your team to focus on feature development. The workflow is designed to propose minimal, targeted fixes to keep changes safe and reviewable.

To explore more capabilities of the Codex CLI, visit the [official repository](https://github.com/openai/codex/).