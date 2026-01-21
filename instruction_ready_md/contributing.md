# Contributing to This Book

Contributions from readers like you are essential for improving this book. Whether you find a typo, a broken link, a missing citation, inelegant code, or an unclear explanation, your contributions help us create a better resource for everyone. Thanks to version control and continuous integration, improvements can be incorporated in hours or days, not years. To contribute, you'll submit a pull request to the book's GitHub repository. Once merged, you'll be listed as a contributor.

This guide walks you through the contribution process, from minor text edits to major changes involving code.

## Prerequisites

Before you begin, ensure you have:
*   A [GitHub account](https://github.com/).
*   (For major changes) [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed on your local machine.
*   (For editing code) Familiarity with [Jupyter Notebook](https://jupyter.org/) and the setup instructions in the book.

## How to Contribute

### 1. Submitting Minor Changes (Typos, Single Sentences)

For small fixes like typos or rephrasing a sentence, you can edit directly on GitHub.

1.  **Find the File:** Navigate to the [book's GitHub repository](https://github.com/d2l-ai/d2l-en). Use the "Find file" button to locate the Markdown (`.md`) source file you want to edit.
2.  **Edit the File:** Click the "Edit this file" (pencil) icon in the upper-right corner of the file viewer.
3.  **Propose Your Change:** Make your edits in the web editor. At the bottom of the page, describe your change in the "Propose file change" form and click the "Propose file change" button.
4.  **Create a Pull Request (PR):** You will be taken to a page comparing your changes. Review them, then click the "Create pull request" button to submit it for review by the maintainers.

### 2. Proposing Major Changes (Text, Code, New Sections)

For larger contributions that involve updating multiple paragraphs, modifying code, or adding new content, a local workflow is recommended.

**Understanding the Format:**
*   The book is written in Markdown, extended with the [D2L-Book](http://book.d2l.ai/user/markdown.html) package for features like cross-references.
*   **If you are changing code,** you **must** open and edit the Markdown files using Jupyter Notebook (as described in the book's setup). This allows you to run and test your changes.
*   **Crucial:** Before submitting, **clear all cell outputs** from your notebook. The CI system will re-execute the code to generate fresh outputs.
*   For multi-framework code blocks, use the `%%tab` directive on the first line of a cell (e.g., `%%tab pytorch`, `%%tab tensorflow`, `%%tab all`).

### 3. Submitting Major Changes via Git

For major changes, follow this standard Git workflow:

#### Step 1: Fork the Repository
1.  Go to the main book repository: `https://github.com/d2l-ai/d2l-en`.
2.  Click the **Fork** button (top-right). This creates your personal copy (e.g., `https://github.com/your_username/d2l-en`).

#### Step 2: Clone Your Fork Locally
Clone your forked repository to your computer.
```bash
# Replace 'your_github_username' with your actual GitHub username
git clone https://github.com/your_github_username/d2l-en.git
cd d2l-en
```

#### Step 3: Make Your Edits
1.  Open the relevant Markdown files in Jupyter Notebook to edit text and code.
2.  Make your changes, run cells to test them, and ensure everything works.
3.  **Remember to clear all cell outputs** before saving.

#### Step 4: Commit and Push Your Changes
1.  Check which files you've modified:
    ```bash
    git status
    ```
    You should see a list of changed files.
2.  Stage the specific files you want to commit. For example, if you edited `chapter_appendix-tools-for-deep-learning/contributing.md`:
    ```bash
    git add chapter_appendix-tools-for-deep-learning/contributing.md
    ```
3.  Commit your changes with a descriptive message:
    ```bash
    git commit -m "Fix a typo in the git documentation section"
    ```
4.  Push the commit to your fork on GitHub:
    ```bash
    git push
    ```

#### Step 5: Create a Pull Request
1.  Go to your fork's page on GitHub (`https://github.com/your_username/d2l-en`).
2.  Click the **"New pull request"** button.
3.  GitHub will show a comparison between your branch and the main `d2l-ai/d2l-en` repository. Ensure the changes look correct.
4.  Click **"Create pull request"**.
5.  Add a clear title and description explaining *what* you changed and *why*. This helps the maintainers review your work efficiently.
6.  Submit the pull request. The maintainers will review it and may provide feedback for you to incorporate.

## Best Practices & Summary

*   **Use GitHub's web editor** for quick, minor fixes.
*   **Use the local Git workflow** for substantial changes involving code or multiple files.
*   **Keep pull requests focused.** Smaller, single-topic PRs are easier to review and merge than large, sweeping changes.
*   **Test code changes** in Jupyter and **clear outputs** before committing.
*   **Use descriptive commit messages** and PR descriptions.

## Exercises

1.  **Star and Fork:** Go to the [`d2l-ai/d2l-en`](https://github.com/d2l-ai/d2l-en) repository, star it, and create your own fork.
2.  **Make Your First Contribution:** Look through the book. If you find anything that needs improvement (e.g., a typo, a missing reference), submit a pull request to fix it.
3.  **Learn Branching (Advanced):** For better organization, create a new Git branch for each pull request. Learn how with the [Git Branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) guide.

---
*Happy contributing! Your efforts help make this resource better for learners worldwide.*