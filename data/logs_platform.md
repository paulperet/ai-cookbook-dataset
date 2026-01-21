# Guide: Downloading Logs with the OpenAI Compliance Logs Platform

This guide provides step-by-step instructions for downloading compliance log files from the OpenAI Compliance Logs Platform. You'll learn how to use the provided scripts to retrieve logs for ingestion into your SIEM or data lake.

## Prerequisites

Before you begin, ensure you have:

1.  **An OpenAI Enterprise Compliance API Key:** This must be exported as an environment variable named `COMPLIANCE_API_KEY`.
2.  **A Principal ID:** This is either your ChatGPT account ID (a UUID) or your API Platform Organization ID (prefixed with `org-`).
3.  **A Compatible Environment:** You will need either a Unix-like shell (e.g., bash) or PowerShell (v5.1+).

## Understanding the Scripts

Two functionally identical scripts are provided: one for Unix-based systems (bash) and one for Windows (PowerShell). Both scripts perform the same core tasks:
1.  **List Logs:** Query the Compliance API for available log files based on event type and a start time (`after` parameter).
2.  **Handle Pagination:** Automatically page through all available results.
3.  **Download Logs:** Retrieve the full content of each log file.

The scripts output the raw log data (in JSON Lines format) to `stdout`, allowing you to pipe it directly to a file or another process.

## Option 1: Using the Unix (bash) Script

### Step 1: Save and Prepare the Script

1.  Copy the bash script content into a new file named `download_compliance_files.sh`.
2.  Make the script executable:
    ```bash
    chmod +x download_compliance_files.sh
    ```

### Step 2: Set Your API Key

Export your Compliance API key as an environment variable in your terminal session:
```bash
export COMPLIANCE_API_KEY='your_api_key_here'
```

### Step 3: Run the Script

The script requires four arguments in order:
1.  `workspace_or_org_id`: Your Principal ID.
2.  `event_type`: The type of event log to retrieve (e.g., `AUTH_LOG`).
3.  `limit`: The maximum number of log file *entries* to fetch per API page (e.g., `100`).
4.  `after`: The ISO 8601 timestamp marking the start of your query window.

**Example 1: Fetch authentication logs from the last 24 hours for a workspace ID**
```bash
./download_compliance_files.sh \
  f7f33107-5fb9-4ee1-8922-3eae76b5b5a0 \
  AUTH_LOG \
  100 \
  "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" > output.jsonl
```

**Example 2: Fetch authentication logs from the last 24 hours for an organization ID**
```bash
./download_compliance_files.sh \
  org-p13k3klgno5cqxbf0q8hpgrk \
  AUTH_LOG \
  100 \
  "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" > output.jsonl
```
*Note: The `date` command syntax above is for macOS/BSD. On Linux, use `date -u -d "1 day ago" +%Y-%m-%dT%H:%M:%SZ`.*

The `>` operator redirects the script's output (the log data) into a file named `output.jsonl`.

## Option 2: Using the Windows (PowerShell) Script

### Step 1: Save the Script

1.  Copy the PowerShell script content into a new file named `download_compliance_files.ps1`.
2.  Save it to a convenient directory.

### Step 2: Set Your API Key

Set the API key as an environment variable in your PowerShell session:
```powershell
$env:COMPLIANCE_API_KEY = 'your_api_key_here'
```

### Step 3: Run the Script

Open PowerShell, navigate to the script's directory, and execute it with the four required arguments (in the same order as the bash script).

**Example 1: Fetch authentication logs from the last 24 hours for a workspace ID**
```powershell
.\download_compliance_files.ps1 `
  f7f33107-5fb9-4ee1-8922-3eae76b5b5a0 `
  AUTH_LOG `
  100 `
  (Get-Date -AsUTC).AddDays(-1).ToString('yyyy-MM-ddTHH:mm:ssZ') |
    Out-File -Encoding utf8 output.jsonl
```

**Example 2: Fetch authentication logs from the last 24 hours for an organization ID**
```powershell
.\download_compliance_files.ps1 `
  org-p13k3klgno5cqxbf0q8hpgrk `
  AUTH_LOG `
  100 `
  (Get-Date -AsUTC).AddDays(-1).ToString('yyyy-MM-ddTHH:mm:ssZ') |
    Out-File -Encoding utf8 output.jsonl
```
The pipeline (`|`) sends the script's output to the `Out-File` cmdlet, which writes it to `output.jsonl`.

## Script Output and Behavior

Both scripts provide feedback on their progress via `stderr` (the console), while the actual log data is written to `stdout`. You will see messages like:
```
Fetching page 1 with after='2024-01-15T10:00:00Z' (local: 2024-01-15 05:00:00 EST)
Fetching logs for ID: log_abc123...
Completed downloading 150 log files up to 2024-01-16T09:45:12Z (local: 2024-01-16 04:45:12 EST)
```

If no logs are found for the given parameters, you will see:
```
No results found for event_type AUTH_LOG after 2024-01-15T10:00:00Z
```

## Next Steps

Once you have successfully downloaded the log files (e.g., `output.jsonl`), you can:
*   Inspect the JSON Lines format.
*   Ingest the file directly into your SIEM, data lake, or analytics platform.
*   Modify the scripts to integrate them into a scheduled job (e.g., using `cron` or Task Scheduler) for regular log collection.

For more detailed information, refer to the official [OpenAI Compliance API documentation](https://chatgpt.com/admin/api-reference#tag/Compliance-API-Logs-Platform).