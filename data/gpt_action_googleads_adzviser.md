# Guide: Building a GPT Action for Google Ads with Adzviser

## Introduction

This guide walks you through building a Custom GPT Action that connects to Google Ads via the Adzviser middleware. This integration allows you to query Google Ads performance data and audit account settings using natural language in ChatGPT, with results returned as structured CSV files for analysis.

**Key Value:** Marketers can instantly retrieve and analyze campaign metrics (impressions, clicks, cost, ROAS) without manual exports or navigating the Google Ads UI.

**How it Works:**
1.  You connect your Google Ads account to Adzviser.
2.  Your Custom GPT uses Adzviser's API to translate natural language questions into Google Ads Query Language (GAQL).
3.  Adzviser executes the query, formats the response into a CSV, and sends it back to ChatGPT.
4.  ChatGPT's Data Analysis feature (Code Interpreter) reads the CSV and provides insights.

### Prerequisites

Before you begin, ensure you have:
1.  **Google Ads Account Access:** Read-only, Standard, or Admin access to at least one Google Ads account.
2.  **Adzviser Account:** A subscribed Adzviser account (plans start at $0.99). [Sign up here](https://adzviser.com/signup).
3.  **Connected Workspace:** Follow the [Adzviser workspace setup guide](https://docs.adzviser.com/getStarted/workspace) to connect your Google Ads account.

---

## Step 1: Configure Your Custom GPT Instructions

The "Instructions" tell your GPT how to behave and the specific logic to follow when processing user requests.

1.  Create a new Custom GPT in the ChatGPT interface.
2.  Navigate to the **Configure** tab and find the **Instructions** field.
3.  Copy and paste the following instructions precisely.

```text
***Context***:
You are a Google Ads specialist who audits account health, retrieves real-time reporting data, and optimizes performances for marketers. When asked for an audit on account health, collect the relevant account settings, provide recommendations to adjust account structures. When asked about reporting data insights, gather relevant metrics and breakdowns, thoroughly analyze the reporting data, and then provide tailored recommendations to optimize performance.

***Instructions for Retrieval of Reporting Data***:
- Workflow to fetch real-time reporting data
Step 1. Calculate the date range with Python and Code Interpreter based on user input, such as "last week", "last month", "yesterday", "last 28 days", "last quarter" or "last year" etc. If no specific timeframe is provided, ask the user to clarify. Adjust for calendar variations. For example, "last week" should cover Monday to Sunday of the previous week.
Step 2. Retrieve workspace information using the 'getWorkspace' function.
Step 3. Fetch the relevant metrics and breakdowns for the inquired data source using functions like 'getGoogleAdsMetricsList' and 'getGoogleAdsBreakdownsList'.
Step 4. Use 'searchQuery' function with the data gathered from the previous steps like available workspace_name and metrics/breakdowns as well as calculated date range to retrieve real-time reporting data.
- Time Granularity: If the user asks for daily/weekly/quarterly/monthly data, please reflect such info in the field time_granularity in searchQueryRequest. No need to add time_granularity if the user did not ask for it explicitly.
- Returned Files: If multiple files are returned, make sure to read all of them. Each file contains data from a segment in a data source or a data source.
- Necessary Breakdowns Only: Add important breakdowns only. Less is more. For example, if the user asks for "which ad is performing the best in Google Ads?", then you only add "Ad Name" in the breakdown list for the google_ads_request. No need to add breakdowns such as "Device" or "Campaign Name".

***Instruction for Auditing****:
- Workflow to audit Google Ads account
Step 1. Retrieve workspace information using the 'getWorkspace' function.
Step 2. Use '/google_ads_audit/<specfic_section_to_check>' function to retrieve account settings.
- Comprehensive Audit: When asked for an comprehensive audit, don't call all the /google_ads_audit/<specfic_section_to_check> all at once. Show the users what you're planning to do next first. Then audit two sections from the Google Ads Audit GPT Knowledge at a time, then proceed to the next two sections following users consent. For the line items in the tables in the Audit Knowledge doc that don't have automation enabled, it is very normal and expected that no relevant data is seen in the retrieved response. Please highlight what needs to be checked by the user manually because these non-automated steps are important too. For example, when checking connections, adzviser only checks if the google ads account is connected with Google Merchant Center. For other connections such as YT channels, please politely ask the user to check them manually.

***Additional Notes***:
- Always calculate the date range please with Code Interpreter and Python. It often is the case that you get the date range 1 year before when the user asks for last week, last month, etc.
- If there is an ApiSyntaxError: Could not parse API call kwargs as JSON, please politely tell the user that this is due to the recent update in OpenAI models and it can be solved by starting a new conversation on ChatGPT.
- If the users asks for Google Ads data, for example, and there is only one workspace that has connected to Google Ads, then use this workspace name in the searchQueryRequest or googleAdsAuditRequest.
- During auditing, part of the process is to retrieve the performance metrics at account, campaign, ad group, keyword, and product levels, remember to also run Python to calculate the date range for last month and the previous period. For retrieving performance metrics at these 5 levels, please send 5 distinct requests with different breakdowns list for each level. More can be found in the audit knowledge doc.
```

---

## Step 2: Add the OpenAPI Schema (Actions)

The OpenAPI Schema defines the specific API endpoints your GPT can call and how to call them.

1.  In your same Custom GPT configuration, find the **Actions** section.
2.  Click **Create new action**.
3.  In the **Import from URL** field, paste the following URL to load the schema directly:
    `https://copter.adzviser.com/openapi.json`

**Alternatively, you can paste the raw schema:**
1.  Select the **Schema** editing mode.
2.  Delete any placeholder content.
3.  Copy and paste the complete OpenAPI schema provided below.

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Adzviser Actions for GPT",
    "description": "Equip GPTs with the ability to retrieve real-time reporting data and account settings from Google Ads",
    "version": "v0.0.1"
  },
  "servers": [
    {
      "url": "https://copter.adzviser.com"
    }
  ],
  "paths": {
    "/google_ads/get_metrics_list": {
      "get": {
        "description": "Get the list of seletable Google Ads metrics, such as Cost, Roas, Impressions, etc.",
        "operationId": "getGoogleAdsMetricsList",
        "parameters": [],
        "deprecated": false,
        "security": [],
        "x-openai-isConsequential": false
      }
    },
    "/google_ads/get_breakdowns_list": {
      "get": {
        "description": "Get the list of seletable Google Ads breakdowns such as Device, Keyword Text, Campaign Name etc.",
        "operationId": "getGoogleAdsBreakdownsList",
        "parameters": [],
        "deprecated": false,
        "security": [],
        "x-openai-isConsequential": false
      }
    },
    "/search_bar": {
      "post": {
        "description": "Retrieve real-time reporting data such as impressions, cpc, etc. from marketing channels such as Google Ads, Fb Ads, Fb Insights, Bing Ads, etc.",
        "operationId": "searchQuery",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/searchQueryRequest"
              }
            }
          },
          "required": true
        },
        "deprecated": false,
        "security": [
          {
            "oauth2": []
          }
        ],
        "x-openai-isConsequential": false
      }
    },
    "/workspace/get": {
      "get": {
        "description": "Retrieve a list of workspaces that have been created by the user and their data sources, such as Google Ads, Facebook Ads accounts connected with each.",
        "operationId": "getWorkspace",
        "parameters": [],
        "deprecated": false,
        "security": [
          {
            "oauth2": []
          }
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/getWorkspaceResponse"
                }
              }
            }
          }
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_merchant_center_connection": {
      "post": {
        "description": "Retrieve whether the Google Merchant Center is connected to the Google Ads account.",
        "operationId": "checkGoogleAdsMerchantCenterConnection",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_account_settings": {
      "post": {
        "description": "Retrieve the Google Ads account settings such as whether auto tagging is enabled, inventory type, etc.",
        "operationId": "checkGoogleAdsAccountSettings",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_negative_keywords_and_placements": {
      "post": {
        "description": "Retrieve the negative keywords and placements set in the Google Ads account.",
        "operationId": "checkGoogleAdsNegativeKeywordsAndPlacements",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_remarketing_list": {
      "post": {
        "description": "Retrieve the remarketing list set in the Google Ads account.",
        "operationId": "checkGoogleAdsRemarketingList",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_conversion_tracking": {
      "post": {
        "description": "Retrieve the conversion tracking status in the Google Ads account.",
        "operationId": "checkGoogleAdsConversionTracking",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_bidding_strategy": {
      "post": {
        "description": "Retrieve the bidding strategy set for each active campaigns in the Google Ads account.",
        "operationId": "checkGoogleAdsBiddingStrategy",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_search_campaign_basic": {
      "post": {
        "description": "Retrieve the basic information of the search campaigns such as campaign structure, language targeting, country targeting, etc.",
        "operationId": "checkSearchCampaignBasic",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_search_campaign_detailed": {
      "post": {
        "description": "Retrieve the detailed information of the search campaigns such as best performing keywords, ad copies, ad extentions, pinned descriptions/headlines etc.",
        "operationId": "checkSearchCampaignDetailed",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_dynamic_search_ads": {
      "post": {
        "description": "Retrieve the dynamic search ads information such as dynamic ad targets, negative ad targets, best performing search terms etc.",
        "operationId": "checkDynamicSearchAds",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    },
    "/google_ads_audit/check_pmax_campaign": {
      "post": {
        "description": "Retrieve the performance of the pmax campaigns such as search themes, country/language targeting, final url expansions, excluded urls.",
        "operationId": "checkPmaxCampaign",
        "parameters": [],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/googleAdsAuditRequest"
              }
            }
          },
          "required": true
        },
        "x-openai-isConsequential": false
      }
    }
  },
  "components": {
    "schemas": {
      "getWorkspaceResponse": {
        "title": "getWorkspaceResponse",
        "type": "array",
        "description": "The list of workspaces created by the user on adzviser.com/main. A workspace can include multiple data sources",
        "items": {
          "type": "object",
          "properties": {
            "name": {
              "title": "name",
              "type": "string",
              "description": "The name of a workspace"
            },
            "data_connections_accounts": {
              "title": "data_connections_accounts",
              "type": "array",
              "description": "The list of data sources that the workspace is connected. The name can be an account name and type can be Google Ads/Facebook Ads/Bing Ads",
              "items": {
                "type": "object",
                "properties": {
                  "name": {
                    "title": "name",
                    "type": "string",
                    "description": "The name of a data connection account"
                  }
                }
              }
            }
          }
        }
      },
      "googleAdsAuditRequest": {
        "description": "Contains details about the Google Ads account audit request.",
        "type": "object",
        "required": [
          "workspace_name"
        ],
        "title": "googleAdsAuditRequest",
        "properties": {
          "workspace_name": {
            "type": "string",
            "title": "workspace_name",
            "description": "Call API getWorkspace first to get a list of available workspaces"
          }
        }
      },
      "searchQueryRequest": {
        "description": "Contains details about queried data source, metrics, breakdowns, time ranges and time granularity, etc.",
        "type": "object",
        "required": [
          "workspace_name",
          "data_source",
          "metrics",
          "breakdowns",
          "start_date",
          "end_date"
        ],
        "title": "searchQueryRequest",
        "properties": {
          "workspace_name": {
            "type": "string",
            "title": "workspace_name",
            "description": "Call API getWorkspace first to get a list of available workspaces"
          },
          "data_source": {
            "type": "string",
            "title": "data_source",
            "description": "The data source to query, such as google_ads, facebook_ads, etc."
          },
          "metrics": {
            "type": "array",
            "title": "metrics",
            "description": "The list of metrics to retrieve, such as impressions, clicks, cost, etc.",
            "items": {
              "type": "string"
            }
          },
          "breakdowns": {
            "type": "array",
            "title": "breakdowns",
            "description": "The list of breakdowns to segment the data by, such as campaign_name, ad_group_name, etc.",
            "items": {
              "type": "string"
            }
          },
          "start_date": {
            "type": "string",
            "title": "start_date",
            "description": "The start date of the query range in YYYY-MM-DD format"
          },
          "end_date": {
            "type": "string",
            "title": "end_date",
            "description": "The end date of the query range in YYYY-MM-DD format"
          },
          "time_granularity": {
            "type": "string",
            "title": "time_granular