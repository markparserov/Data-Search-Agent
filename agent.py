
import os
import json
import re
import time

import pandas as pd
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams
from google.genai import types

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

retry_config = types.HttpRetryOptions(
    attempts=7,            # Maximum number of retry attempts
    exp_base=7,            # Exponential backoff multiplier (delay grows: 1s, 7s, 49s...)
    initial_delay=1,       # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
)

google_search = GoogleSearchTool(bypass_multi_tools_limit=True)

google_search_model = Gemini(model="gemini-2.5-flash", retry_options=retry_config)
thinking_model = Gemini(model="gemini-2.5-pro", retry_options=retry_config)


response_mime_type = "application/json"

class Revenue(BaseModel):
    market_size: Optional[float] = Field(description="Estimated market size, may be null.", default=None)
    currency: Optional[str] = Field(description="Currency code, e.g. USD or EUR, may be null.", default=None)
    year: Optional[int] = Field(description="Year of the market estimate, may be null.", default=None)
    source_title: str = Field(description="Title of the source document or report.")
    source_url: str = Field(description="URL of the source.")
    method_note: Optional[str] = Field(description="Methodological note explaining how the estimate was obtained.", default=None)

json_cfg_revenue = types.GenerateContentConfig(
    response_mime_type=response_mime_type,
    response_json_schema=Revenue.model_json_schema(),
    temperature=0.0,
    seed=42
)

class RevenueTrend(BaseModel):
    cagr: Optional[float] = Field(description="Compound annual growth rate, may be null.", default=None)
    years_range: Optional[str] = Field(description="Range of years used to calculate CAGR, may be null.", default=None)
    source_title: str = Field(description="Title of the source document or report.")
    source_url: str = Field(description="URL of the source.")
    method_note: Optional[str] = Field(description="Methodological note for CAGR estimation.", default=None)

json_cfg_revenue_trend = types.GenerateContentConfig(
    response_mime_type=response_mime_type,
    response_json_schema=RevenueTrend.model_json_schema(),
    temperature=0.0,
    seed=42,
)

class AverageNumberOfEmployees(BaseModel):
    estimate_total: Optional[int] = Field(description="Estimated total number of employees, may be null.", default=None)
    snapshots: List[Dict] = Field(
        description="List of snapshot objects with site, query, observed_count, url."
    )
    notes: Optional[str] = Field(description="Additional notes about the estimation process.", default=None)

json_cfg_average_number_of_employees = types.GenerateContentConfig(
    response_mime_type=response_mime_type,
    response_json_schema=AverageNumberOfEmployees.model_json_schema(),
    temperature=0.0,
    seed=42,
)

class Synthesis(BaseModel):
    technology: str = Field(description="Technology name.")
    description: Optional[str] = Field(description="Technology description, may be null.", default=None)
    revenue: Optional[Dict] = Field(description="Revenue object or null.", default=None)
    revenue_trend: Optional[Dict] = Field(description="Revenue trend object or null.", default=None)
    average_number_of_employees: Optional[Dict] = Field(description="Average number of employees object or null.", default=None)

json_cfg_synthesis = types.GenerateContentConfig(
    response_mime_type=response_mime_type,
    response_json_schema=Synthesis.model_json_schema(),
    temperature=0.0,
    seed=42,
)

mcp_http = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://markparserov-tam-mcp-server.hf.space/mcp",
        protocol="http"
    ),
)

databases_search_agent = Agent(
    name="http_mcp_agent",
    description="Agent that searches socio-economic databases (Alpha Vantage, BLS, Census, FRED, IMF, Nasdaq Data Link, OECD, World Bank)",
    model=thinking_model,
    instruction="Use MCP tools to fetch the requested data",
    tools=[mcp_http])

description_agent = Agent(
    name="TechDescriptor",
    description="Agent that generates the textual description of a technology",
    model=google_search_model,
    instruction="""
        Provide a precise and concise description of the technology specified by the user.
        It should implicitly reflect the maturity level of the technology (do not mention maturity directly).
        Write 8–10 sentences as a continuous text: definition, main components or approaches, key application scenarios, ecosystem or vendors or projects.
        Use Google Search to generate the description and to verify facts.
        Return only Markdown with no explanations.
    """,
    tools=[google_search],
    output_key="description",
)

revenue_agent = Agent(
    name="RevenueEstimator",
    description="Agent that determines the market size for a technology",
    model=google_search_model,
    generate_content_config=json_cfg_revenue,
    instruction="""
        Find the most recent and reliable public estimate of the global market size for the technology specified by the user.
        Use google_search and databases_search_agent tools to generate the answer and to double-check the facts.
        Check consistency of values across sources and refer to the most trustworthy ones.
        Priority for google_search: reports of analytical agencies, public financial reports, regulatory filings.
        For databases_search_agent: ask the agent to estimate the latest TAM for the technology.
        It is critical to check the year and the method; return the market size as a full number, for example 1500000000 rather than 1.5 (billion).
        Return clean JSON with the following keys:
        {
          "market_size": <float|null>,
          "currency": <str|null>,
          "year": <int|null>,
          "source": {
            "title": "...",
            "url": "..."
          },
          "method_note": "..."
        }
    """,
    tools=[google_search, AgentTool(agent=databases_search_agent)],
    output_key="revenue",
)

cagr_agent = Agent(
    name="CAGREstimator",
    description="Agent that determines the CAGR for a technology",
    model=google_search_model,
    generate_content_config=json_cfg_revenue_trend,
    instruction="""
        Find the most reliable public estimate of the CAGR for the technology specified by the user.
        Use google_search and databases_search_agent tools to generate the answer and to verify facts.
        Check the consistency of values across sources and rely on the most trustworthy ones.
        Priority for google_search: reports of analytical agencies, public financial reports, regulatory filings.
        For databases_search_agent: ask the agent to estimate the latest CAGR for the technology.
        It is critical to verify the year and the method.
        Return clean JSON with the following keys:
        {
          "cagr": <float|null>,
          "years_range": <str|null>,
          "source": {
            "title": "...",
            "url": "..."
          },
          "method_note": "..."
        }
    """,
    tools=[google_search, AgentTool(agent=databases_search_agent)],
    output_key="revenue_trend",
)

jobs_agent = Agent(
    name="JobsEstimator",
    model=google_search_model,
    description="Agent that determines the number of job openings for a technology",
    generate_content_config=json_cfg_average_number_of_employees,
    instruction="""
        Estimate the current number of job openings related to the technology specified by the user.
        Use google_search and databases_search_agent tools to form the answer and to build the final estimate.
        Priority sites for google_search: linkedin.com/jobs, indeed.com, glassdoor.com, google.com/about/careers/applications/jobs, etc.
        If platforms hide full job counts, make a conservative estimate based on samples and explicitly note the assumptions.
        For databases_search_agent: ask the agent to estimate the number of jobs for the technology / broader sector.
        Return clean JSON with the following keys:
        {
          "estimate_total": <int|null>,
          "snapshots": [
            {
              "site": "...",
              "query": "...",
              "observed_count": <int|null>,
              "url": "..."
            }
          ],
          "notes": "..."
        }
    """,
    tools=[google_search, AgentTool(agent=databases_search_agent)],
    output_key="average_number_of_employees",
)

# Parallel execution of sub-agents  
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[description_agent, revenue_agent, cagr_agent, jobs_agent]
)

# Synthesis agent: assembles the final JSON for the variables  
synthesis_agent = Agent(
    name="Synthesis",
    model=thinking_model,
    generate_content_config=json_cfg_synthesis,
    tools=[],
    output_key="final_json",
    description="Agent that assembles the final result into a single JSON object",
    instruction="""
        Combine the final variables for the technology specified by the user based on the outputs of the child agents.
        Use the fields from the state with the keys: {description}, {revenue}, {revenue_trend}, {average_number_of_employees}.
        Return a FULLY valid one-line JSON without Markdown or comments, following the schema:
        {
          "technology": <str>,
          "description": <str|null>,
          "revenue": <json|null>,
          "revenue_trend": <json|null>,
          "average_number_of_employees": <json|null>,
        }
        Preserve the same JSON structures as provided by the agents, do not rename the keys.
    """
)

# Sequential pipeline: first the sub-agents run in parallel, then the synthesis agent
root_agent = SequentialAgent(
    name="TechVariablesPipeline",
    sub_agents=[parallel_research_team, synthesis_agent],
    description="Parallel research –> JSON synthesis"
)
