import asyncio
import os
import json
import re
import time
import io

import pandas as pd
import streamlit as st
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams
from google.genai import types
from pydantic import BaseModel, Field
# from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Optional, List, Dict

# Page configuration
st.set_page_config(
    page_title="Technology Data Search Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'output_df' not in st.session_state:
    st.session_state.output_df = None

# API Key setup
st.sidebar.header("Configuration")

api_key_source = st.sidebar.radio(
    "API Key Source",
    ["HF Secret", "Manual Input"],
    index=0  # default = HF Secret
)

google_api_key = None

# 1) Hugging Face Secrets (environment variables)
if api_key_source == "HF Secret":
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.sidebar.error("❌ GOOGLE_API_KEY not found in HF Secrets")

# 2) Manual input
elif api_key_source == "Manual Input":
    google_api_key = st.sidebar.text_input("Enter Google API Key", type="password")

# Apply key if found
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    st.sidebar.success("✅ API key configured")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=7,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Initialize tools and models
@st.cache_resource
def initialize_agents():
    """Initialize agents and models (cached to avoid re-initialization)"""
    google_search = GoogleSearchTool(bypass_multi_tools_limit=True)
    
    google_search_model = Gemini(model="gemini-2.5-flash", retry_options=retry_config)
    thinking_model = Gemini(model="gemini-2.5-pro", retry_options=retry_config)
    
    # # Define Pydantic models
    # class Revenue(BaseModel):
    #     market_size: Optional[float] = Field(description="Estimated market size, may be null.", default=None)
    #     currency: Optional[str] = Field(description="Currency code, e.g. USD or EUR, may be null.", default=None)
    #     year: Optional[int] = Field(description="Year of the market estimate, may be null.", default=None)
    #     source_title: str = Field(description="Title of the source document or report.")
    #     source_url: str = Field(description="URL of the source.")
    #     method_note: Optional[str] = Field(description="Methodological note explaining how the estimate was obtained.", default=None)
    
    # class RevenueTrend(BaseModel):
    #     cagr: Optional[float] = Field(description="Compound annual growth rate, may be null.", default=None)
    #     years_range: Optional[str] = Field(description="Range of years used to calculate CAGR, may be null.", default=None)
    #     source_title: str = Field(description="Title of the source document or report.")
    #     source_url: str = Field(description="URL of the source.")
    #     method_note: Optional[str] = Field(description="Methodological note for CAGR estimation.", default=None)
    
    # class AverageNumberOfEmployees(BaseModel):
    #     estimate_total: Optional[int] = Field(description="Estimated total number of employees, may be null.", default=None)
    #     snapshots: List[Dict] = Field(
    #         description="List of snapshot objects with site, query, observed_count, url."
    #     )
    #     notes: Optional[str] = Field(description="Additional notes about the estimation process.", default=None)
    
    # class Synthesis(BaseModel):
    #     technology: str = Field(description="Technology name.")
    #     description: Optional[str] = Field(description="Technology description, may be null.", default=None)
    #     revenue: Optional[Dict] = Field(description="Revenue object or null.", default=None)
    #     revenue_trend: Optional[Dict] = Field(description="Revenue trend object or null.", default=None)
    #     average_number_of_employees: Optional[Dict] = Field(description="Average number of employees object or null.", default=None)
    
    # # JSON configs
    # json_cfg_revenue = types.GenerateContentConfig(
    #     response_json_schema=Revenue.model_json_schema(),
    #     temperature=0.0,
    #     seed=42
    # )
    
    # json_cfg_revenue_trend = types.GenerateContentConfig(
    #     response_json_schema=RevenueTrend.model_json_schema(),
    #     temperature=0.0,
    #     seed=42,
    # )
    
    # json_cfg_average_number_of_employees = types.GenerateContentConfig(
    #     response_json_schema=AverageNumberOfEmployees.model_json_schema(),
    #     temperature=0.0,
    #     seed=42,
    # )
    
    # json_cfg_synthesis = types.GenerateContentConfig(
    #     response_json_schema=Synthesis.model_json_schema(),
    #     temperature=0.0,
    #     seed=42,
    # )
    
    # MCP toolset
    mcp_http = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="https://markparserov-tam-mcp-server.hf.space/mcp",
            protocol="http"
        ),
    )
    
    # Agents
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
        # generate_content_config=json_cfg_revenue,
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
        # generate_content_config=json_cfg_revenue_trend,
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
        # generate_content_config=json_cfg_average_number_of_employees,
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
    
    # Synthesis agent
    synthesis_agent = Agent(
        name="Synthesis",
        model=thinking_model,
        # generate_content_config=json_cfg_synthesis,
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
    
    # Sequential pipeline
    data_search_agent = SequentialAgent(
        name="TechVariablesPipeline",
        sub_agents=[parallel_research_team, synthesis_agent],
        description="Parallel research –> JSON synthesis"
    )
    
    runner = InMemoryRunner(
        agent=data_search_agent,
        plugins=[LoggingPlugin()]
    )
    
    return runner

def process_response(response):
    """Process agent response and extract data"""
    raw = next(
        (e.actions.state_delta["final_json"] for e in reversed(response)
         if hasattr(e, "actions")
         and e.actions
         and isinstance(e.actions.state_delta, dict)
         and "final_json" in e.actions.state_delta),
        None
    )
    
    if raw is None:
        return {
            "technology": None,
            "description": None,
            "revenue": None,
            "revenue_trend": None,
            "average_number_of_employees": None,
            "notes": None
        }
    
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    raw = json.loads(raw)
    
    technology = raw.get("technology")
    description = raw.get("description")
    
    revenue = (raw.get("revenue") or {}).get("market_size")
    revenue_trend = (raw.get("revenue_trend") or {}).get("cagr")
    average_number_of_employees = (raw.get("average_number_of_employees") or {}).get("estimate_total")
    
    notes = []
    
    try:
        rev_rest = {k: v for k, v in raw["revenue"].items() if k != "market_size"}
        notes.append(f"revenue_notes: {rev_rest}")
    except Exception:
        pass
    
    try:
        rev_trend_rest = {k: v for k, v in raw["revenue_trend"].items() if k != "cagr"}
        notes.append(f"revenue_trend_notes: {rev_trend_rest}")
    except Exception:
        pass
    
    try:
        emp_rest = {k: v for k, v in raw["average_number_of_employees"].items() if k != "estimate_total"}
        notes.append(f"employees_notes: {emp_rest}")
    except Exception:
        pass
    
    return {
        "technology": technology,
        "description": description,
        "revenue": revenue,
        "revenue_trend": revenue_trend,
        "average_number_of_employees": average_number_of_employees,
        "notes": "; ".join(notes) if notes else None
    }

async def run(tech):
    try:
        response = await runner.run_debug(tech)
        return response
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return None

# Main UI
st.title("🔍 Technology Data Search Agent")
st.markdown("Upload an Excel file with a 'technology' column to get detailed information about each technology.")

# File upload
uploaded_file = st.file_uploader(
    "Choose an Excel file",
    type=['xlsx', 'xls'],
    help="The file should contain a column named 'technology' with technology names"
)

if uploaded_file is not None:
    try:
        technologies_df = pd.read_excel(uploaded_file)
        
        if 'technology' not in technologies_df.columns:
            st.error("❌ The uploaded file must contain a column named 'technology'")
            st.stop()
        
        st.success(f"✅ File loaded successfully! Found {len(technologies_df)} technologies.")
        
        # Display preview
        with st.expander("Preview uploaded data"):
            st.dataframe(technologies_df, use_container_width=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            delay_seconds = st.number_input(
                "Delay between requests (seconds)",
                min_value=0,
                max_value=300,
                value=30,
                help="Delay to preserve API quota"
            )
        with col2:
            max_technologies = st.number_input(
                "Max technologies to process (0 = all)",
                min_value=0,
                max_value=len(technologies_df),
                value=min(10, len(technologies_df)),
                help="Limit processing for testing"
            )
        
        if st.button("🚀 Process Technologies", type="primary"):
            if not google_api_key:
                st.error("❌ Please configure your Google API key first!")
                st.stop()
            
            # Initialize agents
            with st.spinner("Initializing agents..."):
                runner = initialize_agents()
            
            # Process technologies
            technologies_to_process = technologies_df["technology"].head(max_technologies if max_technologies > 0 else len(technologies_df))
            total = len(technologies_to_process)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            responses = []
            rows = []
            
            for idx, tech in enumerate(technologies_to_process, 1):
                status_text.text(f"Processing {idx}/{total}: {tech}")
                progress_bar.progress(idx / total)
                
                try:
                    response = asyncio.run(run(tech))
                    responses.append(response)
                    row = process_response(response)
                    rows.append(row)
                    st.success(f"✅ Completed: {tech}")
                except Exception as e:
                    st.error(f"❌ Error processing {tech}: {str(e)}")
                    rows.append({
                        "technology": tech,
                        "description": None,
                        "revenue": None,
                        "revenue_trend": None,
                        "average_number_of_employees": None,
                        "notes": f"Error: {str(e)}"
                    })
                
                if idx < total:
                    time.sleep(delay_seconds)
            
            # Create output DataFrame
            output_df = pd.DataFrame(rows)
            st.session_state.output_df = output_df
            st.session_state.processing_complete = True
            
            progress_bar.empty()
            status_text.empty()
            st.success("🎉 Processing complete!")
            
            # Display results
            st.subheader("Results")
            st.dataframe(output_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Download section
if st.session_state.processing_complete and st.session_state.output_df is not None:
    st.divider()
    st.subheader("📥 Download Results")
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.output_df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    
    st.download_button(
        label="📥 Download Excel File",
        data=output,
        file_name="technology_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )