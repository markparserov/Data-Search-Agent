**Problem Statement**

As part of our work at a research institute, we trained an ML model that classifies technologies by their maturity stage using a set of features. To test the classifier on a case that matches real usage scenarios, we need not just a list of technologies but a full set of characteristics for each one: a textual description, market size, CAGR, and the number of job openings. Manual data collection is inefficient for large technology lists, and generating these data using basic LLMs is unreliable.

**Solution Statement**

AI agents can partially automate the data-gathering process used to verify the accuracy of the ML classifier by locating the required and reliable information in open sources. The agent-based approach offers not only automation but also straightforward use: a simple no-code web application has been created, and users will always have access to it.

**Architecture**

The application takes an Excel file as input that contains a list of technologies in a single column. As output, it produces a file with filled-in values for all features and additional data for optional manual verification. The full testing cycle of the ML classifier will therefore have two stages: generating predictors for the technologies using AI agents and analyzing these data with the trained TRL prediction model (which is out of the scope of this project).

The application is a multi-agent system (six agents in total), which we will call `data_search_agent`.

The system consists of four internet-searching AI agents running in parallel (one agent per feature: `description_agent`, `revenue_agent`, `cagr_agent`, `jobs_agent`, all are constructed using the `Agent` class from the Google ADK), plus an orchestrator agent that launches these four in parallel (`ParallelAgent` from Google ADK named `parallel_research_team`), and a synthesis agent (`synthesis_agent` which is `Agent` class).  

The generation process is iterative, meaning that each technology triggers a new call to the AI agents. Once the agents in `parallel_research_team` finish processing a given technology, their outputs become available to `synthesis_agent`, which assembles the collected data into a structured JSON format.  

When the full pipeline has run for all technologies  
(`data_search_agent` → `parallel_research_team` orchestrates parallel execution of `description_agent`, `revenue_agent`, `cagr_agent`, `jobs_agent` → output is passed to `synthesis_agent`, which produces the final JSON for each technology),  
the user receives a completed Excel file.  

The columns in the file are: `technology`, `revenue` (i.e. market size), `revenue-trend` (i.e. CAGR), `average_number_of_employees` (i.e. job openings), `notes` (i.e. comments from AI agents on data sources, estimation methodologies, etc.).

Each of the four internet-searching AI agents has access to a Google Search tool for fact checking. In addition, the three agents responsible for quantitative predictors (`revenue_agent`, `cagr_agent`, `jobs_agent` → market size, CAGR, number of job openings) have access to the [TAM MCP Server](https://github.com/gvaibhav/TAM-MCP-Server), which connects to socio-economic databases with quantitative information such as Alpha Vantage, the Bureau of Labor Statistics, the Census Bureau, FRED, Nasdaq, the IMF, the OECD, and the World Bank. This server is required for retrieving information that may not appear in simple Google searches, such as niche market estimates or job counts for niche technologies. Other analogous MCP servers fall short because they either cover only a single database, have strong limitations on free API usage, or simply replicate basic Google search.

Summary:

* `description_agent` (tool: Google Search)  
* `revenue_agent` (tools: Google Search and TAM MCP lookup)  
* `cagr_agent` (tools: Google Search and TAM MCP lookup)  
* `jobs_agent` (tools: Google Search and TAM MCP lookup)  
* `parallel_research_team` (sub-agents: `description_agent`, `revenue_agent`, `cagr_agent`, `jobs_agent`)  
* `synthesis_agent` (no tools)  
* `data_search_agent` (sub-agents: `parallel_research_team`, `synthesis_agent`)

**Conclusion**

The architecture distributes the workload across multiple specialized AI agents, which removes the bottleneck of manual data collection and reduces the risk of incomplete or low-quality inputs. Iterative per-technology execution guarantees that each item is processed with the same level of detail. The synthesis agent keeps the output consistent and structured. Integrating both Google Search and the TAM MCP Server broadens the coverage of information sources, so the system can retrieve data that ordinary LLM generation or simple web scraping would miss. As a result, the pipeline produces a reliable dataset that can be directly used to test the ML classifier without additional preprocessing.

**Value Statement**

The system reduces the duration of data collection by several times. For a list of 100 technologies, the automation saves more than 10 hours of manual work, since the agents handle search, fact retrieval, and structuring without direct human involvement.
