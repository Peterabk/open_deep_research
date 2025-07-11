# Open Deep Research - Project Overview

*Generated: 2025-07-07*

This document provides a comprehensive overview of the Open Deep Research project, detailing the purpose and functionality of each component.

## Project Summary

Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. It features two implementation approaches:

1. **Workflow-based Implementation**: A structured plan-and-execute workflow using LangGraph
2. **Multi-agent Architecture**: A system of specialized agents collaborating on research tasks

Both implementations support customization of models, prompts, report structure, and search tools.

## Directory Structure

```
open_deep_research/
├── .env.example          - Example environment variables configuration
├── .gitignore            - Git ignore configuration
├── CLAUDE.md             - Information about Claude model usage
├── LICENSE               - Project license
├── README.md             - Project documentation
├── examples/             - Example research reports
│   ├── arxiv.md          - Example using arXiv search
│   ├── inference-market-gpt45.md   - Example using GPT-4.5
│   ├── inference-market.md         - Example report
│   └── pubmed.md         - Example using PubMed search
├── langgraph.json        - LangGraph configuration
├── pyproject.toml        - Python project dependencies
├── src/                  - Source code
│   └── open_deep_research/ - Main package
├── tests/               - Test suite
└── uv.lock              - Dependency lock file
```

## Core Components

### 1. Implementations

The project offers two distinct implementations for automated research:

#### 1.1 Graph-based Workflow (`graph.py`)

The workflow implementation follows a structured plan-and-execute approach:
- **Planning Phase**: Uses a planner model to analyze the topic and generate a report outline
- **Human-in-the-Loop**: Allows for human feedback on the report plan
- **Sequential Research Process**: Creates sections one by one with reflection between search iterations

Key features:
- Structured report generation with clear sections
- Iterative search and refinement for each section
- Human feedback loop for plan approval
- Quality assessment of each section with additional research as needed

#### 1.2 Multi-agent System (`multi_agent.py`)

The multi-agent implementation uses specialized agents with distinct roles:
- **Supervisor Agent**: Manages the research process and coordinates tasks
- **Research Agent(s)**: Conducts web searches and writes report sections

Key features:
- Autonomous task delegation between agents
- Concurrent research on multiple sections
- Specialized agents focusing on different aspects of the research process
- Tool-based architecture for search and content generation

### 2. Configuration (`configuration.py`)

This file defines configuration classes for both implementations:

- `WorkflowConfiguration`: Settings for the graph-based workflow
- `MultiAgentConfiguration`: Settings for the multi-agent system
- `SearchAPI`: Enum of available search providers

Key configuration options:
- Model selection for different roles (planner, writer, etc.)
- Search API selection and configuration
- Report structure templates
- Processing options for search results

### 3. State Management

#### 3.1 Workflow State (`state.py`)

Defines state classes for the LangGraph workflow:
- `ReportState`: Overall report generation state
- `SectionState`: State for individual section generation
- `Sections`: Structure for report sections
- `Queries`: Structure for search queries

#### 3.2 Multi-agent State (`multi_agent.py`)

Defines state classes for the multi-agent system:
- `ReportState`: Overall report generation state
- `SectionState`: State for section research
- `Section`, `Introduction`, `Conclusion`: Structures for report components

### 4. Search Utilities (`utils.py`)

Provides a comprehensive set of search tools and utilities:
- `tavily_search`: Web search using Tavily API
- `exa_search`: Neural search using Exa API
- `perplexity_search`: Web search using Perplexity API
- `arxiv_search_async`: Academic paper search via ArXiv
- `pubmed_search_async`: Medical literature search via PubMed
- `linkup_search`: Web search using Linkup API
- `duckduckgo_search`: Web search using DuckDuckGo
- `google_search_async`: Web search using Google Custom Search or scraping

Supporting utilities:
- `deduplicate_and_format_sources`: Process and format search results
- `scrape_pages`: Extract content from web pages
- `summarize_webpage`: Create concise summaries of web content
- `split_and_rerank_search_results`: Optimize search results with embeddings

### 5. Prompts (`prompts.py`)

Contains system prompts for different LLM roles:
- `report_planner_instructions`: Guidelines for planning report structure
- `query_writer_instructions`: Instructions for generating search queries
- `section_writer_instructions`: Guidelines for writing report sections
- `section_grader_instructions`: Instructions for evaluating section quality
- `final_section_writer_instructions`: Guidelines for writing conclusions
- `SUPERVISOR_INSTRUCTIONS`: Guidelines for the supervisor agent
- `RESEARCH_INSTRUCTIONS`: Guidelines for research agents

### 6. Workflow-specific Components (`workflow/`)

- `workflow.py`: Implementation of the research workflow
- `configuration.py`: Workflow-specific configuration
- `prompts.py`: Specialized prompts for workflow steps
- `state.py`: State management for the workflow

## Search Tools Integration

The project supports multiple search tools with consistent interfaces:

- **Web Search**: Tavily, Perplexity, Exa, Linkup, DuckDuckGo, Google
- **Academic Search**: ArXiv for physics/CS/math papers, PubMed for biomedical research
- **Azure AI Search**: For customized search implementations

Each search tool:
1. Takes a list of search queries
2. Performs concurrent searches
3. Returns formatted results with consistent structure
4. Optionally includes raw content from pages

## Running the Project

The project can be run using the LangGraph server:

```bash
# Windows / Linux
pip install -e .
pip install -U "langgraph-cli[inmem]" 
langgraph dev
```

This opens the LangGraph Studio UI where users can:
1. Input a research topic
2. Review and approve the generated research plan
3. Monitor section generation
4. Access the final formatted report

## Examples

The `examples/` directory contains sample reports generated using different search tools:
- `arxiv.md`: Research using ArXiv paper search
- `pubmed.md`: Research using PubMed medical literature
- `inference-market.md` and `inference-market-gpt45.md`: Example reports using different models

## Customization Options

The project is highly customizable through:
1. Environment variables or configuration objects
2. Custom prompts for different stages
3. Selection of different LLM models for planning and writing
4. Choice of search tools and parameters
5. Report structure templates
