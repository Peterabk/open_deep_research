# FireCrawl API Integration with Open Deep Research

*Created: 2025-07-07*

This guide explains how to modify Open Deep Research to use your custom FireCrawl API for web scraping instead of its built-in scraping functionality.

## Overview

We will modify the workflow to:
1. Use Tavily API to find relevant URLs based on your research topic
2. Send these URLs to your custom FireCrawl API instead of directly scraping them
3. Process the returned content with Open Deep Research's analysis pipeline

## Implementation Steps

### 1. Create a Custom FireCrawl API Client

First, we'll create a new module to handle communication with your FireCrawl API:

```python
# src/open_deep_research/firecrawl_client.py

import httpx
from typing import List, Dict, Any
import logging
import time

class FireCrawlClient:
    """Client for interacting with the FireCrawl API."""
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 60):
        """
        Initialize the FireCrawl client.
        
        Args:
            base_url: Base URL of the FireCrawl API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logging.getLogger("firecrawl_client")
        
    async def scrape_urls(self, urls: List[str], titles: List[str] = None) -> Dict[str, Any]:
        """
        Send URLs to the FireCrawl API for scraping.
        
        Args:
            urls: List of URLs to scrape
            titles: Optional list of page titles (for reference)
            
        Returns:
            Dictionary containing scraped content
        """
        if not titles:
            titles = [f"Page {i+1}" for i in range(len(urls))]
            
        # Prepare request payload
        payload = {
            "urls": urls,
            "titles": titles
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.logger.info(f"Sending {len(urls)} URLs to FireCrawl API")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/scrape",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
            elapsed_time = time.time() - start_time
            self.logger.info(f"FireCrawl API responded in {elapsed_time:.2f} seconds")
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error calling FireCrawl API: {str(e)}")
            raise
```

### 2. Modify the Open Deep Research Utils Module

Now, let's create a new function in `utils.py` that will use your FireCrawl API instead of the built-in `scrape_pages`:

```python
# Add to src/open_deep_research/utils.py

from open_deep_research.firecrawl_client import FireCrawlClient

async def firecrawl_scrape_pages(titles: List[str], urls: List[str], base_url: str, api_key: str = None) -> str:
    """
    Uses FireCrawl API to scrape content from a list of URLs and formats it into a readable document.
    
    Args:
        titles: A list of page titles corresponding to each URL
        urls: A list of URLs to scrape content from
        base_url: Base URL of the FireCrawl API
        api_key: Optional API key for FireCrawl API
        
    Returns:
        A formatted string containing the content from each page with source attribution
    """
    # Initialize the FireCrawl client
    client = FireCrawlClient(base_url=base_url, api_key=api_key)
    
    # Request scraped content from FireCrawl API
    scraped_data = await client.scrape_urls(urls, titles)
    
    # Format the results similar to the original scrape_pages function
    formatted_output = f"Search results: \n\n"
    
    # Process each page from the FireCrawl response
    for i, (title, url) in enumerate(zip(titles, urls)):
        page_key = url  # Assuming FireCrawl returns data keyed by URL
        if page_key in scraped_data:
            page_content = scraped_data[page_key].get("content", "No content returned")
            formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page_content}"
            formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output
```

### 3. Modify Search Functions to Use FireCrawl

Update the Tavily search function to use your FireCrawl API:

```python
# Modified version of tavily_search in utils.py

@tool(name="tavily_search", description=TAVILY_SEARCH_DESCRIPTION)
def tavily_search_with_firecrawl(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    firecrawl_base_url: str = None,  # Add FireCrawl API URL
    firecrawl_api_key: str = None,   # Add FireCrawl API key
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API and processes them with FireCrawl.
    
    Args:
        queries: List of search queries
        max_results: Maximum number of results to return
        topic: Topic to filter results by
        firecrawl_base_url: Base URL for FireCrawl API
        firecrawl_api_key: API key for FireCrawl
        
    Returns:
        A formatted string of search results
    """
    # Get Tavily API key from environment
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: TAVILY_API_KEY environment variable not found."

    # Get FireCrawl API parameters, fallback to environment variables if not provided
    if not firecrawl_base_url:
        firecrawl_base_url = os.environ.get("FIRECRAWL_BASE_URL")
    if not firecrawl_api_key:
        firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")
    
    # Check if FireCrawl API details are available
    if not firecrawl_base_url:
        return "Error: FireCrawl base URL not provided and FIRECRAWL_BASE_URL environment variable not found."

    # Create an async function to handle the async workflow
    async def async_search_with_firecrawl():
        # Get search results from Tavily
        search_results = await tavily_search_async(
            queries, max_results=max_results, topic=topic, include_raw_content=False
        )
        
        # Extract URLs and titles from Tavily results
        urls = []
        titles = []
        for result in search_results:
            for item in result["results"]:
                urls.append(item["url"])
                titles.append(item["title"])
        
        # Use FireCrawl API to get content
        if urls:
            return await firecrawl_scrape_pages(
                titles, urls, firecrawl_base_url, firecrawl_api_key
            )
        else:
            return "No search results found."
    
    # Run the async function
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_search_with_firecrawl())
```

### 4. Modify Configuration to Include FireCrawl Settings

Update the configuration classes to include FireCrawl parameters:

```python
# Add to src/open_deep_research/configuration.py

@dataclass(kw_only=True)
class WorkflowConfiguration:
    # Existing fields...
    
    # FireCrawl configuration
    use_firecrawl: bool = False
    firecrawl_base_url: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    
    # Rest of the class...

@dataclass(kw_only=True)
class MultiAgentConfiguration:
    # Existing fields...
    
    # FireCrawl configuration
    use_firecrawl: bool = False
    firecrawl_base_url: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    
    # Rest of the class...
```

### 5. Update `select_and_execute_search` to Use FireCrawl

Modify the search selection function to use FireCrawl when enabled:

```python
# Updated select_and_execute_search in utils.py

async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict):
    """Select and execute the appropriate search API."""
    
    # Get FireCrawl configuration if present
    use_firecrawl = params_to_pass.pop("use_firecrawl", False)
    firecrawl_base_url = params_to_pass.pop("firecrawl_base_url", None)
    firecrawl_api_key = params_to_pass.pop("firecrawl_api_key", None)
    
    # Check for search API
    if search_api.lower() == "tavily":
        if use_firecrawl and firecrawl_base_url:
            # Use Tavily with FireCrawl
            search_results = await tavily_search_async(query_list, **params_to_pass)
            
            # Extract URLs and titles from Tavily results
            urls = []
            titles = []
            for result in search_results:
                for item in result["results"]:
                    urls.append(item["url"])
                    titles.append(item["title"])
            
            # Use FireCrawl to get content
            if urls:
                return await firecrawl_scrape_pages(
                    titles, urls, firecrawl_base_url, firecrawl_api_key
                )
            else:
                return "No search results found."
        else:
            # Use standard Tavily search
            search_results = await tavily_search_async(query_list, **params_to_pass)
            return deduplicate_and_format_sources(search_results)
    
    # Other search APIs...
    # (keep existing code)
```

## Setting Up Your Environment

1. Create a `.env` file with the required API keys:

```
# Tavily API Key
TAVILY_API_KEY=your_tavily_api_key

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# FireCrawl API Configuration
FIRECRAWL_BASE_URL=http://your-firecrawl-api-url.com/api/v1
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Enable FireCrawl
USE_FIRECRAWL=true
```

2. Ensure your FireCrawl API implements the expected endpoint:
   - `POST /scrape`
   - Request body: `{"urls": ["url1", "url2"], "titles": ["title1", "title2"]}`
   - Response format: `{"url1": {"content": "scraped content"}, "url2": {"content": "scraped content"}}`

## Usage Example

Here's how to use your modified Open Deep Research with FireCrawl:

```python
from langgraph.graph import StateGraph, START, END
from open_deep_research.state import ReportState
from open_deep_research.graph import build_graph

# Define configuration
config = {
    "configurable": {
        "search_api": "tavily",
        "use_firecrawl": True,
        "firecrawl_base_url": "http://your-firecrawl-api.com/api/v1",
        "firecrawl_api_key": "your-api-key",
        "planner_model": "gpt-4",
        "writer_model": "gpt-4"
    }
}

# Build the graph with FireCrawl configuration
graph = build_graph()

# Run the graph with a topic
result = graph.invoke({"topic": "Advances in quantum computing"}, config=config)
```

## FireCrawl API Implementation

Your FireCrawl API should:

1. Accept a list of URLs and optional titles
2. Handle web scraping through your firewall
3. Return the scraped content in a structured format

Here's a simplified example of how your API might look:

```python
from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict
import httpx
from bs4 import BeautifulSoup
import asyncio
from pydantic import BaseModel

app = FastAPI()

class ScrapeRequest(BaseModel):
    urls: List[str]
    titles: List[str] = None

@app.post("/api/v1/scrape")
async def scrape_urls(request: ScrapeRequest = Body(...)):
    """Scrape content from a list of URLs."""
    
    if not request.titles:
        request.titles = [f"Page {i+1}" for i in range(len(request.urls))]
        
    results = {}
    
    # Scrape URLs concurrently
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        tasks = [fetch_url(client, url) for url in request.urls]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for url, content in zip(request.urls, contents):
            if isinstance(content, Exception):
                # Handle errors
                results[url] = {"content": f"Error: {str(content)}", "error": True}
            else:
                # Process successful response
                results[url] = {"content": content, "error": False}
    
    return results

async def fetch_url(client, url):
    """Fetch and process a single URL."""
    response = await client.get(url)
    response.raise_for_status()
    
    # Convert HTML to text (you can use your preferred method)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()
```

## Conclusion

By following this guide, you've modified Open Deep Research to:
1. Use Tavily API to find relevant URLs
2. Send these URLs through your custom FireCrawl API
3. Process the returned content for analysis

This approach gives you complete control over the web scraping process while still leveraging the powerful research and analysis capabilities of Open Deep Research.

## Logging and Debugging

To help with debugging, add comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("firecrawl_integration.log"),
        logging.StreamHandler()
    ]
)

# Create loggers
logger = logging.getLogger("firecrawl_integration")
```

Use these loggers throughout your code to track the flow of data between Open Deep Research and your FireCrawl API.
