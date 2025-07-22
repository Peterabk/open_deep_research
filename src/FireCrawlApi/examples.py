"""
Example usage of the FireCrawl API module.

This file provides examples of how to use the FireCrawl API module
both as a direct client and as a FastAPI server.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging with timestamp for examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("firecrawl_examples.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("firecrawl_examples")

# Import module components
from FireCrawlApi import (
    ScrapeRequest,
    BatchScrapeRequest, 
    SearchRequest,
    SearchandScrapeRequest,
    client,
    server
)

async def example_direct_api():
    """Example of using the direct API functions"""
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [examples.example_direct_api] Starting direct API example")
    
    # Example 1: Scrape a single URL
    logger.info(f"{timestamp} - [examples.example_direct_api] Example 1: Scraping single URL")
    result = await client.direct_scrape_url("https://www.ibm.com/think/topics/artificial-intelligence")
    if result['status'] == 'success':
        logger.info(f"{timestamp} - [examples.example_direct_api] Successfully scraped URL: {result['url']}")
        logger.info(f"{timestamp} - [examples.example_direct_api] Title: {result.get('title')}")
    else:
        logger.error(f"{timestamp} - [examples.example_direct_api] Failed to scrape URL: {result.get('error')}")
    
    # Example 2: Scrape multiple URLs
    logger.info(f"{timestamp} - [examples.example_direct_api] Example 2: Scraping multiple URLs")
    results = await client.direct_scrape_multiple_urls(["https://google.com", "https://github.com"])
    logger.info(f"{timestamp} - [examples.example_direct_api] Retrieved {len(results)} results")
    
    return "Direct API examples completed successfully"

async def example_search_and_extract():
    """Example of the combined search and extract functionality"""
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [examples.example_search_and_extract] Starting search and extract example")
    
    # Import the server functions directly for this example
    from FireCrawlApi.server import search_and_extract
    
    search_request = SearchandScrapeRequest(
        search_queries=["artificial intelligence"],
        max_results=2,
        include_raw_content=False  # Set to false to reduce response size
    )
    
    logger.info(f"{timestamp} - [examples.example_search_and_extract] Sending search and extract request")
    results = await search_and_extract(search_request)
    
    for i, result in enumerate(results):
        logger.info(f"{timestamp} - [examples.example_search_and_extract] Query {i+1}: {result.query}")
        logger.info(f"{timestamp} - [examples.example_search_and_extract] Found {len(result.results)} results")
        
        # Log first result details if available
        if result.results and len(result.results) > 0:
            first = result.results[0]
            logger.info(f"{timestamp} - [examples.example_search_and_extract] First result: {first.url}")
            logger.info(f"{timestamp} - [examples.example_search_and_extract] Title: {first.title}")
    
    return "Search and extract example completed successfully"

def run_examples():
    """Run all examples"""
    logger.info(f"{datetime.now().isoformat()} - [examples.run_examples] Running FireCrawl API examples")
    
    # Run the async examples
    asyncio.run(example_direct_api())
    asyncio.run(example_search_and_extract())
    
    logger.info(f"{datetime.now().isoformat()} - [examples.run_examples] All examples completed")

def start_server():
    """Start the FastAPI server"""
    logger.info(f"{datetime.now().isoformat()} - [examples.start_server] Starting FastAPI server")
    server.start(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Start the server if requested
        start_server()
    else:
        # Otherwise run the examples
        run_examples()
