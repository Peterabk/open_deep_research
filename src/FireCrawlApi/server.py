"""
Server module for FastAPI implementation of the FireCrawl API wrapper.

This module provides a FastAPI server that can be used to interact with
the FireCrawl API service via HTTP endpoints.
"""

import re
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Import from local package
from .models import (
    ScrapeRequest, 
    BatchScrapeRequest,
    ScrapeResponse, 
    ScrapeOptions,
    SearchRequest, 
    SearchResponse,
    SearchandScrapeRequest, 
    SearchandScrapeResponse
)
from .client import scrape_url, batch_scrape_url
from . import FIRECRAWL_AUTH

logger = logging.getLogger("firecrawl_api.server")

# Create FastAPI app
app = FastAPI(
    title="FireCrawl API Wrapper",
    description="A FastAPI wrapper for the FireCrawl web scraping service",
    version="1.0.0"
)

@app.post("/scrape", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_single_url(request: ScrapeRequest):
    """
    Scrape a single URL using the FireCrawl API
    
    Args:
        request: ScrapeRequest with URL to scrape
        
    Returns:
        ScrapeResponse with scrape results
    """
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.scrape_single_url] Processing request for URL: {request.url}")
    
    result = await scrape_url(request.url, include_raw_content=request.include_raw_content)
    
    logger.info(f"{timestamp} - [server.scrape_single_url] Completed request for URL: {request.url}, status: {result.get('status')}")
    return result

@app.post("/scrape/batch", response_model=List[ScrapeResponse], tags=["Scraping"])
async def scrape_multiple_urls(request: BatchScrapeRequest):
    """
    Scrape multiple URLs in parallel using the FireCrawl API
    
    Args:
        request: BatchScrapeRequest with URLs to scrape
        
    Returns:
        List of ScrapeResponse objects with scrape results
    """
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.scrape_multiple_urls] Processing batch request for {len(request.urls)} URLs")
    
    # Process each URL and collect results
    result = await batch_scrape_url(request.urls, include_raw_content=request.include_raw_content)
    
    success_count = sum(1 for r in result if r.get("status") == "success")
    logger.info(f"{timestamp} - [server.scrape_multiple_urls] Completed batch request: {success_count}/{len(request.urls)} successful")
    
    return result

@app.post("/search", response_model=SearchResponse, tags=["Research"])
async def search_for_urls(request: SearchRequest):
    """
    Search for URLs related to a query using FireCrawl's search API
    
    Args:
        request: SearchRequest with query and optional max_results
        
    Returns:
        SearchResponse with list of relevant URLs
    """
    import httpx
    
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.search_for_urls] Searching for URLs with query: {request.query}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create search request payload
            payload = {
                "query": request.query,
                "limit": request.limit
            }
            
            # Send POST request with Basic Auth to FireCrawl search API
            response = await client.post(
                "http://api-services.theautobot.ca/firecrawl/v1/search",
                json=payload,
                auth=FIRECRAWL_AUTH,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            search_result = response.json()

            # Initialize URLs list
            urls = []
            
            # Extract URLs from the correct nested location in the response
            if 'data' in search_result:
                for result in search_result['data']:
                    if 'url' in result:
                        urls.append(result['url'])
            
            # Log success
            elapsed_time = time.time() - start_time
            logger.info(f"{timestamp} - [server.search_for_urls] Successfully retrieved {len(urls)} URLs for query: {request.query} in {elapsed_time:.2f}s")

            return {
                "query": request.query,
                "urls": urls,
                "status": "success",
            }
            
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors
        error_msg = f"HTTP error {e.response.status_code} for search query: {request.query}"
        logger.error(f"{timestamp} - [server.search_for_urls] {error_msg}")
        return {
            "query": request.query,
            "urls": [],
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Handle other exceptions
        error_msg = f"Error searching for query {request.query}: {str(e)}"
        logger.error(f"{timestamp} - [server.search_for_urls] {error_msg}", exc_info=True)
        return {
            "query": request.query,
            "urls": [],
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/search-and-extract", response_model=List[SearchandScrapeResponse], tags=["Research"])
async def search_and_extract(request: SearchandScrapeRequest):
    """
    Search for URLs related to a query and extract their content
    
    This endpoint combines the search and scrape operations:
    1. First searches for relevant URLs using FireCrawl search API
    2. Then extracts content from those URLs using FireCrawl extract API
    
    Args:
        request: SearchandScrapeRequest with query and optional max_results
        
    Returns:
        SearchandScrapeResponse object with query and results array containing extracted contents
        in a format compatible with Tavily API
    """
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.search_and_extract] Processing request with {len(request.search_queries)} queries")
    
    SearchandScrapeResponseList = []
    # Step 1: Search for relevant URLs
    for query in request.search_queries:
        search_results = await search_for_urls(SearchRequest(query=query, limit=request.max_results))
        
        if search_results["status"] != "success" or len(search_results.get("urls", [])) == 0:
            logger.warning(f"{timestamp} - [server.search_and_extract] No URLs found for query: {query}")
            # Return empty results but maintain the Tavily format
            SearchandScrapeResponseList.append({
                "query": query,
                "results": [],
                "follow_up_questions": None,
                "answer": None,
                "images": []
            })
            continue
        else:
            # Step 2: Extract content from each URL
            urls_to_scrape = search_results.get("urls", [])
            logger.info(f"{timestamp} - [server.search_and_extract] Found {len(urls_to_scrape)} URLs for query: {query}")
            
            # Create a batch request with the URLs
            batch_request = BatchScrapeRequest(urls=urls_to_scrape, include_raw_content=request.include_raw_content)
            
            # Use the existing batch scrape function
            scrape_results = await scrape_multiple_urls(batch_request)
            
            # Format as Tavily-style response
            tavily_style_response = {
                "query": query,
                "results": scrape_results,
                "follow_up_questions": None, # Could generate these with an LLM in the future
                "answer": None, # Could generate this with an LLM in the future
                "images": [] # Could extract images in the future
            }
            
            # Log completion
            logger.info(f"{timestamp} - [server.search_and_extract] Completed processing query '{query}' with {len(scrape_results)} results")
            
            # Add the response to the list
            SearchandScrapeResponseList.append(tavily_style_response)
    
    # Log completion
    logger.info(f"{timestamp} - [server.search_and_extract] Completed processing all {len(request.search_queries)} queries")
    
    return SearchandScrapeResponseList

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API is running"""
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.health_check] Health check requested")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def start(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    timestamp = datetime.now().isoformat()
    logger.info(f"{timestamp} - [server.start] Starting FireCrawl API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start()
