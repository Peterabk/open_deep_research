import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyUrl
from typing import List, Dict, Any, Optional, Union
import httpx
import logging
import time
from datetime import datetime

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("firecrawl_api_wrapper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("firecrawl_api")

# FireCrawl API configuration
FIRECRAWL_API_URL = "http://api-services.theautobot.ca/firecrawl/v1/scrape"
FIRECRAWL_AUTH = ("foo", "123456")  # Username and password for Basic Auth

app = FastAPI(
    title="FireCrawl API Wrapper",
    description="A FastAPI wrapper for the FireCrawl web scraping service",
    version="1.0.0"
)

class ScrapeRequest(BaseModel):
    """Request model for scraping a single URL"""
    url: str
    
class BatchScrapeRequest(BaseModel):
    """Request model for scraping multiple URLs"""
    urls: List[str]

class ScrapeResponse(BaseModel):
    """Response model for scrape results"""
    url: str
    content: Optional[str] = None
    status: str
    timestamp: str
    error: Optional[str] = None

# Direct API functions for use without running the FastAPI server
async def direct_scrape_url(url: str) -> Dict[str, Any]:
    """
    Send a scrape request directly to FireCrawl API without running the FastAPI server
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dict with scrape results
    """
    start_time = time.time()
    logger.info(f"[direct_scrape_url] Sending direct scrape request for URL: {url}")
    
    try:
        # Use the existing scrape_url function
        result = await scrape_url(url)
        
        # Log success with timing information
        elapsed_time = time.time() - start_time
        logger.info(f"[direct_scrape_url] Successfully scraped URL: {url} in {elapsed_time:.2f}s")
        
        return result
    except Exception as e:
        # Enhanced error logging
        error_msg = f"[direct_scrape_url] Error scraping URL {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }

async def direct_scrape_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape multiple URLs directly without running the FastAPI server
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of dictionaries with scrape results
    """
    logger.info(f"[direct_scrape_multiple_urls] Starting batch scrape for {len(urls)} URLs")
    start_time = time.time()
    
    results = []
    for i, url in enumerate(urls):
        try:
            logger.info(f"[direct_scrape_multiple_urls] Processing URL {i+1}/{len(urls)}: {url}")
            result = await direct_scrape_url(url)
            results.append(result)
        except Exception as e:
            # Enhanced error logging for individual URL failures
            error_msg = f"[direct_scrape_multiple_urls] Error processing URL {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results.append({
                "url": url,
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })
    
    # Log completion with timing information
    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"[direct_scrape_multiple_urls] Completed batch scrape: {success_count}/{len(urls)} successful in {elapsed_time:.2f}s")
    
    return results

async def scrape_url(url: str) -> Dict[str, Any]:
    """
    Send a scrape request to FireCrawl API for a single URL
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dict with scrape results
    """
    start_time = time.time()
    logger.info(f"Sending scrape request for URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create request payload
            payload = {"url": url}
            
            # Send POST request with Basic Auth
            response = await client.post(
                FIRECRAWL_API_URL,
                json=payload,
                auth=FIRECRAWL_AUTH,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Log success
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully scraped URL: {url} in {elapsed_time:.2f}s")
            
            # Return response data and status
            return {
                "url": url,
                "content": response.text,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }
            
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors
        error_msg = f"HTTP error {e.response.status_code} for URL: {url}"
        logger.error(error_msg)
        return {
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        # Handle other exceptions
        error_msg = f"Error scraping URL {url}: {str(e)}"
        logger.error(error_msg)
        return {
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }

@app.post("/scrape", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_single_url(request: ScrapeRequest):
    """
    Scrape a single URL using the FireCrawl API
    
    Args:
        request: ScrapeRequest with URL to scrape
        
    Returns:
        ScrapeResponse with scrape results
    """
    result = await scrape_url(request.url)
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
    # Process URLs in parallel
    tasks = [scrape_url(url) for url in request.urls]
    results = []
    
    # Process each URL and collect results
    for url in request.urls:
        result = await scrape_url(url)
        results.append(result)
    
    return results

class ScrapeOptions(BaseModel):
    """Options for controlling scraping behavior"""
    formats: List[str] = ["markdown", "links"]
    onlyMainContent: bool = True

class SearchRequest(BaseModel):
    """Request model for searching and returning URLs"""
    query: str
    limit: int = 2
    scrapeOptions: Optional[ScrapeOptions] = None

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    urls: List[str]
    status: str
    timestamp: str
    error: Optional[str] = None

@app.post("/search", response_model=SearchResponse, tags=["Research"])
async def search_for_urls(request: SearchRequest):
    """
    Search for URLs related to a query using FireCrawl's search API
    
    Args:
        request: SearchRequest with query and optional max_results
        
    Returns:
        SearchResponse with list of relevant URLs
    """
    start_time = time.time()
    logger.info(f"Searching for URLs with query: {request.query}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create search request payload
            payload = {
                "query": request.query,
                "limit": request.limit
            }
            
            # Add scrapeOptions if provided or use default
            # if request.scrapeOptions:
            #     # Use the provided options
            #     payload["scrapeOptions"] = request.scrapeOptions.model_dump(exclude_none=True)
            #     logger.info(f"Using custom scrape options: {payload['scrapeOptions']}")
            # else:
            #     # Use default options
            #     payload["scrapeOptions"] = {
            #         "formats": ["markdown", "links"],
            #         "onlyMainContent": True
            #     }
            
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
            logger.info(f"Successfully retrieved {len(urls)} URLs for query: {request.query} in {elapsed_time:.2f}s")

            return {
                "query": request.query,
                "urls": urls,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors
        error_msg = f"HTTP error {e.response.status_code} for search query: {request.query}"
        logger.error(error_msg)
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
        logger.error(error_msg)
        return {
            "query": request.query,
            "urls": [],
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/search-and-extract", response_model=List[ScrapeResponse], tags=["Research"])
async def search_and_extract(request: SearchRequest):
    """
    Search for URLs related to a query and extract their content
    
    This endpoint combines the search and scrape operations:
    1. First searches for relevant URLs using FireCrawl search API
    2. Then extracts content from those URLs using FireCrawl extract API
    
    Args:
        request: SearchRequest with query and optional max_results
        
    Returns:
        List of ScrapeResponse objects with extracted contents
    """
    # First, search for URLs
    search_result = await search_for_urls(request)
    
    # If search failed, return early with the error
    if search_result["status"] == "error":
        return [{
            "url": "",
            "content": "",
            "status": "error",
            "error": search_result["error"],
            "timestamp": datetime.now().isoformat()
        }]
    
    # If no URLs found, return empty list
    if not search_result["urls"]:
        return [{
            "url": "",
            "content": "",
            "status": "no_results",
            "timestamp": datetime.now().isoformat()
        }]
    
    # For each URL, extract content
    urls = search_result["urls"]
    batch_request = BatchScrapeRequest(urls=urls)
    
    # Use the existing batch scrape function
    return await scrape_multiple_urls(batch_request)

# @app.get("/health", tags=["System"])
# async def health_check():
#     """Check if the API is running"""
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Example usage
def example_scrape():
    import asyncio
    
    async def run_example():
        # Example 1: Scrape a single URL
        # print("\n==== Example 1: Scraping single URL ====")
        # result = await direct_scrape_url("https://google.com")
        # print(f"Single URL result status: {result['status']}")
        # print(f"Content length: {len(result.get('content', ''))} characters")
        
        # # Example 2: Scrape multiple URLs
        # print("\n==== Example 2: Scraping multiple URLs ====")
        # results = await direct_scrape_multiple_urls(["https://google.com", "https://github.com"])
        # print(f"Retrieved {len(results)} results")
        # for i, res in enumerate(results):
        #     print(f"Result {i+1} - URL: {res['url']}, Status: {res['status']}")
        
        # Example 3: Search for URLs by query
        print("\n==== Example 3: Searching for URLs by query ====")
        # Create a mock SearchRequest
        search_request = SearchRequest(query="artificial intelligence", limit=3)
        
        # Call the search function directly
        search_result = await search_for_urls(search_request)
        print(f"Search status: {search_result['status']}")
        print(f"URLs found: {len(search_result['urls'])}")
        for i, url in enumerate(search_result['urls']):
            print(f"URL {i+1}: {url}")
        
        # Example 4: Search and extract content in one go
        # print("\n==== Example 4: Search and extract content ====")
        # search_extract_result = await search_and_extract(search_request)
        # print(f"Search and extract results: {len(search_extract_result)}")
        # for i, res in enumerate(search_extract_result):
        #     print(f"Result {i+1} - URL: {res['url']}, Status: {res['status']}")
        #     if res['status'] == 'success':
        #         # Show a preview of content (first 100 chars)
        #         content_preview = res.get('content', '')[:100] + '...' if len(res.get('content', '')) > 100 else res.get('content', '')
        #         print(f"Content preview: {content_preview}")
    
    # Run the async examples
    asyncio.run(run_example())

if __name__ == "__main__":
    # Uncomment one of the options below
    
    # Option 1: Run the FastAPI server
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Option 2: Run the example scrape directly
    example_scrape()