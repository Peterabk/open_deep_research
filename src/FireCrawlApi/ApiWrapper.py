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

@app.post("/search-and-scrape", response_model=List[ScrapeResponse], tags=["Research"])
async def search_and_scrape(query: str, max_results: int = 5):
    """
    Search for URLs related to a query and scrape them
    
    This endpoint would integrate with search APIs like Tavily or OpenAI
    to find relevant URLs and then scrape them.
    
    NOTE: This is a placeholder that would need to be implemented with
    actual search API integration.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of ScrapeResponse objects with scrape results
    """
    # This is where you would integrate with Tavily or other search APIs
    # For now, return a placeholder response
    return [
        {
            "url": "https://example.com",
            "content": "This is a placeholder. Implement search API integration.",
            "status": "placeholder",
            "timestamp": datetime.now().isoformat(),
        }
    ]

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)