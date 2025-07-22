import re
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyUrl, Field
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
    include_raw_content: bool = True
    
class BatchScrapeRequest(BaseModel):
    """Request model for scraping multiple URLs"""
    urls: List[str]
    include_raw_content: bool = True

class ScrapeResponse(BaseModel):
     # 'results': [                     # List of search results
     # {
     #     'title': str,            # Title of the webpage
     #     'url': str,              # URL of the result
     #     'content': str,          # Summary/snippet of content -> description
     #     'score': float,          # Relevance score
     #     'raw_content': str|None  # Full page content if available -> markdown
     # },
    """Response model for scrape results"""
    title: Optional[str] = None
    url: str
    content: Optional[str] = None
    score: Optional[float] = None
    raw_content: Optional[str] = None
    status: str = "pending"  # Adding default value to make it optional
    error: Optional[str] = None

class ScrapeOptions(BaseModel):
    """Options for controlling scraping behavior"""
    formats: List[str] = ["markdown", "links"]
    onlyMainContent: bool = True
    include_raw_content: bool = True

class SearchRequest(BaseModel):
    """Request model for searching and returning URLs"""
    query: str
    limit: int = 2
    status: str = "pending"  # Adding default value to make it optional

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    urls: List[str]

class SearchandScrapeRequest(BaseModel):
        # Args:
        # search_queries (List[str]): List of search queries to process
        # max_results (int): Maximum number of results to return
        # topic (Literal["general", "news", "finance"]): Topic to filter results by
        # include_raw_content (bool): Whether to include raw content in the results
    search_queries: List[str]
    max_results: int = 5
    topic: str = "general"
    include_raw_content: bool = True

class SearchandScrapeResponse(BaseModel):
    """Response model for search and scrape results matching Tavily API format"""
    query: str
    follow_up_questions: Optional[List[str]] = None
    answer: Optional[str] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[ScrapeResponse]

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

async def scrape_url(url: str, include_raw_content: bool = True) -> Dict[str, Any]:
    """
    Send a scrape request to FireCrawl API for a single URL
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dict with scrape results
    """
    start_time = time.time()
    logger.info(f"Sending scrape request for URL: {url}")
    
    title = None
    description = None
    raw_content = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create request payload
            payload = {
                "url": url,
                "formats": ["html"],
                "onlyMainContent": False,
                #"usePlaywright": True
            }
            
            # Send POST request with Basic Auth
            response = await client.post(
                FIRECRAWL_API_URL,
                json=payload,
                auth=FIRECRAWL_AUTH,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            response_result = response.json()

            #for batch scrape, response_result is going to return data
            #data is holding a list of results so we need to
            #result list = data
            #loop through the result list and do the modifications shown below
            #and return the result list and remove the list in BatchRequest
            
            # Log success
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully scraped URL: {url} in {elapsed_time:.2f}s")

            if(include_raw_content):
                if 'html' in response_result['data']:
                    raw_content = response_result['data']['html']
                else:
                    raw_content = None
            else:
                raw_content = None

            #error handeling:
            if 'title' in response_result['data']['metadata']:
                title = response_result['data']['metadata']['title']

            if 'description' in response_result['data']['metadata']:
                description = response_result['data']['metadata']['description']
            
            # Return response data and status
            return {
                "title": title,
                "url": response_result['data']['metadata']['url'],
                "content": description,
                "raw_content": raw_content,
                "score": None,
                "status": "success"
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

async def batch_scrape_url(urls: List[str], include_raw_content: bool = True) -> List[Dict[str, Any]]:
    """
    Send a scrape request to FireCrawl API for a list of URLs
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        Dict with scrape results
    """
    start_time = time.time()
    logger.info(f"Sending scrape request for URL: {urls}")
    
    title = None
    description = None
    raw_content = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create request payload
            payload = {
                "urls": urls,
                "formats": ["html"],
                "onlyMainContent": False,
                #"usePlaywright": True
            }
            
            # Send POST request with Basic Auth
            response = await client.post(
                "http://api-services.theautobot.ca/firecrawl/v1/batch/scrape",
                json=payload,
                auth=FIRECRAWL_AUTH,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            response_result = response.json()

            # Polling loop to check job status until completion
            logger.info(f"[batch_scrape_url] Started polling job_id={response_result['id']} for completion")
            
            # Setup polling with timeout
            max_attempts = 40  # Maximum polling attempts (5 seconds * 30 = 150 seconds max wait)
            attempts = 0
            poll_interval = 5  # seconds between polls
            job_id = response_result['id']
            url = response_result['url']
            
            while attempts < max_attempts:
                try:
                    logger.info(f"[batch_scrape_url] Polling attempt {attempts+1}/{max_attempts} for job_id={job_id}")
                    response = await client.get(
                        f"http://api-services.theautobot.ca/firecrawl/v1/batch/scrape/{job_id}",
                        auth=FIRECRAWL_AUTH,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if data['success'] == True and data['status'] == 'completed':
                        logger.info(f"[batch_scrape_url] Job completed successfully: job_id={job_id}")
                        break
                    
                    # Status check - log progress if available
                    if data['status'] != 'completed':
                        logger.info(f"[batch_scrape_url] Job progress: {data['status']} for job_id={job_id}")
                        
                    # Wait asynchronously before next poll
                    await asyncio.sleep(poll_interval)
                    attempts += 1
                    
                except Exception as e:
                    logger.error(f"[batch_scrape_url] Error during polling: {str(e)}")
                    await asyncio.sleep(poll_interval)
                    attempts += 1
            
            # Check if we timed out
            if attempts >= max_attempts:
                raise TimeoutError(f"Timed out waiting for batch job to complete after {max_attempts * poll_interval} seconds")

            #for batch scrape, response_result is going to return data
            result_list = data['data']
            #data is holding a list of results so we need to
            #result list = data
            #loop through the result list and do the modifications shown below
            #and return the result list and remove the list in BatchRequest

            results = []
            for result in result_list:
                if(include_raw_content):
                    if 'html' in result:
                        raw_content = result['html']
                    else:
                        raw_content = None
                else:
                    raw_content = None

                #error handeling:
                if 'title' in result['metadata']:
                    title = result['metadata']['title']

                if 'description' in result['metadata']:
                    description = result['metadata']['description']
                
                results.append({
                    "title": title,
                    "url": result['metadata']['url'],
                    "content": description,
                    "raw_content": raw_content,
                    "score": None,
                    "status": "success"
                })
            
            return results
            
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
        
    except TimeoutError as e:
        # Handle timeout on polling
        error_msg = f"[batch_scrape_url] {str(e)}"
        logger.error(error_msg)
        
        # Return error response for each URL in the batch
        return [{
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        } for url in urls]
        
    except Exception as e:
        # Handle other exceptions
        error_msg = f"[batch_scrape_url] Error scraping batch of {len(urls)} URLs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return error response for each URL in the batch
        return [{
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        } for url in urls]

@app.post("/scrape", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_single_url(request: ScrapeRequest):
    """
    Scrape a single URL using the FireCrawl API
    
    Args:
        request: ScrapeRequest with URL to scrape
        
    Returns:
        ScrapeResponse with scrape results
    """
    result = await scrape_url(request.url, include_raw_content=request.include_raw_content)
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
    #tasks = [scrape_url(url) for url in request.urls]
    results = []
    
    # Process each URL and collect results
    result = await batch_scrape_url(request.urls, include_raw_content=request.include_raw_content)
    
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

@app.post("/search-and-extract", response_model=list[SearchandScrapeResponse], tags=["Research"])
async def search_and_extract(request: SearchandScrapeRequest):
    """
    Search for URLs related to a query and extract their content
    
    This endpoint combines the search and scrape operations:
    1. First searches for relevant URLs using FireCrawl search API
    2. Then extracts content from those URLs using FireCrawl extract API
    
    Args:
        request: SearchRequest with query and optional max_results
        
    Returns:
        SearchandScrapeResponse object with query and results array containing extracted contents
        in a format compatible with Tavily API
    """
    SearchandScrapeResponseList = []
    # Step 1: Search for relevant URLs
    for query in request.search_queries:
        search_results = await search_for_urls(SearchRequest(query=query, limit=request.max_results))
        timestamp = datetime.now().isoformat()
        if search_results["status"] != "success" or len(search_results.get("urls", [])) == 0:
            logger.warning(f"{timestamp} - [search_and_extract] No URLs found for query: {query}")
            # Return empty results but maintain the Tavily format
            SearchandScrapeResponseList.append(SearchandScrapeResponse(
                query=request.query,
                results=[],
                follow_up_questions=None,
                answer=None,
                images=[]
            ))
            continue
        else:
            # Step 2: Extract content from each URL
            urls_to_scrape = search_results.get("urls", [])
            logger.info(f"{timestamp} - [search_and_extract] Found {len(urls_to_scrape)} URLs for query: {query}")
            
            # Create a batch request with the URLs
            batch_request = BatchScrapeRequest(urls=urls_to_scrape, include_raw_content=request.include_raw_content)
            
            # Use the existing batch scrape function
            scrape_results = await scrape_multiple_urls(batch_request)
            
            # Format as Tavily-style response
            tavily_style_response = SearchandScrapeResponse(
                query=query,
                results=scrape_results,
                follow_up_questions=None, # Could generate these with an LLM in the future
                answer=None, # Could generate this with an LLM in the future
                images=[] # Could extract images in the future
            )
            
            # Log completion with timing
            # elapsed_time = time.time() - start_time
            logger.info(f"[search_and_extract] Completed processing query with {len(scrape_results)} results")
            
            # Add the response to the list
            SearchandScrapeResponseList.append(tavily_style_response)
    
    # Log completion with timing
    #elapsed_time = time.time() - start_time
    logger.info(f"[search_and_extract] Completed processing all queries")
    
    return SearchandScrapeResponseList

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
        # result = await scrape_single_url(ScrapeRequest(url="https://www.ibm.com/think/topics/artificial-intelligence"))
        # if result['status'] == 'success':
        #     print(f"Single URL result: ")
        #     print(f"Single URL url: {result['url']}")
        #     print(f"Single URL title: {result['title']}")
        #     print(f"Single URL content: {result['content']}")
        #     print(f"Single URL raw_content: {result['raw_content']}")
        
        # # Example 2: Scrape multiple URLs
        # print("\n==== Example 2: Scraping multiple URLs ====")
        # results = await direct_scrape_multiple_urls(["https://google.com", "https://github.com"])
        # print(f"Retrieved {len(results)} results")
        # for i, res in enumerate(results):
        #     print(f"Result {i+1} - URL: {res['url']}, Status: {res['status']}")
        
        # Example 3: Search for URLs by query
        # print("\n==== Example 3: Searching for URLs by query ====")
        # # Create a mock SearchRequest
        # search_request = SearchRequest(query="artificial intelligence", limit=3)
        
        # # Call the search function directly
        # search_result = await search_for_urls(search_request)
        # print(f"Search status: {search_result['status']}")
        # print(f"URLs found: {len(search_result['urls'])}")
        # for i, url in enumerate(search_result['urls']):
        #     print(f"URL {i+1}: {url}")
        
        # Example 4: Search and extract content in one go
        print("\n==== Example 4: Search and extract content ====")
        search_extract_result = await search_and_extract(SearchandScrapeRequest(search_queries=["artificial intelligence"], max_results=3))
        #print(search_extract_result)
        for search in search_extract_result:
            for i in range(len(search.results)):
                print(search.results[i].url)
                print(search.results[i].title)
                print(search.results[i].content)
                print(search.results[i].raw_content)
                print(search.results[i].status)

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