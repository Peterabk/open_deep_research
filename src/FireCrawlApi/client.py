"""
Client module for direct interactions with the FireCrawl API.

This module provides functions for interacting with FireCrawl API
without the need to run the FastAPI server.
"""

import asyncio
import httpx
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Import from local package
from . import FIRECRAWL_API_URL, FIRECRAWL_AUTH

logger = logging.getLogger("firecrawl_api.client")

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
        include_raw_content: Whether to include the raw HTML content in the response
        
    Returns:
        Dict with scrape results
    """
    start_time = time.time()
    logger.info(f"[scrape_url] Sending scrape request for URL: {url}")
    
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
            
            # Log success
            elapsed_time = time.time() - start_time
            logger.info(f"[scrape_url] Successfully scraped URL: {url} in {elapsed_time:.2f}s")

            if(include_raw_content):
                if 'html' in response_result['data']:
                    raw_content = response_result['data']['html']
                else:
                    raw_content = None
            else:
                raw_content = None

            #error handling:
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
        error_msg = f"[scrape_url] HTTP error {e.response.status_code} for URL: {url}"
        logger.error(error_msg)
        return {
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        # Handle other exceptions
        error_msg = f"[scrape_url] Error scraping URL {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
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
        include_raw_content: Whether to include the raw HTML content in the response
        
    Returns:
        List of dictionaries with scrape results
    """
    start_time = time.time()
    logger.info(f"[batch_scrape_url] Sending batch scrape request for {len(urls)} URLs")
    
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
                    logger.error(f"[batch_scrape_url] Error during polling: {str(e)}", exc_info=True)
                    await asyncio.sleep(poll_interval)
                    attempts += 1
            
            # Check if we timed out
            if attempts >= max_attempts:
                raise TimeoutError(f"Timed out waiting for batch job to complete after {max_attempts * poll_interval} seconds")

            #for batch scrape, response_result is going to return data
            result_list = data['data']

            results = []
            for result in result_list:
                if(include_raw_content):
                    if 'html' in result:
                        raw_content = result['html']
                    else:
                        raw_content = None
                else:
                    raw_content = None

                #error handling:
                if 'title' in result['metadata']:
                    title = result['metadata']['title']

                if 'description' in result['metadata']:
                    description = result['metadata']['description']
                
                results.append({
                    "title": title,
                    "url": result['metadata']['url'],
                    "content": description,
                    "raw_content": raw_content,
                    "score": 0.8,
                    "status": "success"
                })
            
            return results
            
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors
        error_msg = f"[batch_scrape_url] HTTP error {e.response.status_code}"
        logger.error(error_msg)
        return [{
            "url": url,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        } for url in urls]
        
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
