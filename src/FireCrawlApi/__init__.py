"""
FireCrawlApi - A module for interacting with the FireCrawl web scraping service

This module provides both direct API client functions and a FastAPI server wrapper
for interacting with the FireCrawl scraping service.

Example usage:
    # As a direct client
    from FireCrawlApi import client
    
    async def example():
        result = await client.direct_scrape_url("https://example.com")
        print(result)
    
    # Run the FastAPI server
    from FireCrawlApi import server
    server.start()
"""

import logging
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

# Import public components
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

from .client import (
    direct_scrape_url,
    direct_scrape_multiple_urls,
    scrape_url,
    batch_scrape_url
)

__all__ = [
    # Models
    "ScrapeRequest", 
    "BatchScrapeRequest",
    "ScrapeResponse", 
    "ScrapeOptions",
    "SearchRequest", 
    "SearchResponse",
    "SearchandScrapeRequest", 
    "SearchandScrapeResponse",
    
    # Client functions
    "direct_scrape_url",
    "direct_scrape_multiple_urls",
    "scrape_url",
    "batch_scrape_url",
    
    # Module components
    "client",
    "server",
    "models"
]

# Version info
__version__ = "1.0.0"
