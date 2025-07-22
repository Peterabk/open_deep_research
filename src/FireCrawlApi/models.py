"""
Pydantic models for the FireCrawl API wrapper.

This module contains all the request and response models used for interacting
with the FireCrawl API service.
"""

import logging
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal

logger = logging.getLogger("firecrawl_api.models")

class ScrapeRequest(BaseModel):
    """Request model for scraping a single URL"""
    url: str
    include_raw_content: bool = True
    
class BatchScrapeRequest(BaseModel):
    """Request model for scraping multiple URLs"""
    urls: List[str]
    include_raw_content: bool = True

class ScrapeResponse(BaseModel):
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
    """Request model for combined search and scrape operations"""
    search_queries: List[str]
    max_results: int = 5
    topic: Literal["general", "news", "finance"] = "general"
    include_raw_content: bool = True

class SearchandScrapeResponse(BaseModel):
    """Response model for search and scrape results matching Tavily API format"""
    query: str
    follow_up_questions: Optional[List[str]] = None
    answer: Optional[str] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[ScrapeResponse]
