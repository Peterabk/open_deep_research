import asyncio
import datetime
from src.open_deep_research.utils import tavily_search_async

async def test_tavily_search():
    """Test function for tavily_search_async"""
    print(f"[{datetime.datetime.now()}] Starting tavily search test...")
    
    # Add breakpoint for debugging if needed
    # breakpoint()
    
    # Sample search queries
    search_queries = ["What are the latest developments in LLM research?", "Advancements in RAG techniques"]
    
    try:
        results = await tavily_search_async(
            search_queries=search_queries,
            max_results=3,
            include_raw_content=True,
            topic="general"
        )
        
        # Print summary of results
        print(f"[{datetime.datetime.now()}] Search completed successfully!")
        print(f"Number of result sets: {len(results)}")
        for i, result in enumerate(results):
            print(f"\nResult set {i+1} for query: '{result['query'] if 'query' in result else search_queries[i]}'")
            print(f"Number of results: {len(result['results']) if 'results' in result else 0}")
            
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Error executing search: {str(e)}")
        # Add debugging breakpoint on error
        breakpoint()

# Run the async test function
if __name__ == "__main__":
    asyncio.run(test_tavily_search())
