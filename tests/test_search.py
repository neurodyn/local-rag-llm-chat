import pytest
import os
from unittest.mock import patch, MagicMock
from app.services.rag_service import WebSearch, RAGService

@pytest.fixture
def mock_ddgs():
    """Mock DuckDuckGo search results."""
    mock_results = [
        {
            'title': 'Test Result 1',
            'body': 'This is a test result from DuckDuckGo'
        },
        {
            'title': 'Test Result 2',
            'body': 'Another test result from DuckDuckGo'
        }
    ]
    with patch('app.services.rag_service.DDGS') as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_instance
        yield mock_ddgs

@pytest.fixture
def mock_serpapi():
    """Mock SerpAPI search results."""
    mock_results = {
        'organic_results': [
            {
                'title': 'Test Result 1',
                'snippet': 'This is a test result from SerpAPI'
            },
            {
                'title': 'Test Result 2',
                'snippet': 'Another test result from SerpAPI'
            }
        ]
    }
    with patch('app.services.rag_service.GoogleSearch') as mock_serpapi:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = mock_results
        mock_serpapi.return_value = mock_instance
        yield mock_serpapi

def test_duckduckgo_search(mock_ddgs):
    """Test DuckDuckGo search functionality."""
    search = WebSearch(search_engine="duckduckgo")
    results = search.run("test query")
    
    # Verify the results format
    assert "Test Result 1" in results
    assert "This is a test result from DuckDuckGo" in results
    assert "Test Result 2" in results
    assert "Another test result from DuckDuckGo" in results
    
    # Verify DuckDuckGo was called correctly
    mock_ddgs.return_value.text.assert_called_once_with("test query", max_results=3)

def test_serpapi_search(mock_serpapi):
    """Test SerpAPI search functionality."""
    # Set up environment variable for SerpAPI
    with patch.dict(os.environ, {'SERPAPI_API_KEY': 'test_key'}):
        search = WebSearch(search_engine="serpapi")
        results = search.run("test query")
        
        # Verify the results format
        assert "Test Result 1" in results
        assert "This is a test result from SerpAPI" in results
        assert "Test Result 2" in results
        assert "Another test result from SerpAPI" in results
        
        # Verify SerpAPI was called correctly
        mock_serpapi.assert_called_once_with({
            "q": "test query",
            "api_key": "test_key",
            "num": 3
        })

def test_serpapi_fallback_to_duckduckgo(mock_ddgs, mock_serpapi):
    """Test that SerpAPI falls back to DuckDuckGo when no API key is present."""
    # Ensure no SERPAPI_API_KEY is set
    with patch.dict(os.environ, {}, clear=True):
        search = WebSearch(search_engine="serpapi")
        results = search.run("test query")
        
        # Verify DuckDuckGo was used instead
        mock_ddgs.return_value.text.assert_called_once_with("test query", max_results=3)
        assert "Test Result 1" in results
        assert "This is a test result from DuckDuckGo" in results

def test_search_error_handling():
    """Test error handling in search functionality."""
    with patch('app.services.rag_service.DDGS') as mock_ddgs:
        # Simulate an error in DuckDuckGo search
        mock_instance = MagicMock()
        mock_instance.text.side_effect = Exception("Search error")
        mock_ddgs.return_value = mock_instance
        
        search = WebSearch(search_engine="duckduckgo")
        results = search.run("test query")
        
        # Verify empty string is returned on error
        assert results == ""

def test_rag_service_search_engine_config():
    """Test RAGService search engine configuration."""
    # Test default search engine
    rag_service = RAGService()
    assert rag_service.search_engine == "duckduckgo"
    
    # Test setting search engine
    rag_service = RAGService(search_engine="serpapi")
    assert rag_service.search_engine == "serpapi"
    
    # Test case insensitivity
    rag_service = RAGService(search_engine="SERPAPI")
    assert rag_service.search_engine == "serpapi"

@pytest.mark.asyncio
async def test_rag_service_search_engine_change():
    """Test changing search engine in RAGService."""
    rag_service = RAGService()
    assert rag_service.search_engine == "duckduckgo"
    
    # Change search engine
    rag_service.search_engine = "serpapi"
    rag_service.search_tool = None
    await rag_service.ensure_initialized()
    
    assert rag_service.search_engine == "serpapi"
    assert isinstance(rag_service.search_tool, WebSearch)
    assert rag_service.search_tool.search_engine == "serpapi" 