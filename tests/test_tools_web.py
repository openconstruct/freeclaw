import pytest
import io
import urllib.request
from freeclaw.freeclaw.tools.web import (
    web_fetch,
    web_search,
    _validate_url,
    _HTMLToText
)

def test_html_parsing():
    parser = _HTMLToText()
    html_raw = "<html><body><h1>Title</h1><p> Paragraph 1 <br> Next </p><script>alert('bad');</script><div>End</div></body></html>"
    parser.feed(html_raw)
    txt = parser.text()
    
    assert "Title" in txt
    assert "Paragraph 1" in txt
    assert "Next" in txt
    assert "End" in txt
    assert "alert('bad')" not in txt

def test_url_validation_ssrf():
    # Valid urls
    assert _validate_url("https://google.com") == "https://google.com"
    
    # Internal Network Blocks (SSRF Prevention)
    with pytest.raises(ValueError, match="Refusing to fetch from private/localhost"):
        _validate_url("http://localhost:8080")
        
    with pytest.raises(ValueError, match="Refusing to fetch from private/localhost"):
        _validate_url("http://127.0.0.1/admin")
        
    # Block Credentials
    with pytest.raises(ValueError, match="must not include credentials"):
        _validate_url("https://user:pass@example.com")
        
    # HTTP/HTTPS only
    with pytest.raises(ValueError, match="Only http/https"):
        _validate_url("ftp://server.com/file")

def test_web_fetch_mocked(tool_ctx, mocker):
    mock_html = b"<html><body><p>This is mocked web content</p></body></html>"
    
    class MockResponse:
        def __init__(self, data):
            self.data = data
            self.status = 200
            self.headers = {"Content-Type": "text/html; charset=utf-8"}
            
        def read(self, amt=None):
            return self.data
            
        def geturl(self):
            return "https://mocked.com"
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock python's urllib
    mocker.patch("urllib.request.OpenerDirector.open", return_value=MockResponse(mock_html))
    
    # Run fetch
    res = web_fetch(ctx=tool_ctx, url="https://mocked.com")
    
    assert res["ok"] is True
    assert res["status"] == 200
    assert "This is mocked web content" in res["text"]

def test_web_search_mocked(tool_ctx, mocker):
    class MockDDGS:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def text(self, keywords, safesearch, max_results):
            # Mock generator of search results
            yield {"title": "Python", "href": "https://python.org", "snippet": "A programming language."}
            yield {"title": "Snake", "href": "https://wiki.org/snake", "snippet": "An animal."}
            
    import sys
    from unittest.mock import MagicMock
    
    mock_module = MagicMock()
    mock_module.DDGS = MockDDGS
    mocker.patch.dict("sys.modules", {"ddgs": mock_module})
    
    res = web_search(tool_ctx, query="python")
    assert res["ok"] is True
    assert len(res["results"]) == 2
    assert res["results"][0]["title"] == "Python"
    assert res["results"][0]["url"] == "https://python.org"
