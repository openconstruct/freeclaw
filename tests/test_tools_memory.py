import pytest
from freeclaw.freeclaw.tools.memory import (
    memory_add,
    memory_get,
    memory_list,
    memory_search,
    memory_delete
)

def test_memory_add_and_get(tool_ctx):
    # Testing meta dictionary and robust inputs
    meta_info = {"app_version": "1.0", "source": "test"}
    add_res = memory_add(
        tool_ctx,
        content="The user's favorite color is blue.",
        key="user_color",
        tags=["user-pref", "color"],
        meta=meta_info
    )
    assert add_res["ok"] is True
    item = add_res["item"]
    assert item["content"] == "The user's favorite color is blue."
    assert item["key"] == "user_color"
    assert "user-pref" in item["tags"]
    assert item["meta"] == meta_info
    
    # Validation by keys
    get_res_key = memory_get(tool_ctx, key="user_color")
    assert get_res_key["ok"] is True
    assert get_res_key["found"] is True
    assert get_res_key["item"]["id"] == item["id"]
    
    # Validation by IDs
    get_res_id = memory_get(tool_ctx, id=item["id"])
    assert get_res_id["ok"] is True
    assert get_res_id["item"]["key"] == "user_color"

def test_memory_upsert(tool_ctx):
    # Verify idempotency
    memory_add(tool_ctx, content="Version 1", key="my_key", tags=["v1"])
    add_res = memory_add(tool_ctx, content="Version 2", key="my_key", tags=["v2"])
    
    assert add_res["ok"] is True
    assert add_res["item"]["content"] == "Version 2"
    assert "v2" in add_res["item"]["tags"]
    
    # Ensure there is only 1 item with this key
    get_res = memory_get(tool_ctx, key="my_key")
    assert get_res["item"]["content"] == "Version 2"

def test_memory_list_pagination(tool_ctx):
    # Insert multiple items
    for i in range(10):
        memory_add(tool_ctx, content=f"Record {i}", key=f"r_{i}")
        
    # Offset & Limit test
    list_res = memory_list(tool_ctx, limit=3, offset=0, include_content=True)
    assert len(list_res["results"]) == 3
    assert list_res["total"] == 10
    assert list_res["results"][0]["key"] == "r_9" # By descending order
    
    list_res_off = memory_list(tool_ctx, limit=5, offset=3, include_content=True)
    assert len(list_res_off["results"]) == 5
    assert list_res_off["results"][0]["key"] == "r_6"

def test_memory_list_and_ttl_with_mock_time(tool_ctx, mocker):
    import time
    base_time = 1000000
    
    # Mock inner _now function so we don't need real sleep
    mock_now = mocker.patch("freeclaw.freeclaw.tools.memory._now", return_value=base_time)
    
    # Add 3 items
    memory_add(tool_ctx, content="Item A", pinned=True)
    memory_add(tool_ctx, content="Item B")
    memory_add(tool_ctx, content="Item C", ttl_seconds=10)
    
    # List immediately (all valid)
    res_now = memory_list(tool_ctx, limit=10, include_content=True)
    contents_now = [i["content"] for i in res_now["results"]]
    assert "Item C" in contents_now
    
    # Time Travel -> +20 seconds
    mock_now.return_value = base_time + 20
    
    list_res = memory_list(tool_ctx, limit=10, include_content=True)
    assert list_res["ok"] is True
    
    # Item C shouldn't appear because it's expired in mocked future
    contents = [i["content"] for i in list_res["results"]]
    assert "Item A" in contents
    assert "Item B" in contents
    assert "Item C" not in contents
    
    # With expired included
    list_res_exp = memory_list(tool_ctx, limit=10, include_expired=True, include_content=True)
    contents_exp = [i["content"] for i in list_res_exp["results"]]
    assert "Item C" in contents_exp

def test_memory_search(tool_ctx):
    memory_add(tool_ctx, content="The secret code is 12345")
    memory_add(tool_ctx, content="The access protocol requires an access badge")
    memory_add(tool_ctx, content="Something totally unrelated")
    
    # Standard fuzzy match
    search_res = memory_search(tool_ctx, query="secret code")
    assert search_res["ok"] is True
    results = search_res["results"]
    assert any("12345" in r["content"] for r in results)
    
    # Empty query check
    with pytest.raises(ValueError, match="query is required"):
        memory_search(tool_ctx, query="   ")
        
def test_memory_delete(tool_ctx):
    res = memory_add(tool_ctx, content="To be deleted", key="del_key")
    
    # Delete by key
    del_res = memory_delete(tool_ctx, key="del_key")
    assert del_res["ok"] is True
    assert del_res["deleted"] == 1
    
    # Non existent deletion doesn't crash but results in 0 rows changed
    del_res_2 = memory_delete(tool_ctx, key="del_key")
    assert del_res_2["deleted"] == 0
    
    # Verify it is gone
    get_res = memory_get(tool_ctx, key="del_key")
    assert get_res["found"] is False
