import pytest
from pathlib import Path
from freeclaw.tools.fs import (
    fs_read,
    fs_write,
    fs_list,
    fs_mkdir,
    fs_rm,
    fs_stat,
    fs_glob,
    fs_diff,
    fs_mv,
    fs_cp
)

# Test content and basic I/O
def test_fs_write_and_read(tool_ctx):
    # Base write
    write_res = fs_write(tool_ctx, path="test.txt", content="L1\nL2\nL3\nL4\n")
    assert write_res["ok"] is True
    
    # Read partial (bounds handling)
    read_res = fs_read(tool_ctx, path="test.txt", start_line=2, end_line=3)
    assert read_res["ok"] is True
    assert read_res["content"] == "L2\nL3\n"
    
    # Write Append mode
    fs_write(tool_ctx, path="test.txt", content="L5\n", mode="append")
    read_all = fs_read(tool_ctx, path="test.txt")
    assert "L5\n" in read_all["content"]

def test_fs_stat_and_glob(tool_ctx):
    fs_write(tool_ctx, path="scripts/test1.py", content="print('hello')", make_parents=True)
    fs_write(tool_ctx, path="scripts/test2.py", content="print('world')", make_parents=True)
    
    # Stat
    st = fs_stat(tool_ctx, path="scripts/test1.py")
    assert st["exists"] is True
    assert st["type"] == "file"
    assert st["size"] > 0
    
    # Glob
    gl = fs_glob(tool_ctx, pattern="scripts/*.py")
    assert gl["ok"] is True
    paths = [r["path"] for r in gl["results"]]
    assert "scripts/test1.py" in paths
    assert "scripts/test2.py" in paths

def test_fs_diff(tool_ctx):
    fs_write(tool_ctx, path="data.json", content='{"a": 1, "b": 2}\n')
    
    # Context diff check
    diff_res = fs_diff(tool_ctx, path="data.json", content='{"a": 1, "b": 3}\n', context_lines=0)
    assert diff_res["ok"] is True
    assert diff_res["changed"] is True
    assert "-{\"a\": 1, \"b\": 2}" in diff_res["diff"]
    assert "+{\"a\": 1, \"b\": 3}" in diff_res["diff"]

def test_fs_mv_cp(tool_ctx):
    # Copy
    fs_write(tool_ctx, path="source.txt", content="copy me")
    fs_cp(tool_ctx, src="source.txt", dst="target.txt")
    
    assert (tool_ctx.root / "target.txt").read_text() == "copy me"
    
    # Move and overwrite boundary
    fs_write(tool_ctx, path="new_source.txt", content="move me")
    # Fail overwrite by default
    with pytest.raises(ValueError):
        fs_mv(tool_ctx, src="new_source.txt", dst="target.txt")
    
    # Force overwrite
    fs_mv(tool_ctx, src="new_source.txt", dst="target.txt", overwrite=True)
    assert (tool_ctx.root / "target.txt").read_text() == "move me"
    assert not (tool_ctx.root / "new_source.txt").exists()

def test_fs_security_boundaries(tool_ctx):
    # Attempt to write outside tool_root
    with pytest.raises(ValueError, match="escapes tool_root"):
        fs_write(tool_ctx, path="../../hack.txt", content="evil")

    # Attempt to read outside tool_root
    with pytest.raises(ValueError, match="escapes tool_root"):
        fs_read(tool_ctx, path="/etc/passwd")
        
    # Attempt to list outside root
    with pytest.raises(ValueError, match="escapes tool_root"):
        fs_list(tool_ctx, path="../../")

def test_fs_mkdir_and_list_recursive(tool_ctx):
    fs_mkdir(tool_ctx, path="deep/nested/folder", parents=True)
    fs_write(tool_ctx, path="deep/nested/folder/file.md", content="Hi")
    
    # Shallow list
    list_res = fs_list(tool_ctx, path=".", recursive=False)
    names = [e["path"] for e in list_res["entries"]]
    assert "deep" in names
    
    # Deep list
    list_deep = fs_list(tool_ctx, path=".", recursive=True, max_depth=5)
    names_deep = [e["path"] for e in list_deep["entries"]]
    assert any("file.md" in n for n in names_deep)

def test_fs_rm_recursive(tool_ctx):
    fs_mkdir(tool_ctx, path="to_delete", parents=True)
    fs_write(tool_ctx, path="to_delete/child.txt", content="bye")
    
    # Missing OK flag avoids crash
    fs_rm(tool_ctx, path="non_existent.txt", missing_ok=True)
    
    # Deleting folder requires recursive
    with pytest.raises(OSError):
        fs_rm(tool_ctx, path="to_delete", recursive=False)
        
    # Force delete
    fs_rm(tool_ctx, path="to_delete", recursive=True)
    assert not (tool_ctx.root / "to_delete").exists()
