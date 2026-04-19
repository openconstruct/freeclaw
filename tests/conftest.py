import pytest
from pathlib import Path
from freeclaw.freeclaw.tools.fs import ToolContext

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

@pytest.fixture
def temp_tool_root(tmp_path):
    root = tmp_path / "tool_root"
    root.mkdir()
    return root

@pytest.fixture
def tool_ctx(temp_tool_root, temp_workspace):
    return ToolContext.from_config_values(
        tool_root=str(temp_tool_root),
        workspace_dir=str(temp_workspace),
        memory_db_path=str(temp_workspace / "memory.sqlite3"),
        max_read_bytes=1000,
        max_write_bytes=1000,
        max_list_entries=10,
        enable_shell=False,
        enable_custom_tools=False
    )
