import pytest
from freeclaw.tools.task_scheduler import (
    task_add,
    task_list,
    task_disable,
    task_enable,
    task_update,
    task_run_now
)

def test_task_add_duplicate(tool_ctx):
    # Add a normal task
    task_add(tool_ctx, minutes=10, task="Review logs")
    
    # Try duplicate without flag
    dup_res = task_add(tool_ctx, minutes=10, task="Review logs")
    assert dup_res["added"] is False 
    assert dup_res["reason"] == "duplicate"
    
    # Force duplicate
    force_res = task_add(tool_ctx, minutes=10, task="Review logs", allow_duplicate=True)
    assert force_res["added"] is True

def test_task_add_and_list(tool_ctx):
    task_add(tool_ctx, minutes=10, task="Review logs")
    task_add(tool_ctx, minutes=60, task="Backup database")
    
    # Format compliance check
    tasks_file = tool_ctx.workspace / "tasks.md"
    content = tasks_file.read_text()
    assert "10-Review logs" in content
    assert "60-Backup database" in content
    
    # List tasks returning parsed items
    list_res = task_list(tool_ctx)
    assert list_res["ok"] is True
    tasks = list_res["tasks"]
    assert any(t["task"] == "Review logs" for t in tasks)

def test_task_malformed_lines_parsing(tool_ctx):
    # Insert garbage into the file
    tasks_file = tool_ctx.workspace / "tasks.md"
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text("# tasks\nRandom garbage here\n[x] invalid format\n10-Valid Task", encoding="utf-8")
    
    # Ensure it doesn't crash and reads what it can
    list_res = task_list(tool_ctx)["tasks"]
    assert len(list_res) == 1
    assert list_res[0]["task"] == "Valid Task"
    assert list_res[0]["minutes"] == 10

def test_task_disable_enable(tool_ctx):
    task_add(tool_ctx, minutes=5, task="Should be disabled")
    list_before = task_list(tool_ctx)["tasks"]
    task_id = next(t["id"] for t in list_before if "disabled" in t["task"])
    
    # Disable
    task_disable(tool_ctx, task_id=task_id)
    content = (tool_ctx.workspace / "tasks.md").read_text()
    assert "# 5-Should be disabled" in content or "#  5-Should be disabled" in content
    
    # Enable back
    task_enable(tool_ctx, task_id=task_id)
    enabled_task = next(t for t in task_list(tool_ctx)["tasks"] if t["id"] == task_id)
    assert enabled_task["enabled"] is True

def test_task_update(tool_ctx):
    task_add(tool_ctx, minutes=15, task="Old task")
    tasks = task_list(tool_ctx)["tasks"]
    t_id = tasks[0]["id"]
    
    # Update minutes and text
    upd_res = task_update(tool_ctx, task_id=t_id, new_minutes=30, new_task="New task")
    assert upd_res["ok"] is True
    
    content = (tool_ctx.workspace / "tasks.md").read_text()
    assert "30-New task" in content
    assert "15-Old task" not in content

def test_task_run_now(tool_ctx):
    task_add(tool_ctx, minutes=10, task="Run me now")
    tasks = task_list(tool_ctx)["tasks"]
    t_id = tasks[-1]["id"]
    
    res = task_run_now(tool_ctx, task_id=t_id)
    assert res["ok"] is True
    assert res["armed"] is True
