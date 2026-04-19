import pytest
import os
import json
from pathlib import Path
from freeclaw.freeclaw.config import (
    load_config,
    write_default_config,
    ClawConfig
)

def test_load_config_defaults(tmp_path):
    # Load from a missing path should return pure defaults
    cfg = load_config(str(tmp_path / "non_existent.json"))
    assert cfg.provider == "nim"
    assert cfg.temperature == 0.7
    assert cfg.web_ui_enabled is True
    assert cfg.assistant_name == "Freebot"

def test_write_and_load_config(tmp_path):
    cfg_file = tmp_path / "config.json"
    written_path = write_default_config(str(cfg_file))
    
    assert written_path.exists()
    
    # Can't write to same path blindly
    with pytest.raises(SystemExit):
        write_default_config(str(cfg_file))
        
    cfg = load_config(str(cfg_file))
    assert cfg.provider == "nim" # default

def test_env_overrides(tmp_path, monkeypatch):
    # Set up a JSON config with some baseline
    cfg_file = tmp_path / "config.json"
    dummy_config = {
        "provider": "openrouter",
        "temperature": 0.5,
        "assistant_name": "OpenAI-Agent",
        "web_ui_enabled": True
    }
    cfg_file.write_text(json.dumps(dummy_config))
    
    # 1. Test json loads correctly
    cfg_json = load_config(str(cfg_file))
    assert cfg_json.provider == "openrouter"
    assert cfg_json.temperature == 0.5
    assert cfg_json.assistant_name == "OpenAI-Agent"
    assert cfg_json.web_ui_enabled is True
    
    # 2. Setup env variables that should override
    monkeypatch.setenv("FREECLAW_PROVIDER", "groq")
    monkeypatch.setenv("FREECLAW_TEMPERATURE", "0.9")
    monkeypatch.setenv("FREECLAW_WEB_UI_ENABLED", "false")
    
    # Load again, env should take priority
    cfg_overridden = load_config(str(cfg_file))
    
    assert cfg_overridden.provider == "groq"
    assert cfg_overridden.temperature == 0.9 # Env beat JSON
    assert cfg_overridden.assistant_name == "OpenAI-Agent" # Untouched, beat Default fallback
    assert cfg_overridden.web_ui_enabled is False # Env beat JSON
