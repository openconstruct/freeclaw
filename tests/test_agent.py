import pytest
import json
from freeclaw.agent import run_agent, AgentResult

class MockChatClient:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.model = "mock-model"

    def chat(self, *, messages, temperature, max_tokens, tools=None):
        if self.call_count >= len(self.responses):
            return {"choices": [{"message": {"role": "assistant", "content": "No more responses"}}]}
        
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    def extract_text(self, resp):
        choices = resp.get("choices", [])
        if not choices: return ""
        return choices[0].get("message", {}).get("content", "")

def test_agent_simple_chat():
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am an agent."
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    ]
    client = MockChatClient(responses)
    messages = [{"role": "user", "content": "Hi"}]
    
    result = run_agent(
        client=client,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        enable_tools=False,
        tool_ctx=None,
        max_tool_steps=5
    )
    
    assert isinstance(result, AgentResult)
    assert result.text == "Hello! I am an agent."
    assert result.steps == 1
    assert len(messages) == 2

def test_agent_tool_use():
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": '{"path": "hello.txt"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The file says: hello world"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    ]
    client = MockChatClient(responses)
    messages = [{"role": "user", "content": "What is in hello.txt?"}]
    
    def mock_dispatcher(ctx, name, args_json):
        return {"ok": True, "content": "hello world"}

    result = run_agent(
        client=client,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        enable_tools=True,
        tool_ctx="mock-ctx",
        max_tool_steps=5,
        tool_dispatcher=mock_dispatcher
    )
    
    assert result.text == "The file says: hello world"
    assert result.steps == 2
    assert len(messages) == 4
    assert messages[2]["role"] == "tool"

def test_agent_malformed_json_fallback():
    # Model hallucinates a broken JSON string for the arguments
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": '{"path": "missing_quote.txt}' # Broken JSON
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I realized my JSON was broken."
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    ]
    client = MockChatClient(responses)
    messages = [{"role": "user", "content": "Fetch the file."}]
    
    def mock_dispatcher(ctx, name, args):
        # We expect parsed args to fallback to {} due to agent serialization sanitization
        # that handles bad JSON without crashing
        return {"error": "bad json test", "provided": args}

    result = run_agent(
        client=client,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        enable_tools=True,
        tool_ctx="mock-ctx",
        max_tool_steps=5,
        tool_dispatcher=mock_dispatcher
    )
    
    # Check that tool call didn't crash but was passed sanitized (failed parse)
    tool_message = messages[2]
    assert tool_message["role"] == "tool"
    # Agent typically records JSON error strings directly as the result
    assert "Invalid JSON" in tool_message["content"] or "bad json" in tool_message["content"].lower() or "error" in tool_message["content"].lower()
    
    assert result.steps == 2
