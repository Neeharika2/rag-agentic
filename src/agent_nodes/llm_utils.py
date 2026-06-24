import os
import json
from typing import Type, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def _load_env_file():
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

_load_env_file()

class StructuredLLMWrapper:
    def __init__(self, structured_llm: Any, schema_json: str):
        self.structured_llm = structured_llm
        self.schema_json = schema_json
        
    def invoke(self, input_data: Any, *args, **kwargs) -> Any:
        injection = (
            f"\n\nIMPORTANT: You MUST respond strictly in valid JSON format matching the following JSON schema:\n"
            f"{self.schema_json}"
        )
        if isinstance(input_data, str):
            input_data += injection
        elif isinstance(input_data, list):
            for i in range(len(input_data) - 1, -1, -1):
                msg = input_data[i]
                if hasattr(msg, "content"):
                    msg.content += injection
                    break
                elif isinstance(msg, tuple) and msg[0] == "user":
                    input_data[i] = (msg[0], msg[1] + injection)
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    msg["content"] = msg.get("content", "") + injection
                    break
        return self.structured_llm.invoke(input_data, *args, **kwargs)

def get_llm(temperature: float = 0.0):
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
    if deepseek_key:
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        return ChatOpenAI(
            model=model_name,
            api_key=deepseek_key,
            base_url=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=0
        )

def get_structured_llm(schema: Type[BaseModel], temperature: float = 0.0):
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    llm = get_llm(temperature=temperature)
    
    if deepseek_key:
        # DeepSeek uses method="json_mode" for structured outputs
        structured_llm = llm.with_structured_output(schema, method="json_mode")
        
        # Build schema JSON string
        if hasattr(schema, "model_json_schema"):
            schema_dict = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            schema_dict = schema.schema()
        else:
            schema_dict = schema
            
        schema_json = json.dumps(schema_dict, indent=2)
        return StructuredLLMWrapper(structured_llm, schema_json)
    else:
        return llm.with_structured_output(schema)
