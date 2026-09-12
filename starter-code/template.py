"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
from google import genai
from google.genai import types
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast
import re

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Bạn phải làm việc theo từng bước.

Ở MỖI LẦN trả lời, CHỈ được làm MỘT trong hai việc:

1. Nếu cần dùng tool:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}

Sau đó DỪNG. Không được tự viết Observation.
Không được tự gọi tool.
Không được viết Final Answer trong cùng lượt.

2. Nếu đã có đủ thông tin:
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>

Observation sẽ được Python cung cấp ở lượt tiếp theo sau khi tool thực sự được thực thi.
"""

GEMINI_MODEL = "gemini-3.1-flash-lite"

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self):
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or "mock-key"
        )
        self.client = genai.Client(api_key=self.api_key)
        
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(tools="Không sử dụng công cụ."),
            temperature=0.0,
        )

    def query(self, user_input: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_input,
                config=self.config,
            )
            return {
                "status": "success",
                "tool_calls": [],
                "response": response.text
            }
        except Exception as e:
            return {
                "status": "error",
                "tool_calls": [],
                "error": str(e)
            }


class ChatbotWithTool:
    def __init__(self):
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or "mock-key"
        )
        self.client = genai.Client(api_key=self.api_key)
        
        tools_str = json.dumps(TOOL_DEFINITIONS, indent=2, ensure_ascii=False)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(tools=tools_str),
            tools=TOOL_DEFINITIONS, # Đã bỏ comment để kích hoạt Native Tools
            temperature=0.0,
        )

    def query(self, user_input: str) -> dict:
        try:
            chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=self.config
            )
            response = chat.send_message(user_input)
            
            tool_calls = []
            if response.function_calls:
                for fc in response.function_calls:
                    tool_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {}
                    })
            
            return {
                "status": "success",
                "tool_calls": tool_calls,
                "response": response.text or ""
            }
        except Exception as e:
            return {
                "status": "error",
                "tool_calls": [],
                "error": str(e)
            }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop (Phiên bản rút gọn)"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
        self.client = genai.Client(api_key=self.api_key)
        
        tools_str = json.dumps(TOOL_DEFINITIONS, ensure_ascii=False)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(tools=tools_str),
            temperature=0.0,
        )

    def run(self, user_input: str) -> dict:
        self.trace = []
        tool_calls = []
        context = f"User: {user_input}\n"

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=context,
                    config=self.config
                )
                llm_response = response.text or ""
            except Exception as e:
                return {
                    "status": "error",
                    "tool_calls": tool_calls,
                    "answer": "",
                    "response": "",
                    "iterations": self._get_iterations(tool_calls),
                    "trace": self.trace,
                    "error": f"Lỗi API: {e}"
                }

            # Một trace entry = một agent step
            step = {
                "iteration": iteration,
                "llm_response": llm_response
            }

            # Nếu LLM đã có Final Answer
            if "Final Answer:" in llm_response:
                final_answer = llm_response.split(
                    "Final Answer:", 1
                )[1].strip()

                step["final_answer"] = final_answer
                self.trace.append(step)

                return {
                    "status": "completed",
                    "tool_calls": tool_calls,
                    "answer": final_answer,
                    "response": final_answer,
                    "iterations": self._get_iterations(tool_calls),
                    "trace": self.trace
                }

            # Parse Action
            match = re.search(
                r"Action:\s*(\{.*\})",
                llm_response,
                re.DOTALL
            )

            if not match:
                observation = (
                    "Bạn phải trả về Action chứa JSON "
                    "hoặc Final Answer."
                )
                step["observation"] = observation
                self.trace.append(step)

                context += f"{llm_response}\n"
                context += f"Observation: {observation}\n"
                continue

            try:
                action_data = json.loads(match.group(1))

                t_name = action_data.get("name")
                t_args = action_data.get("args", {})

                tool_calls.append({
                    "name": t_name,
                    "args": t_args
                })

                step["action"] = {
                    "name": t_name,
                    "args": t_args
                }

                # Execute tool
                if t_name in TOOL_MAP:
                    res = TOOL_MAP[t_name](**t_args)

                    observation = (
                        res
                        if isinstance(res, str)
                        else json.dumps(
                            res,
                            ensure_ascii=False
                        )
                    )
                else:
                    observation = (
                        f"Công cụ '{t_name}' không tồn tại."
                    )

            except Exception as e:
                observation = (
                    f"Lỗi parse JSON hoặc chạy công cụ: {e}"
                )

            step["observation"] = observation
            self.trace.append(step)

            context += f"{llm_response}\n"
            context += f"Observation: {observation}\n"

        # Hết số iteration cho phép
        return {
            "status": "max_iterations_reached",
            "tool_calls": tool_calls,
            "answer": "",
            "response": "",
            "iterations": self._get_iterations(tool_calls),
            "trace": self.trace,
            "error": "Đạt giới hạn vòng lặp."
        }

    def _error_result(self, error_msg: str, tool_calls: list, iterations: int = 0) -> dict:
        self.trace.append({"step": "error", "error": error_msg})
        return {"status": "error", "tool_calls": tool_calls, "response": "", "error": error_msg, "iterations": iterations}
    
    def _get_iterations(self, tool_calls):
        if len(tool_calls) <= 1:
            return 1
        return len(tool_calls) + 1

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()