"""
BẢN MẪU LANGGRAPH: viết lại đúng vòng lặp agent ở agent_demo.py dưới dạng đồ thị.

Cài đặt:
    pip install -U langgraph langchain-google-genai
    (dùng lại biến môi trường GEMINI_API_KEY như trước)

Chạy:
    python agent_langgraph_mau.py

Đối chiếu với bản thủ công:
    contents (list)             ->  State["messages"]
    bước (a) gọi model          ->  node "call_model"
    bước (d) chạy tool          ->  node "run_tools"
    bước (c) có function_call?  ->  conditional edge "should_continue"
    vòng for + MAX_STEPS        ->  cạnh run_tools -> call_model + recursion_limit
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# langchain-google-genai thường đọc GOOGLE_API_KEY; nếu bạn chỉ có GEMINI_API_KEY thì gán sang
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

MODEL = "gemini-3.5-flash-lite"  # đổi sang model bạn đang dùng


# ---------------------------------------------------------------
# 1. TOOL: @tool tự tạo khai báo từ tên hàm, type hint và DOCSTRING
#    (docstring chính là "description" model đọc, hãy viết cho rõ)
# ---------------------------------------------------------------
@tool
def get_weather(city: str) -> dict:
    """Lấy nhiệt độ hiện tại (độ C) của một thành phố."""
    fake_data = {"hà nội": 28, "hanoi": 28, "tokyo": 22, "paris": 15}
    if city.lower() not in fake_data:
        raise ValueError(f"Không có dữ liệu nhiệt độ của thành phố {city}")
    return {"city": city, "temp_c": fake_data[city.lower()]}


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Đổi một số đo từ độ C sang độ F. Chỉ đổi đơn vị, không tra nhiệt độ thành phố."""
    return round(celsius * 1.8 + 32, 2)

TOOLS = [get_weather, celsius_to_fahrenheit]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}  # giống dict TOOL_FUNCS ở bản thủ công


# ---------------------------------------------------------------
# 2. STATE: dữ liệu chạy xuyên suốt đồ thị
#    add_messages = khi node trả về messages mới thì NỐI THÊM vào danh sách
#    (giống bước contents.append(...) bạn tự viết)
# ---------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------
# 3. MODEL: bind_tools = gắn danh sách khai báo tool vào mọi lời gọi
# ---------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model=MODEL)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------
# 4. CÁC NODE: mỗi node là một hàm nhận state, trả về PHẦN CẬP NHẬT state
# ---------------------------------------------------------------
def call_model(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    print(f"[call_model] model xin gọi: {[(c['name'], c['args']) for c in response.tool_calls]}")
    return {"messages": [response]}


def run_tools(state: State) -> dict:
    last_message = state["messages"][-1]  # message của model, chứa các yêu cầu gọi tool
    results = []
    for call in last_message.tool_calls:  # mỗi call có "name", "args", "id"
        print(f"Model yêu cầu gọi: {call['name']}({call['args']})")
        try:
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        except Exception as e:
            output = {"error": str(e)}  # trả lỗi về cho model tự xử lý
        print(f"Kết quả tool: {output}")
        results.append(
            ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"])
        )
    return {"messages": results}


# ---------------------------------------------------------------
# 5. CẠNH ĐIỀU KIỆN: hàm trả về TÊN node kế tiếp (hoặc END)
# ---------------------------------------------------------------
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    return "run_tools" if last_message.tool_calls else END


# ---------------------------------------------------------------
# 6. LẮP ĐỒ THỊ
# ---------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("run_tools", run_tools)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model", should_continue, {"run_tools": "run_tools", END: END}
)
builder.add_edge("run_tools", "call_model")  # đây chính là "vòng lặp"

graph = builder.compile()


# ---------------------------------------------------------------
# 7. CHẠY
# ---------------------------------------------------------------
def to_text(content) -> str:
    # content có thể là chuỗi, hoặc danh sách các khối (block) tùy model/phiên bản
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def run_agent(goal: str) -> str:
    try:
        # recursion_limit đếm số bước node (mỗi vòng = 2 bước: call_model + run_tools),
        # nên 10 ~ tối đa khoảng 5 vòng. Đây là "MAX_STEPS" của LangGraph.
        result = graph.invoke(
            {"messages": [HumanMessage(goal)]}, config={"recursion_limit": 10}
        )
    except GraphRecursionError:
        return "Đã hết số bước tối đa."
    return to_text(result["messages"][-1].content)


if __name__ == "__main__":
    # In sơ đồ đồ thị dạng mermaid, dán vào https://mermaid.live để xem hình
    print(graph.get_graph().draw_mermaid())

    print(run_agent("Giữa Hà Nội và Tokyo nơi nào nóng hơn? Cho tôi nhiệt độ của nơi đó theo độ F, hãy dùng tool để quy đổi."))