"""
KHUNG BÀI TẬP LANGGRAPH: tự dựng lại agent của bạn dưới dạng đồ thị.

Cách làm:
  1. Đóng agent_langgraph_mau.py. Chỉ mở ra xem đúng chỗ đang kẹt.
  2. Điền các TODO. Dùng bản thủ công của chính bạn (agent bài tập đã làm xong)
     làm "bản đồ": mỗi bước cũ tương ứng với một phần của đồ thị.
  3. Chạy được rồi thì làm BÀI TẬP MỞ RỘNG ở cuối file.

Mục tiêu: tự vẽ ra được đồ thị (giấy hoặc draw_mermaid) trước khi viết code,
và giải thích được vì sao cạnh run_tools -> call_model tạo ra "vòng lặp".
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv('.env')
if not load_dotenv:
    raise FileNotFoundError("Can not found env file")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash-lite"  # đổi sang model bạn đang dùng
memory = MemorySaver()

# ---------------- Tool ----------------
# TODO 1: viết lại các tool của bạn bằng @tool.
#         Nhớ: docstring + type hint là thứ model đọc để biết tool làm gì.
@tool
def get_weather(id: str) -> dict:
    """Trả về thông tin nhiệt độ của 1 thành phố thông qua ID của thành phố đấy"""
    fake_data = {
        "hanoi": 25,
        "tokyo": 21,
        "paris": 23
    }
    if id not in fake_data:
        raise ValueError(f"không tìm được dữ liệu của thành phố {id} trong dữ liệu")
    return {"id": id, "temp_c": fake_data[id]}

@tool
def mapping_city(city: str) -> str:
    """Trả về thông tin id của 1 thành phố dựa trên tên của thành phố đó"""
    mapping={
        "Hà Nội": "hanoi",
        "hà nội": "hanoi",
        "hanoi": "hanoi",
        "Hà nội": "hanoi",
        "Tokyo": "tokyo",
        "Paris": "paris"
    }
    if city not in mapping:
        return city
    return mapping[city]

@tool
def celsius_to_fahrenheit(celsius: float) ->float:
    """Đổi một số đo từ độ C sang độ F. Chỉ đổi đơn vị, không tra nhiệt độ thành phố."""
    return (celsius * 1.8) + 32

TOOLS = [get_weather, mapping_city, celsius_to_fahrenheit]
# TOOLS_BY_NAME = {t.name: t for t in TOOLS}  # TODO 2: dict tên tool -> tool
tool_node = ToolNode(TOOLS, handle_tool_errors=True)

# ---------------- State ----------------
# TODO 3: định nghĩa State có khóa "messages", dùng add_messages làm reducer.
#         Tự trả lời: add_messages khác gì với việc ghi đè danh sách?
# add_messages được sử dụng để append message nhằm giữ trọn vẹn thông tin liên quan đến lịch sử trong xuyên suốt cuộc hội thoại
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------- Model ----------------
llm = ChatGoogleGenerativeAI(model=MODEL)
llm_with_tools = llm.bind_tools(TOOLS)  # TODO 4: gắn tool vào model (bind_tools)

# ---------------- Node ----------------
def call_model(state: State) -> dict:
    # TODO 5: gọi llm_with_tools với toàn bộ messages trong state,
    #         trả về {"messages": [response]}
    response = llm_with_tools.invoke(state["messages"])
    print(f"[Call Model] Model xin gọi các tool sau: {[(c["name"], c["args"]) for c in response.tool_calls]}")
    return {"messages": [response]}


# def run_tools(state: State) -> dict:
#     # TODO 6: lấy message cuối, duyệt message.tool_calls,
#     #         chạy tool tương ứng (có try/except), mỗi kết quả đóng gói thành
#     #         ToolMessage(content=..., tool_call_id=call["id"], name=call["name"]),
#     #         trả về {"messages": [danh sách ToolMessage]}
#     last_message = state["messages"][-1]
#     results = []
#     for call in last_message.tool_calls:
#         print(f"Model yêu cầu gọi: {call['name']}, {call["args"]}")
#         try:
#             response = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
#         except Exception as e:
#             response = {"error": str(e)}
#         print(response)
#         results.append(
#             ToolMessage(content=str(response), tool_call_id=call["id"], name=call["name"])
#         )
#     return {"messages": results}


# def should_continue(state: State) -> str:
#     # TODO 7: nếu message cuối có tool_calls -> trả "run_tools", ngược lại trả END
#     last_message = state["messages"][-1]
#     return "run_tools" if last_message.tool_calls else END


# ---------------- Lắp đồ thị ----------------
builder = StateGraph(State)
# TODO 8: add_node cho 2 node, add_edge(START -> call_model),
#         add_conditional_edges(call_model -> should_continue),
#         add_edge(run_tools -> call_model)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")


graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[]      
)   # TODO 9: compile
# print("Đây là ảnh graph của hệ thống")
# print(graph.get_graph().draw_mermaid())

def run_agent(goal: str) -> str:
    # TODO 10: invoke đồ thị với {"messages": [HumanMessage(goal)]}
    #          và config={"recursion_limit": ...}, bắt GraphRecursionError,
    #          trả về nội dung message cuối cùng
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(goal)]}, config={"recursion_limit": 10, "configurable": {"thread_id": "2"}}
        )
    except GraphRecursionError: 
        return "Đã hết số bước tối đa."
    return f"Kết luận: {result["messages"][-1].content[0]["text"]}"


if __name__ == "__main__":
    print(run_agent("Hà Nôi và Paris thời tiết như thế nào? Thành phố nào mát nhất trong số chúng?"))
    print(run_agent("Vậy với tokyo thì sao? Thành phố nào mát nhất?"))


# =====================================================================
# BÀI TẬP MỞ RỘNG (làm lần lượt, tự viết, tra tài liệu LangGraph khi cần)
# =====================================================================
# 1. In graph.get_graph().draw_mermaid(), dán vào mermaid.live. Hình có khớp
#    với hình bạn tự vẽ không? Done
# 2. Dùng graph.stream(..., stream_mode="updates") thay cho invoke. In ra tên
#    node và dữ liệu node trả về ở từng bước. So với việc bạn print contents
#    ở bản thủ công, cái nào dễ theo dõi hơn? tôi thấy cách theo dõi cũ dễ theo dõi hơn
# 3. Thay run_tools tự viết bằng ToolNode có sẵn, thay should_continue bằng
#    tools_condition (đều trong langgraph.prebuilt). Chạy lại với: thành phố
#    không có dữ liệu, hỏi 3 thành phố một lượt. ToolNode xử lý lỗi và gọi
#    song song thế nào so với code của bạn? Nó làm hộ bạn những gì?
#    tool_node làm hộ tôi phần run_tool, tools_condition thì làm hộ tôi phần should continue. 
#    chúng đi liền với nhau nên tôi cũng không cần định nghĩa condition dựa trên danh sách tool_call của LLM nữa. 
#    Sau khi config handle_tool_errors=True thì tôi cũng ko cần thực hiện xử lý lỗi manual nữa. 
#    Hiện tại với chỉ vài tool nên chưa có quá nhiều sự khác biệt, 
#    nhưng tôi nghĩ khi danh sách tool dài hơn thì sẽ có nhiều lợi ích hơn đấy 
# 4. Trí nhớ qua nhiều lượt: compile với checkpointer=MemorySaver() và gọi
#    invoke với config={"configurable": {"thread_id": "1"}}. Hỏi câu thứ nhất
#    rồi hỏi tiếp "còn Paris thì sao?". Bản thủ công của bạn có làm được không?
#    Đổi thread_id thì chuyện gì xảy ra? 
#    Sau khi config memory thì câu trả lời thứ 2 có lưu thông tin của câu thứ nhất, 
#    bản thủ công của tôi không làm được, 
#    còn việc đổi thread_id thì không ảnh hưởng câu trả lời cuối cùng của AI
# 5. Người duyệt giữa chừng: compile với checkpointer và
#    interrupt_before=["run_tools"], rồi chạy. Đồ thị dừng ở đâu? Làm sao tiếp
#    tục (gợi ý: invoke với input None và cùng thread_id)? Thử in ra tool_calls
#    đang chờ duyệt trước khi cho chạy. Sau đó tìm hiểu hàm interrupt() mới hơn.
# 6. Thêm system prompt (SystemMessage) vào đầu lịch sử mỗi lần gọi model mà
#    không làm nó bị lưu lặp trong state. Bạn xử lý ở đâu: trong call_model
#    hay trong State?
# 7. (Chuẩn bị cho bước routing) Thêm một node "classify" đứng trước
#    call_model: gọi model (không bind tool) phân loại câu hỏi là "thoi_tiet"
#    hay "khac". Cạnh điều kiện: "thoi_tiet" đi vào nhánh có tool, "khac" trả
#    lời thẳng. Đây là pattern routing của bài blog, dựng bằng đồ thị.c