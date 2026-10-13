"""
BẢN MẪU: vòng lặp agent tối thiểu với Gemini API (SDK google-genai).

Cài đặt:
    pip install google-genai
    export GEMINI_API_KEY="..."      # Windows PowerShell: $env:GEMINI_API_KEY="..."

Chạy:
    python agent_mau.py
"""

from google import genai
from google.genai import types

client = genai.Client()  # tự đọc GEMINI_API_KEY từ biến môi trường
MODEL = "gemini-3.6-flash"  # đổi sang model bạn đang dùng
MAX_STEPS = 5  # giới hạn số vòng lặp để tránh chạy vô hạn tốn tiền


# ---------------------------------------------------------------
# 1. TOOL THẬT: chỉ là hàm Python bình thường
# ---------------------------------------------------------------
def get_weather(city: str) -> dict:
    fake_data = {"hanoi": 28, "tokyo": 22, "paris": 15}
    return {"city": city, "temp_c": fake_data.get(city.lower(), 25)}


# Bảng tra: tên tool (model gọi) -> hàm Python thật
TOOL_FUNCS = {"get_weather": get_weather}


# ---------------------------------------------------------------
# 2. KHAI BÁO TOOL CHO MODEL: model chỉ thấy mô tả này, không thấy code
# ---------------------------------------------------------------
weather_declaration = {
    "name": "get_weather",
    "description": "Lấy nhiệt độ hiện tại (độ C) của một thành phố",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Tên thành phố, ví dụ: Hanoi"}
        },
        "required": ["city"],
    },
}

config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[weather_declaration])],
    # Tắt vòng lặp tự động của SDK để TỰ viết vòng lặp
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------------------------------------------------------------
# 3. VÒNG LẶP AGENT
# ---------------------------------------------------------------
def run_agent(goal: str) -> str:
    # "contents" là toàn bộ lịch sử hội thoại, mỗi lượt gọi API đều gửi lại cả danh sách
    contents = [types.Content(role="user", parts=[types.Part(text=goal)])]

    for step in range(MAX_STEPS):
        print(f"\n===== Bước {step + 1} =====")

        # (a) Gọi model với toàn bộ lịch sử
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )

        # (b) Thêm NGUYÊN VẸN câu trả lời của model vào lịch sử
        #     (không tự chế lại, vì model có thể kèm metadata cần giữ nguyên)
        model_content = response.candidates[0].content
        contents.append(model_content)

        # (c) Model có yêu cầu gọi tool không?
        calls = [p.function_call for p in (model_content.parts or []) if p.function_call]

        if not calls:
            # Không gọi tool nữa -> đây là câu trả lời cuối
            print("Model trả lời cuối.")
            return response.text

        # (d) Chạy từng tool model yêu cầu (có thể có nhiều tool trong 1 lượt)
        result_parts = []
        for call in calls:
            print(f"Model yêu cầu gọi: {call.name}({dict(call.args)})")
            result = TOOL_FUNCS[call.name](**call.args)
            print(f"Kết quả tool: {result}")
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )

        # (e) Đưa kết quả tool về cho model (role="user"), rồi lặp lại
        contents.append(types.Content(role="user", parts=result_parts))

    return "Đã hết số bước tối đa mà chưa có câu trả lời cuối."


if __name__ == "__main__":
    answer = run_agent("Hà Nội và Tokyo hôm nay nơi nào nóng hơn, chênh bao nhiêu độ?")
    print("\n>>> Đáp án:", answer)