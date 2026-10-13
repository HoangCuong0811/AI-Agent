"""
KHUNG BÀI TẬP: tự viết vòng lặp agent với Gemini API.

Cách làm:
  1. Đóng agent_mau.py lại. Chỉ mở ra xem đúng chỗ đang kẹt, xem xong đóng lại.
  2. Điền các chỗ TODO bên dưới.
  3. Chạy được rồi thì làm tiếp phần BÀI TẬP MỞ RỘNG ở cuối file.

Mục tiêu hoàn thành: viết lại được vòng lặp này với một tool mới
trong ~15 phút mà không nhìn bản mẫu.
"""

from google import genai
from google.genai import types

client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 5


# ---------------- Tool ----------------
def get_weather(city: str) -> dict:
    fake_data = {"hà nội": 28, "tokyo": 22, "paris": 15}
    if not fake_data.get(city.lower()):
        raise ValueError(f"Không có dữ liệu nhiệt độ của thành phố {city} trong database!")
    return {"city": city, "temp_c": fake_data.get(city.lower())}

def get_f_degree(degree: float) -> float:
    return round(float(degree) * 1.8 + 32, 2)

def who_won_wc() -> str:
    return "Tây Ban Nha - Spain"


# TODO 1: tạo dict TOOL_FUNCS ánh xạ tên tool -> hàm Python
TOOL_FUNCS = {
    "get_weather": get_weather,
    "get_f_degree": get_f_degree,
    "who_won_wc": who_won_wc
}

# TODO 2: khai báo tool cho model (name, description, parameters kiểu JSON schema)
tools_declaration = [
    {
        "name": "get_weather",
        "description": "Lấy nhiệt độ (Celsius) cho 1 thành phố",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Tên 1 thành phố"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_f_degree",
        "description": "Tính nhiệt độ theo độ F(Fahrenheit) từ độ C(Celsius) của 1 thành phố",
        "parameters": {
            "type": "object",
            "properties": {
                "degree": {"type": "number", "description": "Nhiệt độ theo thang độ C(Celsius)"}
            },
            "required": ["degree"],
        },
    },
    {
        "name": "who_won_wc",
        "description": "Thông tin quốc gia vô địch World Cup năm 2026"
    }
]

# TODO 3: tạo config có tools=[...] và TẮT automatic function calling
config = types.GenerateContentConfig(
    tools=[get_weather, get_f_degree, who_won_wc],
)


# ---------------- Vòng lặp agent ----------------
def run_agent(goal: str) -> str:
    # TODO 4: khởi tạo lịch sử "contents" chứa câu hỏi đầu tiên của user
    contents = [types.Content(role="user", parts=[types.Part(text=goal)])]
    
    for step in range(MAX_STEPS):
        # TODO 5: gọi model với toàn bộ lịch sử + config
        response = client.models.generate_content(
            model=MODEL, config=config, contents=contents
        )

        print(response.usage_metadata)        

        # TODO 6: lấy content của model từ response và nối vào lịch sử
        model_content = response.candidates[0].content
        contents.append(model_content)

        # TODO 7: tìm các function_call trong model_content.parts
        calls = [p.function_call for p in (model_content.parts or []) if p.function_call]

        # TODO 8: nếu KHÔNG có function_call -> trả về câu trả lời cuối
        if not calls:
            print("Đây là câu trả lời cuối cùng của AI: ")
            return response.text
        
        # TODO 9: với mỗi call -> chạy hàm Python tương ứng bằng call.args
        #         rồi đóng gói kết quả thành Part.from_function_response(...)
        result_parts = []
        for call in calls:
            print(f"Model yêu cầu gọi: {call.name}(dict({call.args}))")
            try:
                result=TOOL_FUNCS[call.name](**call.args)
            except Exception as e:
                print(f"Lỗi về giá trị rồi: {e}")
                result = {"Error": str(e)}
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
            
        # TODO 10: nối result_parts vào lịch sử (role="user") rồi lặp lại
        contents.append(types.Content(role="user", parts=result_parts))
        
    return "Đã hết số bước tối đa."

if __name__ == "__main__":
    print(run_agent("Giữa Hà Nội và paris thành phố nào nóng hơn và nóng hơn bao nhiêu độ. Bên cạnh đó, quốc gia nào vô địch world cup 2026?"))


# =====================================================================
# BÀI TẬP MỞ RỘNG (làm lần lượt, tự viết, đừng nhờ AI viết hộ)
# =====================================================================
# 1. In ra toàn bộ `contents` sau mỗi bước. Quan sát: sau 1 vòng, lịch sử
#    có những phần tử nào, role gì, part gì? DONE
# 2. Thêm tool thứ hai: convert_c_to_f(celsius: float). Hỏi một câu buộc
#    model phải dùng cả hai tool. DONE
# 3. Xử lý lỗi: cho get_weather raise Exception với thành phố lạ. Bắt lỗi
#    và trả {"error": "..."} về cho model. Model phản ứng thế nào? DONE sẽ trả về nội dung giống với error
# 4. Phá thử: hỏi một câu không cần tool; hỏi câu cần tool không tồn tại;
#    đặt MAX_STEPS=1. Chuyện gì xảy ra ở mỗi trường hợp? 
# Đặt MAX_STEPS=1 thì chưa kịp đưa ra câu trả lời vì câu hỏi yêu cầu nhiều lần gọi tool không chỉ 1
# 5. Đếm token/chi phí: in response.usage_metadata mỗi bước. Thấy gì về
#    việc gửi lại toàn bộ lịch sử ở mỗi vòng?
# 6. Thử hỏi "thời tiết ở Hà Nội, Tokyo và Paris": model có gọi nhiều tool
#    trong cùng một lượt không? Code của bạn xử lý đúng chưa? DONE
# 7. Khi xong hết: bật lại automatic_function_calling, truyền thẳng hàm
#    Python vào tools=[get_weather]. So sánh với code tự viết: SDK đã
#    làm hộ bạn những bước nào?