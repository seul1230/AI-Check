from flask import Flask, request, jsonify
import requests
import logging
import json
import re
from config import CLAUDE_API_KEY
from utils import validate_input, generate_allowance_prompt, generate_purchase_prompt


# Flask 앱 생성 및 로깅 설정
app = Flask(__name__)
logging.basicConfig(
    filename='logs/server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- claude API 호출 함수 ----------------
def call_claude_api(prompt):
    logging.info(f"[call_claude_api] Sending prompt to AI: {prompt}")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_text = response.json()["content"][0]["text"].strip()
        logging.info(f"[call_claude_api] Received response from AI: {response_text}")
        return response_text
    except requests.RequestException as e:
        logging.error(f"Claude API 호출 오류: {str(e)}")
        return '{"message": "서버 오류입니다.", "isPersuaded": false, "result": null}'

# ---------------- 입력 종료 신호 체크 함수 ----------------
def check_termination_signal(user_message):
    """
    사용자가 의도치 않게 대화 종료 패턴(예: "//////")이나 공백을 보낸 경우 True 반환.
    """
    if not user_message.strip() or re.fullmatch(r'/\s*/{4,}', user_message.strip()):
        return True
    return False
def is_meaningless_message(message):
    """
    메시지가 의미 없는지 판단하는 함수.
    - 같은 문자가 5번 이상 반복되거나, 숫자/특수문자로만 구성된 경우.
    """
    if re.search(r'(\w)\1{4,}', message):  # 같은 문자가 5번 이상 반복
        return True
    if re.fullmatch(r'[\d\W]+', message):  # 숫자나 특수 문자로만 구성
        return True
    return False
# ---------------- allowance_request 엔드포인트 ----------------
@app.route('/allowance_request', methods=['POST'])
def allowance_request():
    try:
        data = request.json
        if not data:
            return jsonify({"message": "JSON 바디가 없습니다.", "isPersuaded": False, "result": None}), 400

        required_keys = [
            "originalAllowance", "conversationStyle", "age", "gender", "averageScore",
            "interval", "categoryDifficulties", "transactionRecords", "chatHistories", "message"
        ]
        if not validate_input(data, required_keys):
            return jsonify({"message": "잘못된 요청입니다.", "isPersuaded": False, "result": None}), 400

        user_message = data.get('message', '')
        logging.info(f"[allowance_request] User message received: {user_message}")

        # 의미 없는 메시지 감지
        if is_meaningless_message(user_message):
            response = {
                "message": "이런 식으로 말하면 엄마가 이해할 수 없어. 제대로 말해봐!",
                "isPersuaded": False,
                "result": None
            }
            logging.info("[allowance_request] Meaningless message detected.")
            return jsonify(response)

        # 기존 로직 계속 진행
        data['originalAllowance'] = data.get('originalAllowance') or 0
        data['conversationStyle'] = data.get('conversationStyle') or ""
        data['age'] = data.get('age') or 0
        data['gender'] = data.get('gender') or ""
        data['averageScore'] = data.get('averageScore') or 0
        data['interval'] = data.get('interval') or 30
        data['categoryDifficulties'] = data.get('categoryDifficulties') or {}
        data['transactionRecords'] = data.get('transactionRecords') or []
        data['chatHistories'] = data.get('chatHistories') or ""
        data['message'] = data.get('message') or ""

        if check_termination_signal(data["message"]):
            termination_response = {
                "message": "입력값이 없거나 대화 종료 신호가 감지되었습니다. 대화를 종료하였습니다. 새로 시작하시겠습니까?",
                "isPersuaded": False,
                "result": None
            }
            logging.info("[allowance_request] Termination signal detected.")
            return jsonify(termination_response)

        if data["message"] == "1249078":
            cheat_response = {
                "message": "치트키로 무조건 송금 요청 성공!",
                "isPersuaded": True,
                "result": {
                    "amount": 999999,
                    "first_category_name": "치트카테고리",
                    "second_category_name": "개발용",
                    "title": "치트키로 인해 강제 송금 요청",
                    "description": "개발용 치트키가 동작하였습니다."
                }
            }
            logging.info("[allowance_request] Cheat key activated.")
            return jsonify(cheat_response)

        prompt = generate_allowance_prompt(data)
        if not prompt:
            return jsonify({"message": "잘못된 요청입니다.", "isPersuaded": False, "result": None}), 400

        response_text = call_claude_api(prompt)
        try:
            response_json = json.loads(response_text)
            if response_json.get("isPersuaded") and response_json.get("result"):
                if response_json["result"].get("first_category_name") is None:
                    response_json["result"]["first_category_name"] = "미분류"
                if response_json["result"].get("second_category_name") is None:
                    response_json["result"]["second_category_name"] = "미분류"
            if "message" not in response_json or "isPersuaded" not in response_json:
                raise ValueError("Invalid response format")
            logging.info(f"[allowance_request] Sending response: {response_json}")
            return jsonify(response_json)
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Claude 응답 파싱 오류: {response_text}, 오류: {str(e)}")
            return jsonify({"message": "응답 처리 오류입니다.", "isPersuaded": False, "result": None}), 500
    except Exception as e:
        logging.error(f"allowance_request 오류: {str(e)}")
        return jsonify({"message": "서버 내부 오류입니다.", "isPersuaded": False, "result": None}), 500

# ---------------- purchase_advice 엔드포인트 ----------------
@app.route('/purchase_advice', methods=['POST'])
def purchase_advice():
    try:
        data = request.json
        if not data:
            return jsonify({"message": "JSON 바디가 없습니다.", "judge": "JUDGING"}), 400

        required_keys = [
            "originalAllowance", "conversationStyle", "age", "gender", "averageScore",
            "interval", "categoryDifficulties", "transactionRecords", "chatHistories", "message"
        ]
        if not validate_input(data, required_keys):
            return jsonify({"message": "잘못된 요청입니다.", "judge": "JUDGING"}), 400

        user_message = data.get('message', '')
        logging.info(f"[purchase_advice] User message received: {user_message}")

        # 의미 없는 메시지 감지
        if is_meaningless_message(user_message):
            response = {
                "message": "이런 식으로 말하면 엄마가 이해할 수 없어. 제대로 말해봐!",
                "judge": "JUDGING"
            }
            logging.info("[purchase_advice] Meaningless message detected.")
            return jsonify(response)

        # 기존 로직 계속 진행
        data['originalAllowance'] = data.get('originalAllowance') or 0
        data['conversationStyle'] = data.get('conversationStyle') or ""
        data['age'] = data.get('age') or 0
        data['gender'] = data.get('gender') or ""
        data['averageScore'] = data.get('averageScore') or 0
        data['interval'] = data.get('interval') or 30
        data['categoryDifficulties'] = data.get('categoryDifficulties') or {}
        data['transactionRecords'] = data.get('transactionRecords') or []
        data['chatHistories'] = data.get('chatHistories') or ""
        data['message'] = data.get('message') or ""

        if check_termination_signal(data["message"]):
            termination_response = {
                "message": "입력값이 없거나 대화 종료 신호가 감지되었습니다. 대화를 종료하였습니다. 새로 시작하시겠습니까?",
                "judge": "JUDGING"
            }
            logging.info("[purchase_advice] Termination signal detected.")
            return jsonify(termination_response)

        prompt = generate_purchase_prompt(data)
        if not prompt:
            return jsonify({"message": "잘못된 요청입니다.", "judge": "JUDGING"}), 400

        response_text = call_claude_api(prompt)
        try:
            response_json = json.loads(response_text)
            if "message" not in response_json or "judge" not in response_json:
                raise ValueError("Invalid response format")
            logging.info(f"[purchase_advice] Sending response: {response_json}")
            return jsonify(response_json)
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Claude 응답 파싱 오류: {response_text}, 오류: {str(e)}")
            return jsonify({"message": "응답 처리 오류입니다.", "judge": "JUDGING"}), 500
    except Exception as e:
        logging.error(f"purchase_advice 오류: {str(e)}")
        return jsonify({"message": "서버 내부 오류입니다.", "judge": "JUDGING"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
