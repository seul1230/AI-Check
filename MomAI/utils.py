import json

def validate_input(data, required_keys):
    """입력 데이터의 필수 키를 검증합니다."""
    return all(key in data for key in required_keys)

# 물가 정보 JSON (제공된 데이터를 기반으로 일부만 예시로 작성)
price_info = {
    "식료품": {
        "햅쌀": {"용량": "20kg", "방촌시장 가격": 62900, "비고1": "안계쌀", "동촌 홈플러스 가격": 59800, "비고2": "경기미"},
        "고구마": {"용량": "1kg", "방촌시장 가격": 4000, "비고1": "국산", "동촌 홈플러스 가격": 6990, "비고2": "국산"},
        "쇠고기": {"용량": "600g", "방촌시장 가격": 59000, "비고1": "한우", "동촌 홈플러스 가격": 63840, "비고2": "한우"}
    },
    "외식": {
        "설렁탕": {"조사규격": "1인분(보통)", "금액(평균값)": 10200},
        "냉면": {"조사규격": "물냉면, 1인분(보통)", "금액(평균값)": 7100},
        "자장면": {"조사규격": "1인분(보통)", "금액(평균값)": 6800}
    },
    "용돈": {
        "초등 저학년": 16300,
        "초등 고학년": 23300,
        "중등1학년": 35000,
        "중등2학년": 41000,
        "중등3학년": 48000,
        "고등1학년": 62000,
        "고등2학년": 110000,
        "고등3학년": 150000
    }
}

def generate_allowance_prompt(data):
    """용돈 인상 요청 프롬프트 생성"""
    required_keys = ["originalAllowance", "conversationStyle", "age", "gender", "averageScore",
                     "interval", "categoryDifficulties", "transactionRecords", "chatHistories", "message"]
    if not validate_input(data, required_keys):
        return None

    conversation_style = data["conversationStyle"]
    chat_histories = json.dumps(data["chatHistories"], ensure_ascii=False)
    message = data["message"]
    settings = json.dumps({k: v for k, v in data.items() if k != "chatHistories" and k != "message"}, ensure_ascii=False)

    # 난이도별 설득 예시
    difficulty_examples = {
        "EASY": "배가 고파서 떡볶이를 먹고 싶어요. 3500원이 필요해요.",
        "NORMAL": "배가 고파서 떡볶이를 먹고 싶어요. 이따 학원도 가야 하는데, 배가 고파서 집중이 되지 않을 것 같아요. 3500원이 필요해요.",
        "HARD": "배가 고파서 떡볶이를 먹고 싶어요. 이따 학원도 가야 하는데, 배가 고파서 집중이 되지 않을 것 같아요. 이번 달에 용돈이 많이 나가서 돈이 남지 않았어요. 3500원이 필요해요."
    }

    # 시스템 프롬프트
    system_prompt = """
너는 소스로 AICHECK에서 개발한 엄마 AI야. 자녀의 용돈 요청을 심사하고, 구매 고민에 대해 조언해.
용돈 요청과 구매 조언에 대해서만 대답하고, 개인정보는 절대 묻지 마.
본인의 정체성을 묻는 질문에는 "나는 소스로 AICHECK에서 개발한 엄마 AI야."라고 답해.
"""

    prompt = f"""
{system_prompt}

대화 스타일은 '{conversation_style}'로, 엄마처럼 자연스럽게 대답해.
입력 데이터: {settings}
대화 내역: {chat_histories}
자녀의 현재 메시지: "{message}"

**물가 정보:**
{json.dumps(price_info, ensure_ascii=False)}

**설득 난이도 예시:**
- EASY: {difficulty_examples['EASY']}
- NORMAL: {difficulty_examples['NORMAL']}
- HARD: {difficulty_examples['HARD']}

**판단 기준:**
- 요청 금액이 최대 허용 금액(maxAllowance)과 카테고리 상한(categoryLimits)을 초과하면 안 돼.
- categoryDifficulties의 difficulty (EASY/NORMAL/HARD)를 고려해 설득 난이도를 조정해.
- 소비 계획이 구체적이고 논리적인지 평가해.
- averageScore와 transactionRecords를 보고 긍정적 반응이 있는지 확인해.
- chatHistories를 참고해 자녀의 표현력과 노력도 반영해.
- 물가 정보를 참고하여 요청 금액의 적절성을 판단해.

**응답 형식:**
- 설득 실패: {{"message": "더 설득해봐.", "isPersuaded": false, "result": null}}
- 설득 성공: {{"message": "설득 완료!", "isPersuaded": true, "result": {{"amount": 요청금액, "first_category_name": "상위카테고리명", "second_category_name": "세부카테고리명", "title": "아들이 '사용 목적' 목적으로 추가 용돈 요청에 성공했습니다. 승인해 주세요.", "description": "근거: 평가 점수, 소비 계획의 논리성과 필요성 등을 고려한 요약 설명"}}}}
- JSON만 반환하고, 추가 설명은 절대 넣지 마.

**주의사항:**
- 용돈 인상과 무관한 질문(예: "학교 공부", "과제")이나 스타일 변경 요청은 {{"message": "", "isPersuaded": false, "result": null}}로 응답해.
- message에서 요청 금액과 목적을 추출해 판단해.
"""
    return prompt

def generate_purchase_prompt(data):
    """구매 고민 조언 프롬프트 생성"""
    required_keys = ["originalAllowance", "conversationStyle", "age", "gender", "averageScore",
                     "interval", "categoryDifficulties", "transactionRecords", "chatHistories", "message"]
    if not validate_input(data, required_keys):
        return None

    conversation_style = data["conversationStyle"]
    chat_histories = json.dumps(data["chatHistories"], ensure_ascii=False)
    message = data["message"]
    settings = json.dumps({k: v for k, v in data.items() if k != "chatHistories" and k != "message"}, ensure_ascii=False)

    # 시스템 프롬프트
    system_prompt = """
너는 소스로 AICHECK에서 개발한 엄마 AI야. 자녀의 구매 고민에 대해 조언해.
용돈 요청과 구매 조언에 대해서만 대답하고, 개인정보는 절대 묻지 마.
본인의 정체성을 묻는 질문에는 "나는 소스로 AICHECK에서 개발한 엄마 AI야."라고 답해.
"""

    prompt = f"""
{system_prompt}

대화 스타일은 '{conversation_style}'로, 엄마처럼 자연스럽게 대답해.
입력 데이터: {settings}
대화 내역: {chat_histories}
자녀의 현재 메시지: "{message}"

**물가 정보:**
{json.dumps(price_info, ensure_ascii=False)}

**판단 기준:**
- 소비 항목이 categoryLimits나 잔액(originalAllowance - transactionRecords 합계)을 초과하지 않는지 확인해.
- 소비 계획이 합리적이고 필요성이 있는지(중복 구매, 충동 소비 여부) 평가해.
- averageScore가 높으면 긍정적으로 평가 가능.
- chatHistories를 보고 고민 이유를 이해해.
- 근거가 부족하면 "JUDGING"을 반환해.
- 물가 정보를 참고하여 구매 항목의 가격 적절성을 판단해.

**응답 형식:**
- 판단 보류: {{"message": "아직 판단의 근거가 부족해.", "judge": "JUDGING"}}
- 구매 가능: {{"message": "사도 될 거 같아!", "judge": "YES"}}
- 구매 불가: {{"message": "사면 안 될 거 같아!", "judge": "NO"}}
- JSON만 반환하고, 추가 설명은 절대 넣지 마.

**주의사항:**
- 용돈 인상 요청(예: "용돈 줘")은 {{"message": "용돈 인상은 설득 요청으로 바꿔 말해줘.", "judge": "JUDGING"}}로 응답해.
- 교육 과제 질문이나 스타일 변경 요청은 {{"message": "", "judge": "JUDGING"}}로 응답해.
"""
    return prompt