import re
from typing import Dict, List

# [Step 1. 대화 유형 분류] 임베딩 실패 시 fallback 룰
CONVERSATION_TYPE_RULES: Dict[str, List[re.Pattern]] = {
    "중고거래": [
        re.compile(r"(중고|직거래|택배|판매|구매|가격|송장)"),
    ],
    "재테크": [
        re.compile(r"(투자|수익|수익률|코인|암호화폐|주식|fx|선물|레버리지)"),
    ],
    "구직": [
        re.compile(r"(취업|채용|구인|면접|이력서|지원서|급여|계약서)"),
    ],
    "부업": [
        re.compile(r"(부업|재택|알바|아르바이트|단기|건당|수당|파트타임)"),
    ],
}

# [Step 2. 규칙 기반 신호 추출] 공통 위험 신호 패턴
COMMON_RISK_PATTERNS: Dict[str, List[re.Pattern]] = {
    "money_request": [
        re.compile(
            r"(송금|입금|결제|이체)\s*(해|해주세요|해요|해\s*주세요|부탁|부탁드립니다|요청|주시|주시면|주셔야|해야|하셔야)"
        ),
        re.compile(
            r"(돈|현금|카드|상품권|보증금|예약금)\s*(으로|로)?\s*(보내|송금|입금|결제)\s*(해|해주세요|해요|해\s*주세요|주시|주시면|주셔야|해야|하셔야)"
        ),
        re.compile(
            r"(계좌번호|계좌)(?:\s*로)?\s*(먼저|우선)?\s*(알려|알려줘|알려주세요|알려주시면|보내|보내줘|보내주세요|보내주시면|공유|주세요|주시면|주셔야|주실|받아야|받아야\s*합니다|받아야\s*해요)"
        ),
        re.compile(
            r"(소액|선입금)\s*(이라도)?\s*(먼저)?\s*(보내|입금|송금)\s*(주셔야|주세요|해|해야|하셔야|해요|해\s*주세요)"
        ),
    ],
    "credential_request": [
        re.compile(
            r"(비밀번호|인증\s*코드|인증코드|인증번호|otp|일회용|pin)\s*(알려|알려줘|알려주세요|보내|보내줘|보내주세요|입력|입력해|입력해주세요|공유|말해)"
        ),
    ],
    "urgency": [
        re.compile(r"(긴급|급히|급해요|급합니다|급함|서둘러|지체\s*없이)"),
        re.compile(r"(지금\s*당장|지금\s*바로|바로\s*지금|오늘\s*내로|몇\s*시간\s*안에)"),
        re.compile(r"오늘\s*안에(?!.*(안\s*하셔도|안\s*해도|괜찮|무관))"),
        re.compile(r"지금\s*(처리|결정|응답)\s*(해|해주세요|하셔야|해야|해\s*주세요)"),
        re.compile(r"지금\s*(결정|처리|응답)하셔야"),
        re.compile(r"지금\s*.*(안\s*하면|안\s*하시면).*(불이익|기회|취소|다음\s*순번)"),
    ],
}

# [Step 2. 규칙 기반 신호 추출] 유형별 위험 신호 패턴
TYPE_RISK_PATTERNS: Dict[str, Dict[str, List[re.Pattern]]] = {
    "구직": {
        "job_fee_request": [
            re.compile(r"(입사비|교육비|연수비|등록비|보증금)\s*(입금|송금|결제|납부)"),
            re.compile(r"(채용|합격|면접)\s*(확정|진행)\s*.*(비용|수수료)"),
        ],
        "job_personal_info": [
            re.compile(r"(신분증|주민등록번호|계좌정보|통장사본)\s*(제출|전송|공유|보내)"),
        ],
    },
    "중고거래": {
        "usedgoods_safe_payment": [
            re.compile(r"(안전결제|안전거래|에스크로)\s*(링크|결제|확인)"),
            re.compile(r"(결제|거래)\s*링크\s*로\s*진행"),
        ],
        "usedgoods_delivery_fee": [
            re.compile(r"(배송비|택배비)\s*(먼저|선결제|선입금)"),
            re.compile(r"(착불\s*불가|선불\s*만\s*가능)"),
        ],
    },
    "재테크": {
        "investment_guarantee": [
            re.compile(r"(원금\s*보장|손실\s*없음|확정\s*수익)"),
            re.compile(r"(고수익|수익률\s*\d+%|월\s*\d+%\s*보장)"),
        ],
        "investment_recruit": [
            re.compile(r"(리딩방|투자방|전문가\s*추천|VIP\s*회원)"),
            re.compile(r"(지금\s*가입|무료\s*체험|수익\s*인증)"),
        ],
    },
    "부업": {
        "sidejob_fee_request": [
            re.compile(r"(등록비|재료비|보증금|교육비)\s*(입금|송금|결제)"),
            re.compile(r"(작업|업무)\s*시작\s*전\s*(비용|수수료)"),
        ],
        "sidejob_task": [
            re.compile(r"(클릭|캡처|좋아요|리뷰)\s*만\s*하면\s*수익"),
            re.compile(r"(단순\s*작업|재택\s*부업|건당\s*수익)"),
        ],
    },
}


# [Step 2. 규칙 기반 신호 추출] 공통/유형 패턴 합침
def _merge_risk_patterns() -> Dict[str, List[re.Pattern]]:
    merged: Dict[str, List[re.Pattern]] = {}
    for signal, patterns in COMMON_RISK_PATTERNS.items():
        merged[signal] = list(patterns)
    for type_patterns in TYPE_RISK_PATTERNS.values():
        for signal, patterns in type_patterns.items():
            merged.setdefault(signal, []).extend(patterns)
    for signal, patterns in merged.items():
        seen = set()
        deduped: List[re.Pattern] = []
        for pattern in patterns:
            key = pattern.pattern
            if key in seen:
                continue
            seen.add(key)
            deduped.append(pattern)
        merged[signal] = deduped
    return merged


# [Step 2. 규칙 기반 신호 추출] 실제 룰셋
RISK_SIGNAL_RULES: Dict[str, List[re.Pattern]] = _merge_risk_patterns()


# [Step 2. 규칙 기반 신호 추출] 유형별 신호 범위 결정
COMMON_RISK_SIGNALS: List[str] = list(COMMON_RISK_PATTERNS.keys())
TYPE_RISK_SIGNALS: Dict[str, List[str]] = {
    conversation_type: list(patterns.keys()) for conversation_type, patterns in TYPE_RISK_PATTERNS.items()
}


# [Step 2. 규칙 기반 신호 추출] 유형에 맞는 신호 범위 선택
def resolve_risk_signals(conversation_type: str) -> List[str]:
    type_signals = TYPE_RISK_SIGNALS.get(conversation_type, [])
    return list(dict.fromkeys(COMMON_RISK_SIGNALS + type_signals))


# [Step 3. RAG 쿼리 보강] 신호별 확장 키워드
SIGNAL_QUERY_TERMS: Dict[str, List[str]] = {
    "money_request": ["입금", "송금", "계좌", "보증금", "예약금", "결제"],
    "urgency": ["오늘 안에", "지금", "당장", "즉시", "몇 시간 안에"],
    "credential_request": ["인증코드", "otp", "비밀번호", "pin"],
    "job_fee_request": ["입사비", "교육비", "연수비", "등록비", "보증금"],
    "job_personal_info": ["신분증", "주민등록번호", "통장사본", "계좌정보"],
    "usedgoods_safe_payment": ["안전결제", "안전거래", "에스크로", "결제 링크"],
    "usedgoods_delivery_fee": ["배송비", "택배비", "선입금", "착불 불가"],
    "investment_guarantee": ["원금 보장", "확정 수익", "고수익", "수익률"],
    "investment_recruit": ["리딩방", "투자방", "VIP", "수익 인증"],
    "sidejob_fee_request": ["등록비", "재료비", "보증금", "수수료"],
    "sidejob_task": ["클릭", "캡처", "좋아요", "단순 작업", "건당"],
}
