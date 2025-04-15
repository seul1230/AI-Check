# import whois
# from datetime import datetime
# import socket

# # 기본 설정
# # socket.setdefaulttimeout(10)  # 타임아웃 10초

# def test_domain(domain):
#     try:
#         print(f"🔍 도메인 조회 시도: {domain}")
#         w = whois.whois(domain)

#         print("✅ WHOIS 성공!")
#         print("📅 creation_date:", w.creation_date)
#         print("📅 expiration_date:", w.expiration_date)
#         print("📅 updated_date:", w.updated_date)
#         print("🏢 org:", w.org)
#         print("👤 name:", w.name)
#         print("🌐 registrar:", w.registrar)

#     except Exception as e:
#         print(f"❌ WHOIS 실패: {e}")

# # 테스트용 도메인 목록 (정상/무료/의심스러운 도메인 조합)
# test_domains = [
#     "google",
#     "www.naver.com",
#     "www.cloudflare.com",
#     "https://bit.ly",           # 단축 URL
#     "tk",               # 비정상
#     "abc.tk",           # 무료 도메인
#     "198.51.100.1",     # IP
#     "simbanet.co.mw",   # 실험에서 문제됐던 도메인
# ]

# for domain in test_domains:
#     print("=" * 50)
#     test_domain(domain)


import whois

domain = "google.com"
w = whois.whois(domain)  # 내부적으로 소켓 열고 응답 받음

print(w.creation_date)  # 사용자는 단순하게 속성만 접근하면 됨
