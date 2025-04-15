# # import pandas as pd
# # from datetime import datetime
# # import whois
# # from urllib.parse import urlparse
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from tqdm import tqdm

# # def extract_whois_days(domain):
# #     try:
# #         w = whois.whois(domain)
# #         today = datetime.utcnow()

# #         creation_date = w.creation_date
# #         expiration_date = w.expiration_date
# #         updated_date = w.updated_date

# #         if isinstance(creation_date, list):
# #             creation_date = creation_date[0]
# #         if isinstance(expiration_date, list):
# #             expiration_date = expiration_date[0]
# #         if isinstance(updated_date, list):
# #             updated_date = updated_date[0]

# #         domain_age_days = (today - creation_date).days if creation_date else -1
# #         days_until_expiry = (expiration_date - today).days if expiration_date else -1
# #         last_updated_days_ago = (today - updated_date).days if updated_date else -1

# #         return [domain_age_days, days_until_expiry, last_updated_days_ago]

# #     except Exception:
# #         return [-1, -1, -1]

# # def extract_domain(url):
# #     try:
# #         parsed = urlparse(url)
# #         return parsed.netloc or parsed.path.split('/')[0]
# #     except Exception:
# #         return ""

# # def main():
# #     df = pd.read_csv("dataset/cleaned_urls_train.csv")
# #     df["domain"] = df["url"].apply(extract_domain)
# #     domains = df["domain"].tolist()

# #     # # 병렬 처리로 WHOIS 정보 수집
# #     # results = [None] * len(domains)
# #     # with ThreadPoolExecutor(max_workers=20) as executor:  # 병렬 스레드 수 조정 가능
# #     #     future_to_index = {executor.submit(extract_whois_days, domain): i for i, domain in enumerate(domains)}

# #     #     for future in tqdm(as_completed(future_to_index), total=len(domains), desc="🚀 WHOIS 병렬 조회 중"):
# #     #         i = future_to_index[future]
# #     #         results[i] = future.result()

# #     # df[["domain_age_days", "days_until_expiry", "last_updated_days_ago"]] = pd.DataFrame(results, index=df.index)
# #     df.to_csv("dataset/cleaned_urls_train.csv", index=False)
# #     print("✅ 저장 완료: dataset/cleaned_urls_train.csv")

# # if __name__ == "__main__":
# #     main()


# # import pandas as pd
# # from datetime import datetime
# # import whois
# # from urllib.parse import urlparse
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from tqdm import tqdm
# # import time

# # def extract_domain(url):
# #     try:
# #         parsed = urlparse(url)
# #         return parsed.netloc or parsed.path.split('/')[0]
# #     except Exception:
# #         return ""

# # def extract_whois_days(domain):
# #     try:
# #         w = whois.whois(domain)
# #         today = datetime.utcnow()

# #         # 날짜 필드가 리스트로 반환되는 경우 첫 값만 사용
# #         creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
# #         expiration_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
# #         updated_date = w.updated_date[0] if isinstance(w.updated_date, list) else w.updated_date

# #         domain_age_days = (today - creation_date).days if creation_date else -1
# #         days_until_expiry = (expiration_date - today).days if expiration_date else -1
# #         last_updated_days_ago = (today - updated_date).days if updated_date else -1

# #         return [domain_age_days, days_until_expiry, last_updated_days_ago]

# #     except Exception:
# #         return [-1, -1, -1]

# # def main():
# #     df = pd.read_csv("dataset/cleaned_urls_train.csv")
# #     df["domain"] = df["url"].apply(extract_domain)

# #     results = [None] * len(df)
# #     with ThreadPoolExecutor(max_workers=32) as executor:
# #         futures = {
# #             executor.submit(extract_whois_days, domain): i
# #             for i, domain in enumerate(df["domain"])
# #         }

# #         for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 WHOIS 병렬 조회 중"):
# #             idx = futures[future]
# #             try:
# #                 results[idx] = future.result()
# #             except Exception:
# #                 results[idx] = [-1, -1, -1]
# #             time.sleep(0.1)  # 너무 빠른 요청 방지

# #     df[["domain_age_days", "days_until_expiry", "last_updated_days_ago"]] = pd.DataFrame(results, index=df.index)
# #     df.to_csv("dataset/cleaned_urls_train.csv", index=False)
# #     print("✅ 저장 완료: dataset/cleaned_urls_train.csv")

# # if __name__ == "__main__":
# #     main()


# import pandas as pd
# from datetime import datetime
# import whois
# import socket
# import re
# import time
# from urllib.parse import urlparse
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# from ipwhois import IPWhois
# import dns.resolver

# # ✅ 도메인 추출
# def extract_domain(url):
#     try:
#         parsed = urlparse(url)
#         return parsed.netloc or parsed.path.split('/')[0]
#     except Exception:
#         return ""

# # ✅ WHOIS + 추가 도메인 관련 특성 추출
# def extract_domain_features(domain):
#     today = datetime.utcnow()
#     domain_age_days = -1
#     days_until_expiry = -1
#     last_updated_days_ago = -1
#     asn_name = "unknown"
#     netname_flag = 0
#     dns_abnormal_flag = 1
#     dns_abnormal_ns_flag = 1

#     try:
#         # WHOIS 날짜 정보
#         w = whois.whois(domain)
#         creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
#         expiration_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
#         updated_date = w.updated_date[0] if isinstance(w.updated_date, list) else w.updated_date
#         domain_age_days = (today - creation_date).days if creation_date else -1
#         days_until_expiry = (expiration_date - today).days if expiration_date else -1
#         last_updated_days_ago = (today - updated_date).days if updated_date else -1

#         # 조직명에 주요 키워드가 포함되어 있는지 체크
#         org = (w.org or w.name or "").lower()
#         known_orgs = ["google", "microsoft", "cloudflare", "amazon", "kakao", "naver"]
#         netname_flag = int(any(org_keyword in org for org_keyword in known_orgs))

#     except:
#         pass

#     try:
#         # IP 기반 ASN 조회
#         ip = socket.gethostbyname(domain)
#         whois_info = IPWhois(ip).lookup_rdap()
#         asn_name = whois_info.get('asn_description', 'unknown')
#     except:
#         pass

#     try:
#         # DNS SOA 레코드
#         soa = dns.resolver.resolve(domain, 'SOA', lifetime=5)
#         for r in soa:
#             ttl = r.minimum
#             if 300 <= ttl <= 86400:
#                 dns_abnormal_flag = 0
#     except:
#         pass

#     try:
#         # NS 레코드 이상 여부
#         ns_records = dns.resolver.resolve(domain, 'NS')
#         ns_names = [r.to_text().lower() for r in ns_records]
#         known_ns = ["cloudflare", "google", "aws", "kakao", "naver", "azure"]
#         dns_abnormal_ns_flag = int(all(not any(k in ns for k in known_ns) for ns in ns_names))
#     except:
#         pass

#     return [
#         domain_age_days,
#         days_until_expiry,
#         last_updated_days_ago,
#         asn_name,
#         netname_flag,
#         dns_abnormal_flag,
#         dns_abnormal_ns_flag
#     ]

# # ✅ 메인 함수
# def main():
#     df = pd.read_csv("dataset/final_url_dataset.csv")
#     df["domain"] = df["url"].apply(extract_domain)

#     results = [None] * len(df)
#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = {
#             executor.submit(extract_domain_features, domain): i
#             for i, domain in enumerate(df["domain"])
#         }

#         for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 WHOIS & DNS 병렬 조회 중"):
#             i = futures[future]
#             try:
#                 results[i] = future.result()
#             except Exception:
#                 results[i] = [-1, -1, -1, "unknown", 0, 1, 1]
#             time.sleep(0.05)

#     cols = [
#         "domain_age_days", "days_until_expiry", "last_updated_days_ago",
#         "asn_name", "netname_keyword_flag", "dns_abnormal_flag", "dns_abnormal_ns_flag"
#     ]
#     df[cols] = pd.DataFrame(results, index=df.index)
#     df.to_csv("dataset/final_url_dataset_with_days.csv", index=False)
#     print("✅ 저장 완료: dataset/final_url_dataset_with_days_v4.csv")

# if __name__ == "__main__":
#     main()


import pandas as pd
from datetime import datetime
import re
import socket
import time
import subprocess
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ✅ 도메인 추출
def extract_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        return ""

# ✅ IP 도메인 여부 확인
def is_ip(domain: str) -> bool:
    return bool(re.fullmatch(r'(\d{1,3}\.){3}\d{1,3}', domain.strip()))

# # ✅ WHOIS CLI 호출
# def whois_cli_query(domain: str) -> str:
#     try:
#         output = subprocess.check_output(['whois', domain], timeout=1, stderr=subprocess.DEVNULL)
#         return output.decode(errors='ignore')
#     except subprocess.TimeoutExpired:
#         print(f"⏱️ TIMEOUT: {domain}")
#         return ""
#     except:
#         return ""

def whois_cli_query(domain: str) -> str:
    try:
        result = subprocess.run(
            ['whois', domain],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,  # 타임아웃 여유 증가
            check=False,  # 오류 발생해도 예외 던지지 않음
            text=True     # 자동 디코딩
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT 발생: {domain}")
        return ""
    except Exception as e:
        print(f"❌ WHOIS 실패: {domain} | 사유: {e}")
        return ""


# ✅ WHOIS 정보 파싱
def parse_date(text):
    patterns = [
        r'\d{4}-\d{2}-\d{2}',              # 2023-01-01
        r'\d{2}-[A-Za-z]{3}-\d{4}',        # 01-Jan-2023
        r'[A-Za-z]{3} \d{2} \d{4}',        # Jan 01 2023
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            try:
                return pd.to_datetime(match.group(0), errors='coerce')
            except:
                pass
    return None

def parse_whois_dates(whois_text):
    whois_text = whois_text or ""
    creation = parse_date(re.search(r"(?i)(creation|created|registered)[^\n]*", whois_text) or "")
    expiration = parse_date(re.search(r"(?i)(expiration|expiry|expire)[^\n]*", whois_text) or "")
    updated = parse_date(re.search(r"(?i)(updated|modified)[^\n]*", whois_text) or "")
    return creation, expiration, updated

# ✅ 최종 WHOIS 기반 특성 추출
def extract_whois_days(domain: str):
    today = datetime.utcnow()

    if is_ip(domain):
        return [-1, -1, -1]

    whois_text = whois_cli_query(domain)
    creation, expiration, updated = parse_whois_dates(whois_text)

    try:
        domain_age_days = (today - creation).days if creation else -1
        days_until_expiry = (expiration - today).days if expiration else -1
        last_updated_days_ago = (today - updated).days if updated else -1
    except:
        domain_age_days, days_until_expiry, last_updated_days_ago = -1, -1, -1
        print("failed")

    return [domain_age_days, days_until_expiry, last_updated_days_ago]

# ✅ 메인 실행
def main():
    df = pd.read_csv("dataset/final_url_dataset.csv")
    df["domain"] = df["url"].apply(extract_domain)

    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(extract_whois_days, domain): i
            for i, domain in enumerate(df["domain"])
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 WHOIS CLI 병렬 조회 중"):
            i = futures[future]
            try:
                results[i] = future.result()
            except:
                results[i] = [-1, -1, -1]
            time.sleep(0.05)  # WHOIS 서버에 부담 최소화

    df[["domain_age_days", "days_until_expiry", "last_updated_days_ago"]] = pd.DataFrame(results, index=df.index)
    df.to_csv("dataset/final_url_dataset_with_days_v4.csv", index=False)
    print("✅ 저장 완료: dataset/final_url_dataset_with_days_v4.csv")

if __name__ == "__main__":
    main()


