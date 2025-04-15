import re
import math
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm

# 🔑 상수 정의
SUSPICIOUS_KEYWORDS = [
    "signin", "login", "account", "update", "reset", "urgent", "alert", 
    "security", "support", "banking", "secure", "verify", "password",
    "payment", "customer", "service", "confirm", "approve", "activate",
    "activation", "member", "center", "info", "identity", "register",
    "validation", "authenticate", "recovery", "unblock", "resolution",
    "unlock", "verification", "bank", "transfer", "card",
    "free", "event", "prize"
]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "shorturl.at", "rb.gy", "dub.sh"]

COUNTRY_CODE_TLDS = [
    "ru", "cn", "tk", "ml", "ga", "cf", "gq", "ua", "kr", "br", "in", "vn", "ir", "bd", "pk", "ng",
    "uk", "us", "de", "fr", "it", "jp", "gov"
]

SPECIALS_V8 = ["www", "@", "?", "&", "#", "%", ".", "=", "_", "-"]  # http, https 제거됨

TWO_LEVEL_TLDS = {
    "co.uk", "ac.uk", "gov.uk", "co.kr", "or.kr", "ne.kr", "go.kr", "ac.kr",
    "com.tw", "idv.tw", "org.tw", "edu.tw", "net.tw",
    "com.cn", "net.cn", "gov.cn", "org.cn", "edu.cn",
    "com.au", "net.au", "edu.au", "org.au", "gov.au",
    "co.jp", "or.jp", "go.jp", "ac.jp", "ad.jp",
    "com.sg", "net.sg", "org.sg", "edu.sg",
    "com.hk", "net.hk", "org.hk", "edu.hk", "gov.hk",
    "co.in", "firm.in", "net.in", "org.in", "gen.in", "ind.in",
}

# ✅ 타이포스쿼팅 점수
def check_typosquatting(url):
    common_brands = {
        "google", "facebook", "amazon", "microsoft", "apple", "netflix",
        "paypal", "twitter", "instagram", "linkedin", "youtube", "yahoo",
        "gmail", "whatsapp", "tiktok", "geocities", "angelfire", "newadvent", "wikipedia"
    }
    score = 0
    try:
        parsed = urlparse(url if "//" in url else "//" + url)
        domain = parsed.netloc.lower() if parsed.netloc else url.lower()

        number_subs = sum(1 for c in domain if c.isdigit())
        if number_subs > 0:
            score += 0.2

        for brand in common_brands:
            if brand not in domain:
                patterns = [
                    brand.replace("o", "0"),
                    brand.replace("i", "1"),
                    brand.replace("l", "1"),
                    brand.replace("e", "3"),
                    brand.replace("a", "4"),
                    brand.replace("s", "5"),
                    brand + "-", brand + "_",
                    brand[:-1],
                    "".join(c + c for c in brand),
                ]
                if any(pat in domain for pat in patterns):
                    score += 0.3
                    break

        if re.findall(r"(.)\1{2,}", domain):
            score += 0.2
        if len(re.findall(r"[-_.]", domain)) > 2:
            score += 0.2
        if len(domain) > 30:
            score += 0.1

    except:
        return 0
    return min(score, 1.0)

# ✅ 도메인 추출
def extract_domain(prime_url: str) -> str:
    parts = prime_url.split('.')
    if len(parts) >= 3:
        last_two = ".".join(parts[-2:])
        if last_two in TWO_LEVEL_TLDS:
            return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def is_shortener_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(domain == short for short in SHORTENERS)
    except:
        return False

def calculate_entropy(s: str) -> float:
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return round(-sum(p * math.log2(p) for p in prob), 4) if s else 0.0

# ✅ 특징 추출 v8 (is_https ➡ suspicious_http)
def extract_lexical_features_v8(url):
    try:
        if not isinstance(url, str):
            return [0] * 30

        parsed = urlparse(url if "//" in url else "//" + url)
        prime_url = parsed.netloc or parsed.path.split('/')[0]
        domain = extract_domain(prime_url)
        path = parsed.path or ""

        url_len = len(url)
        hostname_len = len(prime_url)
        path_len = len(path)
        tld = domain.split('.')[-1] if '.' in domain else ''
        tld_len = len(tld)

        special_counts = [url.lower().count(s) for s in SPECIALS_V8]
        digit_count = sum(c.isdigit() for c in url)

        has_ip = int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', prime_url)))
        is_short_url = int(is_shortener_url(url))
        tld_country_flag = int(tld.lower() in COUNTRY_CODE_TLDS)

        url_entropy = calculate_entropy(url)
        contains_suspicious_word = int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS))
        subdomain_count = max(0, prime_url.count('.') - 1)
        first_path_token_len = len(path.strip("/").split("/")[0]) if path.strip("/") else 0
        ends_with_exe_or_zip = int(bool(re.search(r'\.(exe|zip)$', url.lower())))

        special_ratio = sum(1 for c in url if not c.isalnum()) / len(url)
        path_is_numeric = int(path.strip("/").isdigit())
        has_query_string = int('?' in url)
        has_base64 = int(bool(re.search(r'(?:[A-Za-z0-9+/]{20,}={0,2})', url)))
        query_len = len(parsed.query)
        fragment_len = len(parsed.fragment)
        other_domain_len = len(path.lstrip('/').split('/')[0]) if path else 0

        suspicious_http = int('http' in url.lower() and not url.lower().startswith('http'))
        has_at_symbol = int("@" in url)
        typo_score = check_typosquatting(url)

        return [
            url_len, hostname_len, tld_len, path_len,
            *special_counts, digit_count,
            has_ip, is_short_url, tld_country_flag,
            url_entropy, contains_suspicious_word, subdomain_count,
            first_path_token_len, ends_with_exe_or_zip,
            special_ratio, path_is_numeric, has_query_string,
            has_base64, query_len, fragment_len,
            other_domain_len, suspicious_http, has_at_symbol, typo_score
        ]
    except:
        return [0] * 30

# ✅ 피처 이름 v8
def get_feature_names_v8():
    return [
        "url_len", "hostname_len", "tld_len", "path_len",
        *[f"count_{s}" for s in SPECIALS_V8],
        "digit_count", "has_ip", "is_short_url", "tld_country_flag",
        "url_entropy", "contains_suspicious_word", "subdomain_count",
        "first_path_token_len", "ends_with_exe_or_zip",
        "special_ratio", "path_is_numeric", "has_query_string",
        "has_base64", "query_len", "fragment_len",
        "other_domain_len", "suspicious_http", "has_at_symbol", "typosquatting_score"
    ]

# ✅ 메인 실행부
if __name__ == "__main__":
    df = pd.read_csv("dataset/train.csv")
    df['url'] = df['URL'].str.replace(r'\[\.\]', '.', regex=True)

    features = [extract_lexical_features_v8(url) for url in tqdm(df["url"], desc="🔍 V8 특징 추출 중")]
    feature_df = pd.DataFrame(features, columns=get_feature_names_v8())
    df_with_features = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    df_with_features.to_csv("dataset/train_dataset_with_features_v8.csv", index=False)
    print("✅ 저장 완료: dataset/train_dataset_with_features_v8.csv")
