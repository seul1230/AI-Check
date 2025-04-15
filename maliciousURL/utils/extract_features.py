# Re-import necessary modules after code execution state reset
import re
import math
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm

# Feature definition constants
SUSPICIOUS_KEYWORDS = ['login', 'verify', 'secure', 'account', 'bank', 'update']
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "shorturl.at", "rb.gy", "dub.sh"]
COUNTRY_CODE_TLDS = [
    "ru", "cn", "tk", "ml", "ga", "cf", "gq", "ua", "kr", "br", "in", "vn", "ir", "bd", "pk", "ng",
    "uk", "us", "de", "fr", "it", "jp", "gov"
]
SPECIALS = ["http", "https", "www", "@", "?", "&", "#", "%", ".", "=", "_", "-"]

def is_shortener_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(domain == short for short in SHORTENERS)
    except:
        return False

def calculate_entropy(s: str) -> float:
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return round(-sum([p * math.log2(p) for p in prob]), 4) if s else 0.0

# version 3
def extract_lexical_features_v3(url):
    try:
        if not isinstance(url, str):
            return [0] * 25  # 25 features

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        path = parsed.path or ""

        url_len = len(url)
        hostname_len = len(domain)
        path_len = len(path)
        tld = domain.split('.')[-1] if '.' in domain else ''
        tld_len = len(tld)

        special_counts = [url.lower().count(item) for item in SPECIALS]
        digit_count = sum(c.isdigit() for c in url)

        has_ip = int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain)))
        is_short_url = int(is_shortener_url(url))
        tld_country_flag = int(tld.lower() in COUNTRY_CODE_TLDS)

        url_entropy = calculate_entropy(url)
        contains_suspicious_word = int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS))
        subdomain_count = max(0, domain.count('.') - 1)
        first_path_token_len = len(path.strip("/").split("/")[0]) if path.strip("/") else 0
        ends_with_exe_or_zip = int(bool(re.search(r'\.(exe|zip)$', url.lower())))

        special_ratio = sum(1 for c in url if not c.isalnum()) / len(url)
        path_is_numeric = int(path.strip("/").isdigit())
        has_query_string = int('?' in url)
        has_base64 = int(bool(re.search(r'(?:[A-Za-z0-9+/]{20,}={0,2})', url)))
        query_len = len(parsed.query)
        fragment_len = len(parsed.fragment)

        return [
            url_len, hostname_len, tld_len, path_len,
            *special_counts, digit_count,
            has_ip, is_short_url, tld_country_flag,
            url_entropy, contains_suspicious_word, subdomain_count,
            first_path_token_len, ends_with_exe_or_zip,
            special_ratio, path_is_numeric, has_query_string,
            has_base64, query_len, fragment_len
        ]
    except:
        return [0] * 25

def get_feature_names_v3():
    return [
        "url_len", "hostname_len", "tld_len", "path_len",
        *[f"count_{s}" for s in SPECIALS],
        "digit_count", "has_ip", "is_short_url", "tld_country_flag",
        "url_entropy", "contains_suspicious_word", "subdomain_count",
        "first_path_token_len", "ends_with_exe_or_zip",
        "special_ratio", "path_is_numeric", "has_query_string",
        "has_base64", "query_len", "fragment_len"
    ]


# version 4
def extract_lexical_features_v4(url):
    try:
        if not isinstance(url, str):
            return [0] * 25  # 25 features

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        path = parsed.path or ""

        url_len = len(url)
        hostname_len = len(domain)
        path_len = len(path)
        tld = domain.split('.')[-1] if '.' in domain else ''
        tld_len = len(tld)

        special_counts = [url.lower().count(item) for item in SPECIALS]
        digit_count = sum(c.isdigit() for c in url)

        has_ip = int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain)))
        is_short_url = int(is_shortener_url(url))
        tld_country_flag = int(tld.lower() in COUNTRY_CODE_TLDS)

        url_entropy = calculate_entropy(url)
        contains_suspicious_word = int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS))
        subdomain_count = max(0, domain.count('.') - 1)
        first_path_token_len = len(path.strip("/").split("/")[0]) if path.strip("/") else 0
        ends_with_exe_or_zip = int(bool(re.search(r'\.(exe|zip)$', url.lower())))

        special_ratio = sum(1 for c in url if not c.isalnum()) / len(url)
        path_is_numeric = int(path.strip("/").isdigit())
        has_query_string = int('?' in url)
        has_base64 = int(bool(re.search(r'(?:[A-Za-z0-9+/]{20,}={0,2})', url)))
        query_len = len(parsed.query)
        fragment_len = len(parsed.fragment)

        return [
            url_len, hostname_len, tld_len, path_len,
            *special_counts, digit_count,
            has_ip, is_short_url, tld_country_flag,
            url_entropy, contains_suspicious_word, subdomain_count,
            first_path_token_len, ends_with_exe_or_zip,
            special_ratio, path_is_numeric, has_query_string,
            has_base64, query_len, fragment_len
        ]
    except:
        return [0] * 25





# ✅ main 함수: 특징 추출 후 저장
if __name__ == "__main__":
    # Load the dataset and extract features
    df = pd.read_csv("dataset/final_url_dataset.csv")
    features = [extract_lexical_features_v3(url) for url in tqdm(df["url"], desc="🔍 V3 특징 추출 중")]
    feature_df = pd.DataFrame(features, columns=get_feature_names_v3())
    df_with_features = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    df_with_features.to_csv("dataset/final_url_dataset_with_features_v4.csv", index=False)
    print("✅ 저장 완료: dataset/final_url_dataset_with_features_v4.csv")
