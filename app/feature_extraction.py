from __future__ import annotations

import ipaddress
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any


SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
}

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "password",
    "bank",
    "confirm",
    "wallet",
    "urgent",
}

TRUSTED_BRANDS = {
    "google",
    "microsoft",
    "apple",
    "paypal",
    "amazon",
    "facebook",
    "instagram",
    "whatsapp",
    "netflix",
    "bank",
}

SUSPICIOUS_TLDS = {"zip", "mov", "xyz", "top", "click", "loan"}


class _SignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors = 0
        self.external_anchors = 0
        self.tag_links = 0
        self.external_tag_links = 0
        self.iframes = 0
        self.mail_forms = 0
        self.blank_forms = 0
        self.popups = 0
        self.mouseover = 0
        self.right_click_blocks = 0
        self.emails = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value or "" for key, value in attrs}
        if tag == "a":
            self.anchors += 1
            href = attr_map.get("href", "")
            if href.startswith("http"):
                self.external_anchors += 1
        if tag in {"link", "script", "img"}:
            self.tag_links += 1
            resource = attr_map.get("href") or attr_map.get("src") or ""
            if resource.startswith("http"):
                self.external_tag_links += 1
        if tag == "iframe":
            self.iframes += 1
        if tag == "form":
            action = attr_map.get("action", "").strip().lower()
            if action.startswith("mailto:"):
                self.mail_forms += 1
            if action in {"", "about:blank", "#"}:
                self.blank_forms += 1

    def handle_data(self, data: str) -> None:
        lowered = data.lower()
        if "window.open" in lowered or "popup" in lowered:
            self.popups += 1
        if "onmouseover" in lowered:
            self.mouseover += 1
        if "event.button==2" in lowered or "contextmenu" in lowered:
            self.right_click_blocks += 1
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", data):
            self.emails += 1


@dataclass
class ExtractionResult:
    features: dict[str, float]
    notes: list[str]
    fetched: bool
    page_text_length: int


def _normalize_url(url: str) -> str:
    candidate = url.strip()
    if not candidate:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = f"https://{candidate}"
    return candidate


def _hostname_parts(url: str) -> tuple[str, list[str]]:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host, [part for part in host.split(".") if part]


def _try_fetch_page(url: str) -> tuple[str, bool, list[str]]:
    if not url:
        return "", False, []

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 WebpageDetection/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(250_000)
            if "text" not in content_type and "html" not in content_type:
                return "", False, [f"Fetched content type was `{content_type}` instead of HTML/text."]
            return body.decode("utf-8", errors="ignore"), True, []
    except urllib.error.URLError as exc:
        return "", False, [f"Automatic page fetch failed: {exc.reason}"]
    except Exception:
        return "", False, ["Automatic page fetch failed for this URL."]


def _safe_ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _triage_ratio(ratio: float, low_bad: float, high_good: float) -> int:
    if ratio <= low_bad:
        return -1
    if ratio >= high_good:
        return 1
    return 0


def _classify_url_length(url: str) -> int:
    length = len(url)
    if length < 54:
        return 1
    if length <= 75:
        return 0
    return -1


def _classify_subdomains(parts: list[str]) -> int:
    if len(parts) <= 2:
        return 1
    if len(parts) == 3:
        return 0
    return -1


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_features_from_source(url: str = "", text: str = "", html: str = "") -> ExtractionResult:
    normalized_url = _normalize_url(url)
    host, host_parts = _hostname_parts(normalized_url)
    parsed = urllib.parse.urlparse(normalized_url) if normalized_url else urllib.parse.urlparse("")
    supplied_text = "\n".join(part for part in [html, text] if part).strip()

    fetched_html = ""
    fetched = False
    notes: list[str] = []
    if normalized_url and not supplied_text:
        fetched_html, fetched, fetch_notes = _try_fetch_page(normalized_url)
        notes.extend(fetch_notes)

    combined_text = f"{supplied_text}\n{fetched_html}".strip()
    parser = _SignalParser()
    if combined_text:
        parser.feed(combined_text)

    lowered_url = normalized_url.lower()
    lowered_text = combined_text.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    suspicious_word_hits = sum(keyword in lowered_url or keyword in lowered_text for keyword in SUSPICIOUS_KEYWORDS)
    brand_hits = [brand for brand in TRUSTED_BRANDS if brand in lowered_text or brand in host]

    tld = host_parts[-1] if host_parts else ""
    contains_brand_mismatch = any(brand in host for brand in TRUSTED_BRANDS) and ("-" in host or len(host_parts) > 3)

    features = {
        "having_IP_Address": -1 if _is_ip_host(host) else 1,
        "URL_Length": _classify_url_length(normalized_url),
        "Shortining_Service": -1 if host in SHORTENER_DOMAINS else 1,
        "having_At_Symbol": -1 if "@" in normalized_url else 1,
        "double_slash_redirecting": -1 if "//" in normalized_url.replace(f"{parsed.scheme}://", "", 1) else 1,
        "Prefix_Suffix": -1 if "-" in host else 1,
        "having_Sub_Domain": _classify_subdomains(host_parts),
        "SSLfinal_State": 1 if parsed.scheme == "https" else -1,
        "Domain_registeration_length": 0,
        "Favicon": 0,
        "port": -1 if parsed.port not in (None, 80, 443) else 1,
        "HTTPS_token": -1 if "https" in host.replace("https://", "") else 1,
        "Request_URL": _triage_ratio(_safe_ratio(parser.external_tag_links, parser.tag_links), 0.22, 0.61),
        "URL_of_Anchor": _triage_ratio(1 - _safe_ratio(parser.external_anchors, parser.anchors), 0.31, 0.67),
        "Links_in_tags": _triage_ratio(1 - _safe_ratio(parser.external_tag_links, parser.tag_links), 0.17, 0.81),
        "SFH": -1 if parser.blank_forms else 1 if not parser.mail_forms else 0,
        "Submitting_to_email": -1 if parser.mail_forms or "mailto:" in lowered_text else 1,
        "Abnormal_URL": -1 if ("@" in normalized_url or contains_brand_mismatch) else 1,
        "Redirect": -1 if query.count("http") > 0 else 1,
        "on_mouseover": -1 if parser.mouseover else 1,
        "RightClick": -1 if parser.right_click_blocks else 1,
        "popUpWidnow": -1 if parser.popups else 1,
        "Iframe": -1 if parser.iframes else 1,
        "age_of_domain": 0,
        "DNSRecord": 0 if not host else 1,
        "web_traffic": 0,
        "Page_Rank": -1 if suspicious_word_hits >= 3 or tld in SUSPICIOUS_TLDS else 0,
        "Google_Index": -1 if suspicious_word_hits >= 4 else 1,
        "Links_pointing_to_page": 1 if parser.anchors >= 3 else 0 if parser.anchors else -1,
        "Statistical_report": -1 if tld in SUSPICIOUS_TLDS else 1,
    }

    if suspicious_word_hits:
        notes.append(f"Detected {suspicious_word_hits} suspicious keyword signals in the URL/content.")
    if parser.iframes:
        notes.append("Iframe usage detected in the supplied webpage content.")
    if parser.mail_forms:
        notes.append("The page appears to submit data through email-style form actions.")
    if contains_brand_mismatch:
        notes.append("Brand-like wording appears in a structurally unusual hostname.")
    if brand_hits:
        notes.append(f"Recognized brand-related terms: {', '.join(sorted(set(brand_hits))[:4])}.")

    return ExtractionResult(
        features=features,
        notes=notes,
        fetched=fetched,
        page_text_length=len(combined_text),
    )


def summarize_feature_flags(features: dict[str, float]) -> list[dict[str, Any]]:
    labels = {
        "having_IP_Address": "Raw IP address used instead of a normal domain",
        "URL_Length": "Very long URL",
        "Shortining_Service": "URL shortening service detected",
        "having_At_Symbol": "`@` symbol present in URL",
        "Prefix_Suffix": "Hyphenated hostname",
        "SSLfinal_State": "Missing HTTPS",
        "HTTPS_token": "Misleading `https` token inside hostname",
        "Submitting_to_email": "Email-based form submission",
        "Iframe": "Embedded iframe usage",
        "popUpWidnow": "Popup-like behavior in content",
        "RightClick": "Right-click blocking behavior",
        "SFH": "Suspicious or blank form handler",
        "Statistical_report": "Suspicious top-level domain",
    }

    findings: list[dict[str, Any]] = []
    for key, description in labels.items():
        value = features.get(key, 0)
        if value == -1:
            findings.append({"feature": key, "severity": "high", "description": description})
        elif value == 0:
            findings.append({"feature": key, "severity": "medium", "description": f"Uncertain signal for {description.lower()}."})
    return findings[:8]
