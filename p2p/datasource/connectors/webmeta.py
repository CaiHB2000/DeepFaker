import os
import socket
import ssl
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import subprocess, shutil, re

import tldextract
from cryptography import x509
from cryptography.hazmat.backends import default_backend

from p2p.datasource.schema import (
    RdapRecord, HttpFingerprint, TlsFingerprint, WebArchiveRecord,
    ProvenanceExternalMeta, now_unix
)
from p2p.datasource.utils import normalize_header_dict, extract_host_from_url
from p2p.datasource.net import http_get, http_head
from cryptography.x509.oid import NameOID


try:
    import socks  # PySocks
except Exception:
    socks = None

# ---------- RDAP ----------
def rdap_lookup(domain: str) -> Optional[RdapRecord]:
    if not domain:
        return None
    try:
        r = http_get(f"https://rdap.org/domain/{domain}", timeout=10)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", []) or []
        events = {}
        for ev in data.get("events", []):
            action = ev.get("eventAction"); date = ev.get("eventDate")
            if action and date:
                events[action] = date  # ISO8601
        return RdapRecord(
            domain=domain,
            registry=data.get("port43", None),
            domain_status=status,
            events=events,
            raw=data,
        )
    except Exception:
        return None

# ---------- Wayback / CDX ----------
def wayback_summary(url: str) -> Optional[WebArchiveRecord]:
    try:
        params = {"url": url, "output": "json", "from": "2000", "filter": "statuscode:200"}
        r = http_get("https://web.archive.org/cdx/search/cdx", params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or len(rows) <= 1:
            return WebArchiveRecord(url=url, first_seen_ts=None, latest_seen_ts=None, snapshots=0)
        timestamps = [row[1] for row in rows[1:] if len(row) > 1]
        timestamps.sort()
        return WebArchiveRecord(url=url, first_seen_ts=timestamps[0], latest_seen_ts=timestamps[-1], snapshots=len(timestamps))
    except Exception:
        return None

# ---------- HTTP HEAD (fallback to GET) ----------
def http_fingerprint(url: str) -> Optional[HttpFingerprint]:
    try:
        resp = http_head(url, allow_redirects=True, timeout=10)
    except Exception:
        try:
            resp = http_get(url, timeout=12)
        except Exception:
            return None

    headers = normalize_header_dict(resp.headers)  # 多数实现会转小写
    # 兼容大小写取值
    def hget(d, key):
        return d.get(key) or d.get(key.lower()) or d.get(key.title())

    return HttpFingerprint(
        url_requested=url,
        final_url=str(resp.url),
        status_code=resp.status_code,
        headers=headers,
        server=hget(headers, "Server"),
        via=hget(headers, "Via"),
        content_type=hget(headers, "Content-Type"),
    )


# ---------- 代理解析 ----------
def _pick_proxy() -> Optional[Dict[str, str]]:
    for key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        v = os.environ.get(key) or os.environ.get(key.lower())
        if v:
            u = urlparse(v)
            if u.hostname and u.port:
                return {"scheme": (u.scheme or "http").lower(), "host": u.hostname, "port": int(u.port)}
    return None

def _connect_via_proxy(host: str, port: int, proxy: Dict[str, str]) -> Optional[socket.socket]:
    if socks is None:
        return None
    scheme = proxy["scheme"]; phost = proxy["host"]; pport = proxy["port"]
    s = socks.socksocket()
    s.settimeout(12)
    if scheme.startswith("socks5"):
        s.set_proxy(socks.SOCKS5, phost, pport, rdns=True)
    elif scheme.startswith("socks4"):
        s.set_proxy(socks.SOCKS4, phost, pport, rdns=True)
    elif scheme.startswith("http"):
        s.set_proxy(socks.HTTP, phost, pport, rdns=True)  # ← 加 rdns=True
    else:
        return None
    s.connect((host, port))
    return s


# ---------- TLS ----------
def _parse_x509_to_fingerprint(cert) -> TlsFingerprint:
    def _name_str(name):
        try:
            # 扁平化所有 RDN
            items = [(attr.oid, attr.value) for rdn in name.rdns for attr in rdn]
            by_oid = {}
            for oid, val in items:
                by_oid.setdefault(oid, val)

            parts = []
            for oid in (NameOID.COMMON_NAME, NameOID.ORGANIZATION_NAME, NameOID.COUNTRY_NAME):
                v = by_oid.get(oid)
                if v:
                    parts.append(f"{oid._name}={v}")

            if parts:
                return "; ".join(parts)

            # 回退：把全部 RDN 展开
            return ", ".join(f"{oid._name}={val}" for oid, val in items)
        except Exception:
            return str(name)

    issuer  = _name_str(cert.issuer)
    subject = _name_str(cert.subject)
    nb = getattr(cert, "not_valid_before_utc", cert.not_valid_before).isoformat()
    na = getattr(cert, "not_valid_after_utc", cert.not_valid_after).isoformat()
    sig_alg = getattr(cert.signature_hash_algorithm, "name", None)
    sans: List[str] = []
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        sans = [str(x) for x in ext.value.get_values_for_type(x509.DNSName)]
    except Exception:
        pass
    return issuer, subject, nb, na, sig_alg, sans

def _openssl_tls_cert(host: str, port: int, sni: str, proxy: Optional[Dict[str,str]]) -> Optional[TlsFingerprint]:
    if not shutil.which("openssl"):
        return None
    cmd = ["openssl", "s_client",
           "-servername", sni or host,
           "-connect", f"{host}:{port}",
           "-showcerts", "-verify", "0"]
    if proxy:
        # OpenSSL 1.1.1+ 支持 -proxy（HTTP CONNECT）
        cmd += ["-proxy", f"{proxy['host']}:{proxy['port']}"]
    try:
        # 传空 stdin，限制执行时间
        out = subprocess.check_output(cmd, input=b"", stderr=subprocess.STDOUT, timeout=15)
        m = re.search(br"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", out, re.S)
        if not m:
            return None
        pem = m.group(0)
        cert = x509.load_pem_x509_certificate(pem, default_backend())
        issuer, subject, nb, na, sig_alg, sans = _parse_x509_to_fingerprint(cert)
        return TlsFingerprint(
            host=host, port=port, sni=sni or host,
            issuer=issuer, subject=subject,
            not_before=nb, not_after=na,
            serial_number=hex(cert.serial_number),
            sig_hash_alg=sig_alg,
            subject_alternative_names=sans,
        )
    except Exception:
        return None

def tls_fingerprint_for_host(host: str, port: int = 443, sni: Optional[str] = None) -> Optional[TlsFingerprint]:
    """
    先用 PySocks 通过代理做 TLS 握手；失败则用 OpenSSL s_client -proxy 回退。
    """
    if not host:
        return None

    # 代理解析
    proxy = _pick_proxy()

    # --- 引擎 1：PySocks ---
    try:
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.check_hostname = True

        sock: Optional[socket.socket] = None
        if proxy and socks is not None:
            sock = _connect_via_proxy(host, port, proxy)
        else:
            sock = socket.create_connection((host, port), timeout=10)

        if sock is not None:
            with sock:
                with ctx.wrap_socket(sock, server_hostname=sni or host) as ssock:
                    der = ssock.getpeercert(binary_form=True)
            cert = x509.load_der_x509_certificate(der, default_backend())
            issuer, subject, nb, na, sig_alg, sans = _parse_x509_to_fingerprint(cert)
            return TlsFingerprint(
                host=host, port=port, sni=sni or host,
                issuer=issuer, subject=subject,
                not_before=nb, not_after=na,
                serial_number=hex(cert.serial_number),
                sig_hash_alg=sig_alg,
                subject_alternative_names=sans,
            )
    except Exception:
        # 继续尝试 openssl 回退
        pass

    # --- 引擎 2：OpenSSL 回退（HTTP 代理走 CONNECT） ---
    try:
        return _openssl_tls_cert(host, port, sni or host, proxy)
    except Exception:
        return None


# ---------- UTIL ----------
def extract_domain(url: str) -> Optional[str]:
    try:
        pr = urlparse(url)
        host = pr.hostname or ""
        ext = tldextract.extract(host)
        if ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return host
    except Exception:
        return None

# ---------- MAIN SCAN ----------
def scan_url(url: str,
             enable_rdap: bool = True,
             enable_wayback: bool = True,
             enable_http: bool = True,
             enable_tls: bool = True) -> ProvenanceExternalMeta:
    domain = extract_domain(url) or ""
    meta = ProvenanceExternalMeta(url=url, domain=domain, ts_collected=now_unix())

    if enable_rdap:
        rd = rdap_lookup(domain)
        if rd is None: meta.errors.append("rdap_lookup_failed")
        meta.rdap = rd

    if enable_wayback:
        wb = wayback_summary(url)
        if wb is None: meta.errors.append("wayback_summary_failed")
        meta.wayback = wb

    http = None
    if enable_http:
        http = http_fingerprint(url)
        if http is None: meta.errors.append("http_fingerprint_failed")
        meta.http = http

    if enable_tls:
        # 优先对最终跳转后的域名握手，避免 CDN/SNI 不匹配
        target_url = (http.final_url if (http and http.final_url) else url)
        host = extract_host_from_url(target_url)
        tls = tls_fingerprint_for_host(host) if host else None
        if tls is None: meta.errors.append("tls_fingerprint_failed")
        meta.tls = tls

    return meta

