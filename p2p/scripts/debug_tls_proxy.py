#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, ssl, socket, argparse, subprocess, shutil, re, json, time
from urllib.parse import urlparse

try:
    import socks  # PySocks
except Exception:
    socks = None

try:
    import requests
except Exception:
    requests = None

def pick_proxy():
    for k in ("ALL_PROXY","HTTPS_PROXY","HTTP_PROXY","all_proxy","https_proxy","http_proxy"):
        v = os.environ.get(k)
        if v:
            u = urlparse(v)
            if u.hostname and u.port:
                return {"raw": v, "scheme": (u.scheme or "http").lower(), "host": u.hostname, "port": int(u.port)}
    return None

def manual_connect_test(proxy, host, port, timeout=10):
    """
    纯 socket 手写 HTTP CONNECT，读取响应首行与若干头部。
    仅验证代理是否放行 CONNECT，不做 TLS。
    """
    start = time.time()
    res = {"ok": False, "error": None, "resp_line": None, "elapsed_ms": None}
    try:
        s = socket.create_connection((proxy["host"], proxy["port"]), timeout=timeout)
        s.settimeout(timeout)
        req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\nProxy-Connection: Keep-Alive\r\n\r\n"
        s.sendall(req.encode("utf-8"))
        buf = s.recv(4096)
        s.close()
        line = buf.split(b"\r\n",1)[0].decode("latin1","ignore")
        res["resp_line"] = line
        res["ok"] = line.startswith("HTTP/1.1 200") or line.startswith("HTTP/1.0 200")
    except Exception as e:
        res["error"] = repr(e)
    res["elapsed_ms"] = int((time.time()-start)*1000)
    return res

def pysocks_tls_test(proxy, host, port, sni=None, timeout=12):
    res = {"ok": False, "error": None, "issuer": None, "subject": None}
    if socks is None:
        res["error"] = "PySocks not installed"
        return res
    try:
        s = socks.socksocket()
        s.settimeout(timeout)
        sch = proxy["scheme"]
        if sch.startswith("socks5"):
            s.set_proxy(socks.SOCKS5, proxy["host"], proxy["port"], rdns=True)
        elif sch.startswith("socks4"):
            s.set_proxy(socks.SOCKS4, proxy["host"], proxy["port"], rdns=True)
        else:
            s.set_proxy(socks.HTTP, proxy["host"], proxy["port"], rdns=True)  # HTTP CONNECT
        s.connect((host, port))
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        with ctx.wrap_socket(s, server_hostname=sni or host) as ssock:
            cert = ssock.getpeercert()
        # human summary
        res["issuer"] = str(cert.get("issuer"))
        res["subject"] = str(cert.get("subject"))
        res["ok"] = True
    except Exception as e:
        res["error"] = repr(e)
    return res

def openssl_tls_test(proxy, host, port, sni=None, timeout=15):
    res = {"ok": False, "error": None, "issuer": None, "subject": None, "version": None}
    exe = shutil.which("openssl")
    if not exe:
        res["error"] = "openssl not found"
        return res
    try:
        ver = subprocess.check_output([exe, "version"], timeout=5).decode().strip()
        res["version"] = ver
    except Exception:
        pass
    cmd = [exe, "s_client", "-servername", sni or host, "-connect", f"{host}:{port}", "-showcerts", "-verify", "0"]
    # OpenSSL 1.1.1+ 支持 -proxy；旧版不支持
    if proxy:
        cmd += ["-proxy", f"{proxy['host']}:{proxy['port']}"]
    try:
        out = subprocess.check_output(cmd, input=b"", stderr=subprocess.STDOUT, timeout=timeout)
        text = out.decode("latin1","ignore")
        if "unknown option -proxy" in text:
            res["error"] = "openssl without -proxy support"
            return res
        # 抽取证书主干
        m_subj = re.search(r"subject\s*=([^\n]+)", text)
        m_iss  = re.search(r"issuer\s*=([^\n]+)", text)
        res["subject"] = m_subj.group(1).strip() if m_subj else None
        res["issuer"]  = m_iss.group(1).strip() if m_iss else None
        # 完成握手常见标志：SSL-Session 或 Verify return code
        res["ok"] = ("SSL-Session:" in text) or ("Verify return code" in text) or (res["subject"] is not None)
    except Exception as e:
        res["error"] = repr(e)
    return res

def requests_head_test(proxy, host, timeout=10):
    res = {"ok": False, "error": None, "status": None}
    if requests is None:
        res["error"] = "requests not installed"
        return res
    try:
        s = requests.Session()
        s.trust_env = False
        s.headers.update({"User-Agent":"TLS-Debug/1.0", "Connection":"close"})
        s.proxies = {
            "http":  f"http://{proxy['host']}:{proxy['port']}",
            "https": f"http://{proxy['host']}:{proxy['port']}",
        }
        r = s.head(f"https://{host}/", timeout=timeout, allow_redirects=True)
        res["status"] = r.status_code
        res["ok"] = True
    except Exception as e:
        res["error"] = repr(e)
    return res

def main():
    ap = argparse.ArgumentParser(description="Debug TLS over HTTP/SOCKS proxy (with reverse tunnel)")
    ap.add_argument("--url", required=True, help="Target URL (used to extract host/SNI)")
    ap.add_argument("--port", type=int, default=443)
    args = ap.parse_args()

    pr = urlparse(args.url)
    host = pr.hostname or args.url
    sni  = host  # 对大部分站点 SNI=host
    proxy = pick_proxy()

    print("=== proxy from env ===")
    print(json.dumps(proxy, indent=2, ensure_ascii=False))

    if not proxy:
        print("No proxy found in env (HTTP(S)_PROXY / ALL_PROXY). Aborting.")
        sys.exit(2)

    print("\n=== step 1: raw HTTP CONNECT ===")
    print(json.dumps(manual_connect_test(proxy, host, args.port), indent=2, ensure_ascii=False))

    print("\n=== step 2: PySocks TLS (Python ssl) ===")
    print(json.dumps(pysocks_tls_test(proxy, host, args.port, sni=sni), indent=2, ensure_ascii=False))

    print("\n=== step 3: OpenSSL s_client -proxy ===")
    print(json.dumps(openssl_tls_test(proxy, host, args.port, sni=sni), indent=2, ensure_ascii=False))

    print("\n=== step 4: requests HEAD via proxy ===")
    print(json.dumps(requests_head_test(proxy, host), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
