import argparse
import sys
from time import sleep
from typing import List

from p2p.datasource.connectors.webmeta import scan_url
from p2p.datasource.utils import append_jsonl, read_lines

def run_webmeta(args):
    urls: List[str] = []
    if args.url:
        urls.append(args.url.strip())
    if args.urls_file:
        urls.extend(list(read_lines(args.urls_file)))
    urls = [u for u in urls if u]
    if not urls:
        print("No input URL(s). Use --url or --urls_file.", file=sys.stderr)
        sys.exit(2)

    out = args.out
    delay = args.delay
    ok = 0

    for i, u in enumerate(urls, 1):
        rec = scan_url(
            u,
            enable_rdap=not args.no_rdap,
            enable_wayback=not args.no_wayback,
            enable_http=not args.no_http,
            enable_tls=not args.no_tls,
        ).to_dict()

        if out:
            append_jsonl(out, rec)
        else:
            print(rec)
        ok += 1

        if delay > 0 and i < len(urls):
            sleep(delay)

    print(f"Done. {ok}/{len(urls)} records written.")

def main():
    p = argparse.ArgumentParser(description="P2P datasource runner (webmeta)")
    p.add_argument("--url", type=str, help="Single URL to scan")
    p.add_argument("--urls_file", type=str, help="A file with one URL per line")
    p.add_argument("--out", type=str, help="Output JSONL path (append mode)")
    p.add_argument("--delay", type=float, default=0.0, help="Optional delay between requests (seconds)")
    p.add_argument("--no-rdap", action="store_true", help="Disable RDAP lookup")
    p.add_argument("--no-wayback", action="store_true", help="Disable Wayback/CDX")
    p.add_argument("--no-http", action="store_true", help="Disable HTTP HEAD/GET fingerprint")
    p.add_argument("--no-tls", action="store_true", help="Disable TLS fingerprint")
    args = p.parse_args()
    run_webmeta(args)

if __name__ == "__main__":
    main()
