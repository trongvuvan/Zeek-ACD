#!/usr/bin/env bash
# Generate ordinary benign web traffic from this host so Zeek sees whole
# connections (conn_state=SF, service=ssl/http). Used to fill the BENIGN-HTTPS
# gap in the training data (see PROGRESS.md "Việc tiếp theo" #1).
# Usage: tools/gen_benign.sh <duration_seconds> [sites_file]
# sites_file (one host per line) replaces the built-in list -- use a disjoint
# list for a held-out test capture.
set -u
DUR=${1:-1500}
END=$(( $(date +%s) + DUR ))
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
SITES=(
 www.wikipedia.org en.wikipedia.org www.google.com www.bing.com duckduckgo.com
 www.github.com raw.githubusercontent.com www.python.org pypi.org docs.python.org
 www.mozilla.org developer.mozilla.org www.cloudflare.com www.microsoft.com learn.microsoft.com
 www.apple.com www.amazon.com www.bbc.com www.bbc.co.uk www.reuters.com www.nytimes.com
 www.theguardian.com stackoverflow.com news.ycombinator.com www.reddit.com
 www.debian.org www.ubuntu.com rockylinux.org www.kernel.org www.gnu.org
 www.nhk.or.jp www.yahoo.co.jp www.rakuten.co.jp vnexpress.net tuoitre.vn
 www.youtube.com www.linkedin.com www.adobe.com www.oracle.com www.ibm.com
 www.w3.org www.iana.org example.com httpbin.org www.speedtest.net
 cdn.jsdelivr.net cdnjs.cloudflare.com fonts.googleapis.com ajax.googleapis.com
)
if [ -n "${2:-}" ]; then mapfile -t SITES < "$2"; fi
PLAIN=(neverssl.com example.com httpforever.com www.iana.org)
POLL=(api.github.com www.google.com/generate_204 connectivitycheck.gstatic.com/generate_204)

fetch() { curl -sS -L --max-time 20 -A "$UA" -o /dev/null "$@" >/dev/null 2>&1; }

# periodic "update check"/connectivity polling -- benign but beacon-like
( while [ "$(date +%s)" -lt "$END" ]; do
    fetch "https://${POLL[RANDOM % ${#POLL[@]}]}"
    sleep $(( 25 + RANDOM % 15 ))
  done ) &

# browsing sessions: a page plus a few sub-requests, keep-alive reused
( while [ "$(date +%s)" -lt "$END" ]; do
    s=${SITES[RANDOM % ${#SITES[@]}]}
    fetch "https://$s/" "https://$s/favicon.ico" "https://$s/robots.txt"
    [ $((RANDOM % 6)) -eq 0 ] && fetch "http://${PLAIN[RANDOM % ${#PLAIN[@]}]}/"
    [ $((RANDOM % 4)) -eq 0 ] && fetch --http1.1 "https://${SITES[RANDOM % ${#SITES[@]}]}/"
    sleep $(( 1 + RANDOM % 8 ))
  done ) &

# occasional larger downloads
( while [ "$(date +%s)" -lt "$END" ]; do
    case $((RANDOM % 3)) in
      0) fetch --max-time 30 "https://www.kernel.org/pub/linux/kernel/v6.x/sha256sums.asc" ;;
      1) fetch --max-time 30 "https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js" ;;
      2) fetch --max-time 30 "https://raw.githubusercontent.com/python/cpython/main/README.rst" ;;
    esac
    sleep $(( 40 + RANDOM % 60 ))
  done ) &
wait
