#!/usr/bin/env bash
# getUserMedia needs a secure context: localhost counts, file:// does not.
cd "$(dirname "$0")" && python3 -m http.server 8000
