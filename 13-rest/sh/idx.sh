#!/usr/bin/env bash
# https://www.google.com
# https://www.storynory.com/little-red-riding-hood-2
# http://paulgraham.com/worked.html
# https://en.wikipedia.org/wiki/Peter_Pan
# https://edition.cnn.com/2025/07/14/politics/obama-democrats-message
curl -XPOST 'http://localhost:5000' \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.google.com"
  }'
