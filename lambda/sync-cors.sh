#!/usr/bin/env bash
#
# sync-cors.sh — Apply CORS origins from cors.js to the S3 bucket.
#
# Usage: ./sync-cors.sh [bucket-name]
#   bucket-name defaults to gamestudy-data

set -euo pipefail

BUCKET="${1:-gamestudy-data}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Extract the ALLOWED_ORIGINS array from cors.js
ORIGINS=$(node -e "
  const { ALLOWED_ORIGINS } = require('${SCRIPT_DIR}/cors');
  ALLOWED_ORIGINS.forEach(o => console.log(o));
")

# Build the CORS JSON
RULES="[]"
for ORIGIN in $ORIGINS; do
  RULES=$(echo "$RULES" | node -e "
    const fs = require('fs');
    const rules = JSON.parse(fs.readFileSync('/dev/stdin','utf8'));
    rules.push({
      AllowedOrigins: ['${ORIGIN}'],
      AllowedMethods: ['GET','PUT','POST'],
      AllowedHeaders: ['*'],
      MaxAgeSeconds: 3600
    });
    console.log(JSON.stringify(rules));
  ")
done

CORS_CONFIG="{\"CORSRules\": ${RULES}}"

echo "Applying CORS config to s3://${BUCKET}:"
echo "$CORS_CONFIG" | python3 -m json.tool

aws s3api put-bucket-cors \
  --bucket "$BUCKET" \
  --cors-configuration "$CORS_CONFIG"

echo "Done. Verifying:"
aws s3api get-bucket-cors --bucket "$BUCKET"
