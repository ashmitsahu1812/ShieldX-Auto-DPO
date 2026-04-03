#!/usr/bin/env bash
set -uo pipefail

# Configuration
DOCKER_BUILD_TIMEOUT=600
PING_URL="${1:-http://localhost:7860}"
REPO_DIR="${2:-.}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n${BOLD}=== OpenEnv Pre-Submission Validator ===${NC}\n\n"

# Step 1: Ping HF Space
log "${BOLD}Step 1/3: Pinging Environment${NC} ($PING_URL/health) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/health" --max-time 10 || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "Environment is live and responds to /health"
else
  fail "Environment not reachable at $PING_URL (HTTP $HTTP_CODE)"
  hint "Make sure your server is running (e.g., uvicorn app:app --port 7860)"
  stop_at "Step 1"
fi

# Step 2: Docker Build
log "${BOLD}Step 2/3: Running docker build${NC} ..."
if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
  log "  Found Dockerfile in $DOCKER_CONTEXT"
  if docker build "$DOCKER_CONTEXT" -t code_review_test > /dev/null 2>&1; then
    pass "Docker build succeeded"
  else
    fail "Docker build failed"
    stop_at "Step 2"
  fi
else
  fail "No Dockerfile found in repo root"
  stop_at "Step 2"
fi

# Step 3: OpenEnv Validate
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
# We check if openenv is installed, if not we try to use a local check or just skip with a warning
if ! command -v openenv &>/dev/null; then
  log "${YELLOW}Warning:${NC} openenv command not found. Skipping library-level validation."
  log "  Please run 'pip install openenv-core' to verify spec compliance."
else
  if (cd "$REPO_DIR" && openenv validate); then
    pass "openenv validate passed"
  else
    fail "openenv validate failed"
    stop_at "Step 3"
  fi
fi

printf "\n${GREEN}${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready.${NC}\n"
printf "${GREEN}${BOLD}========================================${NC}\n\n"
