#!/usr/bin/env bash
uvicorn app:app --port 7860 &
SERVER_PID=$!
sleep 2
./validate.sh http://127.0.0.1:7860 .
VALIDATION_CODE=$?
kill $SERVER_PID
exit $VALIDATION_CODE
