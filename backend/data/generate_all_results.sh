#!/bin/bash

VIDEO_NAMES=(
  "cctv_cafe.mp4"
  "cctv_cafe1.mp4"
  "cctv_cafe2.mp4"
  "cctv_cafe3.mp4"
  "warehouse2.mp4"
  "warehouse_close.mp4"
)

API_URL="http://0.0.0.0:8000"

# for VIDEO in "${VIDEO_NAMES[@]}"
# do
#   SUCCESS=0
#   while [ $SUCCESS -eq 0 ]; do
#     echo "Analyzing video frames for: $VIDEO"
#     STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/analyze-video" \
#       -H "Content-Type: application/json" \
#       -d '{"video_path": "data/'"$VIDEO"'", "frame_interval": 120}')

#     if [[ $STATUS_CODE =~ ^2 ]]; then
#       SUCCESS=1
#     else
#       echo "Failed. Status code: $STATUS_CODE. Retrying in 1 minute..."
#       sleep 60
#     fi
#   done

#   echo "Waiting 1 minute before next call..."
#   sleep 60
# done

for VIDEO in "${VIDEO_NAMES[@]}"
do
  SUCCESS=0
  while [ $SUCCESS -eq 0 ]; do
    echo "Performing LLM analysis for: $VIDEO"
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/analyze-video-llm" \
      -H "Content-Type: application/json" \
      -d '{"video_path": "data/'"$VIDEO"'", "frame_interval": 240, "force":"True"}')

    if [[ $STATUS_CODE =~ ^2 ]]; then
      SUCCESS=1
    else
      echo "Failed. Status code: $STATUS_CODE. Retrying in 1 minute..."
      sleep 30
    fi
  done

  echo "Waiting 3 minutes before next video..."
  sleep 180
done