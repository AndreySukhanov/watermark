#!/bin/bash
# Monitor AI processing progress on RunPod
SSH="ssh -o StrictHostKeyChecking=no root@69.30.85.178 -p 22115 -i ~/.ssh/id_ed25519"
TOTAL=4765

while true; do
    DONE=$($SSH "ls /workspace/temp_web/ai_*/all_inpainted/*.png 2>/dev/null | wc -l" 2>/dev/null)
    CURRENT=$($SSH "ls /workspace/temp_web/ai_*/inpainted/*.png 2>/dev/null | wc -l" 2>/dev/null)
    IOPAINT=$($SSH "pgrep -c iopaint 2>/dev/null || echo 0" 2>/dev/null)
    OUTPUT=$($SSH "ls /workspace/temp_web/output_*.mp4 2>/dev/null | head -3" 2>/dev/null)

    PCT=$(( (DONE * 100) / TOTAL ))
    echo "[$(date +%H:%M:%S)] Done: $DONE/$TOTAL ($PCT%) | Current batch: $CURRENT | iopaint running: $IOPAINT"

    if [ -n "$OUTPUT" ]; then
        echo "OUTPUT READY: $OUTPUT"
        break
    fi

    if [ "$DONE" -ge "$TOTAL" ] 2>/dev/null; then
        echo "ALL FRAMES DONE!"
        break
    fi

    sleep 30
done
