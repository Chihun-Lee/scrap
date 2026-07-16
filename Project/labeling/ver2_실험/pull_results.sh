#!/bin/bash
# 클러스터 학습 결과 자동 회수 (맥북 launchd, 30분 주기) — 2026-07-15
# 오프라인이면 조용히 종료, 연결되면 결과만 증분 rsync. 중지: launchctl unload ~/Library/LaunchAgents/com.chihun.scrap-pull-results.plist
LOCAL="/Users/chihun/Code/철스크랩/scrap/Project/labeling/ver2_실험"
DEST="$LOCAL/cluster_results"
REMOTE="chihun@134.75.147.179"
LOG="$LOCAL/pull_results.log"
mkdir -p "$DEST"

ssh -p 6000 -o ConnectTimeout=6 -o BatchMode=yes "$REMOTE" true 2>/dev/null || exit 0

# 1) 결과 요약: CSV / 완료 마커 / 학습 로그
rsync -a -e "ssh -p 6000" \
  --include="exp*_results.csv" --include="exp*.done" --include="exp*.log" --exclude="*" \
  "$REMOTE:scrap/Project/labeling/ver2_실험/" "$DEST/" >> "$LOG" 2>&1

# 2) 런 산출물: best 가중치 + 학습 곡선 + 설정 (last.pt 등 대용량 중간물 제외)
rsync -a --prune-empty-dirs -e "ssh -p 6000" \
  --include="*/" --include="best.pt" --include="results.csv" --include="args.yaml" \
  --include="*.png" --include="*.jpg" --exclude="*" \
  "$REMOTE:scrap/Project/labeling/ver2_실험/runs/" "$DEST/runs/" >> "$LOG" 2>&1

done_n=$(ls "$DEST"/exp*.done 2>/dev/null | wc -l | tr -d ' ')
echo "$(date '+%F %T') pulled (done markers: $done_n/4)" >> "$LOG"

# 전부 완료(exp2 스윕, exp6 ablation, exp7 + exp3까지 4개 마커)면 macOS 알림 1회
if [ "$done_n" -ge 3 ] && [ ! -f "$DEST/.notified" ]; then
  osascript -e 'display notification "exp2·exp6·exp7 학습 결과 회수 완료 (cluster_results/)" with title "철스크랩 학습 완료"' 2>/dev/null
  touch "$DEST/.notified"
fi
