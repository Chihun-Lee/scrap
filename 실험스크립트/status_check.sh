#!/bin/bash
# node002/005 학습 상태 1줄 요약 — 모니터링 세션이 ssh로 호출 (2026-07-15)
# 크래시 판정: 해당 런의 결과 CSV 행이 없는데 로그 mtime이 45분 이상 정지 (OOM 자동재시도 Traceback 오탐 방지)
cd "$(dirname "$0")"
rows() { if [ -f "$1" ]; then echo $(( $(wc -l < "$1") - 1 )); else echo 0; fi; }
e2=$(rows exp2_results.csv)
e6=$(rows exp6_results.csv)
e7=$(rows exp7_results.csv)
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>5000{c++} END{print c+0}')
now=$(date +%s)
dead=""
check() { # $1=log $2=variant패턴 $3=csv
  [ -f "$3" ] && grep -q "^$2," "$3" && return
  if [ -f "$1" ]; then
    age=$(( (now - $(stat -c %Y "$1")) / 60 ))
    [ "$age" -gt 45 ] && dead="${dead}$1(${age}m),"
  fi
}
check exp2_base.log  base  exp2_results.csv
check exp2_cut8.log  cut8  exp2_results.csv
check exp2_cut10.log cut10 exp2_results.csv
check exp2_cut12.log cut12 exp2_results.csv
check exp2_cut16.log cut16 exp2_results.csv
check exp6_noexc.log cut10_noexc exp6_results.csv
check exp6_w2.log    cut10_w2    exp6_results.csv
check exp7_mr1.log   cut10 exp7_results.csv
ep=$(grep -h -o "[0-9]*/100" exp2_cut10.log 2>/dev/null | tail -1)
echo "exp2done=$e2/5 exp6done=$e6/2 exp7done=$e7/1 gpus_busy=$busy cut10_epoch=${ep:-scan} stalled=${dead:-none}"
