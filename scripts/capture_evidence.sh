#!/usr/bin/env bash
# Capturas de pantalla (nvidia-smi + top por núcleo) en tres momentos: reposo, fase de datos (CPU)
# y fase de entrenamiento (GPU). Abre una terminal kitty, deja correr la carga, fotografía SOLO
# esa ventana y guarda también el texto en results/evidence/screens/*.txt.
#   bash scripts/capture_evidence.sh
set -u
cd "$(dirname "$0")/.."
OUT=results/evidence/screens; mkdir -p "$OUT"

snap() {  # $1 = nombre, $2 = titulo, $3 = segundos de espera antes de leer, $4.. = carga (opcional)
  name=$1; title=$2; wait=$3; shift 3
  if [ $# -gt 0 ]; then "$@" > "$OUT/$name.workload.log" 2>&1 & wl=$!; else wl=""; fi
  kitty --title "$title" -o remember_window_size=no -o initial_window_width=165c \
        -o initial_window_height=60c -o font_size=9 bash -c "
    sleep 1; clear; sleep $((wait - 1))
    echo \"\$ nvidia-smi   [$title]\"; nvidia-smi | tee $OUT/$name.txt
    echo; echo '\$ top (uso por núcleo de CPU)'
    top -b -n2 -d1 -1 -w 200 | awk '/^top -/{n++} n==2' | head -26 | tee -a $OUT/$name.txt
    sleep 6" &
  kp=$!
  wid=$(xdotool search --sync --name "^$title\$" | head -1)
  sleep $((wait + 4))
  import -window "$wid" "$OUT/$name.png"
  wait $kp 2>/dev/null; [ -n "$wl" ] && { kill $wl 2>/dev/null; wait $wl 2>/dev/null; }
  sleep 2
}

snap 1_reposo "Fase 0 - reposo" 2
snap 2_datos_cpu "Fase A - preparacion de datos CPU" 3 python3 scripts/evidence_workload.py data
snap 3_entrenamiento_gpu "Fase B - entrenamiento CUDA GPU" 9 python3 scripts/evidence_workload.py train
ls -la "$OUT"
