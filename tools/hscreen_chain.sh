#!/usr/bin/env bash
# ждать конца очереди runs/ring, затем запустить проверку высоты (2 ядра)
cd /home/user/ring-tests
until grep -q "queue finished" runs/ring/queue.log 2>/dev/null; do sleep 60; done
export CCX_ARCH2=/home/user/ccx-arch2
STAGE=hscreen THREADS=2 tools/queue_stage1.sh 1 F_H5_mu0.07_C3D8I F_H6_mu0.07_C3D8I F_H8_mu0.07_C3D8I \
    F_H5_mu0.05_C3D8I F_H6_mu0.05_C3D8I F_H8_mu0.05_C3D8I
