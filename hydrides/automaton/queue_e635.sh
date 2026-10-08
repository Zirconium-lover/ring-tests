#!/bin/bash
# Э635 против Плясова 2023: метод 1 (подбор форы), сравнение с форой Zry-4.
S=$1
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/mvenv/bin/python e635_plyasov.py $S/e635 m1 4
/opt/mvenv/bin/python e635_plyasov.py $S/e635 zry 4
echo E635_Q1_DONE
