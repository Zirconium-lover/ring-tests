#!/bin/bash
# Э635, метод 2 Плясова (трубы под давлением): предсказание с форой 5 и 8 °C (вилка по смыслу порога метода 1).
S=$1
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
E635_BIAS=5.0 /opt/mvenv/bin/python e635_plyasov.py $S/e635 m2 4
E635_BIAS=8.0 /opt/mvenv/bin/python e635_plyasov.py $S/e635 m2 4
echo E635_Q2_DONE
