# ring-tests

Расчёты разрушения в форке CalculiX 2.23 с моделью повреждения и удаления
элементов (Zirconium-lover/ccx-arch2, коммит c75ad9b).

Решатель подключается только через переменную окружения:

    export CCX_ARCH2=/home/user/ccx-arch2
    tools/build_solver.sh --check      # сборка + проверка на test/fast

Бинарник: `$CCX_ARCH2/build-mkl/ccx_2.23_pardiso` (PARDISO + SPOOLES).

## Состав

| каталог | что там |
|---|---|
| `notes/` | заметки: выписки из источников, решения по модели |
| `tools/` | сборка решателя, общие скрипты |

Большие результаты (`.frd`, `.dat` > ~50 МБ) в git не кладутся, см. `.gitignore`.
