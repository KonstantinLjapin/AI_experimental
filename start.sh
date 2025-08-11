# Увеличить лимит блокировки памяти
ulimit -l unlimited

# Увеличить лимит виртуальной памяти
ulimit -v unlimited

# Проверить текущие лимиты
ulimit -a

# Затем запустить скрипт
python mistral_worker.py