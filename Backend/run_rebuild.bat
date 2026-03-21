@echo off
call venv\Scripts\activate
python -u manual_rebuild.py > rebuild_log.txt 2>&1
echo Done. >> rebuild_log.txt
