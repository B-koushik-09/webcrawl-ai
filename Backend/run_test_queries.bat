@echo off
call venv\Scripts\activate
python -u test_queries.py > test_queries_log.txt 2>&1
echo Done. >> test_queries_log.txt
