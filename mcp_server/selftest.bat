@echo off
rem MCP self-test: spawn server.py over stdio and walk through initialize / tools/list / tools/call
rem Output is also saved to logs\selftest_output.txt
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "..\..\..\.venv\Scripts\python.exe" (
    set "PY=..\..\..\.venv\Scripts\python.exe"
)
if not exist logs mkdir logs
"%PY%" mcp_server\test_client.py %* > logs\selftest_output.txt 2>&1
type logs\selftest_output.txt
pause
