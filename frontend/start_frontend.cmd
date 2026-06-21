@echo off
cd /d D:\Hacks\gridlock\round2\frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 > ..\frontend_ui.log 2> ..\frontend_ui.err.log
