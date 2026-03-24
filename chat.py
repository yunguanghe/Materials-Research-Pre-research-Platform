#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from urllib import error, request

from ase import Atoms
from ase.build import bulk
from ase.io import write
from download_mp_cif import DEFAULT_OUTPUT_ROOT as MP_DOWNLOAD_ROOT
from download_mp_cif import download_query
from parse_cif import compare_cif_payloads, parse_cif

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_URL = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
CHAT_COMPLETIONS_URL = f"{BASE_URL.rstrip('/')}/chat/completions"
MODEL_NAME = os.environ.get("YUNWU_MODEL", "gpt-5.4-mini")
API_KEY = os.environ.get("YUNWU_API_KEY", "")
MATERIALS_PROJECT_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY", "")
HOST = os.environ.get("CHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHAT_PORT", "8000"))
MATERIALS_DIR = PROJECT_ROOT / "ilovepdf_structured_json" / "materials"
LOGO_PATH = PROJECT_ROOT / "PNG" / "20260320130359.jpg"
SIM_SCRIPT_PATH = PROJECT_ROOT / "matesim_dft.py"
SIM_TASKS_DIR = PROJECT_ROOT / "sim_tasks"
SIM_MATERIALS_SUBDIR = "materials"
SIM_RUNTIME_SUBDIR = "runtime"
DEFAULT_SIMULATION_PARAMS = {
    "relax_steps": 200,
    "relax_fmax": 0.05,
    "optimizer": "FIRE",
    "filter": "ExpCellFilter",
    "constrain_symmetry": False,
    "device": "cpu",
    "max_candidates_per_formula": 4,
    "max_formula_queries": 3,
    "custom_params": {},
}
AGENT_STEPS = [
    "收集到指令",
    "读取文件",
    "检查文件",
    "预处理中",
    "找到答案",
    "输出答案",
]
SIMULATION_STEPS = [
    "接收材料设计指令",
    "检索合金成分",
    "生成 input.cif",
    "加载计算代理",
    "执行计算",
    "结果汇总",
]

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Materials Chat</title>
  <style>
    :root {
      --bg: #07111f;
      --bg2: #0e2037;
      --panel: rgba(9, 20, 38, 0.86);
      --panel-strong: rgba(11, 27, 50, 0.96);
      --ink: #ebf6ff;
      --muted: #8cb5d9;
      --accent: #4cc9ff;
      --accent-2: #1d7dff;
      --line: rgba(98, 178, 255, 0.24);
      --glow: 0 0 0 1px rgba(117, 203, 255, 0.1), 0 20px 60px rgba(5, 20, 45, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(53, 143, 255, 0.24), transparent 28%),
        radial-gradient(circle at top right, rgba(76, 201, 255, 0.18), transparent 30%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      color: var(--ink);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }
    .hero {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 20px;
      align-items: center;
      margin-bottom: 22px;
      padding: 22px 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background:
        linear-gradient(135deg, rgba(14, 33, 58, 0.96), rgba(9, 19, 38, 0.92)),
        radial-gradient(circle at top, rgba(76, 201, 255, 0.1), transparent 45%);
      box-shadow: var(--glow);
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: auto -30% -65% auto;
      width: 340px;
      height: 340px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(76, 201, 255, 0.18), transparent 68%);
      pointer-events: none;
    }
    .hero-logo {
      width: 120px;
      height: 84px;
      object-fit: contain;
      border-radius: 16px;
      padding: 8px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(112, 191, 255, 0.2);
      box-shadow: inset 0 0 40px rgba(87, 205, 255, 0.08);
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0.04em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .hero-kicker {
      margin-bottom: 10px;
      color: var(--accent);
      font-size: 13px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    .dashboard {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 20px;
      align-items: stretch;
    }
    .status-panel, .chat-panel {
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--glow);
      overflow: hidden;
      backdrop-filter: blur(16px);
    }
    .panel-head {
      padding: 18px 20px 16px;
      min-height: 88px;
      border-bottom: 1px solid rgba(117, 203, 255, 0.12);
      background: linear-gradient(180deg, rgba(19, 41, 71, 0.95), rgba(14, 29, 53, 0.86));
    }
    .panel-head h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0.06em;
    }
    .panel-head p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .agent-list {
      padding: 14px 14px 10px;
    }
    .settings-panel {
      margin: 4px 14px 16px;
      border: 1px solid rgba(108, 187, 255, 0.12);
      border-radius: 18px;
      background: rgba(8, 17, 32, 0.58);
      overflow: hidden;
    }
    .settings-toggle {
      width: 100%;
      min-width: 0;
      border-radius: 0;
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #dff4ff;
      background: linear-gradient(135deg, rgba(17, 46, 81, 0.95), rgba(9, 23, 41, 0.92));
      box-shadow: none;
    }
    .settings-toggle span:last-child {
      color: #9fd8ff;
      font-size: 12px;
      letter-spacing: 0.08em;
    }
    .settings-body {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field label {
      font-size: 12px;
      color: #9fd8ff;
    }
    .field input,
    .field select {
      width: 100%;
      border: 1px solid rgba(111, 189, 255, 0.18);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: rgba(255, 255, 255, 0.03);
      color: var(--ink);
    }
    .field textarea {
      min-height: 110px;
      border-radius: 12px;
    }
    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border: 1px solid rgba(111, 189, 255, 0.12);
      border-radius: 12px;
      background: rgba(255,255,255,0.02);
      color: #dff4ff;
      font-size: 13px;
    }
    .settings-actions {
      display: flex;
      gap: 10px;
    }
    .preset-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .preset-btn {
      min-width: 0;
      padding: 10px 12px;
      border-radius: 12px;
      color: #dff4ff;
      background: rgba(255, 255, 255, 0.06);
      box-shadow: none;
      border: 1px solid rgba(111, 189, 255, 0.18);
    }
    .settings-actions button.secondary {
      color: #dff4ff;
      background: rgba(255, 255, 255, 0.06);
      box-shadow: none;
      border: 1px solid rgba(111, 189, 255, 0.18);
    }
    .agent-step {
      display: grid;
      grid-template-columns: 34px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
      padding: 12px 12px;
      border-radius: 16px;
      border: 1px solid rgba(108, 187, 255, 0.12);
      background: rgba(8, 17, 32, 0.58);
      transition: 220ms ease;
    }
    .agent-step .idx {
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 50%;
      border: 1px solid rgba(108, 187, 255, 0.25);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      background: rgba(76, 201, 255, 0.05);
    }
    .agent-step .label {
      font-size: 14px;
      color: #dff3ff;
    }
    .agent-step.waiting .label { color: var(--muted); }
    .agent-step.active {
      border-color: rgba(76, 201, 255, 0.45);
      background: linear-gradient(135deg, rgba(11, 34, 61, 0.95), rgba(8, 22, 42, 0.92));
      box-shadow: 0 0 0 1px rgba(76, 201, 255, 0.12), 0 0 28px rgba(76, 201, 255, 0.12);
    }
    .agent-step.active .idx {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #03101d;
      border-color: transparent;
    }
    .agent-step.done {
      border-color: rgba(112, 255, 195, 0.22);
      background: rgba(8, 27, 36, 0.78);
    }
    .agent-step.done .idx {
      background: linear-gradient(135deg, #73ffd6, #42ddb7);
      color: #031918;
      border-color: transparent;
    }
    .panel {
      border: 0;
      border-radius: 0;
      background: transparent;
      overflow: hidden;
      box-shadow: none;
    }
    .panel {
      display: contents;
    }
    .messages {
      min-height: 460px;
      max-height: 62vh;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background:
        linear-gradient(rgba(5, 14, 28, 0.6), rgba(5, 14, 28, 0.6)),
        repeating-linear-gradient(
          0deg,
          transparent,
          transparent 30px,
          rgba(76, 201, 255, 0.045) 31px
        );
    }
    .msg {
      padding: 14px 16px;
      border-radius: 16px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
      width: min(100%, 820px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .msg.user {
      align-self: flex-end;
      background: linear-gradient(135deg, rgba(29, 125, 255, 0.25), rgba(76, 201, 255, 0.18));
      border: 1px solid rgba(96, 186, 255, 0.24);
    }
    .msg.bot {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(111, 189, 255, 0.16);
    }
    .table-card {
      align-self: stretch;
      width: 100%;
      padding: 16px 16px 12px;
      border-radius: 20px;
      border: 1px solid rgba(111, 189, 255, 0.18);
      background: rgba(7, 19, 35, 0.84);
    }
    .table-card h3 {
      margin: 0 0 12px;
      font-size: 15px;
      color: #dff3ff;
      letter-spacing: 0.03em;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.02);
      table-layout: fixed;
    }
    .data-table th,
    .data-table td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(111, 189, 255, 0.12);
      vertical-align: top;
      font-size: 13px;
    }
    .data-table th {
      color: #9fd8ff;
      background: rgba(76, 201, 255, 0.08);
      font-weight: 600;
    }
    .data-table th.section {
      background: linear-gradient(90deg, rgba(76, 201, 255, 0.16), rgba(29, 125, 255, 0.1));
      color: #dff5ff;
      font-size: 13px;
      letter-spacing: 0.04em;
    }
    .data-table td {
      color: #e6f6ff;
    }
    .data-table td:first-child {
      width: 32%;
      color: #9fd8ff;
    }
    .data-table tr:last-child td {
      border-bottom: 0;
    }
    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 16px;
      border-top: 1px solid rgba(117, 203, 255, 0.12);
      background: rgba(6, 16, 30, 0.88);
    }
    textarea {
      width: 100%;
      min-height: 88px;
      resize: vertical;
      border: 1px solid rgba(111, 189, 255, 0.18);
      border-radius: 14px;
      padding: 14px;
      font: inherit;
      background: rgba(255, 255, 255, 0.03);
      color: var(--ink);
    }
    textarea::placeholder { color: #719dc5; }
    button {
      border: 0;
      border-radius: 14px;
      padding: 0 20px;
      min-width: 120px;
      font: inherit;
      font-weight: 600;
      color: #03101d;
      background: linear-gradient(135deg, var(--accent), #7ae7ff);
      cursor: pointer;
      box-shadow: 0 0 28px rgba(76, 201, 255, 0.28);
    }
    .thinking {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: #bfefff;
    }
    .thinking-dots {
      display: inline-flex;
      gap: 6px;
    }
    .thinking-dots span {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px rgba(76, 201, 255, 0.35);
      animation: pulse 1.2s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes pulse {
      0%, 80%, 100% { transform: scale(0.7); opacity: 0.45; }
      40% { transform: scale(1); opacity: 1; }
    }
    .meta {
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      display: inline-block;
      margin: 6px 8px 0 0;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(76, 201, 255, 0.12);
      color: #bfefff;
      border: 1px solid rgba(76, 201, 255, 0.18);
    }
    .progress-card {
      align-self: stretch;
      width: 100%;
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(111, 189, 255, 0.18);
      background: rgba(7, 19, 35, 0.84);
    }
    .progress-bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      overflow: hidden;
      margin: 10px 0 14px;
    }
    .progress-bar > span {
      display: block;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #4cc9ff, #1d7dff);
      box-shadow: 0 0 18px rgba(76, 201, 255, 0.28);
      transition: width 240ms ease;
    }
    .result-image {
      display: block;
      width: 100%;
      margin-top: 14px;
      border-radius: 16px;
      border: 1px solid rgba(111, 189, 255, 0.18);
      background: rgba(255,255,255,0.02);
    }
    .result-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .asset-link {
      display: inline-flex;
      align-items: center;
      padding: 9px 14px;
      border-radius: 999px;
      border: 1px solid rgba(111, 189, 255, 0.28);
      background: rgba(16, 42, 76, 0.72);
      color: #dff4ff;
      text-decoration: none;
    }
    .hero {
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 26px;
      padding: 26px 28px;
      border-radius: 30px;
      background:
        linear-gradient(135deg, rgba(8, 21, 41, 0.98), rgba(6, 17, 31, 0.94)),
        radial-gradient(circle at 20% 10%, rgba(60, 183, 255, 0.18), transparent 30%),
        radial-gradient(circle at 85% 0%, rgba(43, 113, 255, 0.16), transparent 34%);
      box-shadow:
        0 0 0 1px rgba(122, 205, 255, 0.08),
        0 32px 80px rgba(4, 14, 30, 0.52),
        inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .hero-logo {
      width: 118px;
      height: 118px;
      border-radius: 24px;
      padding: 14px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)),
        rgba(5, 17, 33, 0.72);
    }
    .hero-kicker {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid rgba(112, 198, 255, 0.18);
      background: rgba(255,255,255,0.03);
      letter-spacing: 0.14em;
    }
    .hero h1 {
      font-size: 38px;
      letter-spacing: 0.02em;
      margin-bottom: 10px;
    }
    .hero p {
      max-width: 780px;
      font-size: 15px;
      color: #8db7db;
    }
    .hero-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }
    .stat-card {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(111, 189, 255, 0.14);
      background: linear-gradient(180deg, rgba(14, 31, 55, 0.74), rgba(8, 19, 35, 0.66));
    }
    .stat-card .label {
      color: #82b8db;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .stat-card .value {
      font-size: 17px;
      font-weight: 700;
      color: #eff9ff;
    }
    .dashboard {
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }
    .status-panel, .chat-panel {
      border-radius: 28px;
      background:
        linear-gradient(180deg, rgba(8, 21, 40, 0.92), rgba(6, 16, 29, 0.94)),
        rgba(7, 19, 35, 0.86);
      box-shadow:
        0 0 0 1px rgba(117, 203, 255, 0.08),
        0 30px 70px rgba(4, 14, 29, 0.4);
    }
    .status-panel {
      position: sticky;
      top: 22px;
    }
    .hero {
      grid-template-columns: 132px minmax(0, 1fr);
      gap: 26px;
      padding: 28px 30px;
      border-radius: 30px;
      background:
        linear-gradient(135deg, rgba(9, 23, 41, 0.98), rgba(4, 13, 25, 0.9)),
        radial-gradient(circle at top right, rgba(76, 201, 255, 0.18), transparent 32%);
      box-shadow:
        0 0 0 1px rgba(117, 203, 255, 0.08),
        0 36px 90px rgba(2, 10, 22, 0.48);
    }
    .hero::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent, rgba(120, 214, 255, 0.06), transparent);
      transform: translateX(-100%);
      animation: heroScan 12s linear infinite;
      pointer-events: none;
    }
    .hero h1 {
      font-size: 34px;
      letter-spacing: 0.02em;
    }
    .hero p {
      max-width: 900px;
      color: #a8cbeb;
      font-size: 15px;
      line-height: 1.75;
    }
    .hero-kicker {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(76, 201, 255, 0.08);
      border: 1px solid rgba(122, 222, 255, 0.16);
    }
    .hero-stats {
      margin-top: 18px;
      gap: 14px;
    }
    .stat-card {
      min-height: 92px;
      justify-content: space-between;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(17, 41, 71, 0.68), rgba(9, 21, 36, 0.82));
      border: 1px solid rgba(116, 202, 255, 0.14);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .stat-card .value {
      font-size: 18px;
      line-height: 1.3;
    }
    .dashboard {
      grid-template-columns: 370px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }
    .status-panel, .chat-panel {
      border-radius: 30px;
      background: linear-gradient(180deg, rgba(8, 20, 38, 0.96), rgba(6, 15, 28, 0.92));
      box-shadow:
        0 0 0 1px rgba(117, 203, 255, 0.09),
        0 28px 80px rgba(2, 10, 22, 0.42);
    }
    .panel-head {
      min-height: auto;
      padding: 24px 24px 18px;
      background:
        linear-gradient(180deg, rgba(16, 39, 69, 0.92), rgba(10, 25, 45, 0.84)),
        radial-gradient(circle at top right, rgba(76, 201, 255, 0.08), transparent 30%);
    }
    .panel-head h2 {
      font-size: 19px;
      letter-spacing: 0.08em;
    }
    .panel-head p {
      color: #90b9dd;
    }
    .agent-list {
      padding: 18px 18px 8px;
    }
    .agent-step {
      margin-bottom: 12px;
      padding: 14px 14px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(9, 23, 42, 0.88), rgba(7, 19, 35, 0.7));
    }
    .chain-panel {
      margin: 6px 18px 14px;
      padding: 16px;
      border-radius: 22px;
      border: 1px solid rgba(116, 202, 255, 0.14);
      background: linear-gradient(180deg, rgba(12, 28, 49, 0.84), rgba(7, 18, 33, 0.78));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .chain-head h3 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0.08em;
      color: #dff4ff;
    }
    .chain-head p {
      margin: 8px 0 0;
      font-size: 12px;
      line-height: 1.6;
      color: #7fb0d8;
    }
    .chain-switcher {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .chain-btn {
      min-width: 0;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(116, 202, 255, 0.12);
      background: linear-gradient(180deg, rgba(10, 22, 38, 0.92), rgba(7, 18, 31, 0.84));
      color: #dff4ff;
      box-shadow: none;
      transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }
    .chain-btn.active {
      color: #04111d;
      border-color: rgba(142, 231, 255, 0.75);
      background: linear-gradient(135deg, rgba(112, 220, 255, 1), rgba(47, 124, 255, 0.96));
      box-shadow: 0 12px 28px rgba(39, 131, 255, 0.28);
    }
    .chain-btn:hover {
      transform: translateY(-1px);
    }
    .settings-panel {
      margin: 8px 18px 18px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(10, 24, 44, 0.82), rgba(7, 18, 33, 0.76));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .messages {
      min-height: 620px;
      max-height: 68vh;
      padding: 28px;
      background:
        radial-gradient(circle at top center, rgba(57, 149, 255, 0.1), transparent 28%),
        linear-gradient(rgba(4, 12, 24, 0.68), rgba(4, 12, 24, 0.82)),
        repeating-linear-gradient(
          0deg,
          transparent,
          transparent 26px,
          rgba(76, 201, 255, 0.038) 27px
        ),
        repeating-linear-gradient(
          90deg,
          transparent,
          transparent 26px,
          rgba(76, 201, 255, 0.026) 27px
        );
    }
    .msg {
      border-radius: 22px;
      padding: 18px 20px;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03),
        0 14px 34px rgba(3, 10, 20, 0.2);
    }
    .msg.user {
      border-top-right-radius: 8px;
      background: linear-gradient(135deg, rgba(30, 112, 229, 0.4), rgba(58, 192, 255, 0.2));
    }
    .msg.bot {
      border-top-left-radius: 8px;
      background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
    }
    .table-card, .progress-card {
      border-radius: 24px;
      padding: 18px 18px 14px;
      background: linear-gradient(180deg, rgba(10, 24, 44, 0.88), rgba(7, 18, 32, 0.82));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .table-card h3, .progress-card h3 {
      font-size: 16px;
      letter-spacing: 0.03em;
    }
    .data-table {
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
    }
    .data-table th {
      background: linear-gradient(180deg, rgba(76, 201, 255, 0.15), rgba(29, 125, 255, 0.08));
    }
    .composer {
      padding: 18px;
      gap: 14px;
      background:
        linear-gradient(180deg, rgba(7, 17, 32, 0.96), rgba(6, 14, 27, 0.98));
    }
    textarea {
      min-height: 104px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.025));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    button {
      border-radius: 16px;
      min-width: 132px;
      background: linear-gradient(135deg, #53d2ff, #2a7dff);
      box-shadow:
        0 18px 34px rgba(42, 125, 255, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.22);
    }
    .result-image {
      margin-top: 14px;
      border-radius: 18px;
      border: 1px solid rgba(117, 203, 255, 0.1);
      box-shadow: 0 14px 36px rgba(4, 10, 22, 0.28);
    }
    @keyframes heroScan {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(140%); }
    }
    @media (max-width: 900px) {
      .hero { grid-template-columns: 1fr; }
      .dashboard { grid-template-columns: 1fr; }
      .hero-logo { width: 120px; height: 82px; }
      .msg { width: 100%; }
      .settings-grid { grid-template-columns: 1fr; }
      .hero-stats { grid-template-columns: 1fr; }
      .chain-switcher { grid-template-columns: 1fr; }
      .status-panel { position: static; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <img class="hero-logo" src="/logo" alt="Logo" />
      <div>
        <div class="hero-kicker">Autonomous Research Mesh</div>
        <h1>Agent 材料研究工作台</h1>
        <p>围绕开放讨论、知识检索、候选筛选与多 Agent 协同执行构建的研究工作台，强调过程透明、状态可视与结果对话。</p>
        <div class="hero-stats">
          <div class="stat-card">
            <span class="label">执行模式</span>
            <strong class="value" id="heroMode">Open Discussion</strong>
          </div>
          <div class="stat-card">
            <span class="label">当前阶段</span>
            <strong class="value" id="heroPhase">等待指令</strong>
          </div>
          <div class="stat-card">
            <span class="label">数据来源</span>
            <strong class="value" id="heroSource">本地材料库 / MP</strong>
          </div>
        </div>
      </div>
    </div>
    <div class="dashboard">
      <aside class="status-panel">
        <div class="panel-head">
          <h2>Agent 作战链路</h2>
          <p id="agentHint">系统待命，等待新的研究指令。</p>
        </div>
        <div class="hero-stats">
          <div class="stat-card">
            <span class="label">任务状态</span>
            <strong class="value" id="liveStatus">Idle</strong>
          </div>
          <div class="stat-card">
            <span class="label">工作流</span>
            <strong class="value" id="liveWorkflow">问答分析</strong>
          </div>
          <div class="stat-card">
            <span class="label">结果输出</span>
            <strong class="value" id="liveOutput">等待生成</strong>
          </div>
        </div>
        <div id="agentSteps" class="agent-list"></div>
        <div class="chain-panel">
          <div class="chain-head">
            <h3>思维链选择</h3>
            <p id="chainHint">王工链路更偏执行效率与工程判断。</p>
          </div>
          <div class="chain-switcher">
            <button class="chain-btn active" type="button" data-chain="王工">王工</button>
            <button class="chain-btn" type="button" data-chain="李工">李工</button>
          </div>
        </div>
        <div class="settings-panel">
          <button id="settingsToggle" class="settings-toggle" type="button">
            <span>执行参数设置</span>
            <span id="settingsToggleLabel">展开</span>
          </button>
          <div id="settingsBody" class="settings-body" style="display:none;">
            <div class="field">
              <label>参数预设</label>
              <div class="preset-row">
                <button class="preset-btn" id="presetFastBtn" type="button">快速预筛</button>
                <button class="preset-btn" id="presetStrictBtn" type="button">严格收敛</button>
                <button class="preset-btn" id="presetDeepBtn" type="button">深度迭代</button>
              </div>
            </div>
            <div class="settings-grid">
              <div class="field">
                <label for="relaxSteps">迭代步数</label>
                <input id="relaxSteps" type="number" min="1" step="1" value="200" />
              </div>
              <div class="field">
                <label for="relaxFmax">受力阈值 fmax</label>
                <input id="relaxFmax" type="number" min="0.0001" step="0.001" value="0.05" />
              </div>
              <div class="field">
                <label for="optimizer">优化器</label>
                <select id="optimizer">
                  <option value="FIRE">FIRE</option>
                  <option value="BFGS">BFGS</option>
                </select>
              </div>
              <div class="field">
                <label for="filter">晶胞过滤器</label>
                <select id="filter">
                  <option value="ExpCellFilter">ExpCellFilter</option>
                  <option value="FrechetCellFilter">FrechetCellFilter</option>
                  <option value="">无</option>
                </select>
              </div>
              <div class="field">
                <label for="device">设备</label>
                <select id="device">
                  <option value="cpu">cpu</option>
                  <option value="cuda">cuda</option>
                </select>
              </div>
              <div class="field">
                <label for="maxCandidatesPerFormula">每个体系候选数</label>
                <input id="maxCandidatesPerFormula" type="number" min="1" step="1" value="4" />
              </div>
              <div class="field">
                <label for="maxFormulaQueries">最多检索体系数</label>
                <input id="maxFormulaQueries" type="number" min="1" step="1" value="3" />
              </div>
            </div>
            <div class="switch-row">
              <span>保持对称性约束</span>
              <input id="constrainSymmetry" type="checkbox" />
            </div>
            <div class="field">
              <label for="customParams">高级参数 JSON</label>
              <textarea id="customParams" placeholder='例如 {"model_path":"/path/model.pth","extra_note":"test"}'></textarea>
            </div>
            <div class="settings-actions">
              <button id="saveSettingsBtn" type="button">应用参数</button>
              <button id="resetSettingsBtn" class="secondary" type="button">恢复默认</button>
            </div>
          </div>
        </div>
      </aside>
      <section class="chat-panel">
        <div class="panel-head">
          <h2>对话中心</h2>
          <p></p>
        </div>
        <div class="panel">
          <div id="messages" class="messages"></div>
          <form id="form" class="composer">
            <textarea id="question" placeholder="请输入材料牌号、开放讨论问题或计算请求，比如：我想找更有潜力的超导候选，需要具备哪些特性？"></textarea>
            <button type="submit">启动 Agent</button>
          </form>
        </div>
      </section>
    </div>
    <div class="meta" id="meta"></div>
  </div>

  <script>
    const DEFAULT_SIMULATION_PARAMS = {
      relax_steps: 200,
      relax_fmax: 0.05,
      optimizer: 'FIRE',
      filter: 'ExpCellFilter',
      constrain_symmetry: false,
      device: 'cpu',
      max_candidates_per_formula: 4,
      max_formula_queries: 3,
      custom_params: {}
    };
    const AGENT_STEPS = [
      '收集到指令',
      '调用记忆模块',
      '进行分级检索',
      '检索到相关文件',
      '进行文件分析',
    ];
    const SIMULATION_STEPS = [
      '接收材料设计指令',
      '检索合金成分',
      '生成 input.cif',
      '加载Agent模型',
      '执行运算',
    ];
    const STEP_DELAY = 1000;
    const form = document.getElementById('form');
    const question = document.getElementById('question');
    const messages = document.getElementById('messages');
    const meta = document.getElementById('meta');
    const agentSteps = document.getElementById('agentSteps');
    const agentHint = document.getElementById('agentHint');
    const heroMode = document.getElementById('heroMode');
    const heroPhase = document.getElementById('heroPhase');
    const heroSource = document.getElementById('heroSource');
    const liveStatus = document.getElementById('liveStatus');
    const liveWorkflow = document.getElementById('liveWorkflow');
    const liveOutput = document.getElementById('liveOutput');
    const chainHint = document.getElementById('chainHint');
    const chainButtons = Array.from(document.querySelectorAll('.chain-btn'));
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsToggleLabel = document.getElementById('settingsToggleLabel');
    const settingsBody = document.getElementById('settingsBody');
    const relaxStepsInput = document.getElementById('relaxSteps');
    const relaxFmaxInput = document.getElementById('relaxFmax');
    const optimizerInput = document.getElementById('optimizer');
    const filterInput = document.getElementById('filter');
    const deviceInput = document.getElementById('device');
    const maxCandidatesPerFormulaInput = document.getElementById('maxCandidatesPerFormula');
    const maxFormulaQueriesInput = document.getElementById('maxFormulaQueries');
    const constrainSymmetryInput = document.getElementById('constrainSymmetry');
    const customParamsInput = document.getElementById('customParams');
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    const resetSettingsBtn = document.getElementById('resetSettingsBtn');
    const presetFastBtn = document.getElementById('presetFastBtn');
    const presetStrictBtn = document.getElementById('presetStrictBtn');
    const presetDeepBtn = document.getElementById('presetDeepBtn');
    const conversationState = [];
    let currentStepLabels = [...AGENT_STEPS];
    let simulationParams = { ...DEFAULT_SIMULATION_PARAMS };
    let activeThinkingChain = '王工';
    const THINKING_CHAINS = {
      '王工': {
        hint: '王工链路更偏执行效率、收敛判断与工程可落地性。',
        workflow: '工程研判'
      },
      '李工': {
        hint: '李工链路更偏候选比较、机理分析与研究解释性。',
        workflow: '研究推演'
      }
    };
    const SIMULATION_PRESETS = {
      fast: {
        relax_steps: 80,
        relax_fmax: 0.12,
        optimizer: 'FIRE',
        filter: '',
        constrain_symmetry: false,
        device: 'cpu',
        max_candidates_per_formula: 3,
        max_formula_queries: 3,
        custom_params: {}
      },
      strict: {
        relax_steps: 400,
        relax_fmax: 0.02,
        optimizer: 'BFGS',
        filter: 'ExpCellFilter',
        constrain_symmetry: true,
        device: 'cpu',
        max_candidates_per_formula: 4,
        max_formula_queries: 3,
        custom_params: {}
      },
      deep: {
        relax_steps: 800,
        relax_fmax: 0.01,
        optimizer: 'FIRE',
        filter: 'ExpCellFilter',
        constrain_symmetry: false,
        device: 'cpu',
        max_candidates_per_formula: 6,
        max_formula_queries: 5,
        custom_params: {}
      }
    };

    function setSettingsVisibility(open) {
      settingsBody.style.display = open ? 'grid' : 'none';
      settingsToggleLabel.textContent = open ? '收起' : '展开';
    }

    function fillSettingsForm(params) {
      relaxStepsInput.value = params.relax_steps;
      relaxFmaxInput.value = params.relax_fmax;
      optimizerInput.value = params.optimizer;
      filterInput.value = params.filter;
      deviceInput.value = params.device;
      maxCandidatesPerFormulaInput.value = params.max_candidates_per_formula;
      maxFormulaQueriesInput.value = params.max_formula_queries;
      constrainSymmetryInput.checked = Boolean(params.constrain_symmetry);
      customParamsInput.value = Object.keys(params.custom_params || {}).length
        ? JSON.stringify(params.custom_params, null, 2)
        : '';
    }

    function readSettingsForm() {
      let custom = {};
      const raw = customParamsInput.value.trim();
      if (raw) {
        custom = JSON.parse(raw);
      }
      return {
        relax_steps: Number(relaxStepsInput.value || DEFAULT_SIMULATION_PARAMS.relax_steps),
        relax_fmax: Number(relaxFmaxInput.value || DEFAULT_SIMULATION_PARAMS.relax_fmax),
        optimizer: optimizerInput.value || DEFAULT_SIMULATION_PARAMS.optimizer,
        filter: filterInput.value,
        device: deviceInput.value || DEFAULT_SIMULATION_PARAMS.device,
        max_candidates_per_formula: Number(maxCandidatesPerFormulaInput.value || DEFAULT_SIMULATION_PARAMS.max_candidates_per_formula),
        max_formula_queries: Number(maxFormulaQueriesInput.value || DEFAULT_SIMULATION_PARAMS.max_formula_queries),
        constrain_symmetry: Boolean(constrainSymmetryInput.checked),
        custom_params: custom,
      };
    }

    function applySettingsFromForm() {
      simulationParams = readSettingsForm();
      window.localStorage.setItem('mattersim_settings', JSON.stringify(simulationParams));
      addMessage('bot', '执行参数已更新，后续任务将使用新的设置。');
    }

    function applyPreset(name) {
      const preset = SIMULATION_PRESETS[name];
      if (!preset) return;
      simulationParams = { ...DEFAULT_SIMULATION_PARAMS, ...preset };
      fillSettingsForm(simulationParams);
      window.localStorage.setItem('mattersim_settings', JSON.stringify(simulationParams));
      addMessage('bot', `已应用参数预设：${name === 'fast' ? '快速预筛' : name === 'strict' ? '严格收敛' : '深度迭代'}`);
    }

    function loadStoredSettings() {
      const raw = window.localStorage.getItem('mattersim_settings');
      if (!raw) {
        fillSettingsForm(DEFAULT_SIMULATION_PARAMS);
        return;
      }
      try {
        simulationParams = { ...DEFAULT_SIMULATION_PARAMS, ...JSON.parse(raw) };
      } catch {
        simulationParams = { ...DEFAULT_SIMULATION_PARAMS };
      }
      fillSettingsForm(simulationParams);
    }

    function renderSteps(activeIndex = -1, doneCount = 0, labels = currentStepLabels) {
      agentSteps.innerHTML = '';
      currentStepLabels = labels;
      labels.forEach((label, index) => {
        const node = document.createElement('div');
        const state = index < doneCount ? 'done' : index === activeIndex ? 'active' : 'waiting';
        node.className = `agent-step ${state}`;
        node.innerHTML = `
          <div class="idx">${index + 1}</div>
          <div class="label">${label}</div>
        `;
        agentSteps.appendChild(node);
      });
    }

    function updateWorkbenchState({
      mode = heroMode.textContent,
      phase = heroPhase.textContent,
      source = heroSource.textContent,
      status = liveStatus.textContent,
      workflow = liveWorkflow.textContent,
      output = liveOutput.textContent
    } = {}) {
      heroMode.textContent = mode;
      heroPhase.textContent = phase;
      heroSource.textContent = source;
      liveStatus.textContent = status;
      liveWorkflow.textContent = workflow;
      liveOutput.textContent = output;
    }

    function setThinkingChain(name) {
      const next = THINKING_CHAINS[name] ? name : '王工';
      activeThinkingChain = next;
      chainButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.chain === next);
      });
      chainHint.textContent = THINKING_CHAINS[next].hint;
      liveWorkflow.textContent = THINKING_CHAINS[next].workflow;
    }

    function addMessage(role, text) {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      node.textContent = text;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
      return node;
    }

    function addHtmlBlock(html, className = 'table-card') {
      const node = document.createElement('div');
      node.className = className;
      node.innerHTML = html;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
      return node;
    }

    function addThinkingMessage() {
      const node = document.createElement('div');
      node.className = 'msg bot';
      node.innerHTML = `
        <div class="thinking">
          <span>正在思考中</span>
          <span class="thinking-dots"><span></span><span></span><span></span></span>
        </div>
      `;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
      return node;
    }

    function addProgressCard(title) {
      return addHtmlBlock(`
        <h3>${escapeHtml(title)}</h3>
        <div class="progress-bar"><span id="taskProgressBar"></span></div>
        <div id="taskProgressNote" style="color:#9fd8ff;font-size:13px;">等待任务启动...</div>
      `, 'progress-card');
    }

    function setMeta(payload) {
      meta.innerHTML = '';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function objectRows(data) {
      return Object.entries(data || {}).map(([key, value]) => `
        <tr>
          <td>${escapeHtml(key)}</td>
          <td>${escapeHtml(value)}</td>
        </tr>
      `).join('');
    }

    function buildSectionRow(title) {
      return `
        <tr><th class="section" colspan="2">${escapeHtml(title)}</th></tr>
      `;
    }

    function renderMatchedTables(payload) {
      const matchedPayloads = payload.matched_payloads || [];
      if (!matchedPayloads.length) return;

      if (matchedPayloads.length > 1) {
        const filenames = matchedPayloads.map(item => item.filename || '未命名文件');
        const fieldMap = new Map();
        matchedPayloads.forEach((item) => {
          if (!item.data) return;
          const composition = item.data['化学成分（质量分数）'] || {};
          const merged = {
            '合金牌号': item.data['合金牌号'] || '',
            '名义化学成分': item.data['名义化学成分'] || '',
            ...composition['主要成分'],
            ...composition['杂质，不大于']
          };
          Object.entries(merged).forEach(([key, value]) => {
            if (!fieldMap.has(key)) fieldMap.set(key, []);
            fieldMap.get(key).push(value || '');
          });
        });

        const headerHtml = filenames.map(name => `<th>${escapeHtml(name)}</th>`).join('');
        const bodyHtml = Array.from(fieldMap.entries()).map(([field, values]) => `
          <tr>
            <td>${escapeHtml(field)}</td>
            ${values.map(value => `<td>${escapeHtml(value)}</td>`).join('')}
          </tr>
        `).join('');

        addHtmlBlock(`
          <h3>横向对比结果</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>字段</th>
                ${headerHtml}
              </tr>
            </thead>
            <tbody>${bodyHtml || '<tr><td colspan="99">暂无数据</td></tr>'}</tbody>
          </table>
        `);
        return;
      }

      matchedPayloads.forEach((item) => {
        if (!item.data) {
          addHtmlBlock(`
            <h3>${escapeHtml(item.filename || '未命名文件')}</h3>
            <table class="data-table">
              <tbody>
                <tr><td>错误</td><td>${escapeHtml(item.error || '读取失败')}</td></tr>
              </tbody>
            </table>
          `);
          return;
        }

        const info = {
          '文件名': item.filename || '',
          '合金牌号': item.data['合金牌号'] || '',
          '名义化学成分': item.data['名义化学成分'] || ''
        };
        const composition = item.data['化学成分（质量分数）'] || {};
        const major = composition['主要成分'] || {};
        const impurity = composition['杂质，不大于'] || {};
        const tableRows = [
          buildSectionRow('基础信息'),
          objectRows(info),
          buildSectionRow('主要成分'),
          objectRows(major),
          buildSectionRow('杂质，不大于'),
          objectRows(impurity),
        ].join('');

        addHtmlBlock(`
          <h3>${escapeHtml(item.filename || '材料数据')}</h3>
          <table class="data-table">
            <tbody>${tableRows}</tbody>
          </table>
        `);
      });
    }

    function renderCifComparison(payload) {
      const comparison = payload.cif_comparison || null;
      if (!comparison || !Array.isArray(comparison.rows) || !comparison.rows.length) return;

      const rows = comparison.rows.map((row) => `
        <tr>
          <td>${escapeHtml(row['指标'])}</td>
          <td>${escapeHtml(row['初始结构'])}</td>
          <td>${escapeHtml(row['弛豫后结构'])}</td>
          <td>${escapeHtml(row['变化'])}</td>
        </tr>
      `).join('');

      addHtmlBlock(`
        <h3>CIF 横向对比</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>指标</th>
              <th>初始结构</th>
              <th>弛豫后结构</th>
              <th>变化</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `);
    }

    function renderCifPayloads(payload) {
      const cifPayloads = payload.cif_payloads || [];
      if (!cifPayloads.length) return;

      cifPayloads.forEach((item) => {
        if (!item.data) {
          addHtmlBlock(`
            <h3>${escapeHtml(item.filename || 'CIF 文件')}</h3>
            <table class="data-table">
              <tbody>
                <tr><td>路径</td><td>${escapeHtml(item.path || '')}</td></tr>
                <tr><td>错误</td><td>${escapeHtml(item.error || '解析失败')}</td></tr>
              </tbody>
            </table>
          `);
          return;
        }

        const data = item.data || {};
        const baseInfo = {
          '文件名': item.filename || '',
          '路径': item.path || '',
          '化学式': data.formula || '',
          '约化化学式': data.reduced_formula || '',
          '原子数': data.atom_count || '',
          '空间群': Array.isArray(data.space_group) ? data.space_group.map((value) => String(value)).join(' / ') : (data.space_group || ''),
        };
        const lattice = data.lattice || data.cell || {};
        const latticeInfo = {
          'a': lattice.a || '',
          'b': lattice.b || '',
          'c': lattice.c || '',
          'alpha': lattice.alpha || '',
          'beta': lattice.beta || '',
          'gamma': lattice.gamma || '',
          'volume': lattice.volume || '',
        };
        const siteRows = (data.sites || []).slice(0, 12).map((site) => `
          <tr>
            <td>${escapeHtml(site.index)}</td>
            <td>${escapeHtml(site.element)}</td>
            <td>${escapeHtml((site.fractional_position || []).join(', '))}</td>
            <td>${escapeHtml((site.cartesian_position || []).join(', '))}</td>
          </tr>
        `).join('');

        addHtmlBlock(`
          <h3>${escapeHtml(item.filename || 'CIF 文件')}</h3>
          <table class="data-table">
            <tbody>
              ${buildSectionRow('基础信息')}
              ${objectRows(baseInfo)}
              ${buildSectionRow('晶格参数')}
              ${objectRows(latticeInfo)}
            </tbody>
          </table>
          <h3>原子位点（最多展示前 12 个）</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>序号</th>
                <th>元素</th>
                <th>分数坐标</th>
                <th>笛卡尔坐标</th>
              </tr>
            </thead>
            <tbody>${siteRows || '<tr><td colspan="4">暂无位点信息</td></tr>'}</tbody>
          </table>
        `);
      });
    }

    function renderDownloadPayloads(payload) {
      const downloadResults = payload.download_results || [];
      if (!downloadResults.length) return;

      downloadResults.forEach((item) => {
        const rows = (item.results || []).map((row) => `
          <tr>
            <td>${escapeHtml(row.rank)}</td>
            <td>${escapeHtml(row.formula_pretty || '')}</td>
            <td>${escapeHtml(row.material_id || '')}</td>
            <td>${escapeHtml(row.energy_above_hull ?? '')}</td>
            <td>${escapeHtml(row.energy_per_atom ?? '')}</td>
          </tr>
        `).join('');

        addHtmlBlock(`
          <h3>CIF 下载结果: ${escapeHtml(item.query || '')}</h3>
          <table class="data-table">
            <tbody>
              <tr><td>查询类型</td><td>${escapeHtml(item.query_type || '')}</td></tr>
              <tr><td>下载数量</td><td>${escapeHtml(item.count ?? '')}</td></tr>
              <tr><td>输出目录</td><td>${escapeHtml(item.output_dir || '')}</td></tr>
            </tbody>
          </table>
          <table class="data-table">
            <thead>
              <tr>
                <th>排序</th>
                <th>化学式</th>
                <th>材料ID</th>
                <th>能量高出凸包</th>
                <th>每原子能量</th>
              </tr>
            </thead>
            <tbody>${rows || '<tr><td colspan="5">没有返回可下载结果</td></tr>'}</tbody>
          </table>
        `);
      });
    }

    function renderSimulationResult(payload) {
      const result = payload.result || {};
      const summaryText = payload.summary || result.summary || '';
      const queryStats = result.query_stats || result.metadata?.query_stats || [];
      const queryStatsHtml = queryStats.length ? `
        <h3>体系检索概览</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>体系</th>
              <th>返回候选数</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>${queryStats.map((item) => `
            <tr>
              <td>${escapeHtml(item.query || '')}</td>
              <td>${escapeHtml(item.returned_candidates ?? '')}</td>
              <td>${escapeHtml(item.status === 'ok' ? '已返回候选' : '未返回候选')}</td>
            </tr>
          `).join('')}</tbody>
        </table>
      ` : '';
      if ((result.candidates || []).length) {
        const detailBlocks = result.candidates.map((item) => {
          const detailRows = (item.details || []).map((row) => `
            <tr>
              <td>${escapeHtml(row['字段'])}</td>
              <td>${escapeHtml(row['数值'])}</td>
            </tr>
          `).join('');
          const relaxedLink = item.relaxed_structure
            ? `<a class="asset-link" href="/task-asset?task_id=${encodeURIComponent(payload.result?.run_id ? payload.result.run_id : '')}&name=${encodeURIComponent(item.relaxed_structure)}" target="_blank" rel="noopener">查看 ${escapeHtml(item.relaxed_structure)}</a>`
            : '';
          return `
            <div class="table-card">
              <h3>单个候选计算指标: ${escapeHtml(item.name)}</h3>
              <table class="data-table">
                <tbody>${detailRows}</tbody>
              </table>
            </div>
          `;
        }).join('');

        const candidateRows = result.candidates.map((item) => `
          <tr>
            <td>${escapeHtml(item.rank)}</td>
            <td>${escapeHtml(item.name)}</td>
            <td>${escapeHtml(item.display_formula || item.formula)}</td>
            <td>${escapeHtml(item.source)}</td>
            <td>${escapeHtml(item.energy_per_atom)}</td>
            <td>${escapeHtml(item.max_force)}</td>
            <td>${escapeHtml(item.priority_score)}</td>
            <td>${escapeHtml(item.recommendation)}</td>
          </tr>
        `).join('');
        addHtmlBlock(`
          ${summaryText ? `<h3>任务摘要</h3><table class="data-table"><tbody><tr><td>摘要</td><td>${escapeHtml(summaryText)}</td></tr></tbody></table>` : ''}
          ${queryStatsHtml}
          ${detailBlocks}
          <h3>候选快排结果</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>排序</th>
                <th>候选</th>
                <th>化学式</th>
                <th>来源</th>
                <th>每原子能量</th>
                <th>最大受力</th>
                <th>优先级分数</th>
                <th>建议</th>
              </tr>
            </thead>
            <tbody>${candidateRows}</tbody>
          </table>
          <h3>汇总分析</h3>
          <table class="data-table">
            <tbody>${(result.table || []).map((row) => `
              <tr>
                <td>${escapeHtml(row['字段'])}</td>
                <td>${escapeHtml(row['数值'])}</td>
              </tr>
            `).join('')}</tbody>
          </table>
          ${payload.image_url ? `<img class="result-image" src="${payload.image_url}" alt="simulation result" />` : ''}
        `);
        return;
      }

      const rows = (result.table || []).map((row) => `
        <tr>
          <td>${escapeHtml(row['字段'])}</td>
          <td>${escapeHtml(row['数值'])}</td>
        </tr>
      `).join('');
      const links = [
        payload.input_url ? `<a class="asset-link" href="${payload.input_url}" target="_blank" rel="noopener">查看 input.cif</a>` : '',
        payload.relaxed_url ? `<a class="asset-link" href="${payload.relaxed_url}" target="_blank" rel="noopener">查看 relaxed.cif</a>` : ''
      ].filter(Boolean).join('');
      addHtmlBlock(`
        <h3>材料合成结果</h3>
        ${summaryText ? `<table class="data-table"><tbody><tr><td>摘要</td><td>${escapeHtml(summaryText)}</td></tr></tbody></table>` : ''}
        ${queryStatsHtml}
        <table class="data-table">
          <tbody>${rows || '<tr><td>状态</td><td>暂无结果</td></tr>'}</tbody>
        </table>
        ${links ? `<div class="result-actions">${links}</div>` : ''}
        ${payload.image_url ? `<img class="result-image" src="${payload.image_url}" alt="simulation result" />` : ''}
      `);
    }

    function sleep(ms) {
      return new Promise(resolve => window.setTimeout(resolve, ms));
    }

    async function playPipeline() {
      for (let index = 0; index < currentStepLabels.length; index += 1) {
        renderSteps(index, index, currentStepLabels);
        agentHint.textContent = `当前阶段：${currentStepLabels[index]}`;
        updateWorkbenchState({
          phase: currentStepLabels[index],
          status: 'Running',
          workflow: currentStepLabels === SIMULATION_STEPS ? '多Agent计算链路' : THINKING_CHAINS[activeThinkingChain].workflow,
          output: '生成中'
        });
        await sleep(STEP_DELAY);
      }
      renderSteps(-1, currentStepLabels.length, currentStepLabels);
      agentHint.textContent = '处理完成，已输出最终答案。';
      updateWorkbenchState({
        phase: '输出完成',
        status: 'Completed',
        output: '结果已写入控制台'
      });
    }

    async function pollSimulationTask(taskId, progressCard) {
      const progressBar = progressCard.querySelector('#taskProgressBar');
      const progressNote = progressCard.querySelector('#taskProgressNote');

      while (true) {
        const res = await fetch(`/task-status?id=${encodeURIComponent(taskId)}`);
        const payload = await res.json();
        const steps = payload.steps || SIMULATION_STEPS;
        const doneCount = (payload.steps || []).filter(step => step.status === 'done').length;
        const activeIndex = (payload.steps || []).findIndex(step => step.status === 'running');
        renderSteps(activeIndex, doneCount, steps.map(step => step.label));
        agentHint.textContent = payload.note || '任务执行中';
        updateWorkbenchState({
          mode: 'Deep Compute',
          phase: payload.note || '任务执行中',
          source: 'Materials Project / 本地模板',
          status: payload.status === 'completed' ? 'Completed' : payload.status === 'failed' ? 'Failed' : 'Running',
          workflow: '多Agent计算链路',
          output: payload.status === 'completed' ? '返回指标与图像' : '等待计算完成'
        });
        const progress = payload.progress_percent || 0;
        progressBar.style.width = `${progress}%`;
        progressNote.textContent = payload.note || '执行中';

        if (payload.status === 'completed') {
          progressBar.style.width = '100%';
          progressNote.textContent = '模拟完成，正在加载结果';
          progressCard.remove();
          addMessage('bot', payload.summary || '材料合成任务已完成。');
          renderCifComparison(payload);
          renderCifPayloads(payload);
          renderSimulationResult(payload);
          return payload;
        }

        if (payload.status === 'failed') {
          progressNote.textContent = payload.note || '任务失败';
          throw new Error(payload.note || '模拟任务失败');
        }

        await sleep(1000);
      }
    }

    settingsToggle.addEventListener('click', () => {
      setSettingsVisibility(settingsBody.style.display === 'none');
    });

    saveSettingsBtn.addEventListener('click', () => {
      try {
        applySettingsFromForm();
      } catch (err) {
        addMessage('bot', '参数保存失败: ' + err.message);
      }
    });

    resetSettingsBtn.addEventListener('click', () => {
      simulationParams = { ...DEFAULT_SIMULATION_PARAMS };
      fillSettingsForm(simulationParams);
      window.localStorage.setItem('mattersim_settings', JSON.stringify(simulationParams));
      addMessage('bot', '执行参数已恢复默认。');
    });

    presetFastBtn.addEventListener('click', () => applyPreset('fast'));
    presetStrictBtn.addEventListener('click', () => applyPreset('strict'));
    presetDeepBtn.addEventListener('click', () => applyPreset('deep'));
    chainButtons.forEach((button) => {
      button.addEventListener('click', () => {
        setThinkingChain(button.dataset.chain || '王工');
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;

      addMessage('user', text);
      question.value = '';
      updateWorkbenchState({
        mode: 'Agent Active',
        phase: '接收用户指令',
        source: '本地材料库 / MP / 结构代理',
        status: 'Queued',
        workflow: THINKING_CHAINS[activeThinkingChain].workflow,
        output: '尚未生成'
      });
      const thinkingNode = addThinkingMessage();
      const pipelinePromise = playPipeline();

      try {
        const responsePromise = fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: text,
            history: conversationState,
            simulation_params: simulationParams,
            thinking_profile: activeThinkingChain
          })
        });
        const [res] = await Promise.all([responsePromise, pipelinePromise]);
        const payload = await res.json();
        thinkingNode.remove();
        if (payload.mode === 'simulation') {
          renderSteps(-1, 0, SIMULATION_STEPS);
          agentHint.textContent = '已进入Agent计算链路';
          updateWorkbenchState({
            mode: 'Deep Compute',
            phase: '进入计算链路',
            source: 'Materials Project / 本地模板',
            status: 'Running',
            workflow: '多Agent计算链路',
            output: '等待计算结果'
          });
          addMessage('bot', payload.answer || '已启动材料计算任务。');
          const progressCard = addProgressCard('Agent 协同执行中');
          const simulationPayload = await pollSimulationTask(payload.task_id, progressCard);
          conversationState.push({
            role: 'user',
            content: text
          });
          conversationState.push({
            role: 'assistant',
            content: simulationPayload.summary || '',
            simulation_task_id: payload.task_id,
            simulation_result: simulationPayload.result || {},
            cif_payloads: simulationPayload.cif_payloads || [],
            cif_comparison: simulationPayload.cif_comparison || null
          });
          return;
        }
        addMessage('bot', payload.answer || '没有收到回答。');
        renderDownloadPayloads(payload);
        renderCifComparison(payload);
        renderCifPayloads(payload);
        renderMatchedTables(payload);
        setMeta(payload);
        updateWorkbenchState({
          mode: 'Open Discussion',
          phase: '分析完成',
          source: (payload.matched_payloads || []).length ? '本地材料库' : '开放讨论知识链路',
          status: 'Completed',
          workflow: (payload.matched_payloads || []).length ? '材料检索与解析' : THINKING_CHAINS[activeThinkingChain].workflow,
          output: (payload.matched_payloads || []).length ? '表格结果已生成' : '文本分析已生成'
        });
        conversationState.push({
          role: 'user',
          content: text
        });
        conversationState.push({
          role: 'assistant',
          content: payload.answer || '',
          matched_payloads: payload.matched_payloads || [],
          cif_payloads: payload.cif_payloads || [],
          cif_comparison: payload.cif_comparison || null,
          download_results: payload.download_results || []
        });
      } catch (err) {
        thinkingNode.remove();
        renderSteps(-1, 0);
        agentHint.textContent = '处理失败，请检查请求或稍后重试。';
        updateWorkbenchState({
          phase: '处理失败',
          status: 'Failed',
          output: '请求未完成'
        });
        addMessage('bot', '请求失败: ' + err.message);
      }
    });

    loadStoredSettings();
    setSettingsVisibility(false);
    renderSteps(-1, 0);
    setThinkingChain(activeThinkingChain);
    updateWorkbenchState({
      mode: 'Open Discussion',
      phase: '等待指令',
      source: '本地材料库 / Materials Project',
      status: 'Idle',
      workflow: THINKING_CHAINS[activeThinkingChain].workflow,
      output: '等待生成'
    });
  </script>
</body>
</html>
"""


def load_material_index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not MATERIALS_DIR.exists():
        return index
    for path in MATERIALS_DIR.glob("*.json"):
        index[path.stem.upper()] = path
    return index


def get_task_materials_dir(task_id: str) -> Path:
    return SIM_TASKS_DIR / task_id / SIM_MATERIALS_SUBDIR


def get_task_runtime_dir(task_id: str) -> Path:
    return SIM_TASKS_DIR / task_id / SIM_RUNTIME_SUBDIR


def _extract_cif_candidates(question: str) -> List[str]:
    candidates = re.findall(r"(\/[^\s'\"，。；;]+?\.cif)\b", question, flags=re.IGNORECASE)
    candidates.extend(re.findall(r"\b[\w.\-]+\.cif\b", question, flags=re.IGNORECASE))
    return list(dict.fromkeys(candidates))


def _resolve_cif_path_from_history(name: str, history: List[Dict[str, object]]) -> Optional[Path]:
    normalized = name.strip()
    if not normalized:
        return None

    explicit = Path(normalized).expanduser()
    if explicit.is_absolute() and explicit.exists():
        return explicit

    for item in reversed(history[-8:]):
        task_id = str(item.get("simulation_task_id") or "").strip()
        if not task_id:
            continue
        materials_dir = get_task_materials_dir(task_id)
        candidate = materials_dir / normalized
        if candidate.exists():
            return candidate
        lowered = normalized.lower()
        if lowered == "relaxed.cif":
            relaxed_files = sorted(materials_dir.glob("relaxed*.cif"))
            if relaxed_files:
                return relaxed_files[0]
        if lowered == "input.cif":
            input_files = sorted(materials_dir.glob("input*.cif")) + sorted(materials_dir.glob("mp_*.cif"))
            if input_files:
                return input_files[0]
    return None


def maybe_parse_cif_query(question: str, history: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    lowered = question.lower()
    wants_compare = any(keyword in question for keyword in ["对比", "比较", "横向对比"]) or "compare" in lowered
    has_cif_signal = ".cif" in lowered or "晶体结构" in question or "结构文件" in question or wants_compare
    if not has_cif_signal:
        return None

    resolved_paths: List[Path] = []
    if wants_compare and ("当前" in question or "input" in lowered or "relaxed" in lowered):
        before_path = _resolve_cif_path_from_history("input.cif", history)
        after_path = _resolve_cif_path_from_history("relaxed.cif", history)
        for path in [before_path, after_path]:
            if path and path not in resolved_paths:
                resolved_paths.append(path)

    candidates = _extract_cif_candidates(question)
    for candidate in candidates:
        resolved = _resolve_cif_path_from_history(candidate, history)
        if resolved and resolved not in resolved_paths:
            resolved_paths.append(resolved)

    if not resolved_paths and ("relaxed.cif" in lowered or "input.cif" in lowered or "当前" in question):
        fallback_name = "relaxed.cif" if "relaxed" in lowered else "input.cif"
        fallback = _resolve_cif_path_from_history(fallback_name, history)
        if fallback:
            resolved_paths.append(fallback)

    if wants_compare and len(resolved_paths) < 2:
        before_path = _resolve_cif_path_from_history("input.cif", history)
        after_path = _resolve_cif_path_from_history("relaxed.cif", history)
        for path in [before_path, after_path]:
            if path and path not in resolved_paths:
                resolved_paths.append(path)

    if not resolved_paths:
        return None

    parsed_payloads: List[Dict[str, Any]] = []
    parsed_successes: List[Dict[str, Any]] = []
    for path in resolved_paths[:3]:
        try:
            data = parse_cif(path)
            parsed_successes.append(data)
            parsed_payloads.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "data": data,
                }
            )
        except Exception as exc:
            parsed_payloads.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "error": f"解析 CIF 失败: {exc}",
                }
            )

    readable = "、".join(item["filename"] for item in parsed_payloads)
    response: Dict[str, Any] = {
        "answer": f"已解析 CIF 文件：{readable}",
        "cif_payloads": parsed_payloads,
        "steps": AGENT_STEPS,
    }
    if len(parsed_successes) >= 2:
        response["cif_comparison"] = compare_cif_payloads(parsed_successes[0], parsed_successes[1])
        response["answer"] = f"已完成初始 CIF 与弛豫后 CIF 的解析和横向对比：{readable}"
    return response


def detect_download_intent(question: str) -> bool:
    lowered = question.lower()
    has_download = any(keyword in question for keyword in ["下载", "保存到本地", "拉取"])
    has_cif = ".cif" in lowered or "cif" in lowered or "结构文件" in question
    return has_download and has_cif


def _extract_top_k(question: str, default: int = 5) -> int:
    digit_match = re.search(r"前\s*(\d+)\s*个", question)
    if digit_match:
        return max(1, int(digit_match.group(1)))
    if "前五个" in question:
        return 5
    if "前四个" in question:
        return 4
    if "前三个" in question:
        return 3
    if "前两个" in question:
        return 2
    if "前一个" in question or "第一个" in question:
        return 1
    return default


def maybe_handle_cif_download(question: str) -> Optional[Dict[str, object]]:
    if not detect_download_intent(question):
        return None

    queries = _extract_query_formulas(question)
    if not queries:
        return {
            "answer": "没有识别到可下载的化学式或元素体系。请明确写出例如 Nb-Ti、Nb-Ti-Sn、MoS2、FeSe 这类对象。",
            "steps": AGENT_STEPS,
        }

    top_k = _extract_top_k(question, default=5)
    output_root = MP_DOWNLOAD_ROOT
    api_key = MATERIALS_PROJECT_API_KEY.strip()
    if not api_key:
        return {
            "answer": "未配置 Materials Project API key，暂时无法下载 CIF 文件。",
            "steps": AGENT_STEPS,
        }

    download_results: List[Dict[str, object]] = []
    errors: List[str] = []
    for query in queries:
        try:
            download_results.append(download_query(query, top_k, output_root, api_key))
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    if not download_results and errors:
        return {
            "answer": "CIF 下载失败：" + "；".join(errors),
            "steps": AGENT_STEPS,
        }

    summaries = [f"{item['query']} 已下载 {item['count']} 个 CIF 到 {item['output_dir']}" for item in download_results]
    answer = "已完成 CIF 下载。\n\n" + "\n".join(summaries)
    if errors:
        answer += "\n\n未完成部分:\n" + "\n".join(errors)
    return {
        "answer": answer,
        "download_results": download_results,
        "steps": AGENT_STEPS,
    }


def normalize_simulation_params(raw: object) -> Dict[str, object]:
    params = dict(DEFAULT_SIMULATION_PARAMS)
    if isinstance(raw, dict):
        params.update({k: v for k, v in raw.items() if k in params})
        custom = raw.get("custom_params", {})
        params["custom_params"] = custom if isinstance(custom, dict) else {}
    params["relax_steps"] = int(params.get("relax_steps", 200) or 200)
    params["relax_fmax"] = float(params.get("relax_fmax", 0.05) or 0.05)
    params["optimizer"] = str(params.get("optimizer", "FIRE") or "FIRE")
    params["filter"] = str(params.get("filter", "ExpCellFilter") or "")
    params["device"] = str(params.get("device", "cpu") or "cpu")
    params["max_candidates_per_formula"] = int(params.get("max_candidates_per_formula", 4) or 4)
    params["max_formula_queries"] = int(params.get("max_formula_queries", 3) or 3)
    params["constrain_symmetry"] = bool(params.get("constrain_symmetry", False))
    return params


def extract_candidate_tokens(question: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{1,5}[A-Za-z0-9_-]{0,20}", question.upper())
    return list(dict.fromkeys(tokens))


def _looks_like_material_grade(token: str) -> bool:
    normalized = token.strip().upper()
    if not normalized:
        return False
    if re.fullmatch(r"T[ABC][A-Z0-9-]{1,20}", normalized):
        return True
    if re.fullmatch(r"[A-Z]{1,4}\d[A-Z0-9-]{0,20}", normalized):
        return True
    return False


def find_matches(question: str, material_index: Dict[str, Path]) -> List[Tuple[str, Path]]:
    question_upper = question.upper()
    tokens = extract_candidate_tokens(question)
    grade_tokens = [token for token in tokens if _looks_like_material_grade(token)]
    matches: List[Tuple[str, Path]] = []

    for token in grade_tokens:
        if token in material_index:
            matches.append((token, material_index[token]))

    if matches:
        return list(dict.fromkeys(matches))

    for stem, path in material_index.items():
        if stem in question_upper:
            matches.append((stem, path))
    if matches:
        return matches[:5]

    for stem, path in material_index.items():
        if any(token and token in stem for token in grade_tokens):
            matches.append((stem, path))
    return matches[:5]


def load_material_payload(matches: List[Tuple[str, Path]]) -> List[Dict[str, object]]:
    payloads = []
    for stem, path in matches[:5]:
        try:
            payloads.append(
                {
                    "filename": path.name,
                    "data": json.loads(path.read_text(encoding="utf-8")),
                }
            )
        except Exception as exc:
            payloads.append(
                {
                    "filename": path.name,
                    "error": f"Failed to read JSON: {exc}",
                }
            )
    return payloads


def _extract_element_symbol(label: str) -> Optional[str]:
    match = re.search(r"\b([A-Z][a-z]?)\b", label)
    return match.group(1) if match else None


def _parse_numeric_share(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "—", "–", "--"}:
        return None
    if "余量" in text:
        return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace("~", "-"))
    if not numbers:
        return None
    floats = [float(item) for item in numbers]
    if any(token in text for token in ["~", "-", "至"]):
        return sum(floats[:2]) / min(2, len(floats))
    if any(token in text for token in ["<", "≤", "≦", "<="]):
        return floats[0] / 2.0
    return floats[0]


def _extract_main_components(material_data: Dict[str, object]) -> Dict[str, float]:
    composition = material_data.get("化学成分（质量分数）", {}) or {}
    major = composition.get("主要成分", {}) or {}
    parsed: Dict[str, float] = {}
    for label, value in major.items():
        symbol = _extract_element_symbol(str(label))
        if not symbol:
            continue
        number = _parse_numeric_share(value)
        if number is None:
            if "余量" in str(value):
                parsed.setdefault(symbol, -1.0)
            continue
        parsed[symbol] = number
    return parsed


def _host_crystal_structure(symbol: str) -> str:
    if symbol in {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"}:
        return "fcc"
    if symbol in {"Fe", "Cr", "Mo", "W", "V", "Nb", "Ta"}:
        return "bcc"
    if symbol in {"Mg", "Ti", "Co", "Zn"}:
        return "hcp"
    if symbol in {"Si", "Ge", "C"}:
        return "diamond"
    return "hcp" if symbol == "Ti" else "fcc"


def _build_host_template(symbol: str) -> Atoms:
    structure = _host_crystal_structure(symbol)
    if structure == "hcp":
        return bulk(symbol, crystalstructure="hcp") * (2, 2, 2)
    return bulk(symbol, crystalstructure=structure) * (2, 2, 2)


def _expand_structure(atoms: Atoms, minimum_sites: int = 24) -> Atoms:
    expanded = atoms.copy()
    for repeats in [(2, 2, 1), (2, 2, 2), (3, 3, 2)]:
        if len(expanded) >= minimum_sites:
            break
        expanded = atoms * repeats
    return expanded


def _choose_host_element(material_data: Dict[str, object], main_components: Dict[str, float]) -> str:
    if "Ti" in main_components:
        return "Ti"
    nominal = str(material_data.get("名义化学成分", ""))
    if "钛" in nominal or "Titanium" in nominal:
        return "Ti"
    positive = {k: v for k, v in main_components.items() if v >= 0}
    if positive:
        return max(positive.items(), key=lambda item: item[1])[0]
    return next(iter(main_components), "Ti")


def _build_materials_project_query_terms(host: str, main_components: Dict[str, float]) -> List[str]:
    positive = sorted(
        [item for item in main_components.items() if item[0] != host and item[1] >= 0],
        key=lambda item: item[1],
        reverse=True,
    )
    symbols = [host] + [symbol for symbol, _ in positive[:3]]
    chemsys = "-".join(sorted(dict.fromkeys(symbols)))
    queries = [chemsys, host]
    return list(dict.fromkeys([query for query in queries if query]))


def _fetch_materials_project_host_structure(host: str, query_terms: List[str]) -> Tuple[Optional[Atoms], Dict[str, object]]:
    api_key = MATERIALS_PROJECT_API_KEY.strip()
    if not api_key:
        return None, {}

    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        if len(api_key) == 32:
            from mp_api.client import MPRester as ModernMPRester

            with ModernMPRester(api_key=api_key, mute_progress_bars=True) as mpr:
                for query in query_terms:
                    kwargs = {"fields": ["material_id", "formula_pretty", "energy_above_hull", "is_stable"], "all_fields": False}
                    if "-" in query:
                        kwargs["chemsys"] = query
                    else:
                        kwargs["formula"] = query
                    docs = mpr.materials.summary.search(**kwargs)
                    if not docs:
                        continue
                    best = sorted(
                        docs,
                        key=lambda doc: (
                            float(getattr(doc, "energy_above_hull", 999.0) or 999.0),
                            str(getattr(doc, "material_id", "")),
                        ),
                    )[0]
                    material_id = str(getattr(best, "material_id", ""))
                    structure = mpr.materials.get_structure_by_material_id(material_id)
                    return _expand_structure(AseAtomsAdaptor.get_atoms(structure)), {
                        "materials_project_id": material_id,
                        "materials_project_formula": str(getattr(best, "formula_pretty", host)),
                        "materials_project_query": query,
                        "materials_project_source": "mp_api",
                    }

        from pymatgen.ext.matproj import MPRester as LegacyMPRester

        with LegacyMPRester(api_key) as mpr:
            for query in query_terms:
                structures = mpr.get_structures(query, final=True)
                material_ids = mpr.get_materials_ids(query)
                if not structures:
                    continue
                structure = structures[0]
                return _expand_structure(AseAtomsAdaptor.get_atoms(structure)), {
                    "materials_project_id": material_ids[0] if material_ids else "",
                    "materials_project_formula": structure.composition.reduced_formula,
                    "materials_project_query": query,
                    "materials_project_source": "legacy",
                }
    except Exception as exc:
        return None, {"materials_project_error": str(exc)}


def _apply_alloy_substitutions(base_atoms: Atoms, host: str, main_components: Dict[str, float]) -> Atoms:
    atoms = _expand_structure(base_atoms)
    total_sites = len(atoms)
    dopants: List[Tuple[str, int]] = []
    for symbol, share in sorted(main_components.items(), key=lambda item: (item[0] != host, -item[1])):
        if symbol == host or share < 0:
            continue
        count = int(round(total_sites * share / 100.0))
        if share > 0 and count == 0:
            count = 1
        if count > 0:
            dopants.append((symbol, count))

    max_replaceable = max(0, total_sites - 1)
    total_dopants = sum(count for _, count in dopants)
    if total_dopants > max_replaceable and total_dopants > 0:
        scale = max_replaceable / float(total_dopants)
        dopants = [(symbol, max(1, int(round(count * scale)))) for symbol, count in dopants]

    symbols = atoms.get_chemical_symbols()
    used = 0
    for dopant_symbol, dopant_count in dopants:
        for _ in range(dopant_count):
            if used >= total_sites:
                break
            symbols[used] = dopant_symbol
            used += 1
    atoms.set_chemical_symbols(symbols)
    return atoms


def _build_atoms_from_material(material_payload: Dict[str, object]) -> Tuple[Atoms, Dict[str, object]]:
    data = material_payload.get("data", {}) or {}
    main_components = _extract_main_components(data)
    if not main_components:
        raise ValueError("命中的材料 JSON 中没有可用于生成结构模板的主要成分。")

    host = _choose_host_element(data, main_components)
    query_terms = _build_materials_project_query_terms(host, main_components)
    base_atoms, mp_metadata = _fetch_materials_project_host_structure(host, query_terms)
    structure_source = "materials_project_host" if base_atoms is not None else "local_template"
    if base_atoms is None:
        base_atoms = _build_host_template(host)
    alloy_atoms = _apply_alloy_substitutions(base_atoms, host, main_components)

    metadata = {
        "source": structure_source,
        "host_element": host,
        "alloy_grade": data.get("合金牌号", material_payload.get("filename", "")),
        "nominal_composition": data.get("名义化学成分", ""),
        "formula": alloy_atoms.get_chemical_formula(),
        "original_formula": data.get("合金牌号", material_payload.get("filename", "")),
        "template": f"{host}-{_host_crystal_structure(host)}-supercell",
        "material_filename": material_payload.get("filename", ""),
        "major_components": main_components,
    }
    metadata.update(mp_metadata)
    return alloy_atoms, metadata


def _normalize_chemsys(value: str) -> Optional[str]:
    parts = [part for part in re.split(r"[-–—]", value) if part]
    if len(parts) < 2:
        return None
    symbols: List[str] = []
    for part in parts:
        if not re.fullmatch(r"[A-Z][a-z]?", part):
            return None
        if part not in symbols:
            symbols.append(part)
    return "-".join(symbols)


def _extract_query_formulas(question: str) -> List[str]:
    segments = re.split(r"[\n\r,，;；、]+", question)
    chemsys: List[str] = []
    cleaned: List[str] = []

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue
        segment = re.sub(r"^[\-\*\u2022\u2023\u25E6\u2043\u2219]+\s*", "", segment)

        chemsys_match = re.search(r"(?:[A-Z][a-z]?)(?:\s*[-–—]\s*(?:[A-Z][a-z]?)){1,5}", segment)
        if chemsys_match:
            normalized = _normalize_chemsys(chemsys_match.group(0).replace(" ", ""))
            if normalized:
                chemsys.append(normalized)
                continue

        for formula in re.findall(r"\b(?:[A-Z][a-z]?\d*){1,8}\b", segment):
            if len(formula) <= 1:
                continue
            parts = re.findall(r"[A-Z][a-z]?\d*", formula)
            if len(parts) < 2 and not any(ch.isdigit() for ch in formula):
                continue
            cleaned.append(formula)

    return list(dict.fromkeys(chemsys + cleaned))


def _fetch_legacy_chemsys_candidates(mpr: Any, chemsys: str, limit: int) -> List[Tuple[str, Any]]:
    parts = [part for part in chemsys.split("-") if part]
    if len(parts) < 2:
        return []

    try:
        entries = mpr.get_entries_in_chemsys(parts)
    except Exception:
        return []

    ranked: List[Tuple[float, str]] = []
    for entry in entries:
        material_id = str(getattr(entry, "entry_id", "") or "")
        if not material_id:
            continue
        energy = float(getattr(entry, "energy_per_atom", 999.0) or 999.0)
        ranked.append((energy, material_id))

    deduped: List[Tuple[str, Any]] = []
    seen: set[str] = set()
    for _, material_id in sorted(ranked, key=lambda item: item[0]):
        if material_id in seen:
            continue
        try:
            structure = mpr.get_structure_by_material_id(material_id, final=True)
        except Exception:
            continue
        seen.add(material_id)
        deduped.append((material_id, structure))
        if len(deduped) >= limit:
            break
    return deduped


def _prepare_materials_project_candidates(
    materials_dir: Path, question: str, simulation_params: Dict[str, object]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    api_key = MATERIALS_PROJECT_API_KEY.strip()
    if not api_key:
        raise ValueError("未配置 Materials Project API key。")

    formulas = _extract_query_formulas(question)
    if not formulas:
        raise ValueError("未命中合金牌号时，需要在问题中明确给出化学式或化学体系，例如 MgB2、Nb3Sn、FeSe。")

    candidates: List[Dict[str, object]] = []
    query_stats: List[Dict[str, object]] = []
    from pymatgen.io.ase import AseAtomsAdaptor
    max_formula_queries = int(simulation_params.get("max_formula_queries", 3))
    max_candidates_per_formula = int(simulation_params.get("max_candidates_per_formula", 4))
    requested_formulas = formulas[:max_formula_queries]

    if len(api_key) == 32:
        from mp_api.client import MPRester as ModernMPRester

        with ModernMPRester(api_key=api_key, mute_progress_bars=True) as mpr:
            for formula in requested_formulas:
                kwargs = {
                    "fields": ["material_id", "formula_pretty", "energy_above_hull", "is_stable"],
                    "all_fields": False,
                }
                if "-" in formula:
                    kwargs["chemsys"] = formula
                else:
                    kwargs["formula"] = formula
                docs = mpr.materials.summary.search(**kwargs)
                selected_docs = sorted(
                    docs,
                    key=lambda item: float(getattr(item, "energy_above_hull", 999.0) or 999.0),
                )[:max_candidates_per_formula]
                for rank, doc in enumerate(selected_docs, start=1):
                    material_id = str(getattr(doc, "material_id", ""))
                    structure = mpr.materials.get_structure_by_material_id(material_id)
                    atoms = _expand_structure(AseAtomsAdaptor.get_atoms(structure))
                    name = f"{formula}-{material_id}"
                    structure_path = materials_dir / f"mp_{formula}_{rank}.cif"
                    write(structure_path, atoms)
                    candidates.append(
                        {
                            "name": name,
                            "structure_path": str(structure_path),
                            "metadata": {
                                "source": "materials_project",
                                "query_formula": formula,
                                "formula": atoms.get_chemical_formula(),
                                "original_formula": str(getattr(doc, "formula_pretty", formula)),
                                "materials_project_id": material_id,
                                "materials_project_formula": str(getattr(doc, "formula_pretty", formula)),
                            },
                        }
                    )
                query_stats.append(
                    {
                        "query": formula,
                        "returned_candidates": len(selected_docs),
                        "status": "ok" if selected_docs else "empty",
                    }
                )
    else:
        from pymatgen.ext.matproj import MPRester as LegacyMPRester

        with LegacyMPRester(api_key) as mpr:
            for formula in requested_formulas:
                selected_count = 0
                if "-" in formula:
                    legacy_candidates = _fetch_legacy_chemsys_candidates(mpr, formula, max_candidates_per_formula)
                    for rank, (material_id, structure) in enumerate(legacy_candidates, start=1):
                        atoms = _expand_structure(AseAtomsAdaptor.get_atoms(structure))
                        name = f"{formula}-{material_id or rank}"
                        structure_path = materials_dir / f"mp_{formula}_{rank}.cif"
                        write(structure_path, atoms)
                        candidates.append(
                            {
                                "name": name,
                                "structure_path": str(structure_path),
                                "metadata": {
                                    "source": "materials_project",
                                    "query_formula": formula,
                                    "formula": structure.composition.reduced_formula,
                                    "original_formula": structure.composition.reduced_formula,
                                    "materials_project_id": material_id,
                                    "materials_project_formula": structure.composition.reduced_formula,
                                },
                            }
                        )
                    selected_count = len(legacy_candidates)
                else:
                    structures = mpr.get_structures(formula, final=True)[:max_candidates_per_formula]
                    material_ids = mpr.get_materials_ids(formula)[: len(structures)]
                    for rank, structure in enumerate(structures, start=1):
                        atoms = _expand_structure(AseAtomsAdaptor.get_atoms(structure))
                        material_id = material_ids[rank - 1] if rank - 1 < len(material_ids) else ""
                        name = f"{formula}-{material_id or rank}"
                        structure_path = materials_dir / f"mp_{formula}_{rank}.cif"
                        write(structure_path, atoms)
                        candidates.append(
                            {
                                "name": name,
                                "structure_path": str(structure_path),
                                "metadata": {
                                    "source": "materials_project",
                                    "query_formula": formula,
                                    "formula": structure.composition.reduced_formula,
                                    "original_formula": structure.composition.reduced_formula,
                                    "materials_project_id": material_id,
                                    "materials_project_formula": structure.composition.reduced_formula,
                                },
                            }
                        )
                    selected_count = len(structures)
                query_stats.append(
                    {
                        "query": formula,
                        "returned_candidates": selected_count,
                        "status": "ok" if selected_count else "empty",
                    }
                )

    if not candidates:
        raise ValueError("Materials Project 没有返回可用于计算的候选结构。")
    return candidates, query_stats


def prepare_simulation_inputs(task_dir: Path, question: str, matched_payloads: List[Dict[str, object]], simulation_params: Dict[str, object]) -> Dict[str, str]:
    materials_dir = task_dir / SIM_MATERIALS_SUBDIR
    runtime_dir = task_dir / SIM_RUNTIME_SUBDIR
    materials_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = runtime_dir / "candidate_manifest.json"
    source_json_path = materials_dir / "source_materials.json"
    candidates: List[Dict[str, object]] = []

    if matched_payloads:
        for index, material_payload in enumerate([item for item in matched_payloads if "data" in item][:5], start=1):
            atoms, metadata = _build_atoms_from_material(material_payload)
            structure_path = materials_dir / f"input_{index}.cif"
            write(structure_path, atoms)
            candidates.append(
                {
                    "name": metadata.get("alloy_grade") or material_payload.get("filename", f"candidate-{index}"),
                    "structure_path": str(structure_path),
                    "metadata": metadata,
                }
            )
        source_json_path.write_text(
            json.dumps([item["data"] for item in matched_payloads if "data" in item], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        candidates, query_stats = _prepare_materials_project_candidates(materials_dir, question, simulation_params)
    manifest_payload: Dict[str, object] = {"question": question, "candidates": candidates}
    if not matched_payloads:
        manifest_payload["query_stats"] = query_stats
        manifest_payload["requested_queries"] = _extract_query_formulas(question)[: int(simulation_params.get("max_formula_queries", 3))]

    params_path = runtime_dir / "simulation_params.json"
    params_path.write_text(json.dumps(simulation_params, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"candidate_manifest": str(manifest_path), "params_json": str(params_path)}


def detect_simulation_intent(question: str) -> bool:
    question_lower = question.lower()
    explicit_actions = [
        "开始分析",
        "去分析",
        "开始计算",
        "去计算",
        "开始筛选",
        "去筛选",
        "开始快排",
        "跑一下",
        "执行计算",
        "做快排",
        "做筛选",
        "跑一下计算",
        "执行快排",
        "simulate",
        "run mattersim",
    ]
    if any(keyword in question_lower for keyword in explicit_actions):
        return True

    has_targets = bool(_extract_query_formulas(question))
    action_keywords = ["快排", "筛选", "排序", "计算", "模拟"]
    return has_targets and any(keyword in question_lower for keyword in action_keywords)


def _history_has_simulation(history: List[Dict[str, object]]) -> bool:
    return any(item.get("simulation_task_id") for item in history[-8:])


def _initialize_failed_task(task_dir: Path, note: str) -> None:
    runtime_dir = task_dir / SIM_RUNTIME_SUBDIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "current_step": 2,
        "status": "failed",
        "note": note,
        "steps": [
            {"label": SIMULATION_STEPS[0], "status": "done"},
            {"label": SIMULATION_STEPS[1], "status": "done"},
            {"label": SIMULATION_STEPS[2], "status": "failed"},
            {"label": SIMULATION_STEPS[3], "status": "waiting"},
            {"label": SIMULATION_STEPS[4], "status": "waiting"},
            {"label": SIMULATION_STEPS[5], "status": "waiting"},
        ],
        "updated_at": time.time(),
    }
    result = {
        "summary": note,
        "table": [
            {"字段": "状态", "数值": "失败"},
            {"字段": "原因", "数值": note},
        ],
        "artifacts": {},
        "error": note,
    }
    (runtime_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    (runtime_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def start_simulation_task(question: str, matched_payloads: List[Dict[str, object]], simulation_params: Dict[str, object]) -> str:
    SIM_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"sim-{uuid.uuid4().hex[:10]}"
    task_dir = SIM_TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare_simulation_inputs(task_dir, question, matched_payloads, simulation_params)
    except Exception as exc:
        _initialize_failed_task(task_dir, f"input.cif 生成失败: {exc}")
        return task_id
    cmd = [
        sys.executable,
        str(SIM_SCRIPT_PATH),
        "--task-id",
        task_id,
        "--question",
        question,
        "--output-dir",
        str(SIM_TASKS_DIR),
    ]
    if prepared.get("candidate_manifest"):
        cmd.extend(["--candidate-manifest", prepared["candidate_manifest"]])
    if prepared.get("params_json"):
        cmd.extend(["--params-json", prepared["params_json"]])
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return task_id


def load_simulation_status(task_id: str) -> Dict[str, object]:
    task_dir = SIM_TASKS_DIR / task_id
    materials_dir = get_task_materials_dir(task_id)
    runtime_dir = get_task_runtime_dir(task_id)
    progress_path = runtime_dir / "progress.json"
    result_path = runtime_dir / "result.json"
    input_candidates = sorted(materials_dir.glob("input*.cif")) + sorted(materials_dir.glob("mp_*.cif"))
    relaxed_candidates = sorted(materials_dir.glob("relaxed*.cif"))

    if not progress_path.exists():
        return {
            "status": "running",
            "note": "任务初始化中",
            "steps": [{"label": label, "status": "waiting"} for label in SIMULATION_STEPS],
            "progress_percent": 0,
        }

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    steps = progress.get("steps", [])
    progress_percent = 0
    if steps:
      done_count = sum(1 for step in steps if step.get("status") == "done")
      running_count = sum(1 for step in steps if step.get("status") == "running")
      progress_percent = int((done_count + 0.5 * running_count) / len(steps) * 100)

    payload: Dict[str, object] = {
        "status": progress.get("status", "running"),
        "note": progress.get("note", ""),
        "steps": steps,
        "progress_percent": progress_percent,
    }
    parsed_cifs: List[Dict[str, Any]] = []
    parsed_successes: List[Dict[str, Any]] = []
    if input_candidates:
        payload["input_url"] = f"/task-asset?task_id={task_id}&name={input_candidates[0].name}&t={int(time.time())}"
        try:
            input_data = parse_cif(input_candidates[0])
            parsed_successes.append(input_data)
            parsed_cifs.append({
                "filename": input_candidates[0].name,
                "path": str(input_candidates[0]),
                "data": input_data,
            })
        except Exception as exc:
            parsed_cifs.append({
                "filename": input_candidates[0].name,
                "path": str(input_candidates[0]),
                "error": f"解析 CIF 失败: {exc}",
            })
    if relaxed_candidates:
        payload["relaxed_url"] = f"/task-asset?task_id={task_id}&name={relaxed_candidates[0].name}&t={int(time.time())}"
        try:
            relaxed_data = parse_cif(relaxed_candidates[0])
            parsed_successes.append(relaxed_data)
            parsed_cifs.append({
                "filename": relaxed_candidates[0].name,
                "path": str(relaxed_candidates[0]),
                "data": relaxed_data,
            })
        except Exception as exc:
            parsed_cifs.append({
                "filename": relaxed_candidates[0].name,
                "path": str(relaxed_candidates[0]),
                "error": f"解析 CIF 失败: {exc}",
            })
    if parsed_cifs:
        payload["cif_payloads"] = parsed_cifs
    if len(parsed_successes) >= 2:
        payload["cif_comparison"] = compare_cif_payloads(parsed_successes[0], parsed_successes[1])

    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        payload["result"] = result
        payload["summary"] = result.get("summary", "")
        if (runtime_dir / "summary.svg").exists():
            payload["image_url"] = f"/task-asset?task_id={task_id}&name=summary.svg&t={int(time.time())}"
        if progress.get("status") != "failed":
            payload["status"] = "completed"
            payload["progress_percent"] = 100
    return payload


def _collect_history_materials(history: List[Dict[str, object]]) -> List[Dict[str, object]]:
    materials: List[Dict[str, object]] = []
    for item in history[-8:]:
        for material in item.get("matched_payloads", []) or []:
            if material not in materials:
                materials.append(material)
    return materials


def _collect_history_simulation_results(history: List[Dict[str, object]]) -> List[Dict[str, object]]:
    simulations: List[Dict[str, object]] = []
    for item in history[-8:]:
        result = item.get("simulation_result")
        if isinstance(result, dict) and result:
            simulations.append(result)
            continue
        task_id = item.get("simulation_task_id")
        if not task_id:
            continue
        runtime_dir = get_task_runtime_dir(str(task_id))
        result_path = runtime_dir / "result.json"
        if result_path.exists():
            try:
                simulations.append(json.loads(result_path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return simulations


def call_yunwu_api(
    question: str,
    matched_payloads: List[Dict[str, object]],
    history: List[Dict[str, object]],
) -> Tuple[str, str]:
    if not API_KEY:
        fallback = build_fallback_answer(question, matched_payloads)
        return fallback, "未配置 YUNWU_API_KEY，已返回本地检索结果"

    context_materials = matched_payloads or _collect_history_materials(history)
    context_simulations = _collect_history_simulation_results(history)

    system_prompt = (
        "你是材料问答助手。"
        "必须优先根据提 JSON 材料数据回答。"
        "支持多轮对话，如果用户继续分析、追问或者要求对比，要结合历史材料数据继续回答。"
        "如果历史里已经有计算结果，用户继续追问时要优先基于已有计算结果解释，不要误判成新一轮计算。"
        "如果命中了具体文件，总结对应内容。"
        "如果没有命中，不要只说未命中文件；你仍然要继续做开放讨论，给出材料特性分析、候选体系建议、下一步可计算方向。"
        "只有在用户明确说去分析、去计算、开始筛选、开始快排时，才进入计算建议。"
        "当你给候选建议时，优先给出具体化学式或材料体系，方便后续进入 Materials Project 检索。"
        "回答保持简洁，重点做分析和对比，不要输出 markdown 表格。"
    )
    user_prompt = {
        "question": question,
        "matched_materials": context_materials,
        "simulation_results": context_simulations[-3:],
        "history": history[-6:],
    }
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.2,
    }
    req = request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return content, f"已调用 {MODEL_NAME}"
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        fallback = build_fallback_answer(question, matched_payloads)
        return f"{fallback}\n\n模型调用失败: {detail}", f"HTTP {exc.code}"
    except Exception as exc:
        fallback = build_fallback_answer(question, matched_payloads)
        return f"{fallback}\n\n模型调用失败: {exc}", "调用失败"


def build_fallback_answer(question: str, matched_payloads: List[Dict[str, object]]) -> str:
    if not matched_payloads:
        if "超导" in question:
            return (
                "当前没有命中本地合金文件，但可以先做开放讨论。\n\n"
                "如果你是在找更有潜力的超导候选，通常会先关注：结构稳定性、金属态可能性、层状或过渡金属体系、"
                "以及后续电子结构计算是否容易展开。\n\n"
                "你可以下一步直接说：请对 MgB2、Nb3Sn、FeSe 做快排。"
            )
        return (
            f"没有在本地材料目录中找到和问题相关的文件。\n\n问题: {question}\n\n"
            "如果你想继续开放讨论，我可以先给你候选体系建议；"
            "如果你想直接算，请再明确说“开始筛选”或给出化学式。"
        )

    lines = ["已命中以下材料文件，并返回文件内容摘要:"]
    for item in matched_payloads:
        lines.append(f"\n文件: {item['filename']}")
        if "data" in item:
            lines.append(json.dumps(item["data"], ensure_ascii=False, indent=2))
        else:
            lines.append(item["error"])
    return "\n".join(lines)


class ChatHandler(BaseHTTPRequestHandler):
    material_index = load_material_index()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if self.path == "/logo":
                if not LOGO_PATH.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "Logo Not Found")
                    return
                body = LOGO_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/task-status":
                task_id = query.get("id", [""])[0]
                if not task_id:
                    self._send_json({"error": "missing task id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(load_simulation_status(task_id))
                return
            if parsed.path == "/task-asset":
                task_id = query.get("task_id", [""])[0]
                name = query.get("name", [""])[0]
                materials_dir = get_task_materials_dir(task_id).resolve()
                runtime_dir = get_task_runtime_dir(task_id).resolve()
                candidates = [
                    (materials_dir / name).resolve(),
                    (runtime_dir / name).resolve(),
                ]
                asset_path = next((path for path in candidates if path.exists()), None)
                allowed_roots = [str(materials_dir), str(runtime_dir)]
                if asset_path is None or not any(str(asset_path).startswith(root) for root in allowed_roots):
                    self.send_error(HTTPStatus.NOT_FOUND, "Asset Not Found")
                    return
                body = asset_path.read_bytes()
                suffix = asset_path.suffix.lower()
                if suffix == ".svg":
                    content_type = "image/svg+xml"
                elif suffix == ".cif":
                    content_type = "chemical/x-cif; charset=utf-8"
                elif suffix == ".json":
                    content_type = "application/json; charset=utf-8"
                else:
                    content_type = "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            if parsed.path == "/task-status":
                self._send_json(
                    {
                        "status": "failed",
                        "note": f"任务状态读取失败: {exc}",
                        "steps": [{"label": label, "status": "waiting"} for label in SIMULATION_STEPS],
                        "progress_percent": 0,
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self):
        try:
            if self.path != "/chat":
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)

            payload = json.loads(raw.decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            history = payload.get("history", [])
            simulation_params = normalize_simulation_params(payload.get("simulation_params", {}))
            if not question:
                self._send_json({"answer": "请输入问题。"}, status=HTTPStatus.BAD_REQUEST)
                return

            history_list = history if isinstance(history, list) else []

            download_response = maybe_handle_cif_download(question)
            if download_response is not None:
                self._send_json(download_response)
                return

            matches = find_matches(question, self.material_index)
            matched_payloads = load_material_payload(matches)

            cif_response = maybe_parse_cif_query(question, history_list)
            if cif_response is not None:
                self._send_json(cif_response)
                return

            should_simulate = detect_simulation_intent(question)
            if should_simulate and not matched_payloads and not _extract_query_formulas(question) and _history_has_simulation(history_list):
                should_simulate = False

            if should_simulate:
                task_id = start_simulation_task(question, matched_payloads, simulation_params)
                if matched_payloads:
                    answer = "已命中合金牌号，正在使用本地模板生成候选 input.cif，并交给agent做批量快排。"
                else:
                    answer = "未命中合金牌号，正在从外部知识库检索候选结构，并交给agent做批量快排。"
                self._send_json(
                    {
                        "mode": "simulation",
                        "task_id": task_id,
                        "answer": answer,
                    }
                )
                return

            answer, api_status = call_yunwu_api(question, matched_payloads, history_list)

            self._send_json(
                {
                    "answer": answer,
                    "matches": [stem for stem, _ in matches],
                    "matched_payloads": matched_payloads,
                    "api_status": api_status,
                    "steps": AGENT_STEPS,
                }
            )
        except json.JSONDecodeError:
            self._send_json({"answer": "请求体不是合法 JSON。"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"answer": f"请求失败: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _send_json(self, data: Dict[str, object], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ThreadingHTTPServer.allow_reuse_address = True
    server = None
    bound_port = PORT
    for candidate_port in range(PORT, PORT + 20):
        try:
            server = ThreadingHTTPServer((HOST, candidate_port), ChatHandler)
            bound_port = candidate_port
            break
        except OSError as exc:
            if exc.errno != 48:
                raise
    if server is None:
        raise OSError(f"No available port found between {PORT} and {PORT + 19}")

    print(f"Server running at http://{HOST}:{bound_port}")
    print(f"Materials directory: {MATERIALS_DIR}")
    if not API_KEY:
        print("Warning: YUNWU_API_KEY is not set. Only local retrieval fallback will work.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
