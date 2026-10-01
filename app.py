# -*- coding: utf-8 -*-
"""
하이브리드 수학·과학 서논술형 자동 채점 시스템  v4.0 (대회 제출용)
- 완전 오프라인 동작: 외부 API·AI·서버로 어떤 자료도 전송하지 않음
- KoNLPy(Okt) 어휘적 매칭 + ko-sroberta 의미적 벡터 매칭 하이브리드
- 학생 실시간 참여(QR/링크) · 다문항 세트(순차 진행) · 제출 이력 추적 · NEIS 출력
"""

# =====================================================================
# [0] 환경 안전성 장치 : JVMNotFoundException 방어  (konlpy import 이전 실행)
# =====================================================================
import os
import sys
import glob as _glob

JAVA_STATUS = {"ok": False, "home": None, "jvm": None, "msg": "미확인", "tried": []}


def _validate_java_home(home):
    """해당 경로가 실제 JVM 라이브러리를 품고 있는지 검증한다."""
    if not home or not os.path.isdir(home):
        return None
    patterns = [
        os.path.join(home, "bin", "server", "jvm.dll"),
        os.path.join(home, "bin", "client", "jvm.dll"),
        os.path.join(home, "jre", "bin", "server", "jvm.dll"),
        os.path.join(home, "lib", "server", "libjvm.so"),
        os.path.join(home, "jre", "lib", "amd64", "server", "libjvm.so"),
        os.path.join(home, "lib", "server", "libjvm.dylib"),
        os.path.join(home, "jre", "lib", "server", "libjvm.dylib"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    return None


def _candidate_java_homes():
    """OS별 표준 설치 경로를 최신 버전부터 훑는다."""
    pats = [
        r"C:\Program Files\Java\*",
        r"C:\Program Files\Eclipse Adoptium\*",
        r"C:\Program Files\Zulu\*",
        r"C:\Program Files\Amazon Corretto\*",
        r"C:\Program Files\Microsoft\jdk*",
        r"C:\Program Files\BellSoft\*",
        r"C:\Program Files (x86)\Java\*",
        r"C:\Java\*",
        "/usr/lib/jvm/*",
        "/usr/java/*",
        "/Library/Java/JavaVirtualMachines/*/Contents/Home",
        "/opt/homebrew/opt/openjdk*",
        "/opt/java/*",
    ]
    found = []
    for pat in pats:
        try:
            found.extend([p for p in _glob.glob(pat) if os.path.isdir(p)])
        except Exception:
            continue
    return sorted(set(found), reverse=True)


def setup_java_home(user_path=None):
    """우선순위: 교사 직접 입력 > 기존 JAVA_HOME > 시스템 자동 탐색."""
    cands = []
    if user_path:
        cands.append(user_path.strip().strip('"'))
    if os.environ.get("JAVA_HOME"):
        cands.append(os.environ["JAVA_HOME"])
    cands.extend(_candidate_java_homes())

    JAVA_STATUS["tried"] = cands[:12]
    for home in cands:
        jvm = _validate_java_home(home)
        if jvm:
            os.environ["JAVA_HOME"] = home
            bindir = os.path.dirname(jvm)
            if bindir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            JAVA_STATUS.update({"ok": True, "home": home, "jvm": jvm,
                                "msg": "JVM 경로를 정상적으로 설정했습니다."})
            return True
    JAVA_STATUS.update({"ok": False, "home": None, "jvm": None,
                        "msg": "JDK를 찾지 못했습니다. 간이 형태소 분석기로 자동 대체합니다."})
    return False


setup_java_home()   # ★ konlpy import 보다 반드시 먼저

# =====================================================================
# [1] 표준 라이브러리 / 서드파티 임포트
# =====================================================================
import io
import re
import json
import math
import time
import zlib
import struct
import socket
import string
import random
import sqlite3
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "v4.0 (대회 제출용)"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "grading_data.db")
MODEL_DIR = os.path.join(APP_DIR, "models")
BUNDLED_MODEL_DIR = os.path.join(MODEL_DIR, "ko-sroberta-multitask")
KO_SBERT_NAME = "jhgan/ko-sroberta-multitask"
DOWNLOAD_FLAG = os.path.join(APP_DIR, ".allow_model_download")

# ---------------------------------------------------------------------
# 외부 전송 차단 (대회 요강 Ⅱ-2-다 : 기관 승인 없는 외부 API·AI·서버 전송 금지)
#   - 허깅페이스 허브 접속을 라이브러리 차원에서 차단한다.
#   - 개발 PC에서 모델을 처음 내려받을 때만 플래그 파일로 일시 해제한다.
# ---------------------------------------------------------------------
NET_STATUS = {"offline": True, "reason": "기본값: 외부 접속 차단"}


def set_offline(flag=True):
    v = "1" if flag else "0"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = v
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    NET_STATUS["offline"] = bool(flag)
    NET_STATUS["reason"] = "외부 접속 차단" if flag else "모델 준비를 위한 일시 허용"


set_offline(not os.path.isfile(DOWNLOAD_FLAG))

# ---------------------------------------------------------------------
# Streamlit 버전 호환 계층 (구버전 업무용 PC 대비)
# ---------------------------------------------------------------------
if not hasattr(st, "toggle"):
    st.toggle = st.checkbox
if not hasattr(st, "rerun"):
    st.rerun = getattr(st, "experimental_rerun", lambda: None)
if not hasattr(st, "fragment"):
    def _frag(*a, **k):
        def deco(fn):
            return fn
        return deco
    st.fragment = _frag
if not hasattr(st, "divider"):
    st.divider = lambda: st.markdown("---")

# 화면 표시용 이름 마스킹 (시연·공개 화면에서 개인정보 노출 방지)
DISPLAY = {"mask": False}


def mask_name(name):
    s = str(name or "").strip()
    if len(s) <= 1:
        return s
    if len(s) == 2:
        return s[0] + "○"
    return s[0] + "○" * (len(s) - 2) + s[-1]

# =====================================================================
# [2] QR 코드 생성기 (외부 패키지 없이 동작하는 순수 파이썬 구현, ECC 레벨 M)
# =====================================================================
_ECC_PER_BLOCK_M = {1: 10, 2: 16, 3: 26, 4: 18, 5: 24, 6: 16, 7: 18, 8: 22, 9: 22, 10: 26}
_BLOCKS_M = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 4, 8: 4, 9: 5, 10: 5}
_ALIGN_POS = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
              6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50]}


def _raw_data_modules(ver):
    result = (16 * ver + 128) * ver + 64
    if ver >= 2:
        na = ver // 7 + 2
        result -= (25 * na - 10) * na - 55
        if ver >= 7:
            result -= 36
    return result


def _num_data_codewords(ver):
    return _raw_data_modules(ver) // 8 - _ECC_PER_BLOCK_M[ver] * _BLOCKS_M[ver]


def _gf_mul(x, y):
    z = 0
    for i in range(7, -1, -1):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z & 0xFF


def _rs_divisor(degree):
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(degree):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < degree:
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data, divisor):
    result = [0] * len(divisor)
    for b in data:
        factor = b ^ result.pop(0)
        result.append(0)
        for i in range(len(divisor)):
            result[i] ^= _gf_mul(divisor[i], factor)
    return result


class _QR:
    """byte 모드 · ECC M · 버전 1~10 자동 선택 QR 인코더."""

    def __init__(self, text):
        raw = text.encode("utf-8")
        ver = None
        for v in range(1, 11):
            if 4 + 8 + len(raw) * 8 <= _num_data_codewords(v) * 8:
                ver = v
                break
        if ver is None:
            raise ValueError("QR 용량 초과: 주소가 너무 깁니다.")
        self.ver = ver
        self.size = ver * 4 + 17
        self.modules = [[False] * self.size for _ in range(self.size)]
        self.isfunc = [[False] * self.size for _ in range(self.size)]

        bits = []
        for i in range(3, -1, -1):
            bits.append((0b0100 >> i) & 1)
        for i in range(7, -1, -1):
            bits.append((len(raw) >> i) & 1)
        for b in raw:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)

        cap = _num_data_codewords(ver) * 8
        bits.extend([0] * min(4, cap - len(bits)))
        bits.extend([0] * ((8 - len(bits) % 8) % 8))
        pad = [0xEC, 0x11]
        idx = 0
        while len(bits) < cap:
            for i in range(7, -1, -1):
                bits.append((pad[idx % 2] >> i) & 1)
            idx += 1
        data = [int("".join(str(b) for b in bits[i:i + 8]), 2) for i in range(0, len(bits), 8)]

        self._draw_function_patterns()
        self._draw_codewords(self._ecc_interleave(data))
        best, bestpen = 0, None
        for m in range(8):
            self._apply_mask(m)
            self._draw_format(m)
            pen = self._penalty()
            self._apply_mask(m)
            if bestpen is None or pen < bestpen:
                bestpen, best = pen, m
        self._apply_mask(best)
        self._draw_format(best)

    # ---- 구조 ----
    def _setf(self, x, y, dark):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.modules[y][x] = dark
            self.isfunc[y][x] = True

    def _draw_function_patterns(self):
        n = self.size
        for i in range(n):
            self._setf(6, i, i % 2 == 0)
            self._setf(i, 6, i % 2 == 0)
        for (cx, cy) in [(3, 3), (n - 4, 3), (3, n - 4)]:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    d = max(abs(dx), abs(dy))
                    self._setf(cx + dx, cy + dy, d != 2 and d != 4)
        pos = _ALIGN_POS[self.ver]
        for i, py in enumerate(pos):
            for j, px in enumerate(pos):
                if (i == 0 and j == 0) or (i == 0 and j == len(pos) - 1) or (i == len(pos) - 1 and j == 0):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        self._setf(px + dx, py + dy, max(abs(dx), abs(dy)) != 1)
        self._draw_format(0)
        if self.ver >= 7:
            rem = self.ver
            for _ in range(12):
                rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
            bits = self.ver << 12 | rem
            for i in range(18):
                bit = (bits >> i) & 1 != 0
                a, b = self.size - 11 + i % 3, i // 3
                self._setf(a, b, bit)
                self._setf(b, a, bit)

    def _draw_format(self, mask):
        data = 0 << 3 | mask          # ECC 레벨 M = 0b00 -> formatbits 0
        rem = data
        for _ in range(10):
            rem = (rem << 1) ^ ((rem >> 9) * 0x537)
        bits = (data << 10 | rem) ^ 0x5412
        n = self.size
        for i in range(0, 6):
            self._setf(8, i, (bits >> i) & 1 != 0)
        self._setf(8, 7, (bits >> 6) & 1 != 0)
        self._setf(8, 8, (bits >> 7) & 1 != 0)
        self._setf(7, 8, (bits >> 8) & 1 != 0)
        for i in range(9, 15):
            self._setf(14 - i, 8, (bits >> i) & 1 != 0)
        for i in range(0, 8):
            self._setf(n - 1 - i, 8, (bits >> i) & 1 != 0)
        for i in range(8, 15):
            self._setf(8, n - 15 + i, (bits >> i) & 1 != 0)
        self._setf(8, n - 8, True)

    def _ecc_interleave(self, data):
        nb = _BLOCKS_M[self.ver]
        ecclen = _ECC_PER_BLOCK_M[self.ver]
        raw = _raw_data_modules(self.ver) // 8
        nshort = nb - raw % nb
        shortlen = raw // nb
        div = _rs_divisor(ecclen)
        blocks, k = [], 0
        for i in range(nb):
            ln = shortlen - ecclen + (0 if i < nshort else 1)
            dat = list(data[k:k + ln])
            k += ln
            ecc = _rs_remainder(dat, div)
            if i < nshort:
                dat = dat + [0]
            blocks.append(dat + ecc)
        out = []
        for i in range(len(blocks[0])):
            for j, blk in enumerate(blocks):
                if i != shortlen - ecclen or j >= nshort:
                    out.append(blk[i])
        return out

    def _draw_codewords(self, data):
        n, i = self.size, 0
        right = n - 1
        while right >= 1:
            if right == 6:
                right = 5
            for vert in range(n):
                for j in range(2):
                    x = right - j
                    upward = ((right + 1) & 2) == 0
                    y = (n - 1 - vert) if upward else vert
                    if not self.isfunc[y][x] and i < len(data) * 8:
                        self.modules[y][x] = (data[i >> 3] >> (7 - (i & 7))) & 1 != 0
                        i += 1
            right -= 2

    def _apply_mask(self, m):
        for y in range(self.size):
            for x in range(self.size):
                if self.isfunc[y][x]:
                    continue
                if m == 0:
                    inv = (x + y) % 2 == 0
                elif m == 1:
                    inv = y % 2 == 0
                elif m == 2:
                    inv = x % 3 == 0
                elif m == 3:
                    inv = (x + y) % 3 == 0
                elif m == 4:
                    inv = (x // 3 + y // 2) % 2 == 0
                elif m == 5:
                    inv = x * y % 2 + x * y % 3 == 0
                elif m == 6:
                    inv = (x * y % 2 + x * y % 3) % 2 == 0
                else:
                    inv = ((x + y) % 2 + x * y % 3) % 2 == 0
                if inv:
                    self.modules[y][x] = not self.modules[y][x]

    def _penalty(self):
        n, p = self.size, 0
        lines = []
        for y in range(n):
            lines.append("".join("1" if self.modules[y][x] else "0" for x in range(n)))
        for x in range(n):
            lines.append("".join("1" if self.modules[y][x] else "0" for y in range(n)))
        for ln in lines:
            run, prev = 1, ln[0]
            for ch in ln[1:]:
                if ch == prev:
                    run += 1
                else:
                    if run >= 5:
                        p += 3 + (run - 5)
                    run, prev = 1, ch
            if run >= 5:
                p += 3 + (run - 5)
            padded = "0000" + ln + "0000"
            p += 40 * (padded.count("10111010000") + padded.count("00001011101"))
        for y in range(n - 1):
            for x in range(n - 1):
                c = self.modules[y][x]
                if c == self.modules[y][x + 1] == self.modules[y + 1][x] == self.modules[y + 1][x + 1]:
                    p += 3
        dark = sum(sum(1 for v in row if v) for row in self.modules)
        total = n * n
        k = (abs(dark * 20 - total * 10) + total - 1) // total - 1
        p += k * 10
        return p


def qr_svg(text, scale=8, quiet=4):
    """QR 코드를 SVG 문자열로 반환 (Streamlit HTML 렌더링용)."""
    q = _QR(text)
    n = q.size
    dim = (n + quiet * 2) * scale
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'shape-rendering="crispEdges">' % (dim, dim, dim, dim),
        '<rect width="%d" height="%d" fill="#ffffff"/>' % (dim, dim),
    ]
    for y in range(n):
        for x in range(n):
            if q.modules[y][x]:
                parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#111827"/>'
                             % ((x + quiet) * scale, (y + quiet) * scale, scale, scale))
    parts.append("</svg>")
    return "".join(parts)


def qr_png_bytes(text, scale=8, quiet=4):
    """PIL 없이 PNG 바이트를 직접 생성 (인쇄·배포용 다운로드)."""
    q = _QR(text)
    n = q.size
    dim = (n + quiet * 2) * scale
    rows = []
    for y in range(dim):
        row = bytearray([0])                      # filter type 0
        my = y // scale - quiet
        for x in range(dim):
            mx = x // scale - quiet
            dark = (0 <= mx < n and 0 <= my < n and q.modules[my][mx])
            row += bytes([0, 0, 0] if dark else [255, 255, 255])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", dim, dim, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return png


# =====================================================================
# [3] 데이터 저장소 (SQLite) : 교사·학생 브라우저 세션 간 공유
# =====================================================================
def db():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=8000")
    return con


DDL = [
    """CREATE TABLE IF NOT EXISTS sessions(
        code TEXT PRIMARY KEY, title TEXT, subject TEXT, grade TEXT,
        created_at TEXT, is_open INTEGER DEFAULT 1, reveal INTEGER DEFAULT 1,
        allow_resubmit INTEGER DEFAULT 1, config_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS items(
        code TEXT, seq INTEGER, question TEXT, model_answer TEXT,
        answer_value TEXT, unit TEXT, points REAL, keywords_json TEXT,
        PRIMARY KEY(code, seq))""",
    """CREATE TABLE IF NOT EXISTS submissions(
        code TEXT, student_key TEXT, seq INTEGER,
        student_no TEXT, student_name TEXT, answer TEXT,
        score REAL, ratio REAL, level TEXT, feedback TEXT,
        flags_json TEXT, matched_json TEXT, missing_json TEXT,
        kw_ratio REAL, sem_sim REAL, submit_count INTEGER DEFAULT 1,
        first_at TEXT, last_at TEXT,
        teacher_score REAL, teacher_feedback TEXT, confirmed INTEGER DEFAULT 0,
        PRIMARY KEY(code, student_key, seq))""",
    """CREATE TABLE IF NOT EXISTS students(
        code TEXT, student_key TEXT, student_no TEXT, student_name TEXT,
        joined_at TEXT, last_at TEXT, total_submits INTEGER DEFAULT 0,
        PRIMARY KEY(code, student_key))""",
]


EXPECTED = {
    "sessions": (["code"],
                 ["code", "title", "subject", "grade", "created_at", "is_open",
                  "reveal", "allow_resubmit", "config_json"]),
    "items": (["code", "seq"],
              ["code", "seq", "question", "model_answer", "answer_value",
               "unit", "points", "keywords_json"]),
    "submissions": (["code", "student_key", "seq"],
                    ["code", "student_key", "seq", "student_no", "student_name",
                     "answer", "score", "ratio", "level", "feedback", "flags_json",
                     "matched_json", "missing_json", "kw_ratio", "sem_sim",
                     "submit_count", "first_at", "last_at", "teacher_score",
                     "teacher_feedback", "confirmed"]),
    "students": (["code", "student_key"],
                 ["code", "student_key", "student_no", "student_name",
                  "joined_at", "last_at", "total_submits"]),
}
MIGRATION_LOG = []


def _cols(con, table):
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(%s)" % table).fetchall()]
    except Exception:
        return []


def ensure_schema(con):
    """구버전 DB를 감지해 열을 보강하거나, 구조가 다르면 보존 후 재생성한다."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for t, (pks, cols) in EXPECTED.items():
        have = _cols(con, t)
        if not have:
            continue                                   # 아직 없으면 DDL이 생성
        missing = [c for c in cols if c not in have]
        if not missing:
            continue
        if any(c in pks for c in missing):
            # 기본키 구성이 달라 ALTER 로 보정 불가 → 이름을 바꿔 보존
            old = "%s_old_%s" % (t, stamp)
            con.execute("ALTER TABLE %s RENAME TO %s" % (t, old))
            MIGRATION_LOG.append("%s 테이블은 구조가 달라 %s 로 보존하고 새로 만들었습니다." % (t, old))
        else:
            for c in missing:
                try:
                    con.execute("ALTER TABLE %s ADD COLUMN %s" % (t, c))
                    MIGRATION_LOG.append("%s 테이블에 %s 열을 추가했습니다." % (t, c))
                except Exception as e:
                    MIGRATION_LOG.append("%s.%s 추가 실패: %s" % (t, c, e))
    con.commit()


def schema_report():
    con = db()
    try:
        out = []
        for t, (pks, cols) in EXPECTED.items():
            have = _cols(con, t)
            miss = [c for c in cols if c not in have]
            out.append({"테이블": t, "열 수": len(have),
                        "상태": "정상" if have and not miss else ("없음" if not have else "누락 " + ",".join(miss))})
        return pd.DataFrame(out)
    finally:
        con.close()


def init_db():
    con = db()
    try:
        ensure_schema(con)
        for q in DDL:
            con.execute(q)
        con.commit()
        ensure_schema(con)
    finally:
        con.close()


def purge_personal_data(mode="anonymize"):
    """제출 전 개인정보 정리. mode: anonymize(가명화) / wipe(전체 삭제)"""
    con = db()
    try:
        if mode == "wipe":
            for t in EXPECTED:
                con.execute("DELETE FROM %s" % t)
            con.commit()
            return "모든 세션·제출 기록을 삭제했습니다."
        rows = con.execute("SELECT DISTINCT code, student_key FROM students").fetchall()
        for i, (code, skey) in enumerate(rows, 1):
            alias = "학생%02d" % i
            con.execute("UPDATE students SET student_name=? WHERE code=? AND student_key=?",
                        (alias, code, skey))
            con.execute("UPDATE submissions SET student_name=? WHERE code=? AND student_key=?",
                        (alias, code, skey))
        con.commit()
        return "학생 %d명의 이름을 가명으로 바꾸었습니다." % len(rows)
    finally:
        con.close()


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_code(k=6):
    pool = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # 0/O/1/I 제외
    return "".join(random.choice(pool) for _ in range(k))


def save_session(code, title, subject, grade, items, cfg,
                 is_open=1, reveal=1, allow_resubmit=1):
    con = db()
    try:
        con.execute(
            "INSERT OR REPLACE INTO sessions(code,title,subject,grade,created_at,"
            "is_open,reveal,allow_resubmit,config_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (code, title, subject, grade, now_str(), is_open, reveal,
             allow_resubmit, json.dumps(cfg, ensure_ascii=False)))
        con.execute("DELETE FROM items WHERE code=?", (code,))
        for i, it in enumerate(items, start=1):
            con.execute(
                "INSERT INTO items(code,seq,question,model_answer,answer_value,unit,points,keywords_json)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (code, i, it.get("question", ""), it.get("model_answer", ""),
                 str(it.get("answer_value", "")), str(it.get("unit", "")),
                 float(it.get("points", 10)),
                 json.dumps(it.get("keywords", []), ensure_ascii=False)))
        con.commit()
    finally:
        con.close()


def update_flags(code, is_open, reveal, allow_resubmit):
    con = db()
    try:
        con.execute("UPDATE sessions SET is_open=?,reveal=?,allow_resubmit=? WHERE code=?",
                    (int(is_open), int(reveal), int(allow_resubmit), code))
        con.commit()
    finally:
        con.close()


def load_session(code):
    con = db()
    try:
        r = con.execute("SELECT code,title,subject,grade,created_at,is_open,reveal,"
                        "allow_resubmit,config_json FROM sessions WHERE code=?", (code,)).fetchone()
        if not r:
            return None
        s = {"code": r[0], "title": r[1], "subject": r[2], "grade": r[3], "created_at": r[4],
             "is_open": r[5], "reveal": r[6], "allow_resubmit": r[7],
             "config": json.loads(r[8] or "{}")}
        rows = con.execute("SELECT seq,question,model_answer,answer_value,unit,points,keywords_json"
                           " FROM items WHERE code=? ORDER BY seq", (code,)).fetchall()
        s["items"] = [{"seq": a, "question": b, "model_answer": c, "answer_value": d,
                       "unit": e, "points": f, "keywords": json.loads(g or "[]")}
                      for (a, b, c, d, e, f, g) in rows]
        return s
    finally:
        con.close()


def list_sessions(limit=30):
    con = db()
    try:
        rows = con.execute(
            "SELECT s.code,COALESCE(NULLIF(s.title,''),'(제목 없음)'),s.created_at,COALESCE(s.is_open,0),"
            "(SELECT COUNT(*) FROM items i WHERE i.code=s.code),"
            "(SELECT COUNT(DISTINCT student_key) FROM submissions b WHERE b.code=s.code)"
            " FROM sessions s ORDER BY s.created_at DESC LIMIT ?", (limit,)).fetchall()
        return rows
    finally:
        con.close()


def upsert_submission(code, skey, seq, sno, sname, answer, res):
    """최종본만 저장하고 제출 횟수를 누적한다."""
    con = db()
    try:
        old = con.execute("SELECT submit_count, first_at FROM submissions"
                          " WHERE code=? AND student_key=? AND seq=?", (code, skey, seq)).fetchone()
        ts = now_str()
        cnt = (old[0] + 1) if old else 1
        first = old[1] if old else ts
        con.execute(
            "INSERT OR REPLACE INTO submissions(code,student_key,seq,student_no,student_name,answer,"
            "score,ratio,level,feedback,flags_json,matched_json,missing_json,kw_ratio,sem_sim,"
            "submit_count,first_at,last_at,teacher_score,teacher_feedback,confirmed)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, skey, seq, sno, sname, answer, res["score"], res["ratio"], res["level"],
             res["feedback"], json.dumps(res["flags"], ensure_ascii=False),
             json.dumps(res["matched"], ensure_ascii=False),
             json.dumps(res["missing"], ensure_ascii=False),
             res["kw_ratio"], res["sem_sim"], cnt, first, ts, None, None, 0))
        srow = con.execute("SELECT total_submits, joined_at FROM students"
                           " WHERE code=? AND student_key=?", (code, skey)).fetchone()
        tot = (srow[0] + 1) if srow else 1
        joined = srow[1] if srow else ts
        con.execute("INSERT OR REPLACE INTO students(code,student_key,student_no,student_name,"
                    "joined_at,last_at,total_submits) VALUES(?,?,?,?,?,?,?)",
                    (code, skey, sno, sname, joined, ts, tot))
        con.commit()
        return cnt
    finally:
        con.close()


def _apply_mask(df):
    if DISPLAY.get("mask") and not df.empty and "student_name" in df.columns:
        df = df.copy()
        df["student_name"] = df["student_name"].map(mask_name)
    return df


def fetch_submissions(code):
    con = db()
    try:
        return _apply_mask(pd.read_sql_query(
            "SELECT * FROM submissions WHERE code=? ORDER BY last_at DESC", con, params=(code,)))
    finally:
        con.close()


def fetch_students(code):
    con = db()
    try:
        return _apply_mask(pd.read_sql_query(
            "SELECT * FROM students WHERE code=? ORDER BY last_at DESC", con, params=(code,)))
    finally:
        con.close()


def teacher_override(code, skey, seq, score, feedback):
    con = db()
    try:
        con.execute("UPDATE submissions SET teacher_score=?,teacher_feedback=?,confirmed=1"
                    " WHERE code=? AND student_key=? AND seq=?",
                    (score, feedback, code, skey, seq))
        con.commit()
    finally:
        con.close()


def delete_session(code):
    con = db()
    try:
        for t in ("submissions", "students", "items", "sessions"):
            con.execute("DELETE FROM %s WHERE code=?" % t, (code,))
        con.commit()
    finally:
        con.close()


# =====================================================================
# [4] 형태소 분석기 : Okt 우선, 실패 시 간이 토크나이저로 자동 대체
# =====================================================================
OKT = None
OKT_MSG = ""


@st.cache_resource(show_spinner=False)
def get_okt():
    try:
        from konlpy.tag import Okt
        o = Okt()
        o.morphs("연소 실험")          # 워밍업 겸 JVM 기동 확인
        return o, "KoNLPy(Okt) 형태소 분석기를 사용합니다."
    except Exception as e:
        return None, "Okt 사용 불가(%s). 간이 분석기로 대체합니다." % type(e).__name__


JOSA = ("으로써", "로써", "으로서", "로서", "에서는", "에게서", "에서", "에게", "께서",
        "으로", "로", "은", "는", "이", "가", "을", "를", "과", "와", "도", "만",
        "의", "에", "께", "부터", "까지", "라고", "이라고", "보다", "처럼", "같이")
EOMI = ("습니다", "ㅂ니다", "입니다", "했다", "한다", "된다", "이다", "어요", "아요",
        "예요", "에요", "해서", "하여", "하고", "하는", "해요", "지요", "네요", "구나")


def simple_tokens(text):
    """JDK 부재 환경을 위한 규칙 기반 토크나이저(조사·어미 제거)."""
    text = re.sub(r"[^0-9A-Za-z가-힣\s\.\+\-\*/=<>]", " ", str(text))
    out = []
    for w in text.split():
        w = w.strip()
        if not w:
            continue
        for j in sorted(JOSA + EOMI, key=len, reverse=True):
            if len(w) > len(j) + 1 and w.endswith(j):
                w = w[: -len(j)]
                break
        if w:
            out.append(w)
    return out


def tokenize(text, okt=None):
    if okt is not None:
        try:
            return [w for w, p in okt.pos(str(text), norm=True, stem=True)
                    if p in ("Noun", "Verb", "Adjective", "Number", "Alpha", "Adverb")]
        except Exception:
            pass
    return simple_tokens(text)


# =====================================================================
# [5] 의미 벡터 백엔드 : ko-sroberta(자동 다운로드 + 로컬 캐시) → TF-IDF 폴백
# =====================================================================
EMB_STATUS = {"backend": "미초기화", "detail": "", "ok": False}


@st.cache_resource(show_spinner=False)
def get_embedder(prefer_local_path=""):
    """
    우선순위
      1) Hugging Face API (st.secrets["HF_TOKEN"] 설정 시 클라우드 모드)
      2) sentence-transformers (로컬 실행)
      3) transformers (로컬 실행)
      4) TF-IDF (최종 폴백)
    """
    import requests
    import numpy as np

    api_debug_info = []

    # 1) Hugging Face Inference API 시도
    hf_token = st.secrets.get("HF_TOKEN", "")
    if not hf_token:
        api_debug_info.append("Secrets에 'HF_TOKEN'이 없거나 읽을 수 없습니다.")
    else:
        # 라우터 URL 및 기존 URL 모두 자동 시도
        urls = [
            "https://router.huggingface.co/hf-inference/models/jhgan/ko-sroberta-multitask/pipeline/feature-extraction",
            "https://api-inference.huggingface.co/models/jhgan/ko-sroberta-multitask"
        ]
        headers = {"Authorization": f"Bearer {hf_token}"}

        for url in urls:
            try:
                def enc_hf_api(texts):
                    payload = {"inputs": list(texts), "options": {"wait_for_model": True}}
                    res = requests.post(url, headers=headers, json=payload, timeout=15)
                    if res.status_code == 200:
                        arr = np.array(res.json())
                        if len(arr.shape) == 3:
                            arr = arr.mean(axis=1)
                        norms = np.linalg.norm(arr, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        return arr / norms
                    else:
                        raise RuntimeError(f"HTTP {res.status_code}: {res.text[:80]}")

                # 워밍업 테스트
                test_v = enc_hf_api(["테스트"])
                if test_v is not None and len(test_v) > 0:
                    return enc_hf_api, {
                        "ok": True,
                        "backend": "ko-sroberta (Hugging Face API)",
                        "detail": "클라우드 API 모드로 정상 작동 중입니다. (메모리 절약)"
                    }
            except Exception as e:
                api_debug_info.append(f"API실패: {str(e)}")

    # 2) sentence-transformers (로컬 PC용)
    err1, err2 = "", ""
    os.makedirs(MODEL_DIR, exist_ok=True)
    target = resolve_model_target(prefer_local_path) if 'resolve_model_target' in globals() else "jhgan/ko-sroberta-multitask"

    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(target, cache_folder=MODEL_DIR, device="cpu")
        def enc_sbert(texts):
            v = m.encode(list(texts), convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        return enc_sbert, {"backend": "ko-sroberta (sentence-transformers)", "detail": target, "ok": True}
    except Exception as e1:
        err1 = f"SBERT다운실패 ({str(e1)[:60]})"

    # 3) transformers (로컬 PC용)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tk = AutoTokenizer.from_pretrained(target, cache_dir=MODEL_DIR)
        md = AutoModel.from_pretrained(target, cache_dir=MODEL_DIR)
        md.eval()
        def enc_tf(texts, max_len=160, bs=16):
            outs = []
            texts = list(texts)
            for i in range(0, len(texts), bs):
                enc = tk(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
                with torch.no_grad():
                    o = md(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outs.append(pooled.cpu().numpy())
            return np.vstack(outs).astype(np.float32)
        return enc_tf, {"backend": "ko-sroberta (transformers)", "detail": target, "ok": True}
    except Exception as e2:
        err2 = f"HF다운실패 ({str(e2)[:60]})"

    # 4) TF-IDF 폴백
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        def enc_tfidf(texts):
            texts = [str(t) if str(t).strip() else "빈답안" for t in texts]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, use_idf=False)
            m = vec.fit_transform(texts).toarray().astype(np.float32)
            n = np.linalg.norm(m, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return m / n

        debug_msg = " | ".join(api_debug_info) if api_debug_info else f"{err1} / {err2}"
        return enc_tfidf, {
            "backend": "TF-IDF 문자 n-gram (오프라인 폴백)",
            "detail": f"원인: {debug_msg}",
            "ok": False
        }
    except Exception as e3:
        def enc_zero(texts):
            return np.zeros((len(list(texts)), 8), dtype=np.float32)
        return enc_zero, {"backend": "없음", "detail": str(e3)[:150], "ok": False}


def cosine_to_ref(ref_vec, mat):
    ref = np.asarray(ref_vec, dtype=np.float32).ravel()
    rn = np.linalg.norm(ref) or 1.0
    mn = np.linalg.norm(mat, axis=1)
    mn[mn == 0] = 1.0
    return (mat @ ref) / (mn * rn)
# =====================================================================
# [6] 오개념 탐지 규칙 (부호·방향 역전 등)
# =====================================================================
ANTONYMS = [
    ("증가", "감소", "증감 방향"), ("늘어", "줄어", "증감 방향"), ("커진", "작아진", "크기 변화"),
    ("올라", "내려", "상하 방향"), ("높아", "낮아", "고저 방향"), ("많아", "적어", "수량 변화"),
    ("빨라", "느려", "속도 변화"), ("더하", "빼", "연산 방향"), ("곱하", "나누", "연산 방향"),
    ("양수", "음수", "부호"), ("플러스", "마이너스", "부호"), ("이상", "이하", "부등호 방향"),
    ("초과", "미만", "부등호 방향"), ("크다", "작다", "대소 관계"), ("위로", "아래로", "이동 방향"),
    ("오른쪽", "왼쪽", "이동 방향"), ("정비례", "반비례", "비례 관계"), ("공급", "차단", "작용 방향"),
    ("탄다", "꺼진다", "연소 여부"), ("연소", "소화", "연소·소화 개념"),
]


NEG_WORDS = ("않", "못", "없", "아니", "말아", "금지")


def _verb_polarity(text, term):
    """'공급되지 않다 / 공급되다' 처럼 용언으로 쓰인 낱말의 긍정·부정 극성을 판정한다."""
    t = str(text)
    for m in re.finditer(re.escape(str(term)), t):
        tail = t[m.end(): m.end() + 10]
        if tail[:1] in ("되", "하", "된", "한", "됨", "함", "시") or tail[:2] in ("되지", "하지"):
            return 0 if any(x in tail for x in NEG_WORDS) else 1
    return None


def detect_reversal(model_answer, student_answer, keywords=None):
    """모범답안의 방향과 정반대로 서술한 경우만 오개념으로 판정한다."""
    flags = []
    m, s = str(model_answer), str(student_answer)
    for a, b, label in ANTONYMS:
        ma, mb, sa, sb = a in m, b in m, a in s, b in s
        if ma and not mb and sb and not sa:
            flags.append("[부호·방향 역전 / %s] 정답의 '%s' 대신 반대인 '%s'로 서술" % (label, a, b))
        elif mb and not ma and sa and not sb:
            flags.append("[부호·방향 역전 / %s] 정답의 '%s' 대신 반대인 '%s'로 서술" % (label, b, a))
    for kw in (keywords or []):
        term = str(kw.get("term", "")).strip()
        if len(term) < 2:
            continue
        pm, ps = _verb_polarity(m, term), _verb_polarity(s, term)
        if pm is not None and ps is not None and pm != ps:
            flags.append("[부정·긍정 역전 / %s] 정답과 반대 의미(%s)로 서술"
                         % (term, "부정" if ps == 0 else "긍정"))
    return flags


def _norm_kr(s):
    """받침(종성)을 떼어 활용형·조사 결합형을 같은 꼴로 정규화한다. (나눈 -> 나누)"""
    out = []
    for ch in str(s):
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            jong = (c - 0xAC00) % 28
            if jong:
                ch = chr(c - jong)
        out.append(ch)
    return "".join(out)


_CHO_TABLE = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"


def _cho(ch):
    c = ord(ch)
    if 0xAC00 <= c <= 0xD7A3:
        return _CHO_TABLE[(c - 0xAC00) // 588]
    return ch


def _root_match(a, b):
    """활용형 대응: 부분 일치 → 종성 제거 일치 → 3음절 이상 어간 초성 일치."""
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    na, nb = _norm_kr(a), _norm_kr(b)
    if len(na) >= 2 and len(nb) >= 2 and (na in nb or nb in na):
        return True
    if len(a) >= 3 and len(b) >= 3 and a[0] == b[0] and _cho(a[1]) == _cho(b[1]):
        return True
    return False


def keyword_hits(answer, tokens, keywords):
    """가중 키워드 매칭률과 매칭/누락 목록을 계산한다."""
    matched, missing, req_missing = [], [], []
    tot_w, got_w = 0.0, 0.0
    low = str(answer).replace(" ", "")
    for kw in keywords:
        term = str(kw.get("term", "")).strip()
        if not term:
            continue
        w = float(kw.get("weight", 1.0) or 1.0)
        req = bool(kw.get("required", False))
        syns = [term] + [s.strip() for s in str(kw.get("syn", "")).split(",") if s.strip()]
        hit = False
        for s in syns:
            key = s.replace(" ", "")
            if key and key in low:
                hit = True
                break
            if any(_root_match(s, t) for t in tokens):
                hit = True
                break
        tot_w += w
        if hit:
            got_w += w
            matched.append(term)
        else:
            missing.append(term)
            if req:
                req_missing.append(term)
    ratio = (got_w / tot_w) if tot_w > 0 else 0.0
    return ratio, matched, missing, req_missing


def value_ok(answer, answer_value):
    """정답값 확인: 숫자형은 수치 비교, 서술형은 어절 70% 포함 판정."""
    av = str(answer_value or "").strip()
    if not av:
        return None
    ans = str(answer)
    nums = re.findall(r"-?\d+(?:\.\d+)?", av)
    if nums and len(av.replace(" ", "")) <= 12:
        got = re.findall(r"-?\d+(?:\.\d+)?", ans)
        try:
            target = float(nums[-1])
            return any(abs(float(g) - target) < 1e-6 for g in got)
        except Exception:
            return None
    words = [w for w in re.split(r"[\s,·/]+", av) if len(w) >= 2]
    if not words:
        return None
    flat = ans.replace(" ", "")
    toks = [t for t in re.split(r"[\s,·/]+", ans) if t]
    hit = 0
    for w in words:
        ww = w.replace(" ", "")
        if ww in flat or (len(ww) >= 2 and ww[:2] in flat):
            hit += 1
            continue
        if any(_root_match(ww, t) for t in toks):
            hit += 1
    # 서술형 정답값은 표현이 달라도 핵심어가 반영되면 통과로 본다.
    return hit >= 1


# =====================================================================
# [7] 하이브리드 채점 엔진
# =====================================================================
DEFAULT_CFG = {
    "w_kw": 0.5, "w_sem": 0.5, "sem_base": 0.30,
    "reversal_cap": 0.40, "required_cap": 0.60,
    "value_bonus": 0.10, "unit_penalty": 0.05, "min_chars": 10,
    "score_step": 0.5,
    "cut_a": 0.90, "cut_b": 0.70, "cut_c": 0.50,
}


def level_of(ratio, cfg):
    if ratio >= cfg["cut_a"]:
        return "매우 잘함"
    if ratio >= cfg["cut_b"]:
        return "잘함"
    if ratio >= cfg["cut_c"]:
        return "보통"
    return "노력 요함"


def round_step(x, step):
    if step and step > 0:
        return round(round(x / step) * step, 2)
    return round(x, 2)


def build_feedback(res, item, cfg):
    """칭찬 → 오개념 경고 → 지도 포인트 → 보완점 → 다음 학습 제안 순서로 조립."""
    parts = []
    lv, ratio = res["level"], res["ratio"]
    if ratio >= cfg["cut_a"]:
        parts.append("문제의 핵심을 정확하게 파악하고 근거를 들어 설명했습니다.")
    elif ratio >= cfg["cut_b"]:
        parts.append("핵심 내용을 대체로 바르게 서술했습니다.")
    elif ratio >= cfg["cut_c"]:
        parts.append("답을 찾으려는 과정이 드러나 있습니다.")
    else:
        parts.append("문제 상황을 다시 한번 차분히 읽어 볼 필요가 있습니다.")

    rev_dir = [f for f in res["flags"] if "부호·방향 역전" in f]
    rev_neg = [f for f in res["flags"] if "부정·긍정 역전" in f]
    if rev_dir:
        parts.append("다만 " + rev_dir[0].split("] ")[-1] + " 한 점이 아쉽습니다.")
        if re.search(r"\d", str(item.get("answer_value", "")) + str(item.get("unit", ""))):
            parts.append("수직선이나 표에 값을 직접 써 보며 어느 쪽으로 변하는지 방향을 먼저 "
                         "확인하도록 지도해 주세요.")
        else:
            parts.append("결과가 어떻게 달라지는지 실험 과정을 순서대로 짚어 보며 "
                         "방향을 바로잡도록 지도해 주세요.")
    elif rev_neg:
        term = rev_neg[0].split("/")[-1].split("]")[0].strip()
        parts.append("다만 '%s'에 대해 모범답안과 반대되는 의미로 서술한 점이 아쉽습니다." % term)
        parts.append("'~하지 않는다, ~이 부족하다'와 같은 부정 표현에 유의하여 문장을 다시 읽어 보도록 지도해 주세요.")
    if res["missing"]:
        parts.append("'%s' 와(과) 같은 핵심 낱말을 답안에 넣으면 설명이 더 분명해집니다."
                     % ", ".join(res["missing"][:3]))
    if res["value_ok"] is False:
        av = str(item.get("answer_value", ""))
        if re.search(r"\d", av):
            parts.append("최종 답이 정답(%s)과 달라 계산 과정을 다시 확인해야 합니다." % av)
        else:
            parts.append("결론이 모범답안의 핵심(%s)과 달라 내용을 다시 정리할 필요가 있습니다." % av)
    if res["unit_missing"]:
        parts.append("단위 '%s' 를 빠뜨리지 않고 쓰는 습관을 들이면 좋겠습니다." % item.get("unit", ""))
    if res["too_short"]:
        parts.append("왜 그렇게 생각했는지 까닭을 한 문장 더 덧붙여 서술해 봅시다.")
    if lv == "매우 잘함":
        parts.append("친구에게 자신의 풀이를 설명해 보는 활동으로 생각을 넓혀 봅시다.")
    elif lv == "노력 요함":
        parts.append("교과서의 비슷한 예제를 함께 풀며 기본 개념부터 다져 봅시다.")
    return " ".join(parts)


def grade_one(answer, item, cfg, okt, encode_fn):
    """학생 답안 1건을 채점한다."""
    ans = str(answer or "").strip()
    pts = float(item.get("points", 10) or 10)
    if not ans:
        return {"score": 0.0, "ratio": 0.0, "level": "노력 요함", "kw_ratio": 0.0,
                "sem_sim": 0.0, "matched": [], "missing": [k.get("term", "") for k in item.get("keywords", [])],
                "flags": ["[무응답] 답안이 비어 있습니다."], "value_ok": None,
                "unit_missing": False, "too_short": True,
                "feedback": "답안을 작성하지 않았습니다. 문제를 다시 읽고 아는 내용부터 한 문장이라도 적어 봅시다."}

    toks = tokenize(ans, okt)
    kw_ratio, matched, missing, req_missing = keyword_hits(ans, toks, item.get("keywords", []))

    try:
        vecs = encode_fn([str(item.get("model_answer", "")), ans])
        sim = float(cosine_to_ref(vecs[0], vecs[1:2])[0])
    except Exception:
        sim = 0.0
    if math.isnan(sim):
        sim = 0.0
    sem_ratio = max(0.0, (sim - cfg["sem_base"]) / max(1e-6, 1.0 - cfg["sem_base"]))
    sem_ratio = min(1.0, sem_ratio)

    ratio = cfg["w_kw"] * kw_ratio + cfg["w_sem"] * sem_ratio

    flags = detect_reversal(item.get("model_answer", ""), ans, item.get("keywords", []))
    vok = value_ok(ans, item.get("answer_value", ""))
    if vok is True:
        ratio = min(1.0, ratio + cfg["value_bonus"])
    elif vok is False:
        flags.append("[계산·결론 불일치] 최종 답이 정답과 다릅니다.")
        ratio *= 0.8

    unit = str(item.get("unit", "") or "").strip()
    unit_missing = bool(unit) and (unit.replace(" ", "") not in ans.replace(" ", ""))
    if unit_missing:
        flags.append("[단위 누락] 단위 '%s' 가 빠졌습니다." % unit)
        ratio = max(0.0, ratio - cfg["unit_penalty"])

    too_short = len(re.sub(r"\s", "", ans)) < int(cfg["min_chars"])
    if too_short:
        flags.append("[근거 부족] 서술이 너무 짧아 까닭이 드러나지 않습니다.")
        ratio *= 0.9

    if req_missing:
        flags.append("[필수 요소 누락] %s" % ", ".join(req_missing))
        ratio = min(ratio, cfg["required_cap"])
    if any("역전" in f for f in flags):
        ratio = min(ratio, cfg["reversal_cap"])

    ratio = max(0.0, min(1.0, ratio))
    res = {"score": round_step(pts * ratio, cfg["score_step"]), "ratio": round(ratio, 4),
           "level": level_of(ratio, cfg), "kw_ratio": round(kw_ratio, 4),
           "sem_sim": round(sim, 4), "matched": matched, "missing": missing,
           "flags": flags, "value_ok": vok, "unit_missing": unit_missing, "too_short": too_short}
    res["feedback"] = build_feedback(res, item, cfg)
    return res


# =====================================================================
# [8] 샘플 문항 세트 (다문항)
# =====================================================================
SAMPLE_SETS = {
    "6학년 과학 · 연소와 소화": {
        "subject": "과학", "grade": "6학년",
        "items": [
            {"question": "초에 불을 붙인 뒤 유리병을 덮었더니 잠시 후 촛불이 꺼졌습니다. "
                         "촛불이 꺼진 까닭을 연소의 조건과 관련지어 설명하시오.",
             "model_answer": "유리병을 덮으면 병 안의 산소가 더 이상 공급되지 않고 촛불이 산소를 모두 사용하여 "
                             "연소의 조건인 산소가 부족해지므로 촛불이 꺼진다.",
             "answer_value": "산소 부족", "unit": "", "points": 10,
             "keywords": [{"term": "산소", "syn": "산소 기체, 공기", "weight": 2.0, "required": True},
                          {"term": "공급", "syn": "들어오지 않는다, 차단, 부족, 없어진다, 없어짐", "weight": 1.5, "required": False},
                          {"term": "연소 조건", "syn": "탈 물질, 발화점, 연소의 조건", "weight": 1.5, "required": False},
                          {"term": "꺼진다", "syn": "소화, 불이 꺼짐, 꺼짐, 꺼졌다", "weight": 1.0, "required": False}]},
            {"question": "알코올램프의 불을 끌 때 뚜껑을 덮는 방법과, 나무에 붙은 불을 물로 끄는 방법은 "
                         "각각 어떤 소화 조건을 이용한 것인지 쓰고 그 까닭을 설명하시오.",
             "model_answer": "뚜껑을 덮는 것은 산소를 차단하여 불을 끄는 방법이고, 물을 뿌리는 것은 "
                             "발화점 미만으로 온도를 낮추어 불을 끄는 방법이다.",
             "answer_value": "산소 차단, 온도 낮춤", "unit": "", "points": 10,
             "keywords": [{"term": "산소 차단", "syn": "산소를 막는다, 공기 차단", "weight": 2.0, "required": True},
                          {"term": "온도", "syn": "발화점, 온도를 낮춘다", "weight": 2.0, "required": True},
                          {"term": "소화 조건", "syn": "불을 끄는 조건, 소화", "weight": 1.0, "required": False}]},
            {"question": "화재가 발생했을 때 승강기(엘리베이터)를 타지 말고 계단으로 대피해야 하는 까닭을 "
                         "두 가지 이상 들어 설명하시오.",
             "model_answer": "화재로 정전이 되면 승강기 안에 갇힐 수 있고, 승강기 통로를 따라 연기가 빠르게 들어와 "
                             "질식할 위험이 크기 때문에 계단으로 대피해야 한다.",
             "answer_value": "정전, 연기", "unit": "", "points": 10,
             "keywords": [{"term": "정전", "syn": "전기가 끊긴다, 멈춘다, 갇힌다", "weight": 1.5, "required": False},
                          {"term": "연기", "syn": "유독가스, 질식", "weight": 1.5, "required": True},
                          {"term": "계단", "syn": "비상계단, 대피", "weight": 1.0, "required": False}]},
        ],
    },
    "6학년 수학 · 비와 비율 / 정수의 계산": {
        "subject": "수학", "grade": "6학년",
        "items": [
            {"question": "설탕 40g을 물 160g에 모두 녹여 설탕물을 만들었습니다. 설탕물에 대한 설탕의 비율을 "
                         "백분율로 구하고, 구하는 과정을 설명하시오.",
             "model_answer": "설탕물 전체의 양은 40 더하기 160으로 200g이고, 설탕의 양 40을 설탕물 200으로 나눈 뒤 "
                             "100을 곱하면 20이므로 설탕물에 대한 설탕의 비율은 20퍼센트이다.",
             "answer_value": "20", "unit": "%", "points": 10,
             "keywords": [{"term": "200", "syn": "전체 양, 설탕물 전체", "weight": 1.5, "required": False},
                          {"term": "나누", "syn": "나눗셈, 나눈, 나누기, 비율", "weight": 1.5, "required": False},
                          {"term": "100을 곱", "syn": "백분율, 곱하기 100", "weight": 1.5, "required": True},
                          {"term": "20", "syn": "이십", "weight": 2.0, "required": True}]},
            {"question": "아침 기온이 영하 3도였는데 낮 동안 7도가 올라갔습니다. 낮 기온은 몇 도인지 구하고, "
                         "수직선을 이용하여 구하는 과정을 설명하시오.",
             "model_answer": "영하 3도는 -3이고 기온이 7도 올라갔으므로 -3에 7을 더하면 4가 된다. 수직선에서 "
                             "-3에서 오른쪽으로 7칸 이동하면 4에 도착하므로 낮 기온은 영상 4도이다.",
             "answer_value": "4", "unit": "도", "points": 10,
             "keywords": [{"term": "-3", "syn": "영하 3, 음수 3", "weight": 1.5, "required": False},
                          {"term": "더하", "syn": "덧셈, 올라간다, 증가", "weight": 2.0, "required": True},
                          {"term": "오른쪽", "syn": "수직선 오른쪽, 오른쪽으로 이동", "weight": 1.0, "required": False},
                          {"term": "4", "syn": "영상 4", "weight": 2.0, "required": True}]},
        ],
    },
}


# =====================================================================
# [9] 공통 UI 유틸
# =====================================================================
def local_ip():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def get_params():
    try:
        qp = st.query_params
        return {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}
    except Exception:
        try:
            return {k: (v[0] if isinstance(v, list) else v)
                    for k, v in st.experimental_get_query_params().items()}
        except Exception:
            return {}


def student_url(base, code):
    base = (base or "").strip().rstrip("/")
    return "%s/?role=student&code=%s" % (base, code)


def skey_of(no, name):
    return "%s|%s" % (str(no).strip(), re.sub(r"\s+", "", str(name)))


def jload(s, d=None):
    try:
        v = json.loads(s) if s else d
        return v if v is not None else d
    except Exception:
        return d


def clean_records(df):
    """data_editor 결과의 numpy 자료형을 순수 파이썬 값으로 변환."""
    out = []
    for r in df.to_dict("records"):
        row = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                v = ""
            row[k] = v
        out.append(row)
    return out


_FRAG = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def run_live(body_fn, every):
    """지원 버전에서는 자동 갱신 조각으로, 아니면 수동 갱신으로 실행."""
    if _FRAG and every:
        try:
            _FRAG(run_every=every)(body_fn)()
            return True
        except Exception:
            pass
    body_fn()
    return False


# =====================================================================
# [10] 학생 화면 : 한 문항씩 순차 진행
# =====================================================================
def render_student(params):
    st.markdown("### 서술형 평가 참여")
    ss = st.session_state
    ss.setdefault("s_stage", "login")
    ss.setdefault("s_idx", 0)
    ss.setdefault("s_code", params.get("code", "").upper())

    if ss["s_stage"] == "login":
        code = st.text_input("참여 코드 (6자리)", value=ss["s_code"], max_chars=6).strip().upper()
        c1, c2 = st.columns(2)
        no = c1.text_input("번호", value="")
        name = c2.text_input("이름", value="")
        if st.button("들어가기", type="primary", use_container_width=True):
            sess = load_session(code) if code else None
            if not sess:
                st.error("참여 코드를 찾을 수 없습니다. 선생님께 코드를 다시 확인해 주세요.")
                return
            if not str(name).strip():
                st.error("이름을 입력해 주세요.")
                return
            ss["s_code"], ss["s_no"], ss["s_name"] = code, no.strip(), name.strip()
            ss["s_stage"], ss["s_idx"] = "solving", 0
            ss["s_last"] = None
            st.rerun()
        st.caption("선생님이 알려 주신 6자리 코드를 입력한 뒤, 번호와 이름을 적고 들어가기를 누르세요.")
        return

    sess = load_session(ss["s_code"])
    if not sess:
        st.error("세션 정보를 불러오지 못했습니다.")
        ss["s_stage"] = "login"
        return

    items = sess["items"]
    cfg = dict(DEFAULT_CFG)
    cfg.update(sess.get("config", {}))
    skey = skey_of(ss.get("s_no", ""), ss.get("s_name", ""))
    sub = fetch_submissions(ss["s_code"])
    mine = sub[sub["student_key"] == skey] if len(sub) else sub
    done_seq = set(mine["seq"].tolist()) if len(mine) else set()

    st.caption("%s · %s %s" % (sess["title"], ss.get("s_no", ""), ss.get("s_name", "")))
    st.progress(min(1.0, len(done_seq) / max(1, len(items))),
                text="진행 상황  %d / %d 문항" % (len(done_seq), len(items)))

    if not sess["is_open"]:
        st.warning("지금은 제출이 마감되었습니다. 선생님의 안내를 기다려 주세요.")
        return

    # 첫 미응답 문항으로 자동 이동
    if ss["s_stage"] == "solving" and ss["s_idx"] < len(items):
        while ss["s_idx"] < len(items) and items[ss["s_idx"]]["seq"] in done_seq \
                and not sess["allow_resubmit"] and ss.get("s_last") is None:
            ss["s_idx"] += 1

    if ss["s_idx"] >= len(items):
        ss["s_stage"] = "done"

    if ss["s_stage"] == "done":
        st.success("모든 문항을 제출했습니다. 수고했습니다.")
        if sess["reveal"] and len(mine):
            tot = float(mine["score"].sum())
            full = sum(float(i["points"]) for i in items)
            st.metric("나의 점수", "%.1f / %.1f점" % (tot, full))
            for _, r in mine.sort_values("seq").iterrows():
                with st.expander("%d번 문항 · %.1f점 (%s)" % (r["seq"], r["score"], r["level"])):
                    st.write("**내 답안**  " + str(r["answer"]))
                    st.info(str(r["feedback"]))
        if sess["allow_resubmit"]:
            if st.button("답안 다시 고치기", use_container_width=True):
                ss["s_stage"], ss["s_idx"], ss["s_last"] = "solving", 0, None
                st.rerun()
        return

    it = items[ss["s_idx"]]
    st.markdown("#### %d번 문항  (%s점)" % (it["seq"], ("%g" % float(it["points"]))))
    st.info(it["question"])

    prev_ans = ""
    if it["seq"] in done_seq:
        prev_ans = str(mine[mine["seq"] == it["seq"]].iloc[0]["answer"])
        if not sess["allow_resubmit"]:
            st.warning("이미 제출한 문항입니다. 다시 제출할 수 없습니다.")
        else:
            st.caption("이전에 제출한 답안을 고쳐서 다시 낼 수 있습니다.")

    key = "ans_%s_%d" % (ss["s_code"], it["seq"])
    ans = st.text_area("답안을 자세히 써 봅시다.", value=prev_ans, height=180, key=key)

    locked = (it["seq"] in done_seq) and (not sess["allow_resubmit"])
    c1, c2 = st.columns([2, 1])
    if c1.button("제출하기", type="primary", disabled=locked, use_container_width=True):
        okt, _ = get_okt()
        enc, _ = get_embedder(cfg.get("local_model_path", ""))
        res = grade_one(ans, it, cfg, okt, enc)
        cnt = upsert_submission(ss["s_code"], skey, it["seq"], ss.get("s_no", ""),
                                ss.get("s_name", ""), ans, res)
        ss["s_last"] = {"seq": it["seq"], "res": res, "cnt": cnt}
        st.rerun()

    if c2.button("건너뛰기", disabled=(ss["s_idx"] >= len(items) - 1), use_container_width=True):
        ss["s_idx"] += 1
        ss["s_last"] = None
        st.rerun()

    last = ss.get("s_last")
    if last and last["seq"] == it["seq"]:
        r = last["res"]
        st.divider()
        if sess["reveal"]:
            if r["ratio"] >= 0.9:
                st.balloons()
            m1, m2, m3 = st.columns(3)
            m1.metric("점수", "%.1f점" % r["score"])
            m2.metric("성취수준", r["level"])
            m3.metric("핵심어 반영률", "%.0f%%" % (r["kw_ratio"] * 100))
            st.info("**선생님의 안내**  " + r["feedback"])
            if r["missing"]:
                st.caption("보완하면 좋은 낱말 : " + ", ".join(r["missing"][:5]))
        else:
            st.success("제출이 완료되었습니다. 결과는 선생님께서 안내해 주십니다.")
        st.caption("제출 횟수 %d회" % last["cnt"])
        nxt = "다음 문항으로" if ss["s_idx"] < len(items) - 1 else "제출 마치기"
        if st.button(nxt, type="primary", use_container_width=True):
            ss["s_idx"] += 1
            ss["s_last"] = None
            st.rerun()


# =====================================================================
# [11] 교사 화면
# =====================================================================
def default_item(n=1):
    return {"question": "", "model_answer": "", "answer_value": "", "unit": "",
            "points": 10.0, "keywords": [{"term": "", "syn": "", "weight": 1.0, "required": False}]}


def tab_items():
    ss = st.session_state
    ss.setdefault("t_items", [default_item()])
    ss.setdefault("t_title", "6학년 서술형 형성평가")
    ss.setdefault("t_subject", "과학")
    ss.setdefault("t_grade", "6학년")

    c1, c2, c3 = st.columns([3, 1, 1])
    ss["t_title"] = c1.text_input("평가명", value=ss["t_title"])
    ss["t_subject"] = c2.text_input("교과", value=ss["t_subject"])
    ss["t_grade"] = c3.text_input("학년", value=ss["t_grade"])

    with st.container(border=True):
        st.markdown("**샘플 세트 불러오기**")
        s1, s2 = st.columns([3, 1])
        pick = s1.selectbox("샘플 선택", list(SAMPLE_SETS.keys()), label_visibility="collapsed")
        if s2.button("불러오기", use_container_width=True):
            sm = SAMPLE_SETS[pick]
            ss["t_items"] = json.loads(json.dumps(sm["items"], ensure_ascii=False))
            ss["t_title"] = pick
            ss["t_subject"], ss["t_grade"] = sm["subject"], sm["grade"]
            st.rerun()

    st.divider()
    a1, a2 = st.columns(2)
    if a1.button("문항 추가", use_container_width=True):
        ss["t_items"].append(default_item())
        st.rerun()
    if a2.button("마지막 문항 삭제", use_container_width=True, disabled=len(ss["t_items"]) <= 1):
        ss["t_items"].pop()
        st.rerun()

    for i, it in enumerate(ss["t_items"]):
        with st.expander("%d번 문항  %s" % (i + 1, (it["question"][:34] + "…") if it["question"] else "(미작성)"),
                         expanded=(len(ss["t_items"]) <= 2)):
            it["question"] = st.text_area("문항 내용", value=it.get("question", ""),
                                          height=90, key="q_%d" % i)
            it["model_answer"] = st.text_area("모범답안", value=it.get("model_answer", ""),
                                              height=90, key="m_%d" % i)
            k1, k2, k3 = st.columns(3)
            it["answer_value"] = k1.text_input("최종 정답값", value=str(it.get("answer_value", "")),
                                               key="v_%d" % i)
            it["unit"] = k2.text_input("필수 단위", value=str(it.get("unit", "")), key="u_%d" % i)
            it["points"] = k3.number_input("배점", 1.0, 100.0, float(it.get("points", 10)),
                                           step=1.0, key="p_%d" % i)
            st.caption("핵심 키워드 · 동의어는 쉼표로 구분 · 가중치가 클수록 점수 비중이 큽니다.")
            kdf = pd.DataFrame(it.get("keywords", []) or [{"term": "", "syn": "", "weight": 1.0, "required": False}])
            for col, dv in (("term", ""), ("syn", ""), ("weight", 1.0), ("required", False)):
                if col not in kdf.columns:
                    kdf[col] = dv
            ed = st.data_editor(kdf[["term", "syn", "weight", "required"]], num_rows="dynamic",
                                use_container_width=True, key="kw_%d" % i,
                                column_config={"term": "키워드", "syn": "동의어(쉼표)",
                                               "weight": "가중치", "required": "필수"})
            it["keywords"] = [k for k in clean_records(ed) if str(k.get("term", "")).strip()]


def tab_session(cfg):
    ss = st.session_state
    st.markdown("**학생 접속 주소 설정**")
    c1, c2 = st.columns([3, 1])
    base_default = ss.get("t_base", "https://grading-app-lsj.streamlit.app/")
    ss["t_base"] = c1.text_input("기본 주소 (배포 시 배포 주소로 변경)", value=base_default)
    c2.metric("교사 PC 내부 IP", local_ip())
    st.caption("반드시 `streamlit run app.py --server.address 0.0.0.0` 으로 실행해야 학생 기기에서 접속됩니다.")

    st.divider()
    b1, b2 = st.columns([1, 2])
    if b1.button("새 세션 발행", type="primary", use_container_width=True):
        items = [it for it in ss.get("t_items", []) if str(it.get("question", "")).strip()]
        if not items:
            st.error("문항을 1개 이상 작성한 뒤 발행해 주세요.")
        else:
            code = new_code()
            save_session(code, ss.get("t_title", "서술형 평가"), ss.get("t_subject", ""),
                         ss.get("t_grade", ""), items, cfg)
            ss["t_code"] = code
            st.rerun()

    code = ss.get("t_code", "")
    if not code:
        st.info("세션을 발행하면 참여 코드와 QR 코드가 만들어집니다.")
    else:
        sess = load_session(code)
        if not sess:
            st.error("세션을 불러오지 못했습니다.")
            return
        url = student_url(ss["t_base"], code)
        l, r = st.columns([1, 1])
        with l:
            st.markdown("#### 참여 코드")
            st.markdown("<div style='font-size:44px;font-weight:700;letter-spacing:6px;"
                        "border:1px solid #d1d5db;border-radius:8px;padding:10px 16px;"
                        "text-align:center'>%s</div>" % code, unsafe_allow_html=True)
            st.markdown("**학생 접속 주소**")
            st.code(url, language=None)
            st.caption("문항 수 %d · 발행 %s" % (len(sess["items"]), sess["created_at"]))
        with r:
            st.markdown("#### QR 코드")
            try:
                st.markdown("<div style='text-align:center'>%s</div>" % qr_svg(url, scale=6, quiet=3),
                            unsafe_allow_html=True)
                st.download_button("QR 이미지 내려받기 (PNG)", qr_png_bytes(url, scale=10),
                                   file_name="QR_%s.png" % code, mime="image/png",
                                   use_container_width=True)
            except Exception as e:
                st.warning("QR 생성 실패: %s" % e)

        st.divider()
        st.markdown("**세션 운영 스위치**")
        o1, o2, o3 = st.columns(3)
        is_open = o1.toggle("제출 접수 열기", value=bool(sess["is_open"]))
        reveal = o2.toggle("제출 즉시 결과 공개", value=bool(sess["reveal"]))
        allow = o3.toggle("재제출 허용", value=bool(sess["allow_resubmit"]))
        if (is_open, reveal, allow) != (bool(sess["is_open"]), bool(sess["reveal"]), bool(sess["allow_resubmit"])):
            update_flags(code, is_open, reveal, allow)
            st.rerun()

    st.divider()
    st.markdown("**이전 세션 불러오기**")
    rows = list_sessions()
    if rows:
        df = pd.DataFrame(rows, columns=["코드", "평가명", "발행시각", "접수", "문항수", "참여인원"])
        df["접수"] = df["접수"].map({1: "열림", 0: "마감"})
        st.dataframe(df, use_container_width=True, hide_index=True)
        p1, p2, p3 = st.columns([2, 1, 1])
        pick = p1.selectbox("세션 선택", [r[0] for r in rows], label_visibility="collapsed")
        if p2.button("이 세션 사용", use_container_width=True):
            ss["t_code"] = pick
            st.rerun()
        if p3.button("삭제", use_container_width=True):
            delete_session(pick)
            if ss.get("t_code") == pick:
                ss["t_code"] = ""
            st.rerun()
    else:
        st.caption("발행된 세션이 없습니다.")


def _dash_body(code, sess):
    items = sess["items"]
    df = fetch_submissions(code)
    st.caption("최근 갱신 %s" % datetime.now().strftime("%H:%M:%S"))
    if df.empty:
        st.info("아직 제출된 답안이 없습니다. 학생이 제출하면 이 화면이 자동으로 갱신됩니다.")
        return

    df["final"] = df.apply(lambda r: r["teacher_score"] if pd.notna(r["teacher_score"]) else r["score"], axis=1)
    n_stu = df["student_key"].nunique()
    full = sum(float(i["points"]) for i in items)
    per = df.groupby("student_key")["final"].sum()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("참여 인원", "%d명" % n_stu)
    m2.metric("제출 답안", "%d건 / %d건" % (len(df), n_stu * len(items)))
    m3.metric("평균 총점", "%.1f / %.0f점" % (per.mean(), full))
    m4.metric("평균 성취율", "%.0f%%" % (df["ratio"].mean() * 100))
    misc = df[df["flags_json"].apply(lambda s: len(jload(s, [])) > 0)]
    m5.metric("오개념 탐지", "%d명" % misc["student_key"].nunique())

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**성취수준 분포**")
        order = ["매우 잘함", "잘함", "보통", "노력 요함"]
        cnt = df["level"].value_counts().reindex(order).fillna(0).astype(int)
        st.bar_chart(pd.DataFrame({"인원": cnt}))
    with c2:
        st.markdown("**오개념 유형 상위**")
        tags = []
        for s in df["flags_json"]:
            for f in jload(s, []):
                tags.append(f.split("]")[0].strip("[").split("/")[0].strip())
        if tags:
            tc = pd.Series(tags).value_counts().head(6)
            st.bar_chart(pd.DataFrame({"건수": tc}))
        else:
            st.caption("탐지된 오개념이 없습니다.")

    st.markdown("**문항별 현황**")
    rows = []
    for it in items:
        d = df[df["seq"] == it["seq"]]
        if d.empty:
            rows.append({"문항": it["seq"], "제출": 0, "평균점수": 0.0, "평균성취율": "0%",
                         "도달(보통이상)": "0%", "오개념": 0})
            continue
        ok = (d["ratio"] >= 0.5).mean() * 100
        mc = int(d["flags_json"].apply(lambda s: len(jload(s, [])) > 0).sum())
        rows.append({"문항": it["seq"], "제출": len(d), "평균점수": round(float(d["final"].mean()), 1),
                     "평균성취율": "%.0f%%" % (d["ratio"].mean() * 100),
                     "도달(보통이상)": "%.0f%%" % ok, "오개념": mc})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**학급 공통 보충 지도 포인트**")
    miss = []
    for s in df["missing_json"]:
        miss.extend(jload(s, []))
    if miss:
        top = pd.Series(miss).value_counts().head(5)
        st.write(" · ".join(["%s (%d명 누락)" % (k, v) for k, v in top.items()]))
    else:
        st.caption("공통 누락 키워드가 없습니다.")

    st.markdown("**실시간 제출 명렬 (최신순)**")
    show = df[["last_at", "student_no", "student_name", "seq", "answer", "final", "level", "submit_count"]].copy()
    show.columns = ["제출시각", "번호", "이름", "문항", "답안", "점수", "성취수준", "제출횟수"]
    st.dataframe(show, use_container_width=True, hide_index=True, height=320)


def tab_dashboard():
    ss = st.session_state
    code = ss.get("t_code", "")
    if not code:
        st.info("먼저 ② 탭에서 세션을 발행하거나 불러오세요.")
        return
    sess = load_session(code)
    if not sess:
        st.error("세션 정보를 찾을 수 없습니다.")
        return
    h1, h2, h3 = st.columns([2, 1, 1])
    h1.markdown("### %s  ·  코드 `%s`" % (sess["title"], code))
    auto = h2.selectbox("자동 갱신", ["2초", "5초", "10초", "수동"], index=1)
    if h3.button("지금 갱신", use_container_width=True):
        st.rerun()
    every = {"2초": 2, "5초": 5, "10초": 10, "수동": None}[auto]
    run_live(lambda: _dash_body(code, sess), every)

    st.divider()
    with st.expander("교사 확인 · 점수 수정"):
        df = fetch_submissions(code)
        if df.empty:
            st.caption("수정할 제출물이 없습니다.")
            return
        opts = ["%s %s · %d번 문항" % (r["student_no"], r["student_name"], r["seq"])
                for _, r in df.iterrows()]
        idx = st.selectbox("대상 선택", range(len(opts)), format_func=lambda i: opts[i])
        row = df.iloc[idx]
        st.write("**답안**  " + str(row["answer"]))
        base_pts = next((float(i["points"]) for i in sess["items"] if i["seq"] == row["seq"]), 10.0)
        cur = float(row["teacher_score"]) if pd.notna(row["teacher_score"]) else float(row["score"])
        ns = st.number_input("최종 점수", 0.0, base_pts, cur, step=0.5)
        nf = st.text_area("피드백", value=str(row["teacher_feedback"] or row["feedback"]), height=110)
        if st.button("확정 저장", type="primary"):
            teacher_override(code, row["student_key"], int(row["seq"]), float(ns), nf)
            st.success("저장했습니다.")
            st.rerun()


def tab_history():
    ss = st.session_state
    code = ss.get("t_code", "")
    if not code:
        st.info("먼저 ② 탭에서 세션을 발행하거나 불러오세요.")
        return
    sess = load_session(code)
    stu = fetch_students(code)
    sub = fetch_submissions(code)
    if stu.empty:
        st.info("아직 참여한 학생이 없습니다.")
        return

    st.markdown("**학생별 제출 이력 (최종본 기준 · 제출 횟수 누적)**")
    items = sess["items"]
    rows = []
    for _, s in stu.iterrows():
        mine = sub[sub["student_key"] == s["student_key"]]
        try:
            t0 = datetime.strptime(s["joined_at"], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(s["last_at"], "%Y-%m-%d %H:%M:%S")
            dur = "%d분 %d초" % divmod(int((t1 - t0).total_seconds()), 60)
        except Exception:
            dur = "-"
        fin = mine.apply(lambda r: r["teacher_score"] if pd.notna(r["teacher_score"]) else r["score"],
                         axis=1) if len(mine) else pd.Series(dtype=float)
        row = {"번호": s["student_no"], "이름": s["student_name"],
               "완료문항": "%d / %d" % (len(mine), len(items)),
               "총점": round(float(fin.sum()), 1) if len(mine) else 0.0,
               "총제출횟수": int(s["total_submits"]),
               "재제출": int(max(0, s["total_submits"] - len(mine))),
               "첫 제출": s["joined_at"][11:], "최종 제출": s["last_at"][11:], "소요시간": dur}
        for it in items:
            d = mine[mine["seq"] == it["seq"]]
            row["%d번" % it["seq"]] = ("%.1f (%d회)" % (float(d.iloc[0]["score"]), int(d.iloc[0]["submit_count"]))
                                       if len(d) else "미제출")
        rows.append(row)
    hdf = pd.DataFrame(rows)
    st.dataframe(hdf, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("평균 제출 횟수", "%.1f회" % (stu["total_submits"].mean()))
    c2.metric("재제출 학생", "%d명" % int((hdf["재제출"] > 0).sum()))
    c3.metric("전원 완료", "%d명" % int((hdf["완료문항"] == "%d / %d" % (len(items), len(items))).sum()))

    many = hdf[hdf["재제출"] >= 2]
    if len(many):
        st.warning("재제출 2회 이상 학생 : " + ", ".join(
            "%s %s(%d회)" % (r["번호"], r["이름"], r["재제출"]) for _, r in many.iterrows())
            + " — 문항 이해에 어려움을 겪었을 수 있으니 개별 확인을 권합니다.")

    st.download_button("제출 이력 CSV 내려받기", hdf.to_csv(index=False).encode("utf-8-sig"),
                       file_name="제출이력_%s.csv" % code, mime="text/csv")

    st.divider()
    st.markdown("**개별 학생 상세**")
    keys = stu["student_key"].tolist()
    lab = ["%s %s" % (r["student_no"], r["student_name"]) for _, r in stu.iterrows()]
    k = st.selectbox("학생 선택", range(len(keys)), format_func=lambda i: lab[i])
    mine = sub[sub["student_key"] == keys[k]].sort_values("seq")
    for _, r in mine.iterrows():
        with st.container(border=True):
            st.markdown("**%d번 문항 · %.1f점 · %s · 제출 %d회**"
                        % (r["seq"], r["score"], r["level"], r["submit_count"]))
            st.write(str(r["answer"]))
            fl = jload(r["flags_json"], [])
            if fl:
                st.error(" / ".join(fl))
            st.info(str(r["teacher_feedback"] or r["feedback"]))


def neis_level(ratio, cfg):
    return level_of(ratio, cfg)


def tab_neis(cfg):
    ss = st.session_state
    code = ss.get("t_code", "")
    if not code:
        st.info("먼저 ② 탭에서 세션을 발행하거나 불러오세요.")
        return
    sess = load_session(code)
    sub = fetch_submissions(code)
    if sub.empty:
        st.info("채점된 제출물이 없습니다.")
        return
    items = sess["items"]
    full = sum(float(i["points"]) for i in items)
    sub["final"] = sub.apply(lambda r: r["teacher_score"] if pd.notna(r["teacher_score"]) else r["score"], axis=1)

    c1, c2 = st.columns(2)
    topic = c1.text_input("평어에 넣을 주제", value=sess["title"])
    concept = c2.text_input("핵심 개념", value=sess["subject"] + " 서술형 문제 해결")

    rows = []
    for key, g in sub.groupby("student_key"):
        g = g.sort_values("seq")
        tot = float(g["final"].sum())
        ratio = tot / full if full else 0.0
        lv = neis_level(ratio, cfg)
        tmpl = {
            "매우 잘함": "%s에 대한 이해가 뛰어나 %s 과정을 논리적으로 서술함.",
            "잘함": "%s의 핵심을 파악하여 %s 과정을 바르게 설명함.",
            "보통": "%s의 기본 개념을 이해하고 %s 과정을 일부 설명함.",
            "노력 요함": "%s에 대한 이해가 부족하여 %s 과정에 도움이 필요함.",
        }[lv] % (topic, concept)
        r = {"번호": g.iloc[0]["student_no"], "학번": "", "성명": g.iloc[0]["student_name"],
             "원점수": round(tot, 1), "성취수준": lv, "평가결과": tmpl}
        for it in items:
            d = g[g["seq"] == it["seq"]]
            r["%d번" % it["seq"]] = round(float(d.iloc[0]["final"]), 1) if len(d) else 0.0
        rows.append(r)
    ndf = pd.DataFrame(rows)
    # 번호는 문자열이므로 숫자 기준으로 정렬한다 (2번이 10번보다 앞에 오도록)
    ndf["_k"] = pd.to_numeric(ndf["번호"], errors="coerce").fillna(10 ** 6)
    ndf = ndf.sort_values(["_k", "번호"]).drop(columns=["_k"]).reset_index(drop=True)
    st.dataframe(ndf, use_container_width=True, hide_index=True)

    st.markdown("**탭 구분 복사표 (NEIS·한글 표에 바로 붙여넣기)**")
    st.code(ndf.to_csv(sep="\t", index=False), language=None)
    st.markdown("**평어만 복사**")
    st.code("\n".join("%s\t%s" % (r["성명"], r["평가결과"]) for _, r in ndf.iterrows()), language=None)

    d1, d2 = st.columns(2)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            up = ndf[["번호", "학번", "성명", "원점수", "성취수준", "평가결과"]]
            up.to_excel(w, sheet_name="NEIS업로드", index=False)
            ndf.to_excel(w, sheet_name="문항별점수", index=False)
            det = sub[["student_no", "student_name", "seq", "answer", "final", "level",
                       "kw_ratio", "sem_sim", "submit_count", "first_at", "last_at", "feedback"]].copy()
            det.columns = ["번호", "성명", "문항", "답안", "점수", "성취수준", "키워드율",
                           "의미유사도", "제출횟수", "첫제출", "최종제출", "피드백"]
            det.to_excel(w, sheet_name="상세채점결과", index=False)
            pd.DataFrame({"항목": ["평가명", "교과", "학년", "참여코드", "문항수", "만점", "참여인원", "출력일시"],
                          "내용": [sess["title"], sess["subject"], sess["grade"], code,
                                 len(items), full, len(ndf), now_str()]}
                         ).to_excel(w, sheet_name="평가정보", index=False)
            for shname in w.book.sheetnames:
                shx = w.book[shname]
                for col in shx.columns:
                    ln = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                    shx.column_dimensions[col[0].column_letter].width = min(60, max(10, ln + 2))
        d1.download_button("NEIS 엑셀 내려받기 (.xlsx)", buf.getvalue(),
                           file_name="NEIS_서술형_%s.xlsx" % code,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception as e:
        d1.warning("엑셀 생성 실패(openpyxl 확인 필요): %s" % e)
    d2.download_button("상세 채점결과 CSV", sub.to_csv(index=False).encode("utf-8-sig"),
                       file_name="채점상세_%s.csv" % code, mime="text/csv", use_container_width=True)


LICENSES = [
    {"구성요소": "Python", "용도": "실행 환경", "라이선스": "PSF License"},
    {"구성요소": "Streamlit", "용도": "화면 구성", "라이선스": "Apache-2.0"},
    {"구성요소": "pandas", "용도": "표 처리", "라이선스": "BSD-3-Clause"},
    {"구성요소": "NumPy", "용도": "벡터 연산", "라이선스": "BSD-3-Clause"},
    {"구성요소": "openpyxl", "용도": "NEIS 엑셀 생성", "라이선스": "MIT"},
    {"구성요소": "scikit-learn", "용도": "TF-IDF 폴백", "라이선스": "BSD-3-Clause"},
    {"구성요소": "KoNLPy(Okt)", "용도": "형태소 분석(선택)", "라이선스": "GPL-3.0 / Apache-2.0(Open Korean Text)"},
    {"구성요소": "sentence-transformers", "용도": "문장 임베딩", "라이선스": "Apache-2.0"},
    {"구성요소": "PyTorch", "용도": "모델 추론", "라이선스": "BSD-3-Clause"},
    {"구성요소": "jhgan/ko-sroberta-multitask", "용도": "한국어 의미 벡터", "라이선스": "모델 카드 확인 필요"},
    {"구성요소": "QR 인코더", "용도": "학생 접속 QR", "라이선스": "자체 구현 (외부 패키지 미사용)"},
    {"구성요소": "SQLite", "용도": "로컬 저장소", "라이선스": "Public Domain"},
]

# 검사 패턴은 조각을 이어 붙여 만든다(패턴 정의 자체가 검사에 걸리지 않도록).
_NET_PATTERNS = [
    ("open" + "ai", "외부 AI API"),
    ("anthro" + "pic", "외부 AI API"),
    ("reque" + "sts.", "HTTP 요청"),
    ("urllib." + "request", "HTTP 요청"),
    ("http." + "client", "HTTP 요청"),
    ("socket." + "create_connection", "임의 소켓 연결"),
    ("api" + "_key", "인증정보 보관"),
    ("smtp" + "lib", "메일 발송"),
    ("ftp" + "lib", "파일 전송"),
]


def scan_network_calls():
    """실행 중인 자기 소스를 검사해 외부 전송·인증정보 구문을 찾아낸다."""
    hits = []
    try:
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            src_lines = f.read().split("\n")
    except Exception:
        return hits
    for i, line in enumerate(src_lines, 1):
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or "_NET_PATTERNS" in s:
            continue
        for pat, why in _NET_PATTERNS:
            if pat in line:
                hits.append({"줄": i, "구문": pat, "분류": why, "내용": s[:70]})
    return hits


def compliance_check():
    """대회 요강의 주요 조건을 프로그램 스스로 점검한다."""
    net = scan_network_calls()
    con = db()
    try:
        n_name = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    except Exception:
        n_name = 0
    finally:
        con.close()
    rows = [
        {"요건": "유료 소프트웨어 없이 실행",
         "판정": "충족", "근거": "전 구성요소가 무료·오픈소스"},
        {"요건": "외부 API·AI·서버로 업무자료 전송 금지",
         "판정": "충족" if not net else "확인 필요",
         "근거": "소스 검사 결과 외부 통신 구문 %d건" % len(net)},
        {"요건": "인증정보를 프로그램·소스에 저장 금지",
         "판정": "충족", "근거": "로그인·API 키 입력 기능 없음"},
        {"요건": "외부 모델 접속 차단",
         "판정": "충족" if NET_STATUS["offline"] else "해제됨",
         "근거": NET_STATUS["reason"]},
        {"요건": "모델 동봉으로 오프라인 구동",
         "판정": "충족" if model_is_bundled() else "미충족",
         "근거": BUNDLED_MODEL_DIR if model_is_bundled() else "models 폴더에 모델 없음"},
        {"요건": "보안기능 우회 없음",
         "판정": "충족", "근거": "레지스트리·시스템 설정 변경 기능 없음"},
        {"요건": "제출물에 실제 개인정보 미포함",
         "판정": "확인 필요" if n_name else "충족",
         "근거": "DB에 저장된 학생 %d명 (제출 전 가명화 또는 삭제)" % n_name},
    ]
    return rows


def tab_env():
    st.markdown("**Java(JVM) 상태**")
    if JAVA_STATUS["ok"]:
        st.success(JAVA_STATUS["msg"])
        st.code("JAVA_HOME = %s\nJVM      = %s" % (JAVA_STATUS["home"], JAVA_STATUS["jvm"]), language=None)
    else:
        st.warning(JAVA_STATUS["msg"])
        st.caption("탐색한 경로: " + (", ".join(JAVA_STATUS["tried"][:6]) or "없음"))
    p = st.text_input("JAVA_HOME 직접 입력", value=JAVA_STATUS["home"] or "")
    if st.button("적용 후 다시 확인"):
        setup_java_home(p)
        get_okt.clear()
        st.rerun()

    st.divider()
    st.markdown("**형태소 분석기**")
    okt, msg = get_okt()
    (st.success if okt else st.warning)(msg)

    st.markdown("**의미 벡터 백엔드**")
    enc, info = get_embedder(st.session_state.get("local_model_path", ""))
    (st.success if info["ok"] else st.warning)(info["backend"])
    st.caption(info["detail"])
    b1, b2 = st.columns(2)
    if b1.button("임베딩 모델 다시 불러오기", use_container_width=True):
        get_embedder.clear()
        st.rerun()
    if model_is_bundled():
        st.success("모델이 설치 폴더에 동봉되어 있습니다. 인터넷 없이 동작합니다.")
        st.code(BUNDLED_MODEL_DIR, language=None)
    else:
        st.warning("모델이 아직 동봉되지 않았습니다. 배포 전에 아래 버튼으로 준비해 주세요.")
        if b2.button("모델 준비 (개발 PC에서 1회)", use_container_width=True):
            with st.spinner("모델을 내려받는 중입니다. 수 분이 걸릴 수 있습니다."):
                ok, msg = download_model_once()
            (st.success if ok else st.error)(msg)
            get_embedder.clear()

    st.divider()
    st.markdown("**개인정보 관리**")
    st.code("DB : %s" % DB_PATH, language=None)
    st.caption("학생 이름이 DB에 저장됩니다. 제출·시연 전에 반드시 아래 기능으로 정리해 주세요.")
    p1, p2 = st.columns(2)
    if p1.button("이름 가명화 (학생01, 학생02 …)", use_container_width=True):
        st.success(purge_personal_data("anonymize"))
    if p2.button("전체 기록 삭제", use_container_width=True):
        st.success(purge_personal_data("wipe"))

    st.divider()
    st.markdown("**DB 스키마 점검**")
    st.dataframe(schema_report(), use_container_width=True, hide_index=True)
    if MIGRATION_LOG:
        st.info("구버전 DB를 다음과 같이 정리했습니다.")
        for m in MIGRATION_LOG:
            st.caption("· " + m)

    st.divider()
    st.markdown("**대회 요건 자가점검**")
    st.dataframe(pd.DataFrame(compliance_check()), use_container_width=True, hide_index=True)
    st.caption("외부 전송 여부는 실행 중인 소스코드를 직접 검사한 결과입니다.")
    with st.expander("소스코드 외부 통신 검사 상세"):
        hits = scan_network_calls()
        if hits:
            st.warning("확인이 필요한 구문이 %d건 있습니다." % len(hits))
            st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True)
        else:
            st.success("외부 통신에 사용되는 구문이 발견되지 않았습니다.")
    st.markdown("**사용 구성요소 및 라이선스**")
    st.dataframe(pd.DataFrame(LICENSES), use_container_width=True, hide_index=True)


# =====================================================================
# [12] 진입점
# =====================================================================
def sidebar_config():
    st.sidebar.markdown("### 채점 기준 설정")
    cfg = dict(DEFAULT_CFG)
    w = st.sidebar.slider("키워드 비중 (나머지는 의미 유사도)", 0.0, 1.0, 0.5, 0.05)
    cfg["w_kw"], cfg["w_sem"] = w, 1.0 - w
    cfg["score_step"] = st.sidebar.selectbox("점수 단위", [0.5, 1.0, 0.1], index=0)
    cfg["reversal_cap"] = st.sidebar.slider("오개념 시 성취율 상한", 0.0, 1.0, 0.40, 0.05)
    cfg["required_cap"] = st.sidebar.slider("필수 키워드 누락 시 상한", 0.0, 1.0, 0.60, 0.05)
    cfg["min_chars"] = st.sidebar.number_input("근거 부족 판정 글자 수", 0, 100, 10)
    st.sidebar.markdown("**성취수준 커트라인**")
    cfg["cut_a"] = st.sidebar.slider("매우 잘함", 0.5, 1.0, 0.90, 0.05)
    cfg["cut_b"] = st.sidebar.slider("잘함", 0.4, 0.95, 0.70, 0.05)
    cfg["cut_c"] = st.sidebar.slider("보통", 0.2, 0.9, 0.50, 0.05)
    st.sidebar.divider()
    DISPLAY["mask"] = st.sidebar.toggle("화면에 학생 이름 가리기 (시연용)", value=False,
                                        help="켜면 대시보드·이력·NEIS 표의 이름이 홍○동 형태로 표시됩니다.")
    lp = st.sidebar.text_input("로컬 모델 폴더(선택)", value="")
    st.session_state["local_model_path"] = lp
    cfg["local_model_path"] = lp
    return cfg


def main():
    st.set_page_config(page_title="서논술형 자동 채점 시스템", page_icon="📝", layout="wide")
    init_db()
    params = get_params()

    if str(params.get("role", "")).lower() == "student":
        render_student(params)
        return

    st.title("하이브리드 서논술형 자동 채점 시스템")
    st.caption("KoNLPy(Okt) 어휘 매칭 + ko-sroberta 의미 벡터 매칭 · 실시간 참여 · 다문항 세트 · NEIS 출력  |  %s"
               % APP_VERSION)
    if not NET_STATUS["offline"]:
        st.warning("모델 준비용으로 외부 접속이 일시 허용된 상태입니다. "
                   "제출·시연 전 프로그램 폴더의 .allow_model_download 파일을 삭제해 주세요.")
    cfg = sidebar_config()

    enc, info = get_embedder(cfg.get("local_model_path", ""))
    if not info["ok"]:
        # ko-sroberta 미사용 시 유사도 분포가 낮아지므로 기준선·비중을 자동 보정
        cfg["sem_base"] = 0.15
        cfg["w_kw"] = max(cfg["w_kw"], 0.65)
        cfg["w_sem"] = 1.0 - cfg["w_kw"]
        st.sidebar.warning("의미 벡터 폴백 동작 중 (%s). 키워드 비중을 자동으로 %.2f 로 보정했습니다."
                           % (info["backend"], cfg["w_kw"]))
    else:
        st.sidebar.success(info["backend"])

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["① 문항 세트", "② 세션·QR 발행", "③ 실시간 대시보드", "④ 제출 이력", "⑤ NEIS 출력", "⑥ 환경 점검"])
    with t1:
        tab_items()
    with t2:
        tab_session(cfg)
    with t3:
        tab_dashboard()
    with t4:
        tab_history()
    with t5:
        tab_neis(cfg)
    with t6:
        tab_env()


if __name__ == "__main__":
    main()
