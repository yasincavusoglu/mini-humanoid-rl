"""
llm_planner.py — COGNITIVE LAYER (high-level planner).

BIG PICTURE (two-layer brain):
  - LOW LEVEL (reflex / RL): denge_env, yuruyus_env + PPO policies.
    Answers the "How?" question -> drives 20 servos at 50 Hz, stays balanced, takes steps.
  - HIGH LEVEL (this file / LLM): answers the "What to do?" question.
    Natural-language command -> sub-goal (skill) plan.
    Example:  "uc metre ileri yuru sonra dur"
        ->  [{'skill':'walk_forward','sure':15.0,'mesafe':3.0},
             {'skill':'stop','sure':2.0}]

This file produces a plan in TWO ways:
  1) If Ollama (local LLM) is available it uses it  -> flexible, understands free-form language.
  2) If Ollama is NOT available it falls back to the rule-based Turkish parser -> offline, instant.
     The komuttan_plan(komut) function works fully even WITHOUT ollama.

Why LLM + rules together? The LLM is 'smarter' but requires setup/model download and can be
slow. The rule-based fallback is always at hand and provides a deterministic, testable
baseline (teaching + safe default).
"""
from __future__ import annotations
import json
import re
import urllib.request
import urllib.error

# ----------------------------------------------------------------------------
# SKILL DICTIONARY — all sub-goals the high level knows about.
# The planner must only produce these names; the bridge (skill_kopru.py) binds
# them to real RL policies.
# ----------------------------------------------------------------------------
GECERLI_SKILLER = {"walk_forward", "walk_backward", "turn", "stop"}

# Walking speed assumption (m/s). Used for the distance -> duration conversion.
# (Roughly the speed the low-level walking policy reaches; can be updated with training.)
YURUME_HIZI = 0.20            # m/s
DONME_HIZI = 0.80            # rad/s  (approx ~46 deg/s)
VARSAYILAN_YURU_SURE = 3.0   # when no duration/distance in the command
VARSAYILAN_DUR_SURE = 2.0    # wait time for 'stop'
VARSAYILAN_DON_ACI = 1.5708  # when no angle in the turn command -> 90 degrees (pi/2)


# ============================================================================
# SECTION 1 — RULE-BASED TURKISH PARSER  (works WITHOUT ollama)
# ============================================================================

# Turkish number words -> value. Ones + tens are kept separate so we can sum
# compounds like "on bes".
_BIRLER = {
    "sifir": 0, "bir": 1, "iki": 2, "uc": 3, "üc": 3, "üç": 3, "uç": 3,
    "dort": 4, "dört": 4, "bes": 5, "beş": 5, "alti": 6, "altı": 6,
    "yedi": 7, "sekiz": 8, "dokuz": 9,
}
_ONLAR = {
    "on": 10, "yirmi": 20, "yirmı": 20, "otuz": 30, "kirk": 40, "kırk": 40,
    "elli": 50, "altmis": 60, "altmış": 60, "yetmis": 70, "yetmiş": 70,
    "seksen": 80, "doksan": 90, "yuz": 100, "yüz": 100,
}

# Conjunctions that split the command into multiple steps.
_AYIRAC_KALIP = re.compile(
    r"\s*(?:,|;|\bsonra\b|\bardindan\b|\bardından\b|\bdaha\s+sonra\b|\bve\b|\bsonrasinda\b|\bsonrasında\b)\s*"
)


def _turkce_ascii(s: str) -> str:
    """Roughly simplify Turkish characters (for accent-insensitive matching)."""
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")
    return s.translate(tr).lower().strip()


def _sayi_bul(parca: str):
    """
    Returns the first number (digit or Turkish word) in a text fragment.
    None if not found. "bucuk" -> adds +0.5.
    """
    # 1) First look for a digit (including decimals: 3, 2.5, and the comma form 1,5).
    m = re.search(r"\d+(?:[.,]\d+)?", parca)
    if m:
        return float(m.group(0).replace(",", "."))

    # 2) Sum the Turkish number words (e.g. "on bes" = 10 + 5).
    kelimeler = parca.split()
    toplam, bulundu = 0.0, False
    for k in kelimeler:
        if k in _ONLAR:
            toplam += _ONLAR[k]
            bulundu = True
        elif k in _BIRLER:
            toplam += _BIRLER[k]
            bulundu = True
        elif k in ("bucuk", "buçuk"):
            toplam += 0.5
            bulundu = True
    return toplam if bulundu else None


def _parca_to_skill(parca: str):
    """
    Converts a single command fragment into a sub-goal ({skill, sure, ...}).
    An unrecognized fragment -> None (skipped).
    """
    p = _turkce_ascii(parca)
    if not p:
        return None

    sayi = _sayi_bul(p)
    metre_var = "metre" in p or "metrelik" in p or " m " in f" {p} "
    saniye_var = "saniye" in p or "sn" in p.split()

    # --- STOP / WAIT ---
    if re.search(r"\b(dur|durdur|bekle|dur bakalim|sabit kal)\b", p):
        sure = sayi if (sayi and saniye_var) else VARSAYILAN_DUR_SURE
        return {"skill": "stop", "sure": float(sure)}

    # --- TURN (right / left) ---
    if re.search(r"\b(don|dön|donus|dönüs|donus yap)\b", p) or "don" in p:
        if re.search(r"\bsag|saga\b", p) or "sag" in p:
            yon = "sag"
        elif re.search(r"\bsol|sola\b", p) or "sol" in p:
            yon = "sol"
        elif "geri" in p:      # "geri don" -> treated as 180 degrees
            yon = "sag"
        else:
            yon = "sag"        # direction unspecified -> default to right
        # Angle: if degrees are given convert to radians, otherwise 90 degrees.
        if sayi and ("derece" in p or "°" in parca):
            aci = float(sayi) * 3.14159265 / 180.0
        else:
            aci = VARSAYILAN_DON_ACI
        # Right = negative yaw, left = positive yaw (right-hand rule; MuJoCo z-up).
        yaw = -aci if yon == "sag" else +aci
        sure = abs(aci) / DONME_HIZI
        return {"skill": "turn", "yon": yon, "yaw": round(yaw, 4), "sure": round(sure, 2)}

    # --- WALK BACKWARD ---
    if "geri" in p:
        if metre_var and sayi:
            mesafe = float(sayi); sure = mesafe / YURUME_HIZI
        elif saniye_var and sayi:
            sure = float(sayi); mesafe = sure * YURUME_HIZI
        else:
            sure = VARSAYILAN_YURU_SURE; mesafe = sure * YURUME_HIZI
        return {"skill": "walk_backward", "sure": round(sure, 2), "mesafe": round(mesafe, 2)}

    # --- WALK FORWARD (yuru / ileri / git) ---
    if re.search(r"\b(yuru|yürü|ileri|git|ilerle|yuru bakalim)\b", p) or "yuru" in p or "ileri" in p:
        if metre_var and sayi:
            mesafe = float(sayi); sure = mesafe / YURUME_HIZI
        elif saniye_var and sayi:
            sure = float(sayi); mesafe = sure * YURUME_HIZI
        elif sayi and not saniye_var:   # like "3 ileri" -> treat as meters
            mesafe = float(sayi); sure = mesafe / YURUME_HIZI
        else:
            sure = VARSAYILAN_YURU_SURE; mesafe = sure * YURUME_HIZI
        return {"skill": "walk_forward", "sure": round(sure, 2), "mesafe": round(mesafe, 2)}

    return None


def _kural_tabanli_plan(komut: str) -> list:
    """
    Splits the command at conjunctions to produce an ordered skill plan.
    Fully deterministic, offline.
    """
    parcalar = _AYIRAC_KALIP.split(komut)
    plan = []
    for parca in parcalar:
        skill = _parca_to_skill(parca)
        if skill is not None:
            plan.append(skill)
    # If nothing was understood, safe default: stay balanced in place.
    if not plan:
        plan = [{"skill": "stop", "sure": VARSAYILAN_DUR_SURE}]
    return plan


# ============================================================================
# SECTION 2 — OLLAMA (local LLM) PATH  (used if available)
# ============================================================================

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODELLER = ["qwen2.5:3b", "llama3.2:3b"]   # tried in order

_SISTEM_PROMPT = """Sen bir mini-humanoid robotun ust-seviye planlayicisisin.
Kullanicinin Turkce komutunu, robotun bildigi alt-hedeflerin (skill) SIRALI listesine cevir.
SADECE su skill isimlerini kullan:
  - "walk_forward"  : ileri yuru        (alan: sure [saniye], mesafe [metre])
  - "walk_backward" : geri yuru         (alan: sure, mesafe)
  - "turn"          : don               (alan: yon "sag"/"sol", sure)
  - "stop"          : dur / dengede kal (alan: sure)
Yaklasik yurume hizi 0.2 m/s. 1 metre ~ 5 saniye.
CIKTI: yalnizca JSON dizisi. Aciklama yazma. Ornek:
[{"skill":"walk_forward","sure":15,"mesafe":3},{"skill":"stop","sure":2}]
"""


def _ollama_var_mi(timeout: float = 0.5) -> bool:
    """Is the Ollama server up? (short probe)"""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_plan(komut: str, timeout: float = 30.0):
    """
    Request a plan from Ollama. Returns None on failure (the caller falls back).
    """
    istem = _SISTEM_PROMPT + f'\nKomut: "{komut}"\nJSON:'
    for model in _OLLAMA_MODELLER:
        govde = json.dumps({
            "model": model,
            "prompt": istem,
            "stream": False,
            "format": "json",           # force JSON out of ollama
            "options": {"temperature": 0.0},
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                _OLLAMA_URL, data=govde,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                yanit = json.loads(r.read().decode("utf-8"))
            metin = yanit.get("response", "").strip()
            plan = _json_plan_ayikla(metin)
            if plan:
                return plan
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            continue   # didn't work with this model, try the next one / eventually fallback
    return None


def _json_plan_ayikla(metin: str):
    """
    Extract the JSON plan from the LLM output and VALIDATE it.
    format=json may sometimes return {"plan":[...]} and sometimes directly [...].
    Filters out invalid skills; None if there is no valid step.
    """
    try:
        veri = json.loads(metin)
    except json.JSONDecodeError:
        # Look for an array embedded in the text.
        m = re.search(r"\[.*\]", metin, re.DOTALL)
        if not m:
            return None
        try:
            veri = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(veri, dict):
        veri = veri.get("plan") or veri.get("adimlar") or veri.get("steps") or []
    if not isinstance(veri, list):
        return None

    temiz = []
    for adim in veri:
        if isinstance(adim, dict) and adim.get("skill") in GECERLI_SKILLER:
            temiz.append(adim)
    return temiz or None


# ============================================================================
# SECTION 3 — PUBLIC API
# ============================================================================

def komuttan_plan(komut: str, ollama_kullan: bool = True) -> list:
    """
    Natural-language command -> ordered sub-goal (skill) plan (list[dict]).

    If ollama_kullan=True and ollama is up, the LLM is tried first; otherwise
    (or if the LLM fails) the RULE-BASED parser is used.
    This function always returns a valid plan even WITHOUT ollama.
    """
    if ollama_kullan and _ollama_var_mi():
        plan = _ollama_plan(komut)
        if plan:
            return plan
    # Fallback (or ollama_kullan=False): deterministic rule-based plan.
    return _kural_tabanli_plan(komut)


# ---- Simple manual test (shows the rule path WITHOUT ollama) ----------------
if __name__ == "__main__":
    ornekler = [
        "uc metre ileri yuru sonra dur",
        "saga don",
        "dur",
        "iki metre geri git, sonra sola don ve 5 saniye yuru",
    ]
    print("=== RULE-BASED PLANNER (ollama off) ===\n")
    for k in ornekler:
        plan = komuttan_plan(k, ollama_kullan=False)
        print(f"COMMAND : {k}")
        print(f"PLAN  : {json.dumps(plan, ensure_ascii=False)}\n")
