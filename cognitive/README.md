# cognitive/ — Cognitive Layer (High-Level LLM Planner)

This folder is the robot's **"what to do?"** brain. It converts a natural-language command
(`"uc metre ileri yuru sonra dur"`) into an ordered plan of the **sub-goals (skills)** the
robot knows.

## Two-layer brain: RL (low) vs LLM (high)

| Layer | Question | Component | Frequency |
|--------|----------|-----------|-----------|
| **LOW — Reflex / RL** | *How?* (balance, step) | `envs/*.py` + PPO policies (`models/policy_*.zip`) | ~50 Hz, continuous |
| **HIGH — Cognitive / LLM** | *What to do?* (task plan) | `cognitive/llm_planner.py` | sparse, once per command |

The RL policy knows how to drive 20 servos at 50 Hz to stay balanced / take steps, but it
has no idea about a task like "walk 3 meters then stop". The LLM/planner is the opposite: it
cannot drive a single servo but it understands the command and breaks it into skills. The
upper layer produces the plan; the **bridge** (`skill_kopru.py`) binds each skill to an RL
policy.

```
"saga don sonra dur"
        │  llm_planner.komuttan_plan()
        ▼
[{"skill":"turn","yon":"sag","yaw":-1.57,"sure":2.0},
 {"skill":"stop","sure":2.0}]
        │  skill_kopru.plani_yurut()
        ▼
turn  -> policy_yuruyus.zip (yaw target)   |  stop -> policy_denge.zip
```

## Files

- **`llm_planner.py`** — `komuttan_plan(komut, ollama_kullan=True) -> list[dict]`.
  If Ollama is present it uses the local LLM; **if not it falls back to the rule-based
  Turkish parser** (regex + keywords: yuru/ileri/geri/dur/don/sag/sol/metre/saniye).
  It always returns a valid plan even without Ollama.
- **`skill_kopru.py`** — the skill -> policy + parameter table (`SKILL_TABLOSU`) and the
  `plani_yurut()` **skeleton** (to be filled in once the walking policy is ready).

## Known skills

| skill | meaning | policy | extra fields |
|-------|---------|--------|--------------|
| `walk_forward` | walk forward | `policy_yuruyus.zip` | `sure`, `mesafe` |
| `walk_backward` | walk backward | `policy_yuruyus.zip` | `sure`, `mesafe` |
| `turn` | turn | `policy_yuruyus.zip` | `yon` (sag/sol), `yaw`, `sure` |
| `stop` | stop / stay balanced | `policy_denge.zip` | `sure` |

The distance→duration conversion is done with a ~`0.20 m/s` walking-speed assumption (inside
the planner, `YURUME_HIZI`). Update it as training progresses.

## How to run

The rule-based path (offline, instant — **recommended first test**):

```bash
cd /home/yasin/Workspace/humanoid_rl
.venv/bin/python cognitive/llm_planner.py        # 4 example commands -> plan
```

The bridge + skeleton flow:

```bash
cd /home/yasin/Workspace/humanoid_rl/cognitive
../.venv/bin/python skill_kopru.py
```

In your own code:

```python
from cognitive.llm_planner import komuttan_plan
plan = komuttan_plan("iki metre ileri yuru sonra saga don", ollama_kullan=False)
```

## Ollama (local LLM) — optional, more flexible language understanding

The rule-based fallback works **right now**. For free-form/complex commands, if you want a
local LLM, install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b        # downloads ~2 GB   (alternative: llama3.2:3b ~2 GB)
ollama serve                  # http://localhost:11434 (usually starts automatically)
```

Then `komuttan_plan(...)` (default `ollama_kullan=True`) first probes the
`http://localhost:11434` server; if it is up it produces a plan with `qwen2.5:3b` or
`llama3.2:3b`, otherwise it automatically falls back to the rule-based path.

### WARNINGS (specific to this machine)

- **GPU driver broken -> everything on CPU.** Ollama runs the 3B model on CPU but the
  response can take a few seconds. Since planning is sparse (once per command) this is
  acceptable; still, not for real-time control.
- **Disk:** each model is ~2 GB. Before `ollama pull` make sure there is space
  (`df -h ~`). Pulling both models is ~4 GB.
- **Internet:** installation + `pull` require internet. Once the model is downloaded it works
  fully offline.
- **Stability:** the LLM output is validated (`_json_plan_ayikla`) — invalid skills are
  filtered out; if no valid step remains it falls back to the rule-based plan.
```
