"""
skill_kopru.py — SKILL -> RL POLICY bridge (high-level <-> low-level).

llm_planner.py produces abstract sub-goals ("walk_forward", "turn"...).
This file binds each skill to a REAL RL policy + its parameters:
  walk_forward / walk_backward -> walking policy (target forward speed)
  turn                         -> walking policy + torso YAW target
  stop                         -> balance policy (target speed = 0, just stay upright)

NOTE: The walking policy (policy_yuruyus.zip) is NOT trained YET; the runner
below is a SKELETON + comments. The balance policy (policy_denge.zip) exists,
so the 'stop' skill can actually be run today.
"""
from __future__ import annotations
import os

ROOT = "/home/yasin/Workspace/humanoid_rl"

# ----------------------------------------------------------------------------
# SKILL TABLE — defines which policy + which parameters each sub-goal runs
# with. The RL policy knows 'how to walk'; the parameters here set the
# 'which direction / how much' question.
# ----------------------------------------------------------------------------
SKILL_TABLOSU = {
    "walk_forward": {
        "politika": "policy_yuruyus.zip",
        "hedef_hiz": +0.20,           # m/s forward
        "hedef_yaw": 0.0,             # go straight
        "aciklama": "Walk forward (balance + periodic stepping).",
    },
    "walk_backward": {
        "politika": "policy_yuruyus.zip",
        "hedef_hiz": -0.15,           # m/s backward (usually slower/safer)
        "hedef_yaw": 0.0,
        "aciklama": "Walk backward.",
    },
    "turn": {
        "politika": "policy_yuruyus.zip",
        "hedef_hiz": 0.0,             # turn in place (or >0 to trace an arc)
        "hedef_yaw": None,            # filled from the 'yaw' in the plan step
        "aciklama": "Turn the torso to the target yaw angle.",
    },
    "stop": {
        "politika": "policy_denge.zip",
        "hedef_hiz": 0.0,
        "hedef_yaw": 0.0,
        "aciklama": "Stop and stay balanced.",
    },
}


def skill_parametreleri(adim: dict) -> dict:
    """
    Converts a plan step ({'skill':..., 'sure':..., 'yaw':...}) ->
    into an executable parameter dict.
    Unknown skill -> safe 'stop'.
    """
    skill = adim.get("skill", "stop")
    taban = dict(SKILL_TABLOSU.get(skill, SKILL_TABLOSU["stop"]))
    # For 'turn', insert the yaw target coming from the planner.
    if skill == "turn":
        taban["hedef_yaw"] = float(adim.get("yaw", 0.0))
    taban["skill"] = skill
    taban["sure"] = float(adim.get("sure", 2.0))
    taban["politika_yolu"] = os.path.join(ROOT, "models", taban["politika"])
    return taban


# ----------------------------------------------------------------------------
# RUNNER SKELETON — executes the plan in order, running each skill for its
# duration. Real execution needs trained policies and passing parameters to the
# environment (injecting target speed/yaw into the observation); these get wired
# in once yuruyus_env matures. For now it is a commented skeleton showing the flow.
# ----------------------------------------------------------------------------
def plani_yurut(plan: list, kontrol_hz: float = 50.0, render: bool = False):
    """
    plan: output of llm_planner.komuttan_plan(...) (list[dict]).
    Runs each step with its policy for 'sure' seconds.

    SKELETON: the actual policy loading + step loop are commented out; to be
    filled in once the environment and walking policy are ready. For stop, the
    balance policy can be used today.
    """
    # from stable_baselines3 import PPO           # (import at run time)
    # from envs.yuruyus_env import YuruyusEnv
    # from envs.denge_env import DengeEnv

    print("EXECUTING PLAN:")
    for i, adim in enumerate(plan, 1):
        p = skill_parametreleri(adim)
        n_adim = int(p["sure"] * kontrol_hz)
        print(f"  {i}. {p['skill']:<14} | policy={p['politika']:<18} "
              f"speed={p['hedef_hiz']:+.2f} yaw={p['hedef_yaw']} "
              f"dur={p['sure']:.1f}s (~{n_adim} steps)")

        # --- REAL EXECUTION (once the walking policy is ready): --------------
        # model = PPO.load(p["politika_yolu"])
        # env   = _uygun_ortam(p["skill"], p)   # give hedef_hiz/hedef_yaw to the environment
        # obs, _ = env.reset()
        # for _ in range(n_adim):
        #     a, _ = model.predict(obs, deterministic=True)
        #     obs, _, term, trunc, _ = env.step(a)
        #     if render: kareler.append(env.render())
        #     if term or trunc:
        #         obs, _ = env.reset()          # if it fell, re-verify
        # ---------------------------------------------------------------------
    print("PLAN DONE.")


if __name__ == "__main__":
    # Bridge + skeleton demonstration (no policy run, just parsing).
    from llm_planner import komuttan_plan
    plan = komuttan_plan("uc metre ileri yuru sonra saga don sonra dur",
                         ollama_kullan=False)
    plani_yurut(plan)
