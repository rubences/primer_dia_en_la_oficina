from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


@dataclass
class UnitStats:
    code: str
    name: str
    food: float
    wood: float
    gold: float
    power: float


def resolve_data_dir() -> Path:
    candidates = [
        Path(os.getenv("ARCHIVE_DIR", "")),
        Path.cwd() / "archive",
        Path.home() / "Downloads" / "archive",
        Path(__file__).resolve().parent / "data",
    ]

    required = {
        "terrain.csv",
        "battle_durations.csv",
        "front_widths.csv",
        "battle_actors.csv",
        "battles.csv",
    }

    for candidate in candidates:
        if not str(candidate):
            continue
        if candidate.exists() and all((candidate / name).exists() for name in required):
            return candidate

    raise FileNotFoundError(
        "No se encontro un directorio de datos valido. Define ARCHIVE_DIR o coloca los CSV en Downloads/archive."
    )


def load_stats(data_dir: Path) -> Dict[str, UnitStats]:
    terrain = pd.read_csv(data_dir / "terrain.csv", usecols=["isqno", "terra1"])
    durations = pd.read_csv(data_dir / "battle_durations.csv", usecols=["isqno", "duration2"])
    widths = pd.read_csv(data_dir / "front_widths.csv", usecols=["isqno", "wofa", "wofd"])
    actors = pd.read_csv(data_dir / "battle_actors.csv", usecols=["isqno", "attacker", "n", "actor"])
    battles = pd.read_csv(data_dir / "battles.csv", usecols=["isqno", "wina"])

    terrain = terrain[terrain["terra1"].isin(["F", "G", "R"])].dropna(subset=["terra1"])
    durations["duration2"] = pd.to_numeric(durations["duration2"], errors="coerce")
    durations = durations.dropna(subset=["duration2"])

    widths["wofa"] = pd.to_numeric(widths["wofa"], errors="coerce")
    widths["wofd"] = pd.to_numeric(widths["wofd"], errors="coerce")
    widths["front_avg"] = widths[["wofa", "wofd"]].mean(axis=1)
    widths = widths.groupby("isqno", as_index=False)["front_avg"].mean()

    actor_count = (
        actors.groupby("isqno", as_index=False)
        .size()
        .rename(columns={"size": "actor_count"})
    )

    battles["wina"] = pd.to_numeric(battles["wina"], errors="coerce")
    battles["attacker_win"] = (battles["wina"] == 1).astype(int)

    merged = terrain.merge(durations, on="isqno", how="inner")
    merged = merged.merge(widths, on="isqno", how="inner")
    merged = merged.merge(actor_count, on="isqno", how="inner")
    merged = merged.merge(battles[["isqno", "attacker_win"]], on="isqno", how="inner")

    grouped = merged.groupby("terra1", as_index=False).agg(
        food=("duration2", "mean"),
        wood=("front_avg", "mean"),
        gold=("actor_count", "mean"),
        power=("attacker_win", "mean"),
    )

    name_map = {
        "F": "Flat",
        "G": "Rugged",
        "R": "Rolling",
    }

    stats: Dict[str, UnitStats] = {}
    for row in grouped.itertuples(index=False):
        code = row.terra1
        stats[code] = UnitStats(
            code=code,
            name=name_map.get(code, code),
            food=float(row.food),
            wood=float(row.wood),
            gold=float(row.gold),
            power=float(row.power * 100.0),
        )

    missing = [k for k in ["F", "G", "R"] if k not in stats]
    if missing:
        raise ValueError(f"Faltan categorias de terreno para el modelo: {missing}")

    return stats


def default_budgets(stats: Dict[str, UnitStats]) -> Dict[str, float]:
    avg_food = sum(u.food for u in stats.values()) / 3.0
    avg_wood = sum(u.wood for u in stats.values()) / 3.0
    avg_gold = sum(u.gold for u in stats.values()) / 3.0
    scale = 100.0
    return {
        "food": round(avg_food * scale, 3),
        "wood": round(avg_wood * scale, 3),
        "gold": round(avg_gold * scale, 3),
    }


def solve_integer_lp(stats: Dict[str, UnitStats], budgets: Dict[str, float]) -> Dict[str, float]:
    f_stats = stats["F"]
    g_stats = stats["G"]
    r_stats = stats["R"]

    max_f = min(
        int(budgets["food"] // f_stats.food),
        int(budgets["wood"] // f_stats.wood),
        int(budgets["gold"] // f_stats.gold),
    )

    best = {
        "F": 0,
        "G": 0,
        "R": 0,
        "objective": -1.0,
    }

    for f_count in range(max_f + 1):
        rem_food_f = budgets["food"] - f_count * f_stats.food
        rem_wood_f = budgets["wood"] - f_count * f_stats.wood
        rem_gold_f = budgets["gold"] - f_count * f_stats.gold

        if rem_food_f < 0 or rem_wood_f < 0 or rem_gold_f < 0:
            continue

        max_g = min(
            int(rem_food_f // g_stats.food),
            int(rem_wood_f // g_stats.wood),
            int(rem_gold_f // g_stats.gold),
        )

        for g_count in range(max_g + 1):
            rem_food = rem_food_f - g_count * g_stats.food
            rem_wood = rem_wood_f - g_count * g_stats.wood
            rem_gold = rem_gold_f - g_count * g_stats.gold

            if rem_food < 0 or rem_wood < 0 or rem_gold < 0:
                continue

            max_r = min(
                int(rem_food // r_stats.food),
                int(rem_wood // r_stats.wood),
                int(rem_gold // r_stats.gold),
            )

            r_count = max(0, max_r)

            objective = (
                f_count * f_stats.power
                + g_count * g_stats.power
                + r_count * r_stats.power
            )

            if objective > best["objective"]:
                best = {
                    "F": f_count,
                    "G": g_count,
                    "R": r_count,
                    "objective": objective,
                }

    used_food = (
        best["F"] * f_stats.food + best["G"] * g_stats.food + best["R"] * r_stats.food
    )
    used_wood = (
        best["F"] * f_stats.wood + best["G"] * g_stats.wood + best["R"] * r_stats.wood
    )
    used_gold = (
        best["F"] * f_stats.gold + best["G"] * g_stats.gold + best["R"] * r_stats.gold
    )

    best.update(
        {
            "used_food": used_food,
            "used_wood": used_wood,
            "used_gold": used_gold,
            "left_food": budgets["food"] - used_food,
            "left_wood": budgets["wood"] - used_wood,
            "left_gold": budgets["gold"] - used_gold,
        }
    )

    return best


@app.route("/", methods=["GET", "POST"])
def index():
    data_dir = resolve_data_dir()
    stats = load_stats(data_dir)
    defaults = default_budgets(stats)

    food = float(request.form.get("food", defaults["food"]))
    wood = float(request.form.get("wood", defaults["wood"]))
    gold = float(request.form.get("gold", defaults["gold"]))

    budgets = {"food": food, "wood": wood, "gold": gold}
    result = solve_integer_lp(stats, budgets)

    formula = {
        "objective": (
            f"max Z = {stats['F'].power:.2f}F + {stats['G'].power:.2f}G + {stats['R'].power:.2f}R"
        ),
        "food": (
            f"{stats['F'].food:.3f}F + {stats['G'].food:.3f}G + {stats['R'].food:.3f}R <= {food:.3f}"
        ),
        "wood": (
            f"{stats['F'].wood:.3f}F + {stats['G'].wood:.3f}G + {stats['R'].wood:.3f}R <= {wood:.3f}"
        ),
        "gold": (
            f"{stats['F'].gold:.3f}F + {stats['G'].gold:.3f}G + {stats['R'].gold:.3f}R <= {gold:.3f}"
        ),
    }

    return render_template(
        "index.html",
        data_dir=str(data_dir),
        stats=stats,
        defaults=defaults,
        budgets=budgets,
        result=result,
        formula=formula,
    )


if __name__ == "__main__":
    app.run(debug=True)
