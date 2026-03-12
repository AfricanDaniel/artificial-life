"""
Age-Fitness Pareto Optimization (AFPO)
---------------------------------------
Based on: Schmidt & Lipson, "Age-Fitness Pareto Optimization" (2011)

Key idea:
  Every individual has two objectives — maximize FITNESS, minimize AGE.
  Pareto selection preserves young (diverse) individuals alongside fit ones,
  preventing premature convergence without needing an explicit diversity metric.

Each generation:
  1. Age all individuals by +1
  2. Generate N candidate children by mutating Pareto-front parents (age = 0)
     Plus inject a few purely random robots for novelty
  3. Evaluate all children in one batched simulator call
  4. Merge children into the population  →  pool of 2N individuals
  5. Remove Pareto-dominated individuals  (an individual is dominated if
     another exists that is strictly better on at least one objective and
     at least as good on the other)
  6. If pool still exceeds N, remove the oldest individuals until size = N

Usage:
  python afpo_run.py --config config.yaml --generations 20
"""

from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots, build_robot, sample_mask, shift_mask, perturb_p
import numpy as np
import os, json
from scipy import ndimage

# ── Mutation hyper-parameters ─────────────────────────────────────────────────
FLIP_RATE   = 0.15   # per-voxel flip probability (fixed, AFPO handles diversity)
SHIFT_PROB  = 0.30
RESAMPLE_PROB = 0.20
RANDOM_INJECT_FRAC = 0.2  # fraction of children that are purely random new robots


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(robot):
    """Produce one mutant child from a parent robot."""
    mask = robot["mask"].copy()
    p    = robot["p"]

    # Voxel flip (always applied)
    toggled = np.logical_xor(mask, np.random.uniform(size=mask.shape) < FLIP_RATE).astype(int)
    labeled, n = ndimage.label(toggled)
    if n > 0:
        sizes = ndimage.sum(toggled, labeled, range(1, n + 1))
        mask  = (labeled == int(np.argmax(sizes)) + 1).astype(int)

    if np.random.uniform() < SHIFT_PROB:
        mask = shift_mask(mask)

    if np.random.uniform() < RESAMPLE_PROB:
        p    = float(np.clip(p + np.random.normal(scale=0.05), 0.1, 0.9))
        mask = sample_mask(p)

    return build_robot(mask, p)


# ── Simulator helpers ─────────────────────────────────────────────────────────

def build_simulator(robots, config):
    max_masses  = max(r["n_masses"]  for r in robots)
    max_springs = max(r["n_springs"] for r in robots)
    config["simulator"]["n_masses"]  = max_masses
    config["simulator"]["n_springs"] = max_springs

    sim = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )
    sim.initialize(
        [r["masses"]  for r in robots],
        [r["springs"] for r in robots],
    )
    return sim, max_masses, max_springs


def evaluate_batch(robots, config):
    """
    Train a batch of robots and return (fitnesses, control_params, max_masses, max_springs).
    """
    sim, max_masses, max_springs = build_simulator(robots, config)
    fitness_history = sim.train()
    fitnesses       = fitness_history[:, -1]
    control_params  = sim.get_control_params(list(range(len(robots))))
    return fitnesses, control_params, max_masses, max_springs


# ── Pareto helpers ────────────────────────────────────────────────────────────

def dominates(a, b):
    """
    AFPO dominance: a dominates b only if a is STRICTLY younger than b.
    This guarantees every age=0 child survives at least one full generation
    (nothing has age < 0), which is the key AFPO diversity mechanism.

        a dominates b  iff  a.age < b.age  AND  a.fitness >= b.fitness
    """
    return a["age"] < b["age"] and a["fitness"] >= b["fitness"]


def pareto_cull(population):
    """Remove every individual that is dominated by at least one other."""
    n = len(population)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and not dominated[i] and dominates(population[j], population[i]):
                dominated[i] = True
                break
    return [ind for ind, dom in zip(population, dominated) if not dom]


def pareto_front(population):
    """Return the non-dominated subset (the Pareto front)."""
    return pareto_cull(population)


# ── AFPO main loop ────────────────────────────────────────────────────────────

def run_afpo(config, n_generations=20):
    """
    Age-Fitness Pareto Optimization.

    Population size N is taken from config["simulator"]["n_sims"].
    Each generation produces N children (mix of mutants + random injections),
    evaluates them in one batch, then applies Pareto selection.
    """
    N = config["simulator"]["n_sims"]
    n_random = max(1, int(N * RANDOM_INJECT_FRAC))   # random injections per gen

    import shutil
    if os.path.exists("evolution_snapshots"):
        shutil.rmtree("evolution_snapshots")
    os.makedirs("evolution_snapshots")

    # Lineage tracking: every individual gets a unique id and records its parent_id
    lineage_nodes = {}   # id -> {gen, fitness, mask, parent_id}
    _id_counter   = [0]

    def new_id():
        _id_counter[0] += 1
        return f"n{_id_counter[0]}"

    # ── Initialise ────────────────────────────────────────────────────────────
    print("Initializing population (gen 0)...")
    population = load_robots(num_robots=N)
    fitnesses, ctrl_params, max_m, max_s = evaluate_batch(population, config)

    for i, ind in enumerate(population):
        fit = float(fitnesses[i])
        if np.isnan(fit):
            fit = -9999.0
        ind["fitness"]       = fit
        ind["age"]           = 0
        ind["control_params"] = ctrl_params[i]
        ind["max_n_masses"]  = max_m
        ind["max_n_springs"] = max_s
        ind["id"]            = new_id()
        ind["parent_id"]     = None
        lineage_nodes[ind["id"]] = {
            "gen": 0, "fitness": fit, "parent_id": None,
            "mask": ind["mask"].tolist()
        }

    # Do NOT Pareto-cull the initial population — all age=0 means 1 survivor.

    best = max(population, key=lambda x: x["fitness"])
    fitness_log = [fitnesses.copy()]   # list of arrays for plotting

    print(f"[Gen  0] Best fitness: {best['fitness']:.4f} | "
          f"Pop size: {len(population)} | "
          f"Pareto front size: {len(pareto_front(population))}")

    np.save("evolution_snapshots/gen_0.npy",
            {**best, "gen": 0})

    # ── Generation loop ───────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):

        # 1. Age the whole population
        for ind in population:
            ind["age"] += 1

        # 2. Build children
        #    - Select parents from the current Pareto front (if non-empty, else full pop)
        front   = pareto_front(population) or population
        children = []

        # Mutant children from Pareto-front parents
        n_mutants = N - n_random
        for _ in range(n_mutants):
            parent = front[np.random.randint(len(front))]
            child  = mutate(parent)
            child["age"]       = 0
            child["fitness"]   = None
            child["id"]        = new_id()
            child["parent_id"] = parent.get("id")
            children.append(child)

        # Purely random new individuals (fresh blood)
        for new_ind in load_robots(num_robots=n_random):
            new_ind["age"]       = 0
            new_ind["fitness"]   = None
            new_ind["id"]        = new_id()
            new_ind["parent_id"] = None   # no parent — random injection
            children.append(new_ind)

        # 3. Evaluate all N children in one batch
        child_fitnesses, child_ctrl, child_max_m, child_max_s = evaluate_batch(children, config)
        for i, child in enumerate(children):
            fit = float(child_fitnesses[i])
            if np.isnan(fit):
                fit = -9999.0
            child["fitness"]        = fit
            child["control_params"] = child_ctrl[i]
            child["max_n_masses"]   = child_max_m
            child["max_n_springs"]  = child_max_s
            lineage_nodes[child["id"]] = {
                "gen": gen, "fitness": fit,
                "parent_id": child.get("parent_id"),
                "mask": child["mask"].tolist()
            }

        # 4. Merge into one pool
        pool = population + children   # up to 2N individuals

        # 5. Pareto cull
        pool = pareto_cull(pool)

        # 6. Trim to N: remove oldest-and-weakest first
        if len(pool) > N:
            pool = sorted(pool, key=lambda x: (-x["age"], x["fitness"]))
            pool = pool[:N]

        # Guard: if culling was too aggressive, refill with best children
        if len(pool) == 0:
            pool = sorted(children, key=lambda x: -x["fitness"])[:N]

        population = pool

        # ── Track best ────────────────────────────────────────────────────────
        current_best = max(population, key=lambda x: x["fitness"])
        if current_best["fitness"] > best["fitness"]:
            best = current_best

        gen_fitnesses = np.array([ind["fitness"] for ind in population])
        fitness_log.append(gen_fitnesses)

        front_size = len(pareto_front(population))
        print(f"[Gen {gen:2d}] Best: {best['fitness']:.4f} | "
              f"Mean: {gen_fitnesses.mean():.4f} | "
              f"Pop: {len(population)} | Front: {front_size}")

        # Save lineage JSON incrementally
        with open("lineage.json", "w") as f:
            json.dump({"nodes": lineage_nodes}, f)

        np.save(f"evolution_snapshots/gen_{gen}.npy",
                {**best, "gen": gen,
                 "max_n_masses": best.get("max_n_masses", best["n_masses"]),
                 "max_n_springs": best.get("max_n_springs", best["n_springs"])})

    return best, fitness_log


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config",      type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    best, fitness_log = run_afpo(config, n_generations=args.generations)

    print(f"\nAFPO complete.  Best fitness: {best['fitness']:.4f}")

    # ── Save fitness log ──────────────────────────────────────────────────────
    # Pad rows to the same length (population size can vary) then save
    max_len = max(len(row) for row in fitness_log)
    padded  = np.full((len(fitness_log), max_len), np.nan)
    for i, row in enumerate(fitness_log):
        padded[i, :len(row)] = row
    np.save("fitness_history.npy", padded)

    # ── Re-train best robot to get clean control params ───────────────────────
    n_sims = config["simulator"]["n_sims"]
    best_pop = [best] * n_sims
    sim, max_m, max_s = build_simulator(best_pop, config)
    sim.train()
    best["control_params"] = sim.get_control_params([0])[0]
    best["max_n_masses"]   = max_m
    best["max_n_springs"]  = max_s
    np.save("robot_best.npy", best)

    print("Saved best robot   → robot_best.npy")
    print("Saved fitness log  → fitness_history.npy")