from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots, build_robot, sample_mask, shift_mask, perturb_p
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# Mutation rate schedule
# ──────────────────────────────────────────────────────────────────────────────
#
# Unlike your friend's fixed MUTATION_RATE = 0.15, we decay the per-voxel flip
# probability from FLIP_RATE_START down to FLIP_RATE_END over all generations.
# Early generations explore widely; later ones fine-tune promising shapes.
#
FLIP_RATE_START = 0.25   # aggressive early exploration
FLIP_RATE_END   = 0.04   # fine-tuned late exploitation
SHIFT_PROB      = 0.30
RESAMPLE_PROB   = 0.30

def current_flip_rate(gen, n_generations):
    """Cosine decay from FLIP_RATE_START → FLIP_RATE_END over n_generations."""
    t = gen / max(n_generations, 1)
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * t))
    return FLIP_RATE_END + (FLIP_RATE_START - FLIP_RATE_END) * cosine_decay


def mutate_with_rate(robot, flip_rate):
    """
    Same structure as robot.mutate() but accepts an explicit flip_rate,
    so the caller can pass the generation-specific decayed value.
    """
    mask = robot["mask"].copy()
    p    = robot["p"]

    # Voxel flip (always applied, rate controlled externally)
    toggled = np.logical_xor(mask, np.random.uniform(size=mask.shape) < flip_rate).astype(int)
    from scipy import ndimage
    labeled, n = ndimage.label(toggled)
    if n > 0:
        sizes = ndimage.sum(toggled, labeled, range(1, n + 1))
        best  = int(np.argmax(sizes)) + 1
        mask  = (labeled == best).astype(int)
    # else: flip was too destructive — keep original mask

    if np.random.uniform() < SHIFT_PROB:
        mask = shift_mask(mask)

    if np.random.uniform() < RESAMPLE_PROB:
        p    = perturb_p(p)
        mask = sample_mask(p)

    return build_robot(mask, p)


# ──────────────────────────────────────────────────────────────────────────────
# Simulator helpers  (same as friends_run.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_simulator(robots, config):
    """Allocate a fresh simulator sized to fit the current population."""
    max_masses  = max(r["n_masses"]  for r in robots)
    max_springs = max(r["n_springs"] for r in robots)
    config["simulator"]["n_masses"]  = max_masses
    config["simulator"]["n_springs"] = max_springs

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )
    simulator.initialize(
        [r["masses"]  for r in robots],
        [r["springs"] for r in robots],
    )
    return simulator, max_masses, max_springs


def evaluate_population(robots, config):
    """Run one full training pass; return fitnesses, control_params, max_masses, max_springs."""
    simulator, max_masses, max_springs = build_simulator(robots, config)
    fitness_history = simulator.train()
    fitnesses       = fitness_history[:, -1]
    control_params  = simulator.get_control_params(list(range(len(robots))))
    return fitnesses, control_params, max_masses, max_springs


# ──────────────────────────────────────────────────────────────────────────────
# Selection
# ──────────────────────────────────────────────────────────────────────────────

def tournament_select(fitnesses, tournament_size=3):
    candidates = np.random.choice(len(fitnesses), size=tournament_size, replace=False)
    return int(candidates[np.argmax(fitnesses[candidates])])


# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic survival  (simulated-annealing-style acceptance)
# ──────────────────────────────────────────────────────────────────────────────
#
# Your friend always accepts a child only when child_fitness >= loser_fitness.
# We add a small acceptance probability for *worse* children that decays with
# temperature T.  This lets the search escape local optima early on.
#
# Acceptance probability:
#   • child is better  → always accept   (p = 1.0)
#   • child is worse   → accept with     p = exp(Δf / T)
#     where Δf = child_fitness - loser_fitness  (negative)
#     and   T  decays from T_START → T_END over n_generations
#
T_START = 0.05   # high temperature → more likely to accept bad children early
T_END   = 1e-4   # near-zero temperature → almost never accept bad children late

def current_temperature(gen, n_generations):
    """Exponential cooling schedule."""
    if n_generations <= 1:
        return T_END
    decay = (T_END / T_START) ** (gen / (n_generations - 1))
    return T_START * decay


def accept(child_fitness, loser_fitness, temperature):
    """
    Return True if the child should replace the loser.
    Always True when child is better; probabilistic when worse.
    """
    delta = child_fitness - loser_fitness
    if delta >= 0:
        return True
    if temperature < 1e-9:
        return False
    return np.random.uniform() < np.exp(delta / temperature)


# ──────────────────────────────────────────────────────────────────────────────
# Parallel hill climber  (with elitism + decaying mutation + SA survival)
# ──────────────────────────────────────────────────────────────────────────────

def run_parallel_hill_climber(config, n_generations=20):
    """
    Three improvements over friends_run.py:

    1. ELITISM
       The all-time best robot is reinjected into the population at the start
       of every generation, so a good solution can never be permanently lost.

    2. DECAYING MUTATION RATE
       The per-voxel flip probability starts high (wide exploration) and
       follows a cosine schedule down to a small floor value (fine-tuning).

    3. PROBABILISTIC SURVIVAL  (simulated annealing acceptance)
       Slightly worse children can still replace a loser with probability
       exp(Δfitness / T), where T cools exponentially to near-zero.
       This helps the search escape local optima early in the run.
    """
    population_size = config["simulator"]["n_sims"]

    # ── Initialise ───────────────────────────────────────────────────────────
    print("Initializing population...")
    population = load_robots(num_robots=population_size)
    fitnesses, init_control_params, _, _ = evaluate_population(population, config)
    # Store trained control params on each robot for snapshot saving
    for i, robot in enumerate(population):
        robot["control_params"] = init_control_params[i]

    best_idx     = int(np.argmax(fitnesses))
    best_fitness = float(fitnesses[best_idx])
    best_robot   = population[best_idx]

    all_fitness_history = [fitnesses.copy()]
    print(f"[Gen  0] Best: {best_fitness:.4f} | Mean: {fitnesses.mean():.4f} "
          f"| flip_rate: {current_flip_rate(0, n_generations):.3f} "
          f"| T: {current_temperature(0, n_generations):.4f}")

    # Save a snapshot of the gen-0 best so the replay visualizer has it
    os.makedirs("evolution_snapshots", exist_ok=True)
    np.save("evolution_snapshots/gen_0.npy",
            {**best_robot, "gen": 0, "fitness": best_fitness})

    for gen in range(1, n_generations + 1):
        flip_rate   = current_flip_rate(gen, n_generations)
        temperature = current_temperature(gen, n_generations)

        # ── ELITISM: slot 0 is always the all-time best ───────────────────────
        population[0] = best_robot
        fitnesses[0]  = best_fitness

        # ── Produce one child per slot ────────────────────────────────────────
        children         = []
        parent_indices   = []
        for _ in range(population_size):
            parent_idx = tournament_select(fitnesses)
            parent_indices.append(parent_idx)
            children.append(mutate_with_rate(population[parent_idx], flip_rate))

        # ── Evaluate all children in one batched simulator call ───────────────
        child_fitnesses, child_control_params, child_max_masses, child_max_springs = evaluate_population(children, config)
        for i, child in enumerate(children):
            child["control_params"]  = child_control_params[i]
            child["max_n_masses"]    = child_max_masses
            child["max_n_springs"]   = child_max_springs

        # ── SA survival: child competes with a tournament-selected loser ──────
        for i in range(population_size):
            loser_idx = tournament_select(fitnesses)
            if accept(child_fitnesses[i], fitnesses[loser_idx], temperature):
                population[loser_idx] = children[i]
                fitnesses[loser_idx]  = child_fitnesses[i]

        # ── Update global best ────────────────────────────────────────────────
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_robot   = population[gen_best_idx]

        all_fitness_history.append(fitnesses.copy())
        print(f"[Gen {gen:2d}] Best: {best_fitness:.4f} | Mean: {fitnesses.mean():.4f} "
              f"| flip_rate: {flip_rate:.3f} | T: {temperature:.4f}")

        # Save a snapshot of the current best robot for the evolution replay visualizer
        np.save(f"evolution_snapshots/gen_{gen}.npy",
                {**best_robot, "gen": gen, "fitness": best_fitness})

    return best_robot, best_fitness, np.array(all_fitness_history)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config",      type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of hill-climber generations")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    best_robot, best_fitness, fitness_history = run_parallel_hill_climber(
        config, n_generations=args.generations
    )

    print(f"\nEvolution complete.  Best fitness: {best_fitness:.4f}")

    # ── Save fitness history ──────────────────────────────────────────────────
    # Shape: (n_generations + 1, n_robots) — one row per generation incl. gen 0
    np.save("fitness_history.npy", fitness_history)

    # ── Re-train the best robot alone to extract its control parameters ───────
    n_sims       = config["simulator"]["n_sims"]
    best_population = [best_robot] * n_sims
    simulator, max_num_masses, max_num_springs = build_simulator(best_population, config)
    simulator.train()
    best_control_params = simulator.get_control_params([0])

    best_robot["control_params"]  = best_control_params[0]
    best_robot["max_n_masses"]    = max_num_masses
    best_robot["max_n_springs"]   = max_num_springs
    np.save("robot_best.npy", best_robot)

    print("Saved best robot   → robot_best.npy")
    print("Saved fitness log  → fitness_history.npy")