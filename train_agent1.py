import gymnasium as gym
import numpy as np
import argparse
import os
from multiprocessing import Pool, cpu_count


def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    b = params[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)


def evaluate_policy(params, episodes=50, render=False):  # Increased evaluation episodes for stability
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make('LunarLander-v2', render_mode='human' if render else None)
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / episodes


def simulated_binary_crossover(parent1, parent2, eta_c=15):  # More explorative crossover
    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        u = np.random.rand()
        beta = (2 * u) ** (1 / (eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
        child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
    return child


def polynomial_mutation(child, mutation_rate=0.2, eta_m=20, lower_bound=-30,
                        upper_bound=30):  # More aggressive mutation
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            x = child[i]
            delta1 = (x - lower_bound) / (upper_bound - lower_bound)
            delta2 = (upper_bound - x) / (upper_bound - lower_bound)
            rnd = np.random.rand()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rnd < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 + delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1))
                delta_q = 1.0 - val ** mut_pow

            child[i] += delta_q * (upper_bound - lower_bound)
            child[i] = np.clip(child[i], lower_bound, upper_bound)
    return child


def genetic_algorithm(population_size=1000, num_generations=1000, elite_frac=0.2,  # Larger population & generations
                      mutation_rate=0.2, lower_bound=-30, upper_bound=30,
                      initial_params=None):
    gene_size = 8 * 4 + 4
    num_elites = int(population_size * elite_frac)

    # Initialize population with emphasis on existing best parameters
    if initial_params is not None:
        # Use higher mutation rate for initial elites to encourage exploration
        elites = [polynomial_mutation(initial_params.copy(), mutation_rate * 0.5,
                                      eta_m=20, lower_bound=lower_bound,
                                      upper_bound=upper_bound) for _ in range(num_elites)]
        random_pop = [np.random.uniform(lower_bound, upper_bound, gene_size)
                      for _ in range(population_size - num_elites)]
        population = np.vstack([elites, random_pop])
    else:
        population = np.random.uniform(lower_bound, upper_bound, (population_size, gene_size))

    best_reward = -np.inf
    best_params = None

    for gen in range(num_generations):
        with Pool(cpu_count()) as p:
            fitness = np.array(p.map(evaluate_policy, population))

        elite_indices = np.argsort(fitness)[-num_elites:]
        current_best_reward = np.max(fitness)

        if current_best_reward > best_reward:
            best_reward = current_best_reward
            best_params = population[np.argmax(fitness)].copy()
            print("saved ! in best_policy1.npy" )
            np.save("best_policy1.npy", best_params)

        print(f"Generation {gen + 1}/{num_generations}: Best={best_reward:.2f}, Avg={np.mean(fitness):.2f}")

        # Adaptive mutation: Respond faster to stagnation
        if gen > 5 and np.mean(fitness[-5:]) < best_reward * 0.95:
            current_mutation_rate = min(0.5, mutation_rate * 1.5)  # More aggressive adjustment
        else:
            current_mutation_rate = max(0.05, mutation_rate * (1 - gen / (2 * num_generations)))  # Gradual reduction

        elites = population[elite_indices]
        new_population = []
        new_population.extend(elites)

        while len(new_population) < population_size:
            # Tournament selection with larger pool
            tournament_indices = np.random.choice(len(population), 5, replace=False)  # Increased tournament size
            parent1_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
            parent2_idx = tournament_indices[np.argsort(fitness[tournament_indices])[-2]]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            child = simulated_binary_crossover(parent1, parent2, eta_c=15)
            child = polynomial_mutation(child, mutation_rate=current_mutation_rate,
                                        eta_m=20, lower_bound=lower_bound,
                                        upper_bound=upper_bound)
            new_population.append(child)

        population = np.array(new_population[:population_size])

    return best_params


def train_and_save(filename, **kwargs):
    best_params = genetic_algorithm(**kwargs)
    np.save(filename, best_params)
    return best_params


def load_policy(filename):
    if os.path.exists(filename):
        return np.load(filename)
    print(f"File {filename} not found.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--filename", default="best_policy1.npy")
    args = parser.parse_args()

    if args.train:
        initial_params = load_policy(args.filename) if os.path.exists(args.filename) else None
        best_params = train_and_save(
            args.filename,
            population_size=750,
            num_generations=80,
            elite_frac=0.125,
            mutation_rate=0.093,
            lower_bound=-9,
            upper_bound=9,
            initial_params=initial_params
        )
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            avg_reward = evaluate_policy(best_params, episodes=100)
            print(f"Final average over 100 episodes: {avg_reward:.2f}")
    else:
        print("Use --train or --play")


if __name__ == "__main__":
    main()