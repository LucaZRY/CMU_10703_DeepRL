import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import functools


def cmaes(fn, dim, num_iter=10):
  """Optimizes a given function using CMA-ES.

  Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.

  Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
  """
  # Hyperparameters
  sigma = 10
  population_size = 500     # use 500 as recommended for the RL part
  p_keep = 0.10             # Fraction of population to keep
  noise = 0.25              # Isotropic jitter to keep cov well-conditioned

  # Initialize the mean and covariance
  mu = np.zeros(dim)
  cov = sigma**2 * np.eye(dim)

  mu_vec = []
  best_sample_vec = []
  mean_sample_vec = []
  for t in range(num_iter):
    # WRITE CODE HERE
    # 1) Sample candidates from N(mu, cov)
    samples = np.random.multivariate_normal(mean=mu, cov=cov, size=population_size)

    # 2) Evaluate objective
    values = np.array([fn(x) for x in samples])

    # 3) Book-keeping
    best_sample_vec.append(values.max())
    mean_sample_vec.append(values.mean())

    # 4) Select elites (top p_keep fraction)
    elite_size = max(1, int(p_keep * population_size))
    elite_idx = np.argsort(values)[::-1][:elite_size]  # descending sort by value
    elites = samples[elite_idx]

    # 5) Update search distribution
    mu = elites.mean(axis=0)
    cov = np.cov(elites, rowvar=False) + noise * np.eye(dim)

    # 6) Track mu trajectory
    mu_vec.append(mu.copy())

  return mu_vec, best_sample_vec, mean_sample_vec


def test_fn(x):
  goal = np.array([65, 49])
  return -np.sum((x - goal)**2)

mu_vec, best_sample_vec, mean_sample_vec = cmaes(test_fn, dim=2, num_iter=100)

"""Run the following cell to visualize CMA-ES."""

x = np.stack(np.meshgrid(np.linspace(-10, 100, 30), np.linspace(-10, 100, 30)), axis=-1)
fn_value = [test_fn(xx) for xx in x.reshape((-1, 2))]
fn_value = np.array(fn_value).reshape((30, 30))
plt.figure(figsize=(6, 4))
plt.contourf(x[:, :, 0], x[:, :, 1], fn_value, levels=10)
plt.colorbar()
mu_arr = np.array(mu_vec)
plt.plot(mu_arr[:, 0], mu_arr[:, 1], 'o-')
plt.plot([mu_arr[0, 0]], [mu_arr[0, 1]], '+', ms=14, label='initial value')
plt.plot([mu_arr[-1, 0]], [mu_arr[-1, 1]], 'x', ms=14, label='final value')
plt.plot([65], [49], 's', ms=8, label='maximum')
plt.legend()
plt.grid(True)
plt.title("CMA-ES trajectory on test function")
plt.show()


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, params):
  w = params[:4]
  b = params[4]
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])  # 0 = LEFT, 1 = RIGHT
  return a

def rl_fn(params, env):
  assert len(params) == 5
  ## WRITE CODE HERE
  obs, _ = env.reset()
  done, truncated = False, False
  total_rewards = 0.0
  while not done and not truncated:
    a = _get_action(obs, params)
    obs, r, done, truncated, _ = env.step(a)
    total_rewards += r
  return total_rewards


# ===== Part 2: evaluate the three fixed policies (1000 runs each) =====
def evaluate_policies(env):
  policies = {
    "x = (-1, -1, -1, -1, -1)": np.array([-1, -1, -1, -1, -1], dtype=float),
    "x = (1, 0, 1, 0, 1)":       np.array([ 1,  0,  1,  0,  1], dtype=float),
    "x = (0, 1, 2, 3, 4)":       np.array([ 0,  1,  2,  3,  4], dtype=float),
  }
  out = {}
  for name, p in policies.items():
    rets = [rl_fn(p, env) for _ in range(1000)]
    out[name] = float(np.mean(rets))
  return out


# ===== Part 3: run CMA-ES on the RL objective and plot mean/best reward =====
env = gym.make('CartPole-v0')  # v0 semantics (200-step cap) is fine for the assignment
fn_with_env = functools.partial(rl_fn, env=env)
mu_vec_rl, best_vec_rl, mean_vec_rl = cmaes(fn_with_env, dim=5, num_iter=10)

# Plot mean vs best population rewards across iterations
plt.figure()
plt.plot(mean_vec_rl, label="Mean population reward")
plt.plot(best_vec_rl, label="Best sample reward")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("CMA-ES on CartPole-v0")
plt.legend()
plt.grid(True)
plt.show()

# Print Part 2 results for quick verification
results = evaluate_policies(env)
for k, v in results.items():
  print(f"{k}: {v:.2f}")
