from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import lasagne.nonlinearities as NL
from sandbox.smooth.smooth_env import SmoothEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

data_num = 200
after_current = 2
start_index = 2
max_path_length=data_num - start_index - after_current

d3_constaint = 0.3

batch_size=50000
n_itr=500
discount=1
trpo_stepsize = 0.02
seeds = [1, 2, 3, 4, 5]

def run_task(*_):
    env = normalize(SmoothEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        # output_nonlinearity=NL.softmax
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args={
            'hidden_sizes': (64, 32),
            'hidden_nonlinearity': NL.tanh,
            'learn_std': False,
            'step_size': trpo_stepsize,
            'optimizer': ConjugateGradientOptimizer(subsample_factor=0.2)
        }
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        gae_lambda=0.95,
        discount=discount,
        step_size=trpo_stepsize,
        optimizer_args={'subsample_factor': 0.2},
        store_paths=True
        # plot=True
    )
    algo.train()

for seed in seeds:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=72,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        exp_prefix='Smooth3D'+str(d3_constaint)+'F'+str(after_current)+'GaussianLinearProjection-TRPO-N'+str(n_itr)+'BS'+str(batch_size)+ 'S' + str(trpo_stepsize)+'Seed'+str(seed),
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        #plot=True
    )
