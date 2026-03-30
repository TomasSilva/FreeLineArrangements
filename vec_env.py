"""
vec_env.py

Subprocess-based vectorized environment for parallel rollout collection.

Each worker process runs its own FreeArrangementEnv instance. The main
process batches observations for a single model forward pass, then
distributes actions back.  Auto-reset: when an episode ends, the worker
immediately resets and returns the new initial observation.

Usage:
    vec = SubprocVecEnv(env_kwargs_list)   # one dict per worker
    obs_list = vec.reset_all(reset_kwargs_list)
    for step in range(n_steps):
        actions = model.act_batch(obs_list)
        obs_list, rewards, dones, infos = vec.step(actions, next_reset_args)
    vec.close()
"""

import multiprocessing as mp
import os


def _worker(pipe, env_kwargs):
    """Event loop running in a child process."""
    # Prevent numpy/torch from spawning extra threads per worker
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    from environment import FreeArrangementEnv

    env = FreeArrangementEnv(**env_kwargs)

    while True:
        cmd, data = pipe.recv()

        if cmd == 'step':
            action = data['action']
            obs, reward, done, info = env.step(action)
            if done:
                # Enrich info with arrangement data (main process can't access env)
                info['arr_lines'] = [str(l) for l in env.arr.lines]
                info['arr_summary'] = env.arr.summary()
                info['arr_max_mult'] = env.arr.max_multiplicity()
                info['arr_n_pts'] = env.arr.n_intersection_points()
                info['terminal_obs'] = obs
                # Auto-reset with the next episode's triple
                next_tn, next_te = data.get('next_reset_args', (env.target_n, None))
                obs = env.reset(target_n=next_tn, random_start=True,
                                target_exponents=next_te)
            pipe.send((obs, reward, done, info))

        elif cmd == 'reset':
            obs = env.reset(**data)
            pipe.send(obs)

        elif cmd == 'close':
            pipe.close()
            break


class SubprocVecEnv:
    """Vectorized environment using one subprocess per env."""

    def __init__(self, env_kwargs_list):
        self.n_envs = len(env_kwargs_list)
        self.pipes = []
        self.procs = []

        ctx = mp.get_context('fork')
        for kwargs in env_kwargs_list:
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child_conn, kwargs),
                               daemon=True)
            proc.start()
            child_conn.close()
            self.pipes.append(parent_conn)
            self.procs.append(proc)

    def reset_all(self, reset_kwargs_list):
        """Send reset to all workers. Returns list of obs dicts."""
        for pipe, kwargs in zip(self.pipes, reset_kwargs_list):
            pipe.send(('reset', kwargs))
        return [pipe.recv() for pipe in self.pipes]

    def step(self, actions, next_reset_args_list):
        """
        Step all envs in parallel.

        Args:
            actions: list of int, one per env
            next_reset_args_list: list of (target_n, target_exponents) tuples,
                used for auto-reset if the episode ends.

        Returns:
            obs_list, rewards, dones, infos
        """
        for pipe, action, reset_args in zip(self.pipes, actions,
                                            next_reset_args_list):
            pipe.send(('step', {
                'action': action,
                'next_reset_args': reset_args,
            }))
        results = [pipe.recv() for pipe in self.pipes]
        obs_list, rewards, dones, infos = zip(*results)
        return list(obs_list), list(rewards), list(dones), list(infos)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except BrokenPipeError:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
