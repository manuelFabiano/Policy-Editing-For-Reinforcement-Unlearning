Directory Guide
===================================

algos/
  dqn/, ppo-atari/, ppo-mujoco/, sac/, td3/
    offline-retain_experiments.py       run PERU (offline)
    online-retain_experiments.py        run PERU (online) (Can work also as hyperparam grid) search)
    RetainSetGenerator.py               build retain set (online)
    RetainSetGeneratorNoEnv.py          build retain set (offline)
    unlearning.py                       core unlearning algorithm + metrics
    tsne.ipynb                          latent-space visualization (only in ppo-atari)

hyperparams/
  atari/*.json                          hyper-parameter presets for each Atari game
  mujoco/*.json                         hyper-parameter presets for each MuJoCo task

results/
  atari/ , mujoco/
    <algo>/<env>.csv                    avg. returns and forget / retain accuracies (online) 
    <algo>/<env>_est.csv                avg. returns and forget / retain accuracies (offline) 
    <env>-ablation_results.json         extra metrics for ablation study

scripts/
  means.py                               aggregate CSVs and compute mean ± 95 % CI
  plots.ipynb                            generate all paper figures
  rfs_baseline.py                        retrain-from-scratch baseline

