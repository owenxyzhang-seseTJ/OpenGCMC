# OpenGCMC calculation scheduling

An isotherm plan freezes the model, reservoirs, number of independent chains, burn-in and production budgets. The controller records progress in SQLite. Each logical sample block has a unique pressure or window index, chain index and step range. Workers write a checkpoint and a sample block together. Only the current execution attempt can make that block visible to aggregation.

## Local runs and MCP

Install the optional MCP dependency and configure a stdio client to run:

```bash
opengcmc-mcp --root /path/to/runs serve --input-directory /path/to/models --workers 4
```

`opengcmc_plan` takes the same object as the molecular CLI. Set `pressure_pa` for pressure points at fixed reservoir composition, or provide `reservoirs` for individual fugacities or mixtures. Optional `count_mcse` and `log_selectivity_mcse` specify the precision to evaluate at the planned production budget. The plan requires production and block lengths to be multiples of the sampling interval.

Send the returned `plan_id`, a root seed and an idempotency key to `opengcmc_submit`. Query `opengcmc_status` and `opengcmc_result` by `run_id`. The result reports each component's uptake and between-chain diagnostics, with composition-normalized selectivity when two bulk mole fractions are provided. Use `opengcmc_cancel` to stop dispatch and `opengcmc_resume` to resume from the most recent committed blocks. A failed input identifies the affected chain and reports its error.

For a count free-energy profile, supply `count_windows` with at least two harmonic windows at the same reservoir using the Python driver. Each window takes `coefficients`, `center` and `stiffness` in reduced potential units. All windows use the same linear combination of species counts. The aggregate evaluates the cross-window biases on each retained sample and uses MBAR to report the unbiased point estimate, overlap and weight concentration.

The same controller runs without a client connection.

```bash
opengcmc-mcp --root /path/to/runs submit model.json --seed 2026 --key mfi-298K --pressure-pa 20000 100000 500000 --replicas 4 --block-steps 10000
opengcmc-mcp --root /path/to/runs work RUN_ID --workers 4
opengcmc-mcp --root /path/to/runs result RUN_ID
```

## Slurm

Start the controller on a Slurm submission host. Use a run directory and Python installation accessible to the allocated workers.

```bash
opengcmc-mcp --root /shared/opengcmc work RUN_ID --backend slurm --workers 8
```

The controller dispatches a Slurm array with the specified concurrency cap and reads the workers' result markers. Workers read a manifest and write artifacts on the shared filesystem. They do not open the SQLite ledger. If a submission response is interrupted, the controller uses the persisted job name to reconcile with Slurm before dispatching another wave. A terminal scheduler state without a complete block is reported as a failure that can be resumed explicitly.

## Reproducibility

Store the exact code and native binary for strict native checkpoint continuation. If an earlier binary has changed, `FastGCMC.from_checkpoint(..., configuration_only=True)` loads the molecular positions for a new chain. Set a new seed and equilibrate before using the new chain in production. Scheduled delayed runs share raw LJ and electrostatic framework fields across compatible pressure points. Independent chains and separate execution attempts keep distinct random and mutable states.
