# AGENTS.md

Tutorial/documentation site for Universal Differential Equations (SciML: DiffEqFlux.jl, NeuralPDE.jl). Not a Julia package: there is no src/test suite. Verification = the Literate notebooks in `docs/` execute end-to-end.

## Source of truth

- `docs/ude-*.jl` and `docs/index.jl` are Literate notebooks written in Julia. They are the canonical sources; generated `.ipynb` files and the `.cache/` dir are gitignored and must not be committed.
- Literate markdown-string format (`mdstrings=true`): `#===` ... `===#` wraps markdown (LaTeX `$...` allowed), `#---` separates code blocks. Follow existing style in `docs/ude-*.jl`.
- Gitignore does NOT protect against the `ci.jl` footgun: it scans the working tree, so a stale generated `.ipynb` sitting in `docs/` is treated as a source, reverse-converted into the same-named `.jl` (overwriting it), and deleted. Run Literate into a scratch dir if testing, or delete the generated `docs/*.ipynb` before running `ci.jl`.
- All notebook deps live in the root `Project.toml`; notebooks `using` them directly. `Startup/` is an empty placeholder sub-project of the Pkg workspace, not a library to write code in.

## Commands

- Env uses the Pkg workspace at the repo root — always `--project=@.`:
  `julia --project=@. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`
- Rebuild/execute all notebooks like CI (SHA256 cache in `.cache/`, misses rerun automatically):
  `julia --project=@. -p 4 .github/ci.jl` (env vars: `DOCDIR=docs`, `NBCACHE=.cache`)
- Run a single notebook:
  `julia --project=@. -e 'using Literate; Literate.notebook("docs/ude-01-firstexample.jl", "docs"; mdstrings=true, execute=true)'`
  (delete the produced `docs/*.ipynb` afterwards — see above)
- No lint/typecheck/test commands exist. There is no local CI shortcut other than running `ci.jl`.

## Notebook execution gotchas

- Notebooks are compute-heavy (neural ODE training, PINNs, AdvancedHMC sampling). CI pins `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and sets `ENV["GKSwstype"]="100"` for headless Plots.jl — mirror these locally to reproduce CI timing/behavior.
- Plots.jl output is post-processed by `ci.jl` (SVG/HTML stripped when PNG exists); don't hand-edit generated `.ipynb` outputs.

## Julia version & manifests

- Julia is pinned to 1.13 (`[compat]` in `Project.toml`) and CI reads the exact version from the nonstandard `julia_version = "1.13.2"` line in `Manifest.toml` — change it there, not in the workflow.
- `Manifest.toml` is committed on purpose (static environment). It is updated by weekly "Julia Dependency Update" PRs (`Pkg.update()`); Dependabot covers GitHub Actions and pip only. PRs with the `automerge` label are squash-merged by Kodiak, so manifest churn on `main` is expected — don't hand-edit dependency versions in `Manifest.toml`.
- CI's notebook cache is keyed on file hashes; editing a `docs/*.jl` is all that's needed to force its re-execution.

## Site build

- `docs/myst.yml` (MyST) is the site config; its TOC references generated `index.ipynb` / `ude-*.ipynb`, so notebooks must be built first.
- The site uses the **npm** `jupyter-book` package (`npm i -g jupyter-book`), not the Python one. Build in `docs/`: `jupyter-book build --html`. Deploy to GitHub Pages happens on `main` only.
- Python deps (`requirements.txt`) exist only for optional matplotlib in notebooks and are currently empty; the site itself uses no Python.

## Formatting

Per `.editorconfig`: 4-space indent in `.jl`, 2-space in `.yml`/`.toml`/`.md`/`.json`; trailing whitespace trimmed.
