# VisualWebArena Viewport Benchmark

This repository is an experimental fork of
[VisualWebArena](https://github.com/web-arena-x/visualwebarena). It uses the
original VisualWebArena tasks, websites, browser environment, and evaluators to
measure how browser viewport dimensions affect a multimodal web agent's task
success and trajectory.

This is not a replacement for the upstream benchmark. For the original project
description, released trajectories, and baseline results, see the
[upstream repository](https://github.com/web-arena-x/visualwebarena),
[project website](https://jykoh.com/vwa), and
[paper](https://arxiv.org/abs/2401.13649).

![VisualWebArena overview](media/overview.png)

## Research scope

The main experimental variable in this fork is the browser viewport:

- `--viewport_width`
- `--viewport_height`

All other conditions should remain fixed between paired runs. The current
`run.py` always sets `current_viewport_only=True`, so the agent observes the
current viewport rather than the full page. The `--current_viewport_only` CLI
flag therefore does not define a separate condition in this fork.

Tasks with a non-empty `viewport_size` field in their config are excluded
automatically because that task-level setting would override the command-line
viewport dimensions. The excluded task IDs and their configured dimensions are
written to the run log.

For a clean viewport-height experiment, keep the width fixed and compare, for
example, `1280x720` against `1280x2048`. Use a separate result directory for
every condition.

| Change between runs | Keep fixed between runs |
| --- | --- |
| Viewport width and/or height | Task IDs and site snapshot |
| Nothing else | Model and provider |
|  | Prompt, action set, and observation type |
|  | Step limit and provider generation settings |
|  | Evaluation model and environment URLs |

Remote model APIs can remain nondeterministic even with a fixed seed. For a
formal comparison, repeat each condition and report both aggregate success
rates and paired per-task outcome changes.

## What differs from upstream

- The benchmark is organized around paired viewport-size experiments.
- OpenAI and Gemini integrations use current SDK interfaces.
- Gemini uses `google-genai` instead of the legacy Vertex AI generative-model
  classes.
- Provider clients are initialized lazily, so a Gemini-only agent run does not
  require an OpenAI key at import time.
- OpenAI chat and Gemini generation use seed `42` and medium reasoning.
  `--max_tokens` defaults to 384 but is not sent to either API; these paths do
  not set an output-token limit.
- Dependencies are curated for the current Python 3.10/3.11 code instead of
  reproducing the upstream environment's complete historical `pip freeze`.

The task definitions, browser interaction logic, website setup, prompt assets,
trajectory rendering, and success evaluators remain based on VisualWebArena.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
pip install -e .
```

The last command should be `pip install -e .`, not `pip install -e ".[dev]"`.
The legacy development extra in `setup.cfg` still contains upstream test-tool
pins, while `requirements.txt` describes the environment used by this fork.

Check the Python dependency graph with:

```bash
python -m pip check
```

## Website environment

Set up the standalone VisualWebArena websites by following
[environment_docker/README.md](environment_docker/README.md). Then configure
the benchmark process with URLs for that deployment:

```bash
export DATASET=visualwebarena
export CLASSIFIEDS="http://<host>:9980"
export CLASSIFIEDS_RESET_TOKEN="<reset-token>"
export SHOPPING="http://<host>:7770"
export REDDIT="http://<host>:9999"
export WIKIPEDIA="http://<host>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://<host>:4399"
```

Generate per-task configuration files and login cookies after the URLs are set:

```bash
python scripts/generate_test_data.py
bash prepare.sh
```

Generated task directories are:

- `config_files/vwa/test_classifieds` — 234 tasks
- `config_files/vwa/test_reddit` — 210 tasks
- `config_files/vwa/test_shopping` — 466 tasks

Together they contain the 910 VisualWebArena tasks. CLI ranges are half-open:
`--test_start_idx 0 --test_end_idx 10` runs task files `0.json` through
`9.json`.

## Model credentials

For Gemini Developer API access:

```bash
export GEMINI_API_KEY="<api-key>"
```

For Gemini through Vertex AI:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="<project-id>"
export GOOGLE_CLOUD_LOCATION=global
```

For OpenAI or an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="<api-key>"
# Optional:
export OPENAI_BASE_URL="<compatible-api-base-url>"
```

If an OpenAI-compatible multimodal model name does not contain the historical
`gpt-4-...-vision` pattern, opt into image inputs explicitly:

```bash
export VWA_MULTIMODAL=1
```

Some `fuzzy_match` and `ua_match` evaluations use the OpenAI model configured
inside `evaluation_harness/helper_functions.py`. Those tasks require working
OpenAI credentials even when the acting agent uses Gemini.

## Action output

`--instruction_path` selects the format. JSON prompts set
`meta_data.output_format` to `json_schema`, so the runtime requests `reasoning`
and one structured `action`. Two JSON prompts are available in `agent/prompts/jsons/`:

- `p_multimodal_cot_id_actree_3s_json_cot.json`: multimodal accessibility-tree inputs.
- `p_som_cot_id_actree_3s_json_cot.json`: SoM inputs.

Use `--provider openai --mode chat` or `--provider google --mode completion`
with a model that supports the schema. Original prompts keep backtick actions;
evaluator calls stay free text. The three `scripts/run_*_som.sh` scripts select
the SoM JSON prompt and write to separate `json-cot` result folders.

## Run a paired viewport experiment

The example below uses SoM JSON actions and changes only viewport height.
Set `VWA_MODEL` to the exact schema-capable model used for the experiment.

```bash
export VWA_MODEL="replace-with-exact-model-id"

COMMON_ARGS=(
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_json_cot.json
  --test_start_idx 0
  --test_end_idx 10
  --test_config_base_dir config_files/vwa/test_reddit
  --provider google
  --model "$VWA_MODEL"
  --mode completion
  --action_set_tag som
  --observation_type image_som
  --max_steps 30
)

# Start the first condition from the benchmark snapshot.
bash scripts/reset_reddit.sh
bash prepare.sh

python run.py "${COMMON_ARGS[@]}" \
  --viewport_width 1280 \
  --viewport_height 720 \
  --result_dir results/reddit_1280x720

# Restore the website to the same initial state before the paired condition.
bash scripts/reset_reddit.sh
bash prepare.sh

python run.py "${COMMON_ARGS[@]}" \
  --viewport_width 1280 \
  --viewport_height 2048 \
  --result_dir results/reddit_1280x2048
```

For an OpenAI agent, use `--provider openai --mode chat` and the same remaining
arguments. To compare observation representations rather than viewport size,
run a separate experiment; do not change `observation_type` inside a viewport
pair.

Before running the complete benchmark, use a small fixed task range as a smoke
test. Reset each website to the same snapshot before every paired condition,
especially for tasks that mutate shopping, forum, or classifieds state. The
included reset scripts assume the websites run as local Docker containers; use
the equivalent snapshot-restore procedure for a remote deployment.

## Outputs and analysis

Each result directory contains:

- `config.json`: the effective experiment arguments, including viewport size.
- `render_<task_id>.html`: the observation and action trajectory for a task.
- `traces/<task_id>.zip`: the Playwright trace.
- `<result-directory-name>.jsonl`: task logs grouped in `messages` arrays,
  written on completion or error, without timestamps.
- `log_files.txt`: paths to timestamped `.log` files updated at each step,
  including `[Result] (PASS|FAIL)` and the aggregate `Average score`.
- `error.txt`: unhandled task errors, when present.

For a single uninterrupted run, inspect the aggregate score with:

```bash
grep "Average score" "$(tail -n 1 results/reddit_1280x720/log_files.txt)"
grep "Average score" "$(tail -n 1 results/reddit_1280x2048/log_files.txt)"
```

For resumed runs, combine all log files when comparing per-task results.
Report both aggregate success rates and how many tasks changed from fail to
pass or pass to fail between viewport conditions.

## Current execution behavior

`run.py` currently applies the following settings after parsing CLI arguments:

- headless browser execution (`render=False`)
- current-viewport-only observations
- screenshots in rendered trajectories
- Playwright trace capture
- a 2.5-second delay after each browser action

These are benchmark invariants in the current fork. Changing one of them should
be treated as a new experimental condition and documented separately.

## Upstream resources and attribution

This fork intentionally keeps upstream environment and evaluation code so that
results remain comparable to VisualWebArena. Historical announcements, released
human trajectories, baseline trajectories, and the general-purpose demo are
documented in the
[original VisualWebArena README](https://github.com/web-arena-x/visualwebarena#readme)
rather than duplicated here.

When publishing results obtained with this fork, describe the viewport sizes,
task subset, site snapshot, model identifier, prompt, observation type, action
set, number of repeats, and fork commit. Cite VisualWebArena and WebArena:

```bibtex
@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}

@article{zhou2024webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={ICLR},
  year={2024}
}
```

The repository retains the upstream MIT license. See [LICENSE](LICENSE).
