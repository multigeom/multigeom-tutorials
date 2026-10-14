---
name: notebook-skills
description: How to write and edit the demo notebooks of multigeom-tutorials: concise, readable text with measurable budgets, code cells that run on a laptop and on Google Colab, no shell scripts, and edits that keep the notebook tests and the diffs clean. Use whenever a notebook in notebooks/ is created, shortened, reformatted or checked.
---

# Writing the notebooks of multigeom-tutorials

A notebook is for someone who wants to see the package work in a few minutes. Every sentence has to earn its place. When in doubt, cut text and let the code and the figure speak.

## 1. Text: concise and readable

**Budgets** (measure them, section 2; change them only on purpose):

| What | Budget |
|---|---|
| Words in the first markdown cell (title, goal, how to run) | at most 80 |
| Words before the first code cell (all markdown above it) | at most 120 |
| Any other markdown cell | at most 80 words, 5 sentences |
| Sentence | at most 25 words, one idea |
| Flesch-Kincaid grade of the prose | at most 10 |
| Bullets in one cell | at most 5 |

**How to write**
- First cell: a title, one sentence saying what the reader gets, what it needs (data, GPU) and how long it takes. Nothing else. No author lines, no history, no list of every step.
- Say what a step does in one sentence, before its code. Do not repeat what the code or its comments already say.
- Explain a concept once, where it is first used, in one or two sentences, with a link for more. Never re-define it later.
- Active voice, short words, "you" for the reader. Define an acronym or a symbol the first time, or avoid it.
- Prefer a short table or a figure to a paragraph. Do not use bold for emphasis in running text; use it for the few terms a reader must remember.
- Put background (paper, maths, design choices) in the documentation or the README, and link to it from the notebook.
- Numbered section headings only when there are more than four sections.

## 2. Measure it

Run this on the notebook you changed, before and after (it needs no package beyond the standard library; `textstat`, if installed, gives the grade):

```python
import json, re, sys
nb = json.load(open(sys.argv[1]))
strip = lambda t: re.sub(r"```.*?```|`[^`]*`|\$[^$]*\$|\[([^\]]*)\]\([^)]*\)|<[^>]+>|[#*|>_-]", " ", t, flags=re.S)
words = lambda t: re.findall(r"[A-Za-z][A-Za-z'-]+", strip(t))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
md = [t for t, k in cells if k == "markdown"]
first_code = next(i for i, (_, k) in enumerate(cells) if k == "code")
print("markdown words:", sum(len(words(t)) for t in md))
print("words in the first cell:", len(words(cells[0][0])))
print("words before the first code cell:", sum(len(words(t)) for t, k in cells[:first_code]))
print("longest markdown cell:", max(len(words(t)) for t in md), "words")
sentences = [s for t in md for s in re.split(r"(?<=[.!?])\s+", strip(t)) if len(words(s)) > 2]
print("mean sentence length:", round(sum(len(words(s)) for s in sentences) / len(sentences), 1), "words")
try:
    import textstat
    print("Flesch-Kincaid grade:", textstat.flesch_kincaid_grade(" ".join(strip(t) for t in md)))
except ImportError:
    pass
```

Report the numbers in your answer. If a budget is exceeded, shorten the text and measure again; say so if you leave one exceeded and why.

## 3. Code cells

- The first code cell installs the package on Google Colab only: `if "google.colab" in sys.modules: %pip install -q --no-deps --pre multigeom` (plus what Colab lacks, such as `diffusers` for notebook 2). Nothing runs on a laptop.
- The next cell moves to the repository root (the folder with `environment.yml`) and, when the notebook imports from `figures/`, adds the root to `sys.path`.
- Parameters are upper-case names in one cell, with a one-line comment each. The tests override them.
- Data comes from the package: `download_tinyimagenet()`, `download_tvsd()`, `download_things_images(accept_license=True)` in `multigeom.datasets`. Never add a `.sh` script or ask the reader to download by hand.
- Import from `multigeom.<entry point>`: `datasets`, `encoders`, `diffusion`, `geometry`, `analysis`, `pipeline`, `utils`. Plot helpers for the paper's figures are in `figures/` of this repository, not in the package.
- A code cell has one job and fits on a screen. Move repeated logic into the package, with a test, instead of growing the notebook.
- Use `show_figure(fig)` from `multigeom.utils` to display a figure.

## 4. Editing the files safely

- Edit the JSON of a notebook in place with exact string replacements. Do not load and re-dump it: that rewrites the whole file and buries the change in the diff. Check `git diff --stat` before committing; a few lines are expected, hundreds are not. Check with `json.load` that the file is still valid.
- Keep the stored outputs of a notebook unless the task is to refresh them.
- The tests in `tests/` find cells by a unique line (the markers in `tests/nb_helpers.py`, such as `lmf.launch(job)` or `ns["show_figure"]`). If you change one of these lines, change the marker too, and run the tests.
- A notebook that the user has open may have unsaved or uncommitted changes. Look at `git status` first, and do not overwrite or commit changes you did not make.
- `multigeom/docs/source/demo_notebooks/` holds copies of the notebooks; refresh them with the documentation sync and do not edit them by hand.
- Run the tests (`CUDA_VISIBLE_DEVICES= python -m pytest -q tests`, in the `tutorials-env` environment; the GPU of the shared machine is often full) and report what passed, what failed and what you could not run.

## 5. Names

Name modules, functions, files and arguments after what they do, not after the paper (`receptive_field`, not `fig3`). Paper figure numbers may appear in `figures/` of this repository and in short "as in paper Fig. N" notes, never in the package.
