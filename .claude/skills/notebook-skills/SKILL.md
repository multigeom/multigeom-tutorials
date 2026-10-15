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
| Words in the first markdown cell (the template of section 1, author included) | at most 90 |
| Words before the first code cell (all markdown above it, collapsed details excluded) | at most 120 |
| The markdown that introduces a code cell | at most 40 words and 2 sentences (tables, code blocks and `<details>` are not counted) |
| Any other markdown cell (recap, transition) | at most 80 words |
| Sentence | at most 25 words, one idea |
| Flesch-Kincaid grade of the prose | at most 10 |
| Bullets in one cell | at most 5 |

**The first cell** has this form, and nothing else (the Colab badge comes first and is not counted). Each bullet is its own paragraph:

```markdown
[![Open In Colab](...)](...)

# Title of the notebook

* **Author**: laquitainesteeve@gmail.com

* **Goal**: one or two sentences: what the reader gets.

* **Requirements**: data, checkpoints, packages, GPU or CPU.

* **Time**: about how long it takes, and on what hardware.
```

- **Author is required, and it is the point of contact.** A reader who has a question must know whom to ask. Use a working email address, keep it when you edit a notebook, and never remove it to save words. Co-authors go on the same line. Use the address of the person who maintains the notebook (laquitainesteeve@gmail.com unless told otherwise).
- Use the labels Author, Goal, Requirements, Time, in this order. Add `* **Next**: ...` only to point to the next notebook.
- No history, no list of every step (the section headings are the list), no description of the machine it was tested on.
- If a notebook needs more explanation (cost, how to read the result, the formula, the supporting functions), put it in a second markdown cell inside `<details><summary>Details</summary> ... </details>`. It stays collapsed, so the reader sees the short version first and nothing is lost. Do not count it in the budgets.

**How to write**
- **Introduce a code cell in one or two sentences**: what it does and why, nothing the code or its comments already say. Anything more (how it works, caveats, parameters, differences with the paper, a long table of options) goes in a `<details><summary>Details</summary> ... </details>` block at the end of the same cell, so it is there for the reader who wants it and hidden for the one who does not. A table is for what the reader must choose or compare, never for prose in disguise.
- Do not write execution times in the text: every code cell shows its own time (section 3).
- Explain a concept once, where it is first used, in one or two sentences, with a link for more. Never re-define it later.
- Active voice, short words, "you" for the reader. Define an acronym or a symbol the first time, or avoid it.
- Prefer a short table or a figure to a paragraph. Do not use bold for emphasis in running text; use it for the few terms a reader must remember.
- Put background (paper, maths, design choices) in the documentation or the README, and link to it from the notebook.
- **Number every section, so that it can be referred to** ("see section 4.2"). The title is `#` and unnumbered. Sections are `## 1. Title`, subsections `### 1.1. Title`, with a dot after each number and the same format in every notebook. No heading is left unnumbered: the recap too (`## 6. Recap`), and so is a subsection (`### 2.1. What the readout learned`). When you insert or remove a section, renumber the ones after it and fix every "section N" mentioned in text or in comments.

## 2. Measure it

Run this on the notebook you changed, before and after (it needs no package beyond the standard library; `textstat`, if installed, gives the grade). Collapsed `<details>` blocks are not counted:

```python
import json, re, sys
nb = json.load(open(sys.argv[1]))
strip = lambda t: re.sub(r"(?m)^\|.*$|<details>.*?</details>|```.*?```|`[^`]*`|\$[^$]*\$|\[([^\]]*)\]\([^)]*\)|<[^>]+>|[#*|>_-]", " ", t, flags=re.S)
words = lambda t: re.findall(r"[A-Za-z][A-Za-z'-]+", strip(t))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
md = [t for t, k in cells if k == "markdown"]
first_code = next(i for i, (_, k) in enumerate(cells) if k == "code")
print("markdown words:", sum(len(words(t)) for t in md))
print("words in the first cell:", len(words(cells[0][0])))
print("words before the first code cell:", sum(len(words(t)) for t, k in cells[:first_code]))
print("longest markdown cell:", max(len(words(t)) for t in md), "words")
intro = [len(words(cells[i][0])) for i in range(1, len(cells) - 1) if cells[i][1] == "markdown" and cells[i + 1][1] == "code"]
print("longest introduction of a code cell:", max(intro), "words")
def run(t):
    longest = best = 0
    for l in t.splitlines():
        longest = longest + 1 if l.strip() else 0
        best = max(best, longest)
    return best

print("longest run of code lines without a blank line:", max(run(t) for t, k in cells if k == "code"))
print("code lines over 88 characters:", sum(1 for t, k in cells if k == "code" for l in t.splitlines() if len(l) > 88))
print("code cells right after a code cell:", sum(1 for i in range(1, len(cells)) if cells[i][1] == cells[i - 1][1] == "code"))
print("code cells without %%time:", sum(1 for t, k in cells if k == "code" and not t.lstrip().startswith("%%time")))
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

**Structure**
- **Every code cell is preceded by a markdown cell** that says, in a sentence, what the cell does (a section or subsection heading with its sentence counts). Never put two code cells in a row.
- **Definitions have their own cell.** Classes and functions are defined in a code cell that comes before the cells that use them, and that cell contains nothing else: no scripted steps. Helpers go in the setup section (`### 1.3. Custom functions`); a class that is the point of a section (a backbone you write, a model you test) goes in the cell right before the script that uses it, with its own sentence of introduction.
- A script cell has one job and fits on a screen (about 30 lines). Longer: split it in two cells, each with its sentence.

**Line length: at most 88 characters** (code and comments), the same limit as the `multigeom` package.
- Format the code cells with `ruff format` and check them with `ruff check` (the settings are in `ruff.toml`; both work on the `.ipynb` files and keep the `%%time` line). Do not wrap lines by hand to imitate the formatter.
- Markdown cells are not limited: they wrap at the width of the window.
- Write the explanation of a parameter **above** the assignment, not after it: the formatter wraps `NAME = value  # long comment` into parentheses, and the tests find the line `NAME = value`. A short inline comment (it fits in 88 characters) is fine.
- Break a long message into adjacent strings inside the call: `print("first half " "second half")`.

**Layout of a cell: short paragraphs, each with a comment**
- A cell reads as a sequence of paragraphs of 2 to 8 lines, separated by **one blank line**, and each paragraph starts with a comment that names its step. Never leave a wall of more than about 12 lines without a blank line.
- Put a blank line before every comment, before and after a loop or an `if` block, and after a statement that the formatter wrapped over several lines. Keep together the simple statements that belong to one step (for example the `set_xlabel`, `set_xticks`, `set_yticks` of one axis).
- Keep statements short enough for the formatter to leave them on one to three lines. If a call is exploded to one argument per line, name a part of it (`rows = {"height_ratios": [1, 1.1]}`, `first = axes[0, 0]`), leave out the arguments that are defaults, or move the drawing into a function defined in the custom-functions cell. Ten lines to read one call is a sign that the code needs a name.
- Put the comment of a long statement **above** it. At the end of the line it forces the formatter to wrap the statement around the comment.

Before (a wall, and a call with one argument per line):

```python
fig, axes = plt.subplots(2, 6, figsize=(12, 4.3), gridspec_kw={"height_ratios": [1, 1.1]})
for n, ax in enumerate(axes[0]):
    ax.imshow(teacher.filters[n, 0].numpy(), cmap="gray")
    ax.set_title(f"neuron {n}", fontsize=9)
axes[0, 0].text(-0.1, 0.5, "receptive\nfields", transform=axes[0, 0].transAxes, ha="right", va="center")
for n, ax in enumerate(axes[1]):
    ax.scatter(c_train[:300, n], r_train[:300, n], s=4, alpha=0.6)
```

After (three paragraphs, each with its comment, every call on one to three lines):

```python
# 1. a figure of two rows: receptive fields (top), recorded responses (bottom)
rows = {"height_ratios": [1, 1.1]}
fig, axes = plt.subplots(2, 6, figsize=(12, 4.3), gridspec_kw=rows)

# 2. top row: the receptive field of the first six neurons
for n, ax in enumerate(axes[0]):
    ax.imshow(teacher.filters[n, 0].numpy(), cmap="gray")
    ax.set_title(f"neuron {n}", fontsize=9)

first = axes[0, 0]
first.text(-0.1, 0.5, "receptive\nfields", transform=first.transAxes, ha="right")

# 3. bottom row: recorded against true response
for n, ax in enumerate(axes[1]):
    ax.scatter(c_train[:300, n], r_train[:300, n], s=4, alpha=0.6)
```

**Comments**
- Comment the abstract structure of the cell, not every line. In a cell of more than about eight lines, start each key step with a numbered comment: `# 1. load the models`, `# 2. estimate G(x) and J(x)`, `# 3. plot`. A reader should understand the whole cell by reading only these comments.
- Say what a step does and why. Do not restate the code (`# loop over the images`). Keep the short inline comments that explain a constant or a unit.
- Group the imports under one comment per group (`# libraries and the package`).

**Content**

- The first code cell installs the package on Google Colab only: `if "google.colab" in sys.modules: %pip install -q --no-deps --pre multigeom` (plus what Colab lacks, such as `diffusers` for notebook 2). Nothing runs on a laptop.
- The next cell moves to the repository root (the folder with `environment.yml`) and, when the notebook imports from `figures/`, adds the root to `sys.path`.
- Parameters are upper-case names in one cell, with a one-line comment each. The tests override them.
- Data comes from the package: `download_tinyimagenet()`, `download_tvsd()`, `download_things_images(accept_license=True)` in `multigeom.datasets`. Never add a `.sh` script or ask the reader to download by hand.
- Import from `multigeom.<entry point>`: `datasets`, `encoders`, `diffusion`, `geometry`, `analysis`, `pipeline`, `utils`. Plot helpers for the paper's figures are in `figures/` of this repository, not in the package.
- Move repeated logic into the package, with a test, instead of growing the notebook.
- Every code cell starts with `%%time`, so that each cell shows how long it takes. (The tests ignore the `%%` lines.)
- Use `show_figure(fig)` from `multigeom.utils` to display a figure.

## 4. Editing the files safely

- Editing the JSON: `json.load` then `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` reproduces our notebooks byte for byte, so structural edits (splitting or inserting cells) can be done on the loaded notebook. Check `git diff --stat` before committing: a few lines are expected, hundreds are not (then undo and use exact string replacements instead). Check that the file is still valid JSON.
- When you split a code cell, the outputs stay with the cell that produces them (the script cell); give every new cell a unique `id` and `"execution_count": null`.
- Keep the stored outputs of a notebook unless the task is to refresh them. When the user re-runs a notebook and saves it, their saved version is the reference: read it to see how they want it written, and learn from it (this is how the first-cell template above was set).
- The tests in `tests/` find cells by a unique line (the markers in `tests/nb_helpers.py`, such as `lmf.launch(job)` or `ns["show_figure"]`). If you change one of these lines, change the marker too, and run the tests.
- A notebook that the user has open may have unsaved or uncommitted changes. Look at `git status` first, and do not overwrite or commit changes you did not make.
- `multigeom/docs/source/demo_notebooks/` holds copies of the notebooks; refresh them with the documentation sync and do not edit them by hand.
- **Check that every name is defined by an earlier cell**, in the order of the notebook (moving a line between cells easily leaves an import behind; this is how a `snapshot_download` that was never imported reached a reader). The check below prints each name that is used before it is defined; it must print nothing. Then run the setup, import and download cells of the notebook, with any multi-gigabyte download replaced by a `print`.

```python
import ast, builtins, glob, json, sys

for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "notebooks/*.ipynb")):
    nb = json.load(open(f)); defined = set(dir(builtins)) | {"get_ipython"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines = "".join(c["source"]).splitlines()
        code = "\n".join((l[: len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("%", "!")) else l for l in lines)
        tree = ast.parse(code)
        inside = {id(m) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda)) for m in ast.walk(n) if m is not n}   # run later, when called
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in inside}
        stored = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): stored.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)): stored.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)): stored |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.arg): stored.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name: stored.add(n.name)
        for name in sorted(used - defined - stored):
            print(f"{f.split('/')[-1]} cell {i}: {name} is not defined by this cell or an earlier one")
        defined |= stored
```

- Run `ruff format --check .` and `ruff check .` from the repository root, then the tests (`CUDA_VISIBLE_DEVICES= python -m pytest -q tests`, in the `tutorials-env` environment; the GPU of the shared machine is often full) and report what passed, what failed and what you could not run.

## 5. Names

Name modules, functions, files and arguments after what they do, not after the paper (`receptive_field`, not `fig3`). Paper figure numbers may appear in `figures/` of this repository and in short "as in paper Fig. N" notes, never in the package.
