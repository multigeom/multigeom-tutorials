"""
Add `time: <duration>` at the end of the markdown cell that introduces a code cell,
from the `%%time` output stored in the notebook by its last run (the cells that have
no stored time are listed). Lines already there are kept. Usage, from the repository
root:

    python scripts/add_cell_times.py
"""

import glob
import json
import re


def seconds(text):
    m = re.search(r"Wall time:\s*([^\n]+)", text)
    if not m:
        return None
    s = m.group(1).strip()
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)\s*(h|min|ms|μs|µs|us|s)\b", s):
        v = float(value)
        total += (
            v
            * {
                "h": 3600,
                "min": 60,
                "s": 1,
                "ms": 1e-3,
                "μs": 1e-6,
                "µs": 1e-6,
                "us": 1e-6,
            }[unit]
        )
    return total


def label(sec):
    if sec < 1:
        return "<1 sec"
    if sec < 60:
        return "1 sec" if int(sec) == 1 else f"{int(sec)} secs"
    return f"{int(round(sec / 60))} min"


added, missing = 0, []
for f in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(f))
    cells = nb["cells"]
    changed = False
    for i in range(len(cells) - 1):
        a, b = cells[i], cells[i + 1]
        if a["cell_type"] != "markdown" or b["cell_type"] != "code":
            continue
        text = "".join(a["source"])
        if re.search(r"(?m)^time: ", text):
            continue
        out = "".join(
            "".join(o.get("text", []))
            for o in b.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        sec = seconds(out)
        if sec is None:
            missing.append((f.split("/")[-1][:14], i + 1))
            continue
        text = text.rstrip("\n") + f"\n\ntime: {label(sec)}"
        ls = text.split("\n")
        a["source"] = [l + "\n" for l in ls[:-1]] + [ls[-1]]
        changed = True
        added += 1
    if changed:
        open(f, "w").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("time lines added:", added)
print("code cells without a stored time:", missing)
