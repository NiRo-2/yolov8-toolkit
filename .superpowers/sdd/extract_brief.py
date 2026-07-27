from pathlib import Path
import re
import sys

plan = Path(sys.argv[1])
n = int(sys.argv[2])
out = Path(sys.argv[3])
text = plan.read_text(encoding="utf-8")
lines = text.splitlines(True)
result = []
intask = False
infence = False
pat = re.compile(rf"^#+\s+Task\s+{n}([^0-9]|$)")
any_task = re.compile(r"^#+\s+Task\s+[0-9]+")
for line in lines:
    if line.startswith("```"):
        infence = not infence
    if not infence and any_task.match(line):
        intask = bool(pat.match(line))
    if intask:
        result.append(line)
if not result:
    raise SystemExit(f"task {n} not found")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(result), encoding="utf-8")
print(f"wrote {out}: {len(result)} lines")
