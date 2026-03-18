"""
Patch vLLM's llama_tool_parser.py to handle both:
  - Llama format:  {"name": "func", "parameters": {...}}
  - OpenAI format: {"type": "function", "function": {"name": "func", "arguments": "{...}"}}

Llama 3.3 70B generates the OpenAI format when given large tool sets.
The regex in vLLM also finds nested JSON objects (e.g. argument dicts), so we
must skip any parsed object that has no resolvable tool name rather than
falling back to an empty string.
"""
import sys

FILEPATH = "/opt/conda/lib/python3.12/site-packages/vllm/entrypoints/openai/tool_parsers/llama_tool_parser.py"

with open(FILEPATH) as f:
    lines = f.readlines()

# Find the line: obj = json.loads(json_obj)
json_loads_idx = None
for i, line in enumerate(lines):
    if "obj = json.loads(json_obj)" in line:
        json_loads_idx = i
        break

if json_loads_idx is None:
    print("ERROR: could not find 'obj = json.loads(json_obj)'")
    sys.exit(1)

# Check if already patched with fn_name logic
for line in lines:
    if "fn_name = obj.get" in line:
        print("Already patched with fn_name logic, nothing to do")
        sys.exit(0)

# Get indentation of the json_loads line (should be 16 spaces)
indent = len(lines[json_loads_idx]) - len(lines[json_loads_idx].lstrip())
ind = " " * indent

# Find the name= line after json_loads (look in next 10 lines)
name_idx = None
for i in range(json_loads_idx + 1, json_loads_idx + 10):
    if "name=" in lines[i] and i < len(lines):
        name_idx = i
        break

if name_idx is None:
    print("ERROR: could not find name= line after json.loads")
    sys.exit(1)

# Build the replacement block.
# Insert fn_name extraction + continue after json_loads, then replace name= line.
new_lines = list(lines)

# 1. Insert fn_name + continue after json_loads line
insert = [
    f"{ind}fn_name = (obj.get(\"name\") or\n",
    f"{ind}           (obj.get(\"function\") or {{}}).get(\"name\"))\n",
    f"{ind}if not fn_name:\n",
    f"{ind}    continue  # skip nested JSON objects that aren't tool calls\n",
]
new_lines = new_lines[:json_loads_idx + 1] + insert + new_lines[json_loads_idx + 1:]

# 2. name= line is now shifted by len(insert)
name_idx += len(insert)
old_name_line = new_lines[name_idx]

# Find the name= expression — it may span two lines (our previous patch made it multiline)
# Collect all lines of the name= expression until we see the closing comma
name_end = name_idx
paren_depth = 0
started = False
for i in range(name_idx, name_idx + 5):
    for ch in new_lines[i]:
        if ch == '(':
            paren_depth += 1
            started = True
        elif ch == ')':
            paren_depth -= 1
    if started and paren_depth <= 0:
        name_end = i
        break

# Replace the name= block (possibly multi-line) with a single line
name_indent = len(old_name_line) - len(old_name_line.lstrip())
name_ind = " " * name_indent
new_name_line = f"{name_ind}name=fn_name,\n"
new_lines = new_lines[:name_idx] + [new_name_line] + new_lines[name_end + 1:]

with open(FILEPATH, "w") as f:
    f.writelines(new_lines)

print(f"Patched {FILEPATH}")
print(f"  - Inserted fn_name + continue after line {json_loads_idx + 1}")
print(f"  - Replaced multi-line name= expression with name=fn_name")
