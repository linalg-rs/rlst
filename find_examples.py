"""
Script to make .sh file to run all examples.

Type of run can be controlled by including a comment starting with `//?` in
a file. For example, including:

```rust
//? mpirun -n {{NPROCESSES}} --features "mpi"
```

would make this script use mpirun with the -n flag set to a number of processes
and the "mpi" features turned on.
"""

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

files = []
for folder in os.listdir(root_dir):
    if not folder.startswith("."):
        sub_dir = os.path.join(root_dir, folder)
        example_dir = os.path.join(sub_dir, "examples")
        if os.path.isdir(sub_dir) and os.path.isdir(example_dir):
            for file in os.listdir(example_dir):
                if (
                    not file.startswith(".")
                    and file.endswith(".rs")
                    and os.path.isfile(os.path.join(example_dir, file))
                ):
                    files.append((os.path.join(example_dir, file), file[:-3]))

# Check that two examples do not share a name
for i, j in files:
    assert len([a for a, b in files if b == j]) == 1

lines = []
for file, example_name in files:
    with open(file) as f:
        for line in f:
            if line.startswith("//?"):
                line = line[3:].strip()
                if " " in line:
                    cmd, options = line.split(" ", 1)
                else:
                    cmd = line
                    options = None
                break
        else:
            cmd = "run"
            options = None

    command = f"cargo {cmd} --example {example_name} --release"
    if options is not None:
        command += f" {options}"
    if "{{NPROCESSES}}" in command:
        for n in [2, 4]:
            info = f"Running {example_name} on {n} processes"
            lines.append(f"echo \"\n{'=' * len(info)}\n{info}\n{'=' * len(info)}\n\"")
            lines.append(command.replace("{{NPROCESSES}}", f"{n}"))
    else:
        info = f"Running {example_name}"
        lines.append(f"echo \"\n{'=' * len(info)}\n{info}\n{'=' * len(info)}\n\"")
        lines.append(command)

with open("examples.sh", "w") as f:
    f.write(" && \\\n".join(lines))
