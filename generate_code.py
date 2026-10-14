import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "code.md"
CONTEXT = ROOT / "context.md"

# Common environment / build / cache folders that cause context bloat
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "ENV",
    "virtualenv",
    ".tox",
    ".nox",
    "site-packages",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
}

# Common env activation / helper scripts (usually not useful in context dumps)
SKIP_FILE_NAMES = {
    "activate",
    "activate.bat",
    "activate.ps1",
    "activate.fish",
    "deactivate.bat",
    "deactivate.ps1",
    "pyvenv.cfg",
}

def extract_tree_block(text: str) -> str:
    marker = "Folder structure (full, all files):"
    if marker not in text:
        return ""
    after = text.split(marker, 1)[1]
    start = after.find("```")
    if start == -1:
        return ""
    after = after[start + 3 :]
    end = after.find("```")
    if end == -1:
        return ""
    return after[:end].strip()


def is_in_skipped_dir(path: Path) -> bool:
    # If any directory component matches SKIP_DIRS, skip the file
    return any(part in SKIP_DIRS for part in path.parts)


def iter_source_files(root: Path):
    extensions = {".py", ".yaml", ".yml", ".md"}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix not in extensions:
            continue

        # Skip anything inside env/build/cache folders
        if is_in_skipped_dir(path.relative_to(root)):
            continue

        # Skip typical env helper files even if they match extensions elsewhere
        if path.name in SKIP_FILE_NAMES:
            continue

        # Exclude this script itself
        if path.name == os.path.basename(__file__):
            continue

        # Exclude the output file to prevent infinite recursion/loops
        if path.resolve() == OUTPUT.resolve():
            continue

        yield path


def main():
    tree_block = ""
    if CONTEXT.exists():
        tree_block = extract_tree_block(CONTEXT.read_text(encoding="utf-8"))

    with OUTPUT.open("w", encoding="utf-8") as out:
        out.write("# Application Code\n\n")
        if tree_block:
            out.write("## File Structure\n")
            out.write("```text\n")
            out.write(tree_block)
            out.write("\n```\n\n")

        for path in iter_source_files(ROOT):
            rel = path.relative_to(ROOT)
            out.write(f"## {rel}\n")

            suffix = path.suffix
            if suffix in {".yaml", ".yml"}:
                lang = "yaml"
            elif suffix == ".md":
                lang = "markdown"
            else:
                lang = "python"

            out.write(f"```{lang}\n")

            content = path.read_text(encoding="utf-8")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("```\n\n")

    print(f"✅ Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
