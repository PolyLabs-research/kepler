#!/usr/bin/env python3
"""Quick fix for physics_laws generator bug"""

from pathlib import Path

generators_file = Path("generators.py")
content = generators_file.read_text(encoding='utf-8')  # ← ADD ENCODING

# Fix the N_MOLES line
old_line = '                "N_MOLES": np.random.uniform(0.1 if tier == 3 else 0.5, 5),'
new_line = '                "N_MOLES": np.random.uniform(0.1 if tier == 3 else 0.5, 5, self.n_samples),'

if old_line in content:
    content = content.replace(old_line, new_line)
    generators_file.write_text(content, encoding='utf-8')  # ← ADD ENCODING
    print("✓ Fixed physics_laws generator")
else:
    print("⚠ Could not find line to fix (maybe already fixed?)")