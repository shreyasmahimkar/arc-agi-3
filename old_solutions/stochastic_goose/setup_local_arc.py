import os
import zipfile
import re
import subprocess
import sys

def patch_wheel(wheel_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Patching {wheel_path}...")
    with zipfile.ZipFile(wheel_path, 'r') as zin, zipfile.ZipFile(out_path, 'w') as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename.endswith('METADATA'):
                text = content.decode('utf-8')
                text = re.sub(r'Requires-Python: >=3\.12', 'Requires-Python: >=3.9', text)
                text = re.sub(r'Requires-Dist: pillow >=12\.1\.1', 'Requires-Dist: pillow', text)
                content = text.encode('utf-8')
            zout.writestr(item, content)

wheels_dir = "./arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
patched_dir = "./patched_wheels"

arc_agi_orig = os.path.join(wheels_dir, "arc_agi-0.9.8-py3-none-any.whl")
arc_agi_patched = os.path.join(patched_dir, "arc_agi-0.9.8-py3-none-any.whl")

arcengine_orig = os.path.join(wheels_dir, "arcengine-0.9.3-py3-none-any.whl")
arcengine_patched = os.path.join(patched_dir, "arcengine-0.9.3-py3-none-any.whl")

patch_wheel(arc_agi_orig, arc_agi_patched)
patch_wheel(arcengine_orig, arcengine_patched)

print("\nUninstalling any broken packages...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "arc-agi", "arcengine"])

print("\nInstalling base dependencies from PyPI (with Pillow override)...")
subprocess.run([sys.executable, "-m", "pip", "install", "flask", "matplotlib", "pillow<12", "pydantic", "python-dotenv", "requests"])

print("\nInstalling patched ARC wheels directly...")
subprocess.run([sys.executable, "-m", "pip", "install", arcengine_patched, "--no-deps"])
subprocess.run([sys.executable, "-m", "pip", "install", arc_agi_patched, "--no-deps"])

print("\nSetup Complete!")

print("\nUninstalling any broken packages...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "arc-agi", "arcengine"])

print("\nInstalling base dependencies from PyPI (with Pillow override)...")
subprocess.run([sys.executable, "-m", "pip", "install", "flask", "matplotlib", "pillow<12", "pydantic", "python-dotenv", "requests"])

print("\nInstalling patched ARC wheels directly...")
subprocess.run([sys.executable, "-m", "pip", "install", arcengine_patched, "--no-deps"])
subprocess.run([sys.executable, "-m", "pip", "install", arc_agi_patched, "--no-deps"])

print("\nSetup Complete!")
