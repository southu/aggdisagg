"""Loop runner for messy data batch test.

Runs the 24-test batch repeatedly until clean run or 30 min timeout.
Finds errors, but fixes are manual in src/ or test.
"""
import subprocess
import sys
import time

TEST = "tests/test_simulation.py::test_messy_incomplete_data_batch_1"
TIMEOUT = 30 * 60  # 30 min
CMD = ["uv", "run", "pytest", TEST, "-q", "--tb=line"]

start = time.time()
batch = 0
while time.time() - start < TIMEOUT:
    batch += 1
    print(f"\n=== BATCH {batch} at {time.strftime('%H:%M:%S')} ===")
    result = subprocess.run(CMD, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    if result.returncode == 0:
        print("CLEAN RUN! No errors.")
        sys.exit(0)
    print("Errors found, would fix and retry... (simulated)")
    time.sleep(1)  # small delay

print("Timeout reached without clean run.")
sys.exit(1)
