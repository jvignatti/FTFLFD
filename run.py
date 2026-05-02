import subprocess
import sys

COMMANDS = {
    "install": [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    "test": [sys.executable, "-m", "pytest", "tests/"],
    "lint": [sys.executable, "-m", "ruff", "check", "src/"],
    "check-leakage": [sys.executable, "-m", "src.leakage.checks"],
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("FTFLFD — available commands:")
        for name in COMMANDS:
            print(f"  python run.py {name}")
        sys.exit(0)

    cmd = COMMANDS[sys.argv[1]]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
