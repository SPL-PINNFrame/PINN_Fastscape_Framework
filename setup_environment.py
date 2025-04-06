# PINN_Fastscape_Framework/setup_environment.py

import os
import subprocess
import sys
import platform

def run_command(command, cwd=None, check=True):
    """Runs a shell command and prints output/errors."""
    print(f"\nExecuting command: {' '.join(command)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    try:
        # Use shell=True cautiously, especially on non-Windows for complex commands or variable expansion.
        # For basic pip commands, it's often needed on Windows.
        use_shell = platform.system() == "Windows"
        process = subprocess.run(
            command,
            cwd=cwd,
            check=check, # Raise exception on non-zero exit code if True
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', # Specify encoding
            shell=use_shell
        )
        # Print stdout only if it contains non-whitespace characters
        if process.stdout and process.stdout.strip():
            print("Output:\n", process.stdout)
        # Print stderr only if it contains non-whitespace characters
        if process.stderr and process.stderr.strip():
            print("Error output:\n", process.stderr)

        if check and process.returncode != 0:
             print(f"Command failed with exit code {process.returncode}")
             return False
        print("Command executed successfully.")
        return True
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it installed and in your PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stdout and e.stdout.strip():
             print("Output:\n", e.stdout)
        if e.stderr and e.stderr.strip():
             print("Error output:\n", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return False

def main():
    """Main function to setup the environment."""
    # Use absolute path for project root based on this script's location
    project_root = os.path.dirname(os.path.abspath(__file__))
    fortran_lib_path = os.path.join(project_root, 'external', 'fastscapelib-fortran')
    requirements_path = os.path.join(project_root, 'requirements.txt')

    print("--- Starting Environment Setup for PINN Fastscape Framework ---")
    print(f"Project Root: {project_root}")

    # 1. Check for Fortran Compiler (Informational)
    print("\nStep 1: Checking for Fortran Compiler (Informational)")
    print("This script requires a Fortran compiler (like gfortran) to be installed")
    print("and available in your system's PATH for compiling 'fastscapelib-fortran'.")
    print("Please ensure a compiler is installed before proceeding if compilation fails.")
    # Example check (optional, might not be robust across all systems)
    try:
        run_command(['gfortran', '--version'], check=False) # Check=False as we only want to see if it runs
    except FileNotFoundError:
        print("Warning: 'gfortran' command not found. Fortran compilation might fail.")

    # 2. Compile and install fastscapelib-fortran
    print(f"\nStep 2: Compiling and installing 'fastscapelib-fortran' from {fortran_lib_path}")
    if not os.path.isdir(fortran_lib_path):
        print(f"Error: Directory not found: {fortran_lib_path}")
        print("Please ensure you have moved 'fastscapelib-fortran' into the 'external' directory.")
        sys.exit(1)

    # Command: pip install --no-build-isolation --editable .
    # Using sys.executable ensures we use the pip associated with the current python interpreter
    compile_command = [
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation", # Avoids issues with isolated build environments
        "--editable", "."       # Install in editable mode from current dir
    ]
    if not run_command(compile_command, cwd=fortran_lib_path):
        print("\nError: Failed to compile/install 'fastscapelib-fortran'.")
        print("Please check the output above for errors. Ensure a Fortran compiler is installed and accessible.")
        sys.exit(1)

    # 3. Install Python dependencies
    print(f"\nStep 3: Installing Python requirements from {requirements_path}")
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        sys.exit(1)

    # Command: pip install -r requirements.txt
    install_command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    if not run_command(install_command, cwd=project_root):
        print("\nError: Failed to install Python requirements.")
        print("Please check the output above for errors.")
        sys.exit(1)

    print("\n--- Environment Setup Completed Successfully ---")
    print(f"Setup was performed using Python interpreter: {sys.executable}")
    print("You should now be able to run the data generation and training scripts within the same environment.")

if __name__ == "__main__":
    main()
