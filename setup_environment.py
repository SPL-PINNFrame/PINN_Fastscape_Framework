# PINN_Fastscape_Framework/setup_environment.py
# NOTE: This script is now primarily for informational purposes.
# Environment setup should be done using Conda and the environment.yml file.

import os
import subprocess
import sys
import platform

def run_command(command, cwd=None, check=True):
    """Runs a shell command and prints output/errors."""
    print(f"\nExecuting command: {' '.join(command)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    try:
        use_shell = platform.system() == "Windows"
        process = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            shell=use_shell
        )
        if process.stdout and process.stdout.strip():
            print("Output:\n", process.stdout)
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
        traceback.print_exc()
        return False

def main():
    """Main function to guide environment setup using Conda."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(project_root, 'environment.yml')
    requirements_path = os.path.join(project_root, 'requirements.txt') # For dev tools

    print("--- Environment Setup Guide for PINN Fastscape Framework ---")
    print("\nIMPORTANT: Environment setup is now managed using Conda.")
    print("Please ensure you have Anaconda or Miniconda installed.")

    if not os.path.exists(env_file_path):
        print(f"\nError: environment.yml not found at {env_file_path}")
        print("Cannot proceed with Conda environment creation.")
        sys.exit(1)

    print(f"\n1. Create the Conda environment from '{os.path.basename(env_file_path)}':")
    print("   Open your terminal (Anaconda Prompt, PowerShell, bash, etc.)")
    print(f"   Navigate to the project root directory: cd \"{project_root}\"")
    print("   Run the following command:")
    print("   conda env create -f environment.yml")
    print("\n   This will create a new environment (likely named 'pinn-fastscape-env')")
    print("   and install Python, PyTorch, fastscape, and other core dependencies.")
    print("   Note: This step might take some time.")

    print("\n2. Activate the Conda environment:")
    print("   conda activate pinn-fastscape-env  (Replace 'pinn-fastscape-env' if you changed the name in environment.yml)")

    print("\n3. (Optional) Install development/testing tools using pip:")
    if os.path.exists(requirements_path):
        print(f"   Make sure the 'pinn-fastscape-env' environment is activated.")
        print(f"   Run the following command to install tools like pytest:")
        # Construct the pip install command using sys.executable to ensure correct pip
        pip_install_command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        print(f"   {' '.join(pip_install_command)}")
        # Optionally, run the command here, but it's better to guide the user
        # print("\n   Attempting to install development requirements...")
        # run_command(pip_install_command, cwd=project_root)
    else:
        print("   requirements.txt not found, skipping pip install step for dev tools.")


    print("\n--- Setup Guide Finished ---")
    print("After completing the steps above, your environment should be ready.")
    print("Remember to activate the 'pinn-fastscape-env' environment each time you work on the project.")

if __name__ == "__main__":
    main()
