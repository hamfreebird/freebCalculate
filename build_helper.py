#!/usr/bin/env python3
"""
Build script for freebirdcal package

This script helps with building, testing, and uploading the freebirdcal package.
It provides an interactive menu for common packaging tasks.

Usage:
    python build.py                     # Interactive menu
    python build.py --build             # Just build the package
    python build.py --clean             # Just clean build files
    python build.py --install           # Build and install locally
    python build.py --upload            # Build and upload to PyPI
    python build.py --test-pypi         # Build and upload to TestPyPI
    python build.py --all               # Full build and test cycle
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Colors for terminal output (optional)
try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = ""

    class Style:
        BRIGHT = RESET_ALL = ""


class BuildHelper:
    """Helper class for building and packaging freebirdcal"""

    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.egg_info_dir = self.project_root / "freebirdcal.egg-info"

        # Package info from pyproject.toml
        self.package_name = "freebirdcal"
        self.version = "0.1.0"  # Should be read from pyproject.toml

    def print_header(self, text: str) -> None:
        """Print a formatted header"""
        print(f"\n{'=' * 60}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        print(f"{'=' * 60}")

    def print_success(self, text: str) -> None:
        """Print success message"""
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

    def print_warning(self, text: str) -> None:
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

    def print_error(self, text: str) -> None:
        """Print error message"""
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

    def print_info(self, text: str) -> None:
        """Print info message"""
        print(f"{Fore.CYAN}ℹ {text}{Style.RESET_ALL}")

    def run_command(
        self, cmd: str, check: bool = True, capture: bool = False
    ) -> Tuple[bool, str]:
        """
        Run a shell command and return success status and output

        Args:
            cmd: Command to run
            check: Whether to raise an exception on failure
            capture: Whether to capture and return output

        Returns:
            Tuple of (success, output)
        """
        self.print_info(f"Running: {cmd}")

        try:
            if capture:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=check,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
                output = result.stdout + result.stderr
            else:
                result = subprocess.run(
                    cmd, shell=True, check=check, cwd=self.project_root
                )
                output = ""

            return True, output

        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError output:\n{e.stderr}"
            self.print_error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running command: {e}"
            self.print_error(error_msg)
            return False, error_msg

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        self.print_header("Checking Prerequisites")

        required_packages = ["build", "twine", "setuptools", "wheel"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                self.print_success(f"{package} is installed")
            except ImportError:
                missing_packages.append(package)
                self.print_warning(f"{package} is not installed")

        if missing_packages:
            self.print_info(f"Missing packages: {', '.join(missing_packages)}")
            install_cmd = (
                f"{sys.executable} -m pip install {' '.join(missing_packages)}"
            )
            print(f"\nTo install missing packages:")
            print(f"  {install_cmd}")

            response = input(f"\nInstall missing packages now? (y/n): ").lower().strip()
            if response == "y":
                success, _ = self.run_command(install_cmd)
                return success
            else:
                return False

        return True

    def clean_build_files(self) -> bool:
        """Clean previous build files"""
        self.print_header("Cleaning Build Files")

        dirs_to_remove = [self.dist_dir, self.build_dir, self.egg_info_dir]
        files_to_remove = list(self.project_root.glob("*.egg-info"))

        all_dirs = dirs_to_remove + files_to_remove

        removed_count = 0
        for path in all_dirs:
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    self.print_success(f"Removed: {path.name}")
                    removed_count += 1
                except Exception as e:
                    self.print_error(f"Failed to remove {path}: {e}")

        if removed_count == 0:
            self.print_info("No build files to clean")

        return True

    def build_package(self) -> bool:
        """Build the package (sdist and wheel)"""
        self.print_header("Building Package")

        # Check if build tool is available
        try:
            import build_helper
        except ImportError:
            self.print_warning("Build package not installed, installing...")
            success, _ = self.run_command(f"{sys.executable} -m pip install build")
            if not success:
                return False

        # Run the build
        success, output = self.run_command(f"{sys.executable} -m build", capture=True)

        if success:
            # List generated files
            if self.dist_dir.exists():
                self.print_success("Build completed successfully!")
                print("\nGenerated files:")
                for file in sorted(self.dist_dir.iterdir()):
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.name} ({size_mb:.2f} MB)")

                # Read version from pyproject.toml if possible
                pyproject_path = self.project_root / "pyproject.toml"
                if pyproject_path.exists():
                    try:
                        import tomllib

                        with open(pyproject_path, "rb") as f:
                            data = tomllib.load(f)
                        self.version = data.get("project", {}).get(
                            "version", self.version
                        )
                    except ImportError:
                        try:
                            import tomli as tomllib

                            with open(pyproject_path, "rb") as f:
                                data = tomllib.load(f)
                            self.version = data.get("project", {}).get(
                                "version", self.version
                            )
                        except ImportError:
                            pass

                self.print_info(f"Package: {self.package_name} v{self.version}")
            return True
        else:
            self.print_error("Build failed!")
            if output:
                print(f"\nBuild output:\n{output[:1000]}...")  # Show first 1000 chars
            return False

    def test_install(self) -> bool:
        """Test install the package locally"""
        self.print_header("Testing Local Installation")

        # Find the latest wheel file
        wheel_files = list(self.dist_dir.glob("*.whl"))
        if not wheel_files:
            self.print_error("No wheel files found in dist/ directory")
            self.print_info("You need to build the package first")
            return False

        latest_wheel = max(wheel_files, key=lambda f: f.stat().st_mtime)

        print(f"Installing: {latest_wheel.name}")

        # First uninstall any existing version
        self.run_command(
            f"{sys.executable} -m pip uninstall {self.package_name} -y", check=False
        )

        # Install the wheel
        success, output = self.run_command(
            f"{sys.executable} -m pip install {latest_wheel}", capture=True
        )

        if success:
            self.print_success("Installation successful!")

            # Test the installation
            print("\nTesting package import...")
            test_code = f"""
import {self.package_name}
print("Package imported successfully!")
print(f"Version: {{ {self.package_name}.__version__ }}")
print(f"Author: {{ {self.package_name}.__author__ }}")

# Test a simple import
try:
    from {self.package_name} import AstronomicalSimulator
    print("✓ AstronomicalSimulator imported")
except ImportError:
    print("✗ Could not import AstronomicalSimulator (requires optional dependencies)")
"""

            try:
                result = subprocess.run(
                    [sys.executable, "-c", test_code],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
                print(result.stdout)
                if result.stderr:
                    print(f"Warnings/Errors:\n{result.stderr}")
            except Exception as e:
                self.print_warning(f"Test import had issues: {e}")

            return True
        else:
            self.print_error("Installation failed!")
            if output:
                print(f"\nInstallation output:\n{output[:1000]}...")
            return False

    def upload_to_pypi(self, test: bool = False) -> bool:
        """Upload package to PyPI or TestPyPI"""
        repo = "testpypi" if test else "pypi"
        self.print_header(f"Uploading to {repo.upper()}")

        # Check if twine is available
        try:
            import twine
        except ImportError:
            self.print_warning("Twine not installed, installing...")
            success, _ = self.run_command(f"{sys.executable} -m pip install twine")
            if not success:
                return False

        # Check if dist directory exists and has files
        if not self.dist_dir.exists() or not list(self.dist_dir.iterdir()):
            self.print_error(f"No files found in dist/ directory")
            self.print_info("You need to build the package first")
            return False

        # Upload command
        upload_cmd = f"{sys.executable} -m twine upload --repository {repo} dist/*"

        self.print_info("Note: You'll need PyPI credentials")
        self.print_info("Username: __token__")
        self.print_info("Password: Your API token")
        print()

        response = input(f"Proceed with upload to {repo}? (y/n): ").lower().strip()
        if response != "y":
            self.print_info("Upload cancelled")
            return False

        success, output = self.run_command(upload_cmd, capture=True)

        if success:
            self.print_success(f"Upload to {repo} completed successfully!")
            if test:
                print(f"\nTest installation command:")
                print(
                    f"  pip install --index-url https://test.pypi.org/simple/ {self.package_name}"
                )
            else:
                print(f"\nInstallation command:")
                print(f"  pip install {self.package_name}")
            return True
        else:
            self.print_error(f"Upload to {repo} failed!")
            if output:
                print(f"\nUpload output:\n{output[:1000]}...")
            return False

    def run_full_build(self) -> bool:
        """Run full build and test cycle"""
        self.print_header("Running Full Build Cycle")

        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Cleaning build files", self.clean_build_files),
            ("Building package", self.build_package),
            ("Testing installation", self.test_install),
        ]

        for step_name, step_func in steps:
            print(f"\n{Fore.MAGENTA}▶ {step_name}{Style.RESET_ALL}")
            if not step_func():
                self.print_error(f"Step '{step_name}' failed, aborting")
                return False

        self.print_success("Full build cycle completed successfully!")

        # Ask about upload
        print(f"\n{Fore.YELLOW}Upload Options:{Style.RESET_ALL}")
        print("1. Upload to TestPyPI (for testing)")
        print("2. Upload to PyPI (production)")
        print("3. Skip upload")

        choice = input("\nSelect upload option (1-3): ").strip()

        if choice == "1":
            return self.upload_to_pypi(test=True)
        elif choice == "2":
            return self.upload_to_pypi(test=False)
        else:
            self.print_info("Skipping upload")
            return True

    def interactive_menu(self) -> None:
        """Display interactive menu"""
        while True:
            self.print_header(f"{self.package_name} Build Helper")

            print(f"{Fore.CYAN}Select an option:{Style.RESET_ALL}")
            print("1. Run full build cycle (clean, build, test)")
            print("2. Clean build files only")
            print("3. Build package only")
            print("4. Test local installation")
            print("5. Upload to PyPI (production)")
            print("6. Upload to TestPyPI (for testing)")
            print("7. Check prerequisites")
            print("8. Exit")

            try:
                choice = input(
                    f"\n{Fore.GREEN}Enter choice (1-8): {Style.RESET_ALL}"
                ).strip()

                if choice == "1":
                    self.run_full_build()
                elif choice == "2":
                    self.clean_build_files()
                elif choice == "3":
                    self.build_package()
                elif choice == "4":
                    self.test_install()
                elif choice == "5":
                    self.upload_to_pypi(test=False)
                elif choice == "6":
                    self.upload_to_pypi(test=True)
                elif choice == "7":
                    self.check_prerequisites()
                elif choice == "8":
                    self.print_info("Exiting...")
                    break
                else:
                    self.print_error(f"Invalid choice: {choice}")

                if choice != "8":
                    input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}End of input{Style.RESET_ALL}")
                break


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build helper for freebirdcal package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                   # Interactive menu
  %(prog)s --build           # Build package only
  %(prog)s --clean           # Clean build files only
  %(prog)s --install         # Build and test install
  %(prog)s --upload          # Build and upload to PyPI
  %(prog)s --test-pypi       # Build and upload to TestPyPI
  %(prog)s --all             # Full build cycle (clean, build, test)
        """,
    )

    parser.add_argument("--build", action="store_true", help="Build package only")
    parser.add_argument("--clean", action="store_true", help="Clean build files only")
    parser.add_argument(
        "--install", action="store_true", help="Build and test install locally"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Build and upload to PyPI"
    )
    parser.add_argument(
        "--test-pypi", action="store_true", help="Build and upload to TestPyPI"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run full build cycle (clean, build, test)"
    )
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    parser.add_argument(
        "--interactive", action="store_true", help="Launch interactive menu (default)"
    )

    args = parser.parse_args()

    helper = BuildHelper()

    # If no arguments, show interactive menu
    if not any(vars(args).values()):
        helper.interactive_menu()
        return

    # Process command line arguments
    success = True

    if args.check:
        success = helper.check_prerequisites()

    if args.clean:
        success = success and helper.clean_build_files()

    if args.build:
        success = success and helper.build_package()

    if args.install:
        if not helper.build_package():
            success = False
        else:
            success = success and helper.test_install()

    if args.upload:
        if not helper.build_package():
            success = False
        else:
            success = success and helper.upload_to_pypi(test=False)

    if args.test_pypi:
        if not helper.build_package():
            success = False
        else:
            success = success and helper.upload_to_pypi(test=True)

    if args.all:
        success = helper.run_full_build()

    if args.interactive:
        helper.interactive_menu()
        return

    # Exit with appropriate code
    if success:
        helper.print_success("All operations completed successfully!")
        sys.exit(0)
    else:
        helper.print_error("Some operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
