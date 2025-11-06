from __future__ import annotations

import ast
import hashlib
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import docker
import toml
from docker.errors import ImageNotFound
from fastmcp import FastMCP  # type: ignore
from rich import print

import tiny_scientist

PACKAGE_ROOT = Path(tiny_scientist.__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}

DEFAULT_BASE_PACKAGES = [
    "torch",
    "datasets",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "requests",
    "pillow",
]


def _resolve_configured_packages() -> List[str]:
    docker_cfg = config.get("docker") if isinstance(config, dict) else None
    configured = None
    if isinstance(docker_cfg, dict):
        configured = docker_cfg.get("base_packages")
    if isinstance(configured, list) and all(isinstance(pkg, str) for pkg in configured):
        return configured
    return DEFAULT_BASE_PACKAGES.copy()


class DockerExperimentRunner:
    def __init__(
        self,
        docker_image: str = "tiny-scientist-ml",
        docker_base: str = "python:3.11-slim",
    ):
        self.docker_image = docker_image
        self.docker_base = docker_base
        self.docker_client = None
        self.use_docker = False
        self.base_packages = _resolve_configured_packages()
        self._base_package_set: Set[str] = set(self.base_packages)
        self._base_fingerprint = self._compute_base_fingerprint()
        last_colon = self.docker_image.rfind(":")
        last_slash = self.docker_image.rfind("/")
        if last_colon > last_slash:
            self._base_image_repo = self.docker_image[:last_colon]
        else:
            self._base_image_repo = self.docker_image
        self.base_image_tag = f"{self._base_image_repo}:{self._base_fingerprint}"

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.use_docker = True
            print("[Docker] Docker client initialized.")
        except Exception as e:
            print(f"[Docker] Docker not available: {e}")
            self.use_docker = False

    @staticmethod
    def detect_required_packages(
        pyfile: str, base_packages: Optional[Set[str]] = None
    ) -> List[str]:
        """Detect required packages from import statements in a Python file."""
        if base_packages is None:
            base_packages = set(DEFAULT_BASE_PACKAGES)

        # Common package name mappings (import_name -> pip_package_name)
        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "plotly": "plotly",
            "bokeh": "bokeh",
            "dash": "dash",
            "streamlit": "streamlit",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2-binary",
            "pymongo": "pymongo",
            "redis": "redis",
            "celery": "celery",
            "flask": "flask",
            "django": "django",
            "scipy": "scipy",
            "statsmodels": "statsmodels",
            "seaborn": "seaborn",
            "plotnine": "plotnine",
            "altair": "altair",
            "holoviews": "holoviews",
            "folium": "folium",
            "geopandas": "geopandas",
            "shapely": "shapely",
            "fiona": "fiona",
            "rasterio": "rasterio",
            "xarray": "xarray",
            "netcdf4": "netcdf4",
            "h5py": "h5py",
            "tables": "tables",
            "pyarrow": "pyarrow",
            "fastparquet": "fastparquet",
            "openpyxl": "openpyxl",
            "xlrd": "xlrd",
            "xlwt": "xlwt",
            "odf": "odfpy",
            "tabula": "tabula-py",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
            "pypdf": "pypdf",
            "reportlab": "reportlab",
            "weasyprint": "weasyprint",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "rst": "docutils",
            "sphinx": "sphinx",
            "mkdocs": "mkdocs",
            "jupyter": "jupyter",
            "ipython": "ipython",
            "notebook": "notebook",
            "jupyterlab": "jupyterlab",
            "voila": "voila",
            "nbconvert": "nbconvert",
            "papermill": "papermill",
            "nbdime": "nbdime",
            "nbstripout": "nbstripout",
            "jupytext": "jupytext",
        }

        # Standard library modules that don't need pip installation
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "random",
            "math",
            "statistics",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "argparse",
            "subprocess",
            "tempfile",
            "shutil",
            "glob",
            "re",
            "urllib",
            "http",
            "socket",
            "threading",
            "multiprocessing",
            "asyncio",
            "logging",
            "pickle",
            "copy",
            "typing",
            "abc",
            "enum",
            "dataclasses",
            "contextlib",
            "weakref",
            "gc",
            "inspect",
            "traceback",
            "pdb",
            "unittest",
            "doctest",
            "hashlib",
            "hmac",
            "base64",
            "zlib",
            "gzip",
            "bz2",
            "lzma",
            "zipfile",
            "tarfile",
            "csv",
            "configparser",
            "xml",
            "html",
            "email",
            "smtplib",
            "ftplib",
            "telnetlib",
            "poplib",
            "imaplib",
            "nntplib",
            "socketserver",
            "xmlrpc",
            "webbrowser",
            "cgi",
            "cgitb",
            "wsgiref",
            "urllib3",
            "ssl",
            "select",
            "signal",
            "pwd",
            "grp",
            "pwd",
            "spwd",
            "crypt",
            "termios",
            "tty",
            "pty",
            "fcntl",
            "pipes",
            "posix",
            "nt",
            "msvcrt",
            "winreg",
            "winsound",
            "msilib",
            "win32com",
            "win32api",
            "win32gui",
            "win32con",
            "win32file",
            "win32pipe",
            "win32event",
            "win32process",
            "win32security",
            "win32service",
            "win32serviceutil",
            "win32timezone",
            "pythoncom",
            "pywintypes",
            "win32ui",
            "win32print",
            "win32clipboard",
            "win32api",
            "win32gui",
            "win32con",
            "win32file",
            "win32pipe",
            "win32event",
            "win32process",
            "win32security",
            "win32service",
            "win32serviceutil",
            "win32timezone",
            "pythoncom",
            "pywintypes",
            "win32ui",
            "win32print",
            "win32clipboard",
        }

        try:
            with open(pyfile, "r") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"[Docker] Failed to parse {pyfile}: {e}")
            return []

        pkgs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    pkg_name = n.name.split(".")[0]
                    if pkg_name not in stdlib_modules:
                        pkgs.add(pkg_name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkg_name = node.module.split(".")[0]
                if pkg_name not in stdlib_modules:
                    pkgs.add(pkg_name)

        # Map import names to pip package names and filter out base packages
        mapped_pkgs = []
        for pkg in pkgs:
            if pkg not in base_packages:
                pip_pkg = package_mapping.get(pkg, pkg)
                if pip_pkg not in mapped_pkgs:
                    mapped_pkgs.append(pip_pkg)

        return sorted(mapped_pkgs)

    def get_or_build_base_image(self) -> Optional[str]:
        """Build or get the base Docker image with common ML packages."""
        if not self.use_docker:
            return None
        if self.docker_client is not None:
            try:
                self.docker_client.images.get(self.base_image_tag)
                print(f"[Docker] Using existing image: {self.base_image_tag}")
            except ImageNotFound:
                print(f"[Docker] Building image: {self.base_image_tag}")
                dockerfile_lines = [f"FROM {self.docker_base}"]
                if self.base_packages:
                    joined = " ".join(sorted(self.base_packages))
                    dockerfile_lines.append(f"RUN pip install --no-cache-dir {joined}")
                dockerfile = "\n".join(dockerfile_lines) + "\n"
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                        f.write(dockerfile)
                    self.docker_client.images.build(
                        path=tmpdir, tag=self.base_image_tag, rm=True
                    )
            return self.base_image_tag
        return None

    def get_or_build_experiment_image(self, experiment_py_path: str) -> Optional[str]:
        """Build or get experiment-specific Docker image with required packages."""
        if not self.use_docker:
            return None
        base_image = self.get_or_build_base_image()
        extra_pkgs = self.detect_required_packages(
            experiment_py_path, base_packages=self._base_package_set
        )
        if extra_pkgs:
            extras_fingerprint = hashlib.sha256(
                "|".join(sorted(extra_pkgs)).encode("utf-8")
            ).hexdigest()[:12]
            image_name = (
                f"tiny-scientist-exp-{self._base_fingerprint}-{extras_fingerprint}"
            )
            if self.docker_client is not None:
                try:
                    self.docker_client.images.get(image_name)
                    print(f"[Docker] Using cached experiment image: {image_name}")
                    return image_name
                except ImageNotFound:
                    print(
                        f"[Docker] Building experiment image: {image_name} with extra packages: {extra_pkgs}"
                    )
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Write requirements.txt
                        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                            for pkg in extra_pkgs:
                                f.write(pkg + "\n")
                        # Write Dockerfile with better error handling
                        dockerfile = f"""
FROM {base_image}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || (echo "Failed to install packages:" && cat requirements.txt && exit 1)
COPY experiment.py .
"""
                        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                            f.write(dockerfile)
                        # Copy experiment.py
                        shutil.copy(
                            experiment_py_path, os.path.join(tmpdir, "experiment.py")
                        )
                        try:
                            # Build with detailed logging
                            image, build_logs = self.docker_client.images.build(
                                path=tmpdir, tag=image_name, rm=True
                            )
                            print(f"[Docker] Successfully built image: {image_name}")
                        except Exception as e:
                            print(f"[Docker] Failed to build image {image_name}: {e}")
                            # Fallback to base image
                            print(f"[Docker] Falling back to base image: {base_image}")
                            return base_image
                    return image_name
            return base_image
        else:
            return base_image

    def _compute_base_fingerprint(self) -> str:
        """Fingerprint the base image definition so cache invalidation is automatic."""
        normalized = "\n".join([self.docker_base] + sorted(self.base_packages))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]

    def run_experiment_in_docker(
        self, experiment_code: str, run_num: int, output_dir: str, timeout: int = 7200
    ) -> Optional[Tuple[int, str]]:
        """Run experiment in a Docker container."""
        if not self.use_docker:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy experiment.py
            exp_py = os.path.join(temp_dir, "experiment.py")
            with open(exp_py, "w") as f:
                f.write(experiment_code)

            # Detect and build experiment image
            image_name = self.get_or_build_experiment_image(exp_py)

            # Mount output dir for results
            output_dir = os.path.abspath(output_dir)

            container = None
            try:
                # Create container without auto-removal
                if self.docker_client is not None:
                    container = self.docker_client.containers.run(
                        image=image_name,
                        command=f"python experiment.py --out_dir=run_{run_num}",
                        volumes={
                            temp_dir: {"bind": "/experiment", "mode": "rw"},
                            output_dir: {"bind": "/experiment/output", "mode": "rw"},
                        },
                        working_dir="/experiment",
                        detach=True,
                        remove=False,  # Don't auto-remove
                        mem_limit="2g",
                        cpu_period=100000,
                        cpu_quota=50000,
                    )

                    print(f"[Docker] Container {container.id[:12]} started")

                    # Wait for container to finish
                    try:
                        result = container.wait(timeout=timeout)
                        print(
                            f"[Docker] Container {container.id[:12]} finished with status {result['StatusCode']}"
                        )
                    except Exception as wait_e:
                        print(f"[Docker] Container wait failed: {wait_e}")
                        # Try to get logs and stop container
                        try:
                            logs = container.logs().decode("utf-8")
                            container.stop(timeout=10)
                        except Exception:
                            logs = "Container failed to start or stopped unexpectedly"
                        return (
                            1,
                            f"Container execution failed: {wait_e}\nLogs: {logs}",
                        )

                    # Get logs before removing container
                    try:
                        logs = container.logs().decode("utf-8")
                    except Exception as log_e:
                        print(f"[Docker] Failed to get logs: {log_e}")
                        logs = "Failed to retrieve container logs"

                    if result["StatusCode"] == 0:
                        # Copy results from temp_dir/output/run_{run_num} to output_dir/run_{run_num}
                        src = os.path.join(temp_dir, f"run_{run_num}")
                        dst = os.path.join(output_dir, f"run_{run_num}")
                        if os.path.exists(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        return (0, logs)
                    else:
                        # Print error logs to stderr so they're visible
                        print(
                            f"[Docker] Container failed with logs:\n{logs}",
                            file=sys.stderr,
                        )

                        # Check if it's a missing package error
                        if "ModuleNotFoundError" in logs:
                            missing_pkg = self.extract_missing_package(logs)
                            print(f"[Docker] Missing package detected: {missing_pkg}")
                            # Try to install the missing package and retry
                            try:
                                # Create a new image with the missing package
                                retry_image_name = (
                                    f"tiny-scientist-retry-{hash(missing_pkg)}"
                                )
                                with tempfile.TemporaryDirectory() as retry_tmpdir:
                                    # Write requirements.txt with the missing package
                                    with open(
                                        os.path.join(retry_tmpdir, "requirements.txt"),
                                        "w",
                                    ) as f:
                                        f.write(missing_pkg + "\n")

                                    # Write Dockerfile
                                    retry_dockerfile = f"""
FROM {image_name}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""
                                    with open(
                                        os.path.join(retry_tmpdir, "Dockerfile"), "w"
                                    ) as f:
                                        f.write(retry_dockerfile)

                                    # Build retry image
                                    if self.docker_client is not None:
                                        self.docker_client.images.build(
                                            path=retry_tmpdir,
                                            tag=retry_image_name,
                                            rm=True,
                                        )

                                        # Run with retry image
                                        retry_container = self.docker_client.containers.run(
                                            image=retry_image_name,
                                            command=f"python experiment.py --out_dir=run_{run_num}",
                                            volumes={
                                                temp_dir: {
                                                    "bind": "/experiment",
                                                    "mode": "rw",
                                                },
                                                output_dir: {
                                                    "bind": "/experiment/output",
                                                    "mode": "rw",
                                                },
                                            },
                                            working_dir="/experiment",
                                            detach=True,
                                            remove=False,  # Don't auto-remove
                                            mem_limit="2g",
                                            cpu_period=100000,
                                            cpu_quota=50000,
                                        )

                                        print(
                                            f"[Docker] Retry container {retry_container.id[:12]} started"
                                        )
                                        retry_result = retry_container.wait(
                                            timeout=timeout
                                        )
                                        retry_logs = retry_container.logs().decode(
                                            "utf-8"
                                        )

                                        # Clean up retry container
                                        try:
                                            retry_container.remove(force=True)
                                        except Exception:
                                            pass

                                        if retry_result["StatusCode"] == 0:
                                            # Copy results
                                            src = os.path.join(
                                                temp_dir, f"run_{run_num}"
                                            )
                                            dst = os.path.join(
                                                output_dir, f"run_{run_num}"
                                            )
                                            if os.path.exists(src):
                                                shutil.copytree(
                                                    src, dst, dirs_exist_ok=True
                                                )
                                            return (0, retry_logs)
                                        else:
                                            return (
                                                retry_result["StatusCode"],
                                                retry_logs,
                                            )

                            except Exception as retry_e:
                                print(f"[Docker] Retry failed: {retry_e}")
                                return (result["StatusCode"], logs)

                        return (result["StatusCode"], logs)

            except Exception as e:
                print(f"[Docker] Experiment failed: {e}")
                return (1, f"Docker experiment failed: {e}")
            finally:
                # Clean up container
                if container:
                    try:
                        container.remove(force=True)
                        print(f"[Docker] Container {container.id[:12]} cleaned up")
                    except Exception as cleanup_e:
                        print(f"[Docker] Failed to cleanup container: {cleanup_e}")

        return (1, "Docker not available")

    def cleanup_docker_images(self) -> None:
        """Clean up Docker images created during experiments."""
        if not self.use_docker:
            return
        if self.docker_client is not None:
            try:
                images = self.docker_client.images.list()
                for image in images:
                    if image.tags and any(
                        "tiny-scientist" in tag for tag in image.tags
                    ):
                        print(f"[Docker] Removing image: {image.tags[0]}")
                        self.docker_client.images.remove(image.id, force=True)
            except Exception as e:
                print(f"[Docker] Failed to cleanup images: {e}")

    @staticmethod
    def extract_missing_package(stderr: str) -> str:
        """Extract missing package name from stderr with package name mapping."""

        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "plotly": "plotly",
            "bokeh": "bokeh",
            "dash": "dash",
            "streamlit": "streamlit",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2-binary",
            "pymongo": "pymongo",
            "redis": "redis",
            "celery": "celery",
            "flask": "flask",
            "django": "django",
            "scipy": "scipy",
            "statsmodels": "statsmodels",
            "seaborn": "seaborn",
            "plotnine": "plotnine",
            "altair": "altair",
            "holoviews": "holoviews",
            "folium": "folium",
            "geopandas": "geopandas",
            "shapely": "shapely",
            "fiona": "fiona",
            "rasterio": "rasterio",
            "xarray": "xarray",
            "netcdf4": "netcdf4",
            "h5py": "h5py",
            "tables": "tables",
            "pyarrow": "pyarrow",
            "fastparquet": "fastparquet",
            "openpyxl": "openpyxl",
            "xlrd": "xlrd",
            "xlwt": "xlwt",
            "odf": "odfpy",
            "tabula": "tabula-py",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
            "pypdf": "pypdf",
            "reportlab": "reportlab",
            "weasyprint": "weasyprint",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "rst": "docutils",
            "sphinx": "sphinx",
            "mkdocs": "mkdocs",
            "jupyter": "jupyter",
            "ipython": "ipython",
            "notebook": "notebook",
            "jupyterlab": "jupyterlab",
            "voila": "voila",
            "nbconvert": "nbconvert",
            "papermill": "papermill",
            "nbdime": "nbdime",
            "nbstripout": "nbstripout",
            "jupytext": "jupytext",
        }

        for line in stderr.splitlines():
            if "ModuleNotFoundError" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    import_name = parts[1]
                    # Return the mapped package name if it exists, otherwise return the original
                    return package_mapping.get(import_name, import_name)
        return "unknown-package"


app = FastMCP(
    "tiny-scientist-docker-runner", description="Docker experiment MCP server"
)


@app.tool(
    name="docker_runner.status", description="Report Docker availability on the server"
)
def docker_status(
    docker_image: str = "tiny-scientist-ml",
    docker_base: str = "python:3.11-slim",
) -> Dict[str, bool]:
    with redirect_stdout(sys.stderr):
        runner = DockerExperimentRunner(
            docker_image=docker_image, docker_base=docker_base
        )
        return {"use_docker": runner.use_docker}


@app.tool(
    name="docker_runner.run_experiment", description="Run an experiment inside Docker"
)
def run_experiment(
    experiment_code: str,
    run_num: int,
    output_dir: str,
    timeout: int = 7200,
    docker_image: str = "tiny-scientist-ml",
    docker_base: str = "python:3.11-slim",
) -> Optional[Dict[str, Any]]:
    with redirect_stdout(sys.stderr):
        runner = DockerExperimentRunner(
            docker_image=docker_image, docker_base=docker_base
        )
        result = runner.run_experiment_in_docker(
            experiment_code=experiment_code,
            run_num=run_num,
            output_dir=output_dir,
            timeout=timeout,
        )
    if result is None:
        return None
    status_code, logs = result
    return {"status_code": status_code, "logs": logs}


@app.tool(
    name="docker_runner.cleanup",
    description="Cleanup docker images created by the runner",
)
def cleanup(
    docker_image: str = "tiny-scientist-ml", docker_base: str = "python:3.11-slim"
) -> Dict[str, str]:
    with redirect_stdout(sys.stderr):
        runner = DockerExperimentRunner(
            docker_image=docker_image, docker_base=docker_base
        )
        runner.cleanup_docker_images()
    return {"status": "ok"}


if __name__ == "__main__":
    app.run()
