import datetime
import json
import logging
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import docker
import requests
from websocket import WebSocketTimeoutException, create_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EnvironmentProcess = Union[Popen[Any], None, docker.models.containers.Container]

class HasHostPort(Protocol):
    HOST: str
    PORT: int

C = TypeVar('C', bound=HasHostPort)

@dataclass
class Environment:
    """Holds different environment objects that might at some point need to be
    shut down.

    Attributes:
        proc: The underlying process running in the environment.

    """

    proc: EnvironmentProcess

    def close(self) -> None:
        """Closes the environment if necessary (e.g., if running docker or active subprocess)"""
        try:
            if hasattr(self, "proc") and self.proc is not None:
                self.proc.terminate()
        except Exception:
            try:
                self.proc.stop()
                self.proc.remove()
                logger.info("Shut down the container")
            except Exception:
                pass


def wait_for_jupyter(host: str, port: int, timeout: int = 20) -> bool:
    """Waits for the jupyter notebook to load and for it to be accessible.

    Raises:
        Generic exception when timeout is exceeded (i.e., the jupyter environment
            is not accessible within a fixed time)

    """
    url = f"http://{host}:{port}/api"
    for i in range(timeout * 2):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise Exception("Timed out waiting for Jupyter Kernel Gateway to be up")


def launch_local_jupyter(host: str = "127.0.0.1", port: int = 8888) -> Environment:
    """Launches a jupyter server locally.

    Arguments:
        host: The host address.
        port: The port address.

    Returns:
        The subprocess object that will later need to be terminated when
            the session in complete.

    """
    cmd = [
        "jupyter",
        "kernelgateway",
        f"--KernelGatewayApp.ip={host}",
        f"--KernelGatewayApp.port={port}",
        "--KernelGatewayApp.allow_origin=*",
    ]
    process = subprocess.Popen(cmd)
    logger.info(f"Started kernel gateway on {host}:{port}, PID={process.pid}")
    wait_for_jupyter(host, port)

    return Environment(process)


def launch_docker_jupyter(
    host: str = "127.0.0.1", port: int = 8888, image_name: str = "jupyter-kernel"
) -> None:
    """Launches docker and initilizes jupyter serer inside.

    Large adapted from: https://github.com/huggingface/smolagents/blob/f17167778f3d9f977fb6728b7fca6785618d8128/src/smolagents/remote_executors.py#L96


    Arguments:
        host: The host address.
        port: The port address.
        image_name: The name of the docker image.

    Raises:
        DockerException when docker daemon is not running.

    """
    build_new_image = False
    try:
        client = docker.from_env()
    except docker.errors.DockerException:
        raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.")

    try:
        client.images.get(image_name)
        logger.info(f"Using existing Docker image: {image_name}")
    except docker.errors.ImageNotFound:
        build_new_image = True

    try:
        if build_new_image:
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            logger.info(f"Building image: {image_name} with docker file: {dockerfile_path}")

            with open(dockerfile_path, "w") as f:
                f.write(
                    dedent(
                        f"""\
                        FROM python:3.11-slim

                        RUN pip install jupyter_kernel_gateway jupyter_client ipykernel

                        EXPOSE {port}
                        CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port={port}", "--KernelGatewayApp.allow_origin='*'"]
                        """
                    )
                )
            _, build_logs = client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path),
                tag=image_name,
            )
            for log_chunk in build_logs:
                if log_message := log_chunk.get("stream", "").rstrip():
                    logger.info(log_message)

        ## run the container
        container: Optional[docker.models.containers.Container] = None
        container_kwargs: Dict[str, Any] = {}
        container_kwargs["ports"] = {}
        container_kwargs["ports"]["8888/tcp"] = (host, port)
        container_kwargs["detach"] = True
        container = client.containers.run(image_name, **container_kwargs)

        retries = 0
        while container.status != "running" and retries < 5:
            logger.info(f"Container status: {container.status}, waiting...")
            time.sleep(1)
            container.reload()
            retries += 1

    except Exception as e:
        if container is not None:
            container.stop()
            container.remove()

        raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}")

    return Environment(container)


C = TypeVar("C", bound="CodeExecutor")


class CodeExecutor:
    """Base class for code execution utilities."""

    def __init__(self):
        self._setup_executor()

    def _setup_executor(self) -> None:
        """Does any auxiliary setup needed for initializing the executor."""
        raise NotImplementedError

    def run_command(self, comand: str) -> str:
        """Executes a command or piece of code using the internal executor.

        Arguments:
            command: The command to execute.

        """
        raise NotImplementedError

    @classmethod
    def setup(cls: Type[C], **kwargs) -> Tuple[Environment, Type[C]]:
        """Class method for setting up the code executor (e.g., launching jupyter
        servers, instantiating docker) then returning the executor class.

        Returns:
            A tuple containing the environment process (or None) and the
                the code executor class.

        """
        raise NotImplementedError

    def __call__(self, code: str) -> str:
        return self.run_command(code)


def prepare_execute_message(code: str) -> Dict[str, Any]:
    """Prepare message to pass to Jupyter kernel

    Args:
        code: The main code to pass to jupyter

    Returns:
        Jupyter compatible representation of input code.

    """
    hdr = {
        "msg_id": uuid.uuid1().hex,
        "username": "test",
        "session": uuid.uuid1().hex,
        "data": datetime.datetime.now().isoformat(),
        "msg_type": "execute_request",
        "version": "5.0",
    }
    msg = {
        "header": hdr,
        "parent_header": hdr,
        "metadata": {},
        "content": {"code": code, "silent": False},
    }
    return msg


def timeout_call(
    function: Callable[..., Any],
    timeout_seconds: Optional[int] = None,
    *args: tuple,
    **kwargs: Any,
) -> Any:
    """Runs call to Jupyter with timeout handler.

    Args:
        function: the function call to Jupyter.
        timeout_seconds: The timeout value.

    Raises:
        TimeoutError

    """
    if timeout_seconds is None:
        return function(*args, **kwargs)
    if not hasattr(signal, "SIGALRM"):
        logger.warning("*** WARNING: timeout_call is not supported on this platform. ***")
        return function(*args, **kwargs)

    timeout_seconds = int(timeout_seconds)
    timeout_message = (
        f"Function {function.__name__} execution timed out after {timeout_seconds} seconds."
    )

    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(timeout_message)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = function(*args, **kwargs)
    except TimeoutError as exception:
        raise TimeoutError(timeout_message) from exception
    finally:
        signal.alarm(0)
    return result


class Notebook(CodeExecutor):
    """Creates a jupyter notebook for stateful Python and bash coding. Adapated
    from smolagents `RemotePythonExecutor` and the SUPER codebase. This will
    pick up the jupyter server regardless of where it is being run (e.g.,
    Docker, locally, ...) and the particular CodeEnvironment.

    """

    PORT: int = 8888
    HOST: str = "127.0.0.1"

    def __init__(self):
        self._jupyter_server = None
        self._kernel_id = None
        self.ws = None
        self._headers = {"Authorization": "Token password"}
        self.timed_out_unanswered = False
        self._intermediate_output = []
        self._msgs_history = []
        self._timeout_per_command = 60 * 5
        super().__init__()

    @classmethod
    def setup(
        cls: Type[C],
        run_docker: bool = False,
        port: int = 8888,
        host: str = "127.0.0.1",
        **kwargs,
    ) -> Tuple[Environment, Type[C]]:
        """Sets up the notebook environment.

        Arguments:
            run_docker: Switch to determine is docker is used or not.
            port: The port address for the jupyter server.
            host: The host address fro the jupyter server.

        """
        cls.PORT = port
        cls.HOST = host

        initializer = launch_docker_jupyter if run_docker else launch_local_jupyter
        proc = initializer(host=cls.HOST, port=cls.PORT)

        return (proc, cls)

    def _setup_executor(self) -> None:
        """Creates connection with Jupyter server to set up new notebook.

        Raises:
            Exception when the jupyter kernel fails to start.

        """
        self._jupyter_server = f"{self.HOST}:{self.PORT}"
        protocol = "http"
        url = f"{protocol}://{self._jupyter_server}/api/kernels"
        logger.info(f"Creating a new kernel at {url}")

        response = requests.post(url)
        if "id" not in json.loads(response.text):
            raise Exception(
                f"Failed to create a kernel! No id in {response.text}. Check host, port and token."
            )
        else:
            self.kernel_id = json.loads(response.text)["id"]
            logger.info(f"new kernel id: {self.kernel_id}")

        self._create_connection()

    def _create_connection(self) -> None:
        """Creates Jupyter connection."""
        if self.ws:
            self.ws.close()  # Close existing connection if any before creating a new one
        protocol = "ws"
        self.ws = create_connection(
            f"{protocol}://{self._jupyter_server}/api/kernels/{self.kernel_id}/channels",
            header=self._headers,
        )

    def interrupt_kernel(self) -> None:
        """Interrupts the jupyter kernel."""
        self._jupyter_server = f"{self.HOST}:{self.PORT}"
        protocol = "http"
        url = f"{protocol}://{self._jupyter_server}/api/kernels/{self.kernel_id}/interrupt"
        with requests.post(url) as response:
            assert response.status_code == 204, f"Failed to interrupt the kernel: {response}"
        self._create_connection()
        self.timed_out_unanswered = False
        self.run_command("pass")

    def timed_out(self) -> None:
        self.timed_out_unanswered = True
        self._intermediate_output.append(self.timeout_message)

    def _initiate_ws(self, command: str) -> None:
        try:
            self.ws.send(json.dumps(prepare_execute_message(command)))
        except BrokenPipeError:
            self._create_connection()
            self.ws.send(json.dumps(prepare_execute_message(command)))

        self._intermediate_output = []

    def run_command(self, command: str, continue_after_timeout: Optional[bool] = False) -> str:
        """Executes a command or piece of code using the internal executor.

        Arguments:
            command: The command to execute.

        """
        if continue_after_timeout:
            self.timed_out_unanswered = False
            self._intermediate_output = []
        else:
            if self.timed_out_unanswered:
                self.timed_out_unanswered = False
                self.interrupt_kernel()
            self._initiate_ws(command)

        try:
            return timeout_call(self.retrieve_output, self._timeout_per_command)
        except TimeoutError:
            self.timed_out()
            return "".join(self._intermediate_output)

    def retrieve_output(self) -> str:
        """Retrieve output from Jupyter kernel and returns the string output."""
        status = "success"
        execute_reply_msg_id = None
        self.ws.settimeout(60 * 60 * 24)

        while True:
            try:
                recv_obj = self.ws.recv()
            except WebSocketTimeoutException:
                self.timed_out()
                status = "timeout"
                break
            except BrokenPipeError as e:
                logger.error("Broken pipe error. Message history:")
                for msg in self._msgs_history:
                    logger.info(msg)
                raise e
            if not recv_obj:
                continue
            rsp = json.loads(recv_obj)
            self._msgs_history.append(rsp)
            msg_type = rsp["msg_type"]
            if msg_type == "error":
                self._intermediate_output.append(
                    f"Error/Traceback: {rsp['content']['ename']}: {rsp['content']['evalue']}"
                )
                for tb in rsp["content"]["traceback"]:
                    self._intermediate_output.append(tb)
                status = "error"
                break
            elif msg_type == "stream":
                self._intermediate_output.append(rsp["content"]["text"])
                logger.info(rsp["content"]["text"])
            elif msg_type == "execute_result":
                self._intermediate_output.append(rsp["content"]["data"]["text/plain"])

            current_msg_id = self._get_msg_id_from_rsp(rsp)
            if execute_reply_msg_id and current_msg_id == execute_reply_msg_id - 1:
                break
            if (
                msg_type == "execute_reply"
                and rsp["metadata"].get("status") == "ok"
                and rsp["metadata"].get("dependencies_met", False)
            ):
                # check if we didn't skip any stream messages - unfortunately, sometimes stream messages come after the execute_reply message. This is not the most robust way to handle this, but it seems stable enough.
                # we check the message id of the execute_reply (finished message) and the message id of the last stream message. If the stream message id is higher, we continue receiving messages until we get to the message id of execute_reply minus 1.
                try:
                    execute_reply_msg_id = self._get_msg_id_from_rsp(rsp)
                    last_message_id = self._get_msg_id_from_rsp(self._msgs_history[-2])
                    if execute_reply_msg_id - last_message_id > 1:
                        # timeout can now be short since last message was already received
                        logger.info(
                            "*** execute_reply was received before the last stream message. Continuing to receive messages until the last stream message is received. ***"
                        )
                        self.ws.settimeout(5)
                        continue
                except Exception:
                    break
                break
        return "".join(self._intermediate_output)

    @staticmethod
    def _get_msg_id_from_rsp(rsp: Dict[str, Any]) -> int:
        return int(rsp["header"]["msg_id"].split("_")[-1])


CODING_ENVS = {
    "notebook": Notebook,
    # "executor" : Executor,
}


def CodingEnvironment(env_type: str, **kwargs) -> Tuple[Environment, CodeExecutor]:
    """Factory method for setting up code environment and launching any
    required auxiliary services.

    Arguments:
        env_type: The particular coding environment to use.

    Raises:
        ValueError when unknown `env_type` is passed.

    >>> env, executor = CodingEnvironment(
           env_type="notebook",
           run_docker=True  # <--- turn to `False` to run locally
    )
    >>> notebook = executor()
    >>> notebook("a = 20")
    >>> output = notebook("print(a)")
    >>> assert output.strip() == "20"

    >>> new_notebook = executor()  ## create another one
    >>> new_notebook("a = 100")
    >>> output = new_notebook("print(a)")
    >>> assert output.strip() == "100"
    >>> work_dir = new_notebook("!pwd")  ## bash

    ## close jupyter, container, etc.
    >>> env.close()
    """
    exc_class = CODING_ENVS.get(env_type, None)
    if exc_class is None:
        raise ValueError(f"Uknown coding environment: {exc_class}")

    env, executor = exc_class.setup(**kwargs)
    return (env, executor)


def test_local_jupyter() -> None:
    """Run code execution inside of jupyter running on the local computer"""
    env, executor = CodingEnvironment(env_type="notebook", run_docker=False)
    notebook = executor()  ## create new notebook instance
    notebook("a = 20")
    output = notebook("print(a)")
    assert output.strip() == "20"

    new_notebook = executor()  ## create another one
    new_notebook("a = 100")
    output = new_notebook("print(a)")
    assert output.strip() == "100"
    work_dir = new_notebook("!pwd")  ## bash

    env.close()


def test_docker_jupyter() -> None:
    """Run code execution inside of jupyter running on the local computer"""
    env, executor = CodingEnvironment(env_type="notebook", run_docker=True)
    notebook = executor()  ## create new notebook instance
    notebook("a = 20")
    output = notebook("print(a)")
    assert output.strip() == "20"

    new_notebook = executor()  ## create another one
    new_notebook("a = 100")
    output = new_notebook("print(a)")
    assert output.strip() == "100"
    work_dir = new_notebook("!pwd")  ## bash

    env.close()


if __name__ == "__main__":
    test_local_jupyter()
    test_docker_jupyter()