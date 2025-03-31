:orphan:

.. _install-docker:

############################
AIMET installation in Docker
############################

This page describes how to install AIMET in a development Docker container.

Prerequisites
=============

Install Docker from https://docs.docker.com/engine/install/ubuntu/

For GPU variants, install the NVIDIA Container Toolkit from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


Installing AIMET
================

To install an AIMET development Docker container, you:

1. Decide on an AIMET variant to install
2. Choose to either:

   - Download a prebuilt Docker image
   - Build a Docker image

Follow the instructions below to install AIMET within a Docker container. Depending on your
installation choices, you will skip various sections on this page.


Choose your AIMET variant
-------------------------

**Step 1:** Choose a variant.

Choose a variant (a combination of framework and runtime environment) from the following table.
Copy the **<variant_string>**.

.. list-table::
   :widths: 12 12 20
   :header-rows: 1

   * - Framework
     - Runtime Environment
     - `<variant_string>`
   * - PyTorch 2.1
     - GPU
     - `torch-gpu`
   * - PyTorch 2.1
     - CPU
     - `torch-cpu`
   * - TensorFlow
     - GPU
     - `tf-gpu`
   * - TensorFlow
     - CPU
     - `tf-cpu`
   * - ONNX
     - GPU
     - `onnx-gpu`
   * - ONNX
     - CPU
     - `onnx-cpu`

**Step 2:** Set the AIMET_VARIANT environment variable.

Set the ``AIMET_VARIANT`` shell variable to your chosen variant string.

.. code-block:: bash

    export AIMET_VARIANT=<variant_string>


Choose to download or build an image
------------------------------------

**Step 3:** Choose one of the following options. We recommend using a prebuilt Docker image unless your
installation requires custom dependencies.

Continue to :ref:`Download a prebuilt Docker image (step 4)<docker-install-download>`

or

Skip to :ref:`Build a Docker image (step 5)<docker-install-build>`.

.. _docker-install-download:

Download a prebuilt Docker image
--------------------------------

**Step 4:** Set environment variables.

Set the following shell variables to define the Docker image installation.

.. code-block:: bash

    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.${AIMET_VARIANT}"
    docker_container_name="aimet-dev-<container_name>"

where:

**<absolute_path_to_workspace>**
    is the absolute path to the directory where the AIMET Git repository resides on your local machine.
**<container_name>**
    is whatever name you want to assign the AIMET Docker container.

**${AIMET_VARIANT}** is the shell variable you set in the previous section. Type the variable as shown.

When you start the Docker container, it will be downloaded from the image library located at **docker_image_name**.

Skip to :ref:`Start the Docker container (step 7)<docker-install-start>`.


.. _docker-install-build:

Build a Docker image
--------------------

**Step 5:**  Set environment variables.

Set the following shell variables to define the Docker image installation.


.. code-block:: bash

    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="aimet-dev-docker:<any_tag>"
    docker_container_name="aimet-dev-<any_name>"

where:

**<absolute_path_to_workspace>**
    is the absolute path to the directory where the AIMET Git repository resides on your local machine.
**<any_tag>**
    is whatever unique name suffix you want to append to the Docker image.
**<container_name>**
    is whatever name you want to assign the AIMET Docker container.


**Step 6:**  Build the Docker image from code in the the AIMET repo.

.. code-block:: bash

    docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .

.. _docker-install-start:

Start the docker container
--------------------------

**Step 7:**  Check that a Docker container named $docker_container_name is not already running. Remove the container if it is.

.. code-block:: bash

    docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

**Step 8:** (optional) Specify a port to use for port forwarding if you plan to run the Visualization APIs.

.. code-block:: bash

    port_id="<port-number>"

where **<port-number>** is any unused port on the host.

**Step 9:**  Run the Docker container.

.. code-block:: bash

    [docker_run_command] -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
    -v "/local/mnt/workspace":"/local/mnt/workspace" \
    --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}

where:

**[docker_run_command]**
    is ``docker run --gpus all`` if using a GPU varint with nvidia-docker 2.0, or ``nvidia-docker run`` with nvidia-docker 1.0
**-p ${port_id}:${port_id}**
    is the port forwarding option. Omit this if you did not specify a port in the previous step
**WORKSPACE**, **docker_container_name**, and **docker_image_name**
    are variables defined in previous steps.

As a convenience, the following block contains the *first line* of the Docker run command above for all combinations of nvidia-docker with and without port forwarding.

.. code-block:: bash

    # nvidia-docker 2.0 with port forwarding:
    docker run --gpus all -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

    # nvidia-docker 1.0 with port forwarding:
    nvidia-docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

    # CPU only, with port forwarding:
    docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

    # nvidia-docker 2.0 without port forwarding:
    docker run --gpus all --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

    # nvidia-docker 1.0 without port forwarding:
    nvidia-docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

    # CPU only, without port forwarding:
    docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \

Install AIMET packages
----------------------

**Choose an option to install the AIMET package on the Docker container.**

1.  From PyPI (PyTorch only)
2.  Any framework variant (hosted **.whl** files)

**Step 10:**  To install the most recent PyTorch AIMET package with GPU support (the most common option) from PyPI, type the following commands in the Docker container.

.. code-block:: bash

    python3 -m pip install aimet-torch

Skip to :ref:`Environment setup (step 12) <docker-install-setup>`.

**Step 11:**  To install the latest version of any AIMET variant from the .whl files, follow the substeps below.

**Step 11.1:** Select the release tag for the version you want to install, for example, "|version|".

Releases are listed at: https://github.com/quic/aimet/releases

- Identify the .whl file corresponding to the package variant that you want to install
- Continue with the instructions below to install AIMET from the .whl file

**Step 11.2:** Set the package details.

.. parsed-literal::
    
    # Set the release tag, for example "|version|"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="\https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension,
    # for example "aimet_torch-|version|\+cu121\ |torch_whl_suffix|"
    export wheel_file_name="<wheel file name>"

    # NOTE: Do the following only for the PyTorch and ONNX variant packages!
    export find_pkg_url_str="-f https://download.pytorch.org/whl/torch_stable.html"

**Step 11.3:** Install the selected AIMET package.

.. note::

    Python dependencies are automatically installed.

.. code-block:: bash

    # Install the wheel package
    python3 -m pip install ${download_url}/${wheel_file_name} ${find_pkg_url_str}

.. _docker-install-setup:

Environment setup
-----------------

**Step 12:** Run the environment setup script to set common environment variables.

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh


.. |torch_whl_suffix| replace:: \-py38-none-any.whl
