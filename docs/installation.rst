Installation
============

Requirements
------------

* Python 3.10 or higher
* Internet connection for OAuth authentication
* Compatible with Windows, macOS, and Linux

Install from PyPI
-----------------

The easiest way to install the IDTAP API is using pip:

.. code-block:: bash

   pip install idtap

This will install the package and all required dependencies:

* ``requests`` - HTTP client library
* ``requests-toolbelt`` - Additional HTTP utilities  
* ``pyhumps`` - Case conversion utilities
* ``keyring`` - Secure token storage
* ``cryptography`` - Encryption for token security
* ``PyJWT`` - JWT token handling
* ``google-auth-oauthlib`` - Google OAuth integration
* ``pymongo`` - MongoDB database access (for advanced features)

Install from Source
-------------------

To install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/HAI-Res/idtap-client.git

Or clone the repository and install locally:

.. code-block:: bash

   git clone https://github.com/HAI-Res/idtap-client.git
   cd idtap-client
   pip install -e .

Development Installation
------------------------

For development, install with optional development dependencies:

.. code-block:: bash

   pip install idtap[dev]

This includes additional packages for testing and development:

* ``pytest`` - Testing framework
* ``responses`` - HTTP request mocking
* ``black`` - Code formatting
* ``isort`` - Import sorting
* ``mypy`` - Type checking
* ``flake8`` - Code linting

Platform-Specific Notes
------------------------

Linux
~~~~~

On Linux systems, you may need to install additional packages for secure keyring support:

.. code-block:: bash

   pip install idtap[linux]

This installs ``secretstorage`` for integration with the system keyring.

macOS/Windows
~~~~~~~~~~~~~

The standard installation works out of the box on macOS and Windows with built-in keyring support.

Verification
------------

To verify your installation, run:

.. code-block:: python

   import idtap
   print(idtap.__version__)

This should print the installed version number (e.g., ``0.1.7``).
