Examples
========

Real-world examples of using the IDTAP API for common tasks.

.. toctree::
   :maxdepth: 2

   basic-usage
   audio-management
   data-analysis
   batch-processing

Code Examples
-------------

All examples are available in the `examples/ directory <https://github.com/HAI-Res/idtap-client/tree/main/examples>`_ of the GitHub repository.

Common Patterns
---------------

Authentication and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

Every script starts with authentication:

.. code-block:: python

   from idtap import login_google, SwaraClient
   
   # One-time authentication (opens browser)
   login_google()
   
   # Create client for API calls  
   client = SwaraClient()

Data Processing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

Typical workflow for analyzing transcriptions:

.. code-block:: python

   from idtap import Piece
   
   # 1. Get available transcriptions
   transcriptions = client.get_transcriptions()
   
   # 2. Load specific transcription
   piece_id = transcriptions[0]['_id']
   piece_data = client.get_transcription(piece_id)
   
   # 3. Convert to rich data model
   piece = Piece.from_json(piece_data)
   
   # 4. Analyze musical content
   for phrase in piece.phrases:
       # Your analysis code here
       pass

Error Handling Pattern
~~~~~~~~~~~~~~~~~~~~~~

Robust error handling for API calls:

.. code-block:: python

   import requests
   from idtap import SwaraClient
   
   client = SwaraClient()
   
   try:
       transcriptions = client.get_transcriptions()
   except requests.exceptions.ConnectionError:
       print("Cannot connect to IDTAP server")
   except requests.exceptions.HTTPError as e:
       print(f"HTTP error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Next Steps
----------

* Browse the detailed examples in the sections above
* Check the :doc:`../api/index` for complete API reference
* Visit the `GitHub repository <https://github.com/HAI-Res/idtap-client>`_ for more examples