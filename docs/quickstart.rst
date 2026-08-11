Quickstart Guide
================

This guide will get you up and running with the IDTAP API in minutes.

Installation and Authentication
-------------------------------

1. **Install the package**:

   .. code-block:: bash

      pip install idtap

2. **Authenticate with Google**:

   .. code-block:: python

      from idtap import login_google
      
      # Opens browser for Google OAuth
      login_google()

3. **Create a client**:

   .. code-block:: python

      from idtap import SwaraClient
      
      client = SwaraClient()

Basic Operations
----------------

Get Your Transcriptions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get all transcriptions you have access to
   transcriptions = client.get_transcriptions()
   
   for transcription in transcriptions:
       print(f"Title: {transcription['title']}")
       print(f"ID: {transcription['_id']}")

Load a Specific Transcription
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from idtap import Piece
   
   # Load a transcription by ID
   piece_id = "your_transcription_id_here"
   piece = client.get_transcription(piece_id)
   
   # Convert to rich data model
   transcription = Piece.from_json(piece)
   
   print(f"Title: {transcription.title}")
   print(f"Number of phrases: {len(transcription.phrases)}")

Working with Musical Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access musical elements
   for phrase in transcription.phrases:
       print(f"Phrase {phrase.phrase_number}:")
       
       for trajectory in phrase.trajectories:
           print(f"  Trajectory: {len(trajectory.pitches)} pitches")
           
           # Access individual pitches
           for pitch in trajectory.pitches:
               print(f"    Time: {pitch.time}, Frequency: {pitch.freq}")

Audio Management
---------------

Upload Audio File
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from idtap import AudioMetadata, Musician, Location
   
   # Prepare metadata
   metadata = AudioMetadata(
       title="My Recording",
       raga_name="Yaman",
       musicians=[
           Musician(name="Artist Name", instrument="Sitar")
       ],
       location=Location(city="New York", country="USA"),
       notes="Concert recording"
   )
   
   # Upload audio file
   result = client.upload_audio("path/to/audio.wav", metadata)
   print(f"Uploaded: {result.audio_id}")

Download Audio
~~~~~~~~~~~~~~

.. code-block:: python

   # Download audio file
   audio_data = client.download_audio(audio_id, format='wav')
   
   # Save to file
   with open('downloaded_audio.wav', 'wb') as f:
       f.write(audio_data)

Data Export
-----------

Export to JSON
~~~~~~~~~~~~~~

.. code-block:: python

   # Export transcription data
   json_data = client.export_transcription_json(piece_id)
   
   # Save to file
   import json
   with open('transcription.json', 'w') as f:
       json.dump(json_data, f, indent=2)

Export to Excel
~~~~~~~~~~~~~~~

.. code-block:: python

   # Export as Excel file
   excel_data = client.export_transcription_excel(piece_id)
   
   # Save to file
   with open('transcription.xlsx', 'wb') as f:
       f.write(excel_data)

Common Patterns
--------------

Batch Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple transcriptions
   transcriptions = client.get_transcriptions()
   
   for trans_info in transcriptions:
       piece_id = trans_info['_id']
       piece = client.get_transcription(piece_id)
       transcription = Piece.from_json(piece)
       
       # Process each transcription
       print(f"Processing: {transcription.title}")
       # Your analysis code here...

Error Handling
~~~~~~~~~~~~~

.. code-block:: python

   try:
       piece = client.get_transcription(piece_id)
   except Exception as e:
       print(f"Error loading transcription: {e}")

Search and Filter
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get transcriptions with filtering
   transcriptions = client.get_transcriptions()
   
   # Filter by raga
   yaman_transcriptions = [
       t for t in transcriptions 
       if t.get('raga', {}).get('name') == 'Yaman'
   ]

Next Steps
----------

* Read the :doc:`user-guide` for detailed workflows
* Explore the :doc:`api/index` for complete API reference  
* Check out :doc:`examples/index` for real-world use cases
* Visit the `GitHub repository <https://github.com/HAI-Res/idtap-client>`_ for source code