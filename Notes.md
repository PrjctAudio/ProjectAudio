# 1.1 Breakdown

## Preprocessing

Before extracting meaningful information from audio files, it's essential to standardize the format and quality of these files to ensure consistency across the analyses. Here's what typically happens during preprocessing:

- **Format Conversion**: Convert audio files to a common format like WAV because it's a lossless format, which means it has not been compressed and retains all of the original data. This is important for analysis because no information is lost.

- **Sampling Rate Normalization**: Make sure all audio files have the same sampling rate. The sampling rate defines how many samples per second are taken from a continuous signal to make a discrete signal. For audio CD quality, this is typically 44.1 kHz.

- **Mono Conversion**: Convert stereo tracks to mono. Many audio analysis tasks do not require stereo information, and working in mono can reduce computational complexity.

- **Bit Depth Standardization**: Ensure that all audio files have the same bit depth (e.g., 16-bit, 24-bit), which determines the resolution of the sound.

- **Normalization**: Apply normalization to ensure consistent volume levels across all tracks, which could otherwise bias the analysis.

## Feature Extraction

Now let's look at each feature you want to extract:

### Key Signature Detection

Detecting the key signature of a song is about finding the tonic (the root note) and mode (major or minor) that best represent the tonality of the music. Here's how it might be done:

- **Chroma Feature Extraction**: Compute a chromagram from the audio signal, which is a representation of the energy present in each of the 12 different pitch classes (C, C#, D, etc.).

- **Key Detection Algorithm**: Using the chromagram, apply an algorithm to estimate the key. This could be a simple correlation with template key profiles or a more sophisticated machine learning model.

Example code using `librosa` to estimate the key:

```python
import librosa
import librosa.display

y, sr = librosa.load('audio_file.wav')
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
key = librosa.key.key_to_notes(chroma)
print(f"Estimated Key: {key}")
```

### BPM Detection

BPM (beats per minute) detection is about finding the tempo of the track, which is the speed at which it is played:

- **Onset Detection**: Identify moments in the audio where beats are likely to occur.

- **Tempo Estimation**: Use the timing of these onsets to estimate the tempo of the piece.

The code snippet you provided is a straightforward example of using `librosa` to estimate the BPM.

### Time Signature Detection

The time signature of a piece of music defines how many beats are in each measure and which note value constitutes one beat:

- **Beat Tracking**: First, detect the beats in the audio signal.

- **Pattern Recognition**: Analyze the pattern of strong and weak beats to infer the time signature.

Detecting the time signature automatically can be complex and may not be as reliable as other features without a more sophisticated model.

### Genre Classification

To automate the classification of the genre, you'll need to extract a variety of features that are relevant to the genre and then apply a classification algorithm:

- **MFCCs (Mel-frequency cepstral coefficients)**: Capture the timbral aspects of the audio signal.

- **Chroma Features**: Can give information about the harmonic content.

- **Spectral Contrast**: Looks at the difference in amplitude between peaks and valleys in the spectrum, which can be indicative of genre.

- **Zero-crossing Rate**: The rate at which the signal changes sign can be indicative of the amount of noise or percussion in a track, which can be useful for genre classification.

You would then feed these features into a machine learning model that has been trained to classify genres based on these feature sets. Here's a simplified example of how you might start this process:

```python
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the audio file
y, sr = librosa.load('audio_file.wav')

# Extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

# Stack features into a single array
features = np.vstack([mfccs, chroma_stft, spectral_contrast, zero_crossing_rate]).T

# Prepare labels and split data for training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Normalize features
scscaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the classifier
clf = MLPClassifier(hidden_layer_sizes=(500,), learning_rate='adaptive', max_iter=500)
clf.fit(X_train, y_train)

# Test the classifier
score = clf.score(X_test, y_test)
print(f"Classification accuracy: {score}")

```

In this simplified example, `labels` would be an array containing the genre labels for each audio file, which you would have to provide. The `train_test_split` function splits the dataset into a training set and a test set. The `StandardScaler` is used to normalize the features, and `MLPClassifier` is a type of neural network provided by the `scikit-learn` library.

Keep in mind that genre classification is a high-level task that involves significant complexity and subtlety. The quality of the classifier will depend heavily on the size and diversity of the dataset used for training, the features selected, and the architecture of the model itself. It's a task that often requires substantial computational resources and domain expertise to achieve high accuracy.

---

# 1.2 - 1.3

## 1.2 Analyze Audio Fingerprint

Audio fingerprinting is a fascinating area of audio processing that allows for the identification and matching of audio files based on their content. Here's how it generally works:

- **Feature Extraction**: The audio is processed to extract distinctive features, which represent the audio in a compact form. These features should be robust to changes in audio quality, encoding, and other transformations.

- **Fingerprint Generation**: These features are then used to create a fingerprint, a unique identifier for the audio.

- **Database Matching**: The generated fingerprint is compared with a database of known fingerprints to find matches.

### Using AcoustID

AcoustID is a service that uses the Chromaprint algorithm to create fingerprints. Here's a high-level overview of how you might use it:

- **Fingerprint Calculation**: Use `chromaprint` to analyze the audio and create a fingerprint.

- **Database Query**: Send this fingerprint to the AcoustID database to find a matching track ID.

- **Metadata Retrieval**: Use the track ID to retrieve metadata associated with the matched audio file (if it exists in the database).

Here's a Python example using the `pyacoustid` library:

```python
import acoustid
import chromaprint

# Function to identify the song
def identify_audio(file_path):
    # Calculate the fingerprint
    duration, fingerprint = acoustid.fingerprint_file(file_path)

    # Use the AcoustID web service to look up the fingerprint
    results = acoustid.lookup('<your_api_key>', fingerprint, duration)

    # Process results and return
    for score, recording_id, title, artist in results:
        print(f"Found: {artist} - {title} (score: {score})")

# Now call the function with an audio file path
identify_audio('path_to_audio_file.mp3')
```

In this example, replace `'<your_api_key>'` with your actual AcoustID API key. The `acoustid.lookup` function queries the AcoustID database with the generated fingerprint.

## 1.3 Create Song Structure Labels

To analyze the structure of a song, you typically need to:

- **Feature Extraction**: Identify and extract features that can help differentiate between different parts of a song, such as verses, choruses, and bridges.

- **Segmentation**: Use these features to segment the song into its constituent parts. This might involve unsupervised methods like clustering or supervised methods if labeled training data is available.

- **Labeling**: Assign labels to the identified segments. In a supervised learning context, this would involve training a classifier to recognize different types of segments.

### Example of Song Structure Analysis

Here's a simplified example of how you might approach this task using machine learning:

- **Feature Extraction**: Use `librosa` to extract features like MFCCs, chroma features, and spectral contrast over short time frames across the song.

- **Segmentation Algorithm**: Use an algorithm like k-means clustering to group similar frames together, or employ a more sophisticated algorithm like Structural Features (SF) or Temporal Structural Features (TSF).

- **Model Training**: If you have labeled data, you could train a supervised model. For example, a CNN or an RNN could learn to recognize patterns associated with different song parts.

- **Prediction**: Use the trained model to predict the structure of unseen songs.

Example code might look like this:

```python
import librosa
from sklearn.cluster import KMeans

# Load the audio file
y, sr = librosa.load('song.wav')

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# Apply KMeans to segment the song
n_clusters = 5  # example number of clusters
kmeans = KMeans(n_clusters=n_clusters)
mfccs_transposed = mfccs.T  # transpose to get frame-by-feature matrix
kmeans.fit(mfccs_transposed)

# The labels now represent the segment id for each frame
segment_ids = kmeans.labels_
```

In the above code, `n_clusters` would be the number of distinct segments you expect to find in the song. The `segment_ids` array holds the cluster assignment for each frame, which you could then use to infer the structure.

For more sophisticated analysis, you might use a sequence model like an RNN, which could capture temporal dependencies between frames, or you might combine multiple feature types to give the model a richer understanding of the audio content.

Remember, creating accurate song structure labels automatically is a complex task and may require a large amount of labeled training data to achieve satisfactory results.

---

## 2. Organize Analyzed Audio Library

Organizing an analyzed audio library involves several steps, primarily focused on parsing metadata, structuring the data for storage, and handling the file system operations. Below is a more detailed explanation of each of these steps:

### Parse Extracted Metadata

The metadata extracted from audio files typically includes details like artist name, album title, track number, genre, key signature, beats per minute (BPM), and other audio features. Parsing this metadata accurately is crucial for effective organization.

In Python, you might use a library such as `mutagen` to extract metadata from audio files. Here's an example of how you might parse this data:

```python
from mutagen.easyid3 import EasyID3

# Function to extract metadata from an audio file
def get_metadata(file_path):
    audio_metadata = EasyID3(file_path)
    return {
        'artist': audio_metadata.get('artist', ['Unknown Artist'])[0],
        'album': audio_metadata.get('album', ['Unknown Album'])[0],
        'title': audio_metadata.get('title', ['Unknown Title'])[0],
        'genre': audio_metadata.get('genre', ['Unknown Genre'])[0],
        'key': audio_metadata.get('initialkey', ['Unknown Key'])[0],
        # add more metadata fields as needed
    }

# Example usage
file_metadata = get_metadata('path_to_audio_file.mp3')
```

### Organize Files into Directory Structure

Once you have the metadata, you can use it to organize the files. The directory structure is typically organized by artist, album, and track number. However, in this example, you're organizing by detected key signatures.

Using Python's `os` and `shutil` libraries, you can create directories and move files, as shown in your provided code snippet. If you want to organize files by multiple metadata attributes, you might create nested directories.

### Handle Duplicates and Conflicts

Files with the same name or metadata may cause conflicts when organizing your library. To handle duplicates, you could:

- Append a unique identifier to the file name.
- Check if the file already exists in the target directory and compare their sizes or checksums to determine if they're identical.
- Prompt the user for action or log the incident for later review.

A simple way to handle duplicates is to rename the moved file with a counter or timestamp if a file with the same name already exists:

```python
import time

def move_file_with_care(src, dest):
    if not os.path.exists(dest):
        shutil.move(src, dest)
    else:
        base, extension = os.path.splitext(dest)
        i = 1
        # Add a counter to the filename if a file with the same name exists
        new_dest = f"{base}_{i}{extension}"
        while os.path.exists(new_dest):
            i += 1
            new_dest = f"{base}_{i}{extension}"
        shutil.move(src, new_dest)

# Example usage
move_file_with_care(file_path, destination_dir + os.path.basename(file_path))
```

### Automate and Validate the Organization Process

For a large library, manual organization isn't practical. You'd want to automate the process with scripts that can be run on a schedule or triggered when new files are added.

Additionally, you should validate the results of the organization process. Automated tests can verify that files end up in the correct folders and that duplicate handling works as expected.

### Example Script

Combining the metadata parsing and file organization into a script might look something like this:

```python
import os
import shutil
from mutagen.easyid3 import EasyID3

# Define the directory containing the audio files
audio_library_path = 'path_to_audio_library'

# Function to get metadata
def get_metadata(file_path):
    # (implementation as above)

# Function to move files
def move_file_with_care(src, dest):
    # (implementation as above)

# Organize files
for root, dirs, files in os.walk(audio_library_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        try:
            metadata = get_metadata(file_path)
            key = metadata['key']
            destination_dir = f"sorted_library/{key}/"
            os.makedirs(destination_dir, exist_ok=True)
            move_file_with_care(file_path, destination_dir + filename)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Confirmation message
print("Audio library organized by key signature.")
```

This script will walk through the library, read the metadata, and move the files accordingly. It's designed to be robust, handling potential errors and duplicates without crashing.

### Conclusion

Organizing an audio library is a complex task that requires careful consideration of metadata, file operations, and error handling. The above steps and code examples provide a framework for creating a script that can automatically sort an audio library, ensuring that it is both organized and scalable as the### 2. Organize Analyzed Audio Library

Organizing an analyzed audio library involves several steps, primarily focused on parsing metadata, structuring the data for storage, and handling the file system operations. Below is a more detailed explanation of each of these steps:

### Parse Extracted Metadata

The metadata extracted from audio files typically includes details like artist name, album title, track number, genre, key signature, beats per minute (BPM), and other audio features. Parsing this metadata accurately is crucial for effective organization.

In Python, you might use a library such as `mutagen` to extract metadata from audio files. Here's an example of how you might parse this data:

```python
from mutagen.easyid3 import EasyID3

# Function to extract metadata from an audio file
def get_metadata(file_path):
    audio_metadata = EasyID3(file_path)
    return {
        'artist': audio_metadata.get('artist', ['Unknown Artist'])[0],
        'album': audio_metadata.get('album', ['Unknown Album'])[0],
        'title': audio_metadata.get('title', ['Unknown Title'])[0],
        'genre': audio_metadata.get('genre', ['Unknown Genre'])[0],
        'key': audio_metadata.get('initialkey', ['Unknown Key'])[0],
        # add more metadata fields as needed
    }

# Example usage
file_metadata = get_metadata('path_to_audio_file.mp3')
```

### Organize Files into Directory Structure

Once you have the metadata, you can use it to organize the files. The directory structure is typically organized by artist, album, and track number. However, in this example, you're organizing by detected key signatures.

Using Python's `os` and `shutil` libraries, you can create directories and move files, as shown in your provided code snippet. If you want to organize files by multiple metadata attributes, you might create nested directories.

### Handle Duplicates and Conflicts

Files with the same name or metadata may cause conflicts when organizing your library. To handle duplicates, you could:

- Append a unique identifier to the file name.
- Check if the file already exists in the target directory and compare their sizes or checksums to determine if they're identical.
- Prompt the user for action or log the incident for later review.

A simple way to handle duplicates is to rename the moved file with a counter or timestamp if a file with the same name already exists:

```python
import time

def move_file_with_care(src, dest):
    if not os.path.exists(dest):
        shutil.move(src, dest)
    else:
        base, extension = os.path.splitext(dest)
        i = 1
        # Add a counter to the filename if a file with the same name exists
        new_dest = f"{base}_{i}{extension}"
        while os.path.exists(new_dest):
            i += 1
            new_dest = f"{base}_{i}{extension}"
        shutil.move(src, new_dest)

# Example usage
move_file_with_care(file_path, destination_dir + os.path.basename(file_path))
```

### Automate and Validate the Organization Process

For a large library, manual organization isn't practical. You'd want to automate the process with scripts that can be run on a schedule or triggered when new files are added.

Additionally, you should validate the results of the organization process. Automated tests can verify that files end up in the correct folders and that duplicate handling works as expected.

### Example Script

Combining the metadata parsing and file organization into a script might look something like this:

```python
import os
import shutil
from mutagen.easyid3 import EasyID3

# Define the directory containing the audio files
audio_library_path = 'path_to_audio_library'

# Function to get metadata
def get_metadata(file_path):
    # (implementation as above)

# Function to move files
def move_file_with_care(src, dest):
    # (implementation as above)

# Organize files
for root, dirs, files in os.walk(audio_library_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        try:
            metadata = get_metadata(file_path)
            key = metadata['key']
            destination_dir = f"sorted_library/{key}/"
            os.makedirs(destination_dir, exist_ok=True)
            move_file_with_care(file_path, destination_dir + filename)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Confirmation message
print("Audio library organized by key signature.")
```

This script will walk through the library, read the metadata, and move the files accordingly. It's designed to be robust, handling potential errors and duplicates without crashing.

### Conclusion

Organizing an audio library is a complex task that requires careful consideration of metadata, file operations, and error handling. The above steps and code examples provide a framework for creating a script that can automatically sort an audio library, ensuring that it is both organized and scalable as the

---

## 3. Search Features

Implementing search features in an application involves creating a system that allows users to query a dataset and obtain relevant results. This process typically involves several key components and steps:

### Define a RESTful API Endpoint for Search

A RESTful API is designed to handle HTTP requests and responses. A search feature would typically include a GET endpoint that accepts search terms and other parameters as query strings.

```plaintext
GET /api/search?query=term&filter=genre&sort=popularity
```

In this example, the search endpoint might accept several parameters:

- `query`: The user's search term.
- `filter`: Criteria to filter the results (e.g., genre, date range).
- `sort`: The order in which to sort the results (e.g., popularity, relevance).

### Process API Requests

When the server receives a search request, it needs to:

1. **Parse the Request**: Extract the search parameters from the query string.
2. **Validate Inputs**: Check if the inputs are valid and sanitize them to prevent issues like SQL injection.
3. **Translate into a Query**: Convert the search parameters into a database query.

### Database Query Execution

Depending on the database used (SQL or NoSQL), the search parameters are translated into a query that the database can execute.

- **SQL Databases**: Use `SELECT` statements with `WHERE` clauses to filter results, and `ORDER BY` to sort results.

```sql
  SELECT * FROM songs WHERE title LIKE '%term%' AND genre='filter' ORDER BY popularity DESC;
  ```

- **NoSQL Databases**: Use the database's query language or API to fetch the results. For example, in MongoDB, you might use the `find` method with a query document.

  ```javascript
  db.songs.find({ "title": { $regex: 'term', $options: 'i' }, "genre": 'filter'}).sort({ "popularity": -1 });
  ```

### Implementing Filtering and Sorting Logic

Filtering and sorting are essential for refining search results to match the user's needs.

- **Filtering**: Allows users to narrow down search results based on specific criteria. This is usually done by adding conditions to the database query.

  For example, if a user wants to only see songs from a particular genre, the query should include a filter condition for that genre.

- **Sorting**: Allows users to view results in a particular order. Common sorting parameters might include relevance, date, popularity, etc.

  Sorting is typically implemented as part of the database query, specifying how the results should be ordered before they are returned.

### Pagination

For large sets of results, pagination is crucial. It involves splitting the results into pages and only returning a subset (page) of results at a time.

Implementing pagination generally requires:

- Keeping track of the page number and size (number of results per page).
- Using database features like `LIMIT` and `OFFSET` in SQL or their equivalents in NoSQL databases to fetch only the subset of results for the current page.

### Search Indexes

For improved search performance, especially with large datasets, you may need to use search indexes. Indexes are special data structures that the database uses to quickly look up values without scanning the entire dataset.

- In SQL databases, you would create indexes on columns that are frequently searched or used to sort results.
- Some NoSQL databases, like Elasticsearch, are designed specifically for search operations and build indexes by default.

### Example API Implementation

Here's a pseudo-code example of how a simple REST API endpoint for search might be implemented in a web framework like Flask for Python:

```python
from flask import Flask, request
from my_database import query_database

app = Flask(__name__)

@app.route('/api/search', methods=['GET'])
def search():
    # Parse search parameters
    query = request.args.get('query')
    filter = request.args.get('filter')
    sort = request.args.get('sort')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    # Sanitize and validate parameters
    # (code to sanitize and validate)

    # Formulate the database query
    db_query = formulate_query(query, filter, sort, page, per_page)

    # Execute the query
    results = query_database(db_query)

    # Return the results as JSON
    return jsonify(results)

def formulate_query(query, filter, sort, page, per_page):
    # Translate parameters into database query
    # (code to create the database query with filters and sorting)

    # Add logic for pagination (LIMIT and OFFSET or equivalent)
    offset = (page - 1) * per_page
    limit = per_page

    # Return the query
    return {'query': query, 'filter': filter, 'sort': sort, 'offset': offset, 'limit': limit}

if __name__== '__main__':
    app.run(debug=True)
```

In this implementation:

- The `search` function handles incoming GET requests to the `/api/search` endpoint.
- It extracts search parameters from the query string.
- It calls a `formulate_query` function to translate these parameters into a database query, including pagination.
- It executes the query against the database using a hypothetical `query_database` function.
- Finally, it returns the results in JSON format.

### Security Considerations

When implementing search features, it is crucial to consider security:

- **Input Sanitization**: Ensure that user inputs are sanitized to prevent injection attacks.
- **Rate Limiting**: Implement rate limiting to prevent abuse of the search endpoint.
- **Authentication and Authorization**: If the search feature exposes sensitive data, you should implement proper authentication and authorization checks.

### Performance and Scalability

As your application grows, you may need to consider the performance and scalability of your search feature:

- **Caching**: Cache frequent queries and their results to reduce database load.
- **Search Engines**: For complex search requirements or very large datasets, use dedicated search engines like Elasticsearch or Apache Solr.
- **Load Balancing**: Distribute requests across multiple servers to balance the load.

Implementing search features with these considerations in mind will help ensure that your application can efficiently and securely handle user search queries.

---

## 4. Optional Features

In addition to the basic functions of organizing an audio library, you might want to include features that manipulate the audio files themselves. These can enhance the usability of your library for various applications, such as music production, DJing, or personal listening.

### 4.1 Extract Song Sections

To extract specific sections from songs, you will need to analyze the song structure and allow users to select the parts they want. This can be accomplished by combining audio analysis to detect sections with a tool like `pydub` to perform the extraction.

#### Using `pydub` to Extract Sections

`pydub` is a Python library that makes it easy to work with audio files, thanks to its simple and intuitive API. Here's how you might use it to extract a section:

```python
from pydub import AudioSegment

# Load the full song
full_song = AudioSegment.from_file('full_song.mp3')

# Define start and end times for the section in milliseconds
start_time = 60000  # e.g., 1 minute
end_time = 120000   # e.g., 2 minutes

# Extract the desired section
extracted_section = full_song[start_time:end_time]

# Export the extracted section to a file
extracted_section.export('extracted_section.mp3', format='mp3')
```

To integrate this with song structure analysis, you would need to use an audio analysis tool or library to detect the structures (like intro, verse, chorus, etc.) and then allow users to select these sections by name rather than by time.

### 4.2 Split Songs into Stems

`Spleeter` by Deezer is an advanced tool that uses machine learning to separate a song into different stems, such as vocals, drums, bass, and other instruments.

Here's a more detailed example of how to use `Spleeter` in Python:

```python
from spleeter.separator import Separator

# Initialize the separator for 2 stems separation: vocals and accompaniment
separator = Separator('spleeter:2stems')

# Perform separation on the audio file
separator.separate_to_file('audio_file.mp3', 'output_directory')

# After separation, you will have two files in the 'output_directory':
# - vocals.wav
# - accompaniment.wav
```

`Spleeter` supports different stem models, like 2 stems, 4 stems, and 5 stems, depending on how detailed you want the separation to be.

### 4.3 Change Tempo

Tempo changes can be achieved with `pyrubberband`. This library provides an interface to `rubberband-cli`, a command-line tool for audio stretching and pitch shifting.

Here's how to use `pyrubberband` to change the tempo of an audio signal:

```python
import pyrubberband as pyrb
import librosa  # Librosa is often used with pyrubberband for handling audio data

# Load an audio file
y, sr = librosa.load('audio_file.wav')

# Change tempo without changing pitch
# Increase tempo by 50%
y_fast = pyrb.time_stretch(y, sr, 1.5)

# Export the tempo-changed audio
librosa.output.write_wav('audio_file_fast.wav', y_fast, sr)
```

When changing the tempo, it's important to maintain the pitch to keep the musical key consistent, which `pyrubberband` does by default.

### 4.4 Change Key

Changing the key of an audio file involves pitch shifting, which can also be done with `pyrubberband`. This can be useful for transposing music to match the vocal range of a singer or to blend tracks in a DJ set.

Here's an example of changing the key:

```python
# Shift the pitch up by 2 semitones
y_higher_pitch = pyrb.pitch_shift(y, sr, n_steps=2)

# Export the key-changed audio
librosa.output.write_wav('audio_file_higher_pitch.wav', y_higher_pitch, sr)
```

Pitch shifting by a certain number of semitones changes the musical key. Positive values for `n_steps` will raise the pitch, while negative values will lower it.

### Conclusion

When adding these optional features to your audio library, consider the user interface and experience. For example, you might provide a graphical interface or command-line options that allow users to select sections or specify the number of semitones for key changes. Automation and batch processing could also be beneficial, especially for users dealing with large numbers of files.

---
