# Examples of Different Processess Used

## 1. Audio Analysis and Feature Extraction

```
import librosa
import numpy as np

def estimate_key(chromagram, smoothing_window=7):
    """
    Estimates the key signature from a chromagram using the Krumhansl-Schmuckler key-finding algorithm.

    Args:
        chromagram (np.ndarray): A 12-dimensional chroma feature vector (e.g., from librosa.feature.chroma_stft).
        smoothing_window (int): Size of the moving average window to smooth chroma features (optional).

    Returns:
        tuple: (key, correlation) where 'key' is the estimated key (e.g., "Cmaj") and 'correlation'
               is a measure of how well the chromagram correlates with the key profile.
    """

    # Key profiles (Krumhansl-Schmuckler key-finding algorithm profiles)
    # These are normalized major and minor key profiles representing the expected
    # distribution of pitch-class energy in each key.
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Key names for easy indexing
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Smooth the chromagram to reduce noise (optional but recommended)
    if smoothing_window > 1:
        chroma_smooth = np.zeros_like(chromagram)
        for i in range(12):
            chroma_smooth[i, :] = np.convolve(chromagram[i, :], np.ones(smoothing_window)/smoothing_window, mode='same')
    else:
        chroma_smooth = chromagram

    # Calculate total chroma energy for normalization
    total_energy = np.sum(chroma_smooth, axis=1)
    chroma_normalized = chroma_smooth / (total_energy[:, np.newaxis] + 1e-9) #Adding a small number to avoid zero division.

    # Rotate chroma to each key and compute correlation with major/minor profiles
    major_correlations = []
    minor_correlations = []
    for i in range(12):
        rotated_chroma = np.roll(chroma_normalized, -i, axis=0)  # rotate chromagram
        major_correlations.append(np.sum(rotated_chroma * major_profile[:, np.newaxis])) # Correlation with major profile
        minor_correlations.append(np.sum(rotated_chroma * minor_profile[:, np.newaxis])) # Correlation with minor profile

    # Find the best key and mode based on correlation
    major_max = np.max(major_correlations)
    minor_max = np.max(minor_correlations)

    if major_max > minor_max:
        best_key_index = np.argmax(major_correlations)
        key = key_names[best_key_index] + "maj"
        correlation = major_max
    else:
        best_key_index = np.argmax(minor_correlations)
        key = key_names[best_key_index] + "min"
        correlation = minor_max

    return key, correlation

def analyze_audio(file_path):
    """
    Analyzes an audio file to extract tempo, key signature, and other features.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: A dictionary containing tempo, key signature, and other extracted features.
    """
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Estimate Key using chroma features and Krumhansl-Schmuckler algorithm
    key, _ = estimate_key(chroma) #  _ = correlation (not used)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased MFCCs for better genre classification
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    features = np.vstack([mfccs, chroma, contrast, zcr]).T

    return {"tempo": tempo, "key": key, "features": features, "sr": sr}
```

## 2. Genre Classification (Placeholder - uses dummy data)

```
import librosa
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load  # Corrected import

def classify_genre(features, genre_model_path=None, scaler_path=None):
    """
    Classifies the genre of an audio file using a pre-trained machine learning model.

    Args:
        features (np.ndarray): Extracted audio features from the audio file.
        genre_model_path (str, optional): Path to the serialized genre classification model. Defaults to None.
        scaler_path (str, optional): Path to the serialized scaler. Defaults to None.

    Returns:
        str: The predicted genre of the audio file. Returns "Unknown" if the model or scaler are not found.
    """
    try:
        # Load the genre model and scaler if paths are provided
        if genre_model_path and scaler_path:
            genre_model = load(genre_model_path)
            scaler = StandardScaler()
            scaler.scale_ = np.load(scaler_path)
            scaler.mean_ = np.load(scaler_path.replace("npy","mean_.npy"))


            # Aggregate features across frames (mean is a common approach)
            X = np.mean(features, axis=0).reshape(1, -1)

            # Scale the features using the loaded scaler
            X_scaled = scaler.transform(X)

            # Predict the genre using the loaded model
            genre_prediction = genre_model.predict(X_scaled)[0] # The model knows which number is which Genre.
            return genre_prediction

        else:
            print("Error: Model or scaler path not provided. Using a pre-trained model greatly increases accuracy!")
            return "Unknown"

    except FileNotFoundError:
        print("Error: Model or scaler file not found. Make sure the paths are correct.")
        return "Unknown"
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Unknown"
```

## 3. Audio Fingerprinting

```
def fingerprint_audio(file_path, acoustid_api_key):
    try:
        duration, fingerprint = acoustid.fingerprint_file(file_path)
        results = acoustid.lookup(acoustid_api_key, fingerprint, duration)
        # Check for results before accessing them
        if results:
            for score, recording_id, title, artist in results:
                return {"artist": artist, "title": title}
        else:
            return {"artist": "Unknown", "title": "Unknown"}
    except acoustid.AcoustidError as e:
        print(f"AcoustID error: {e}")
        return {"artist": "Unknown", "title": "Unknown"}
```

## 4. Database Query (Simplified - uses hardcoded data as example)

```
def search_similar_songs(genre, tempo_range):
    # Assume we have a database 'songs' that has genres and tempos.
    songs = [
        {"title": "Song1", "genre": "Pop", "tempo": 120},
        {"title": "Song2", "genre": "Rock", "tempo": 130},
        {"title": "Song3", "genre": "Pop", "tempo": 125},
        {"title": "Song4", "genre": "Jazz", "tempo": 90},
    ]

    similar_songs = [song for song in songs
                     if song["genre"] == genre and
                     tempo_range[0] <= song["tempo"] <= tempo_range[1]]

    return similar_songs
```

## 5. File Organization

```
def organize_file(file_path, genre, key, tempo):
    base_dir = "organized_music"
    genre_dir = os.path.join(base_dir, genre)
    key_dir = os.path.join(genre_dir, key)
    tempo_dir = os.path.join(key_dir, str(int(tempo)))  # Simplified tempo folder naming

    os.makedirs(tempo_dir, exist_ok=True)
    new_path = os.path.join(tempo_dir, os.path.basename(file_path))
    try:
        shutil.move(file_path, new_path)
        print(f"Moved '{os.path.basename(file_path)}' to '{new_path}'")
    except Exception as e:
        print(f"Error moving file: {e}")
```

## 6. Model Training

```
def train_genre_model(audio_files, labels, model_output_path, test_size=0.2):

    all_features = []
    for file_path in audio_files:
        try:
            features = analyze_audio(file_path)['features']
            all_features.append(np.mean(features, axis=0))  # Aggregate across frames
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue  # Skip files that cause errors

    X = np.array(all_features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) # Test/Train Split

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    genre_model = MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, random_state=42)  # Hidden Layers, Iterations.
    genre_model.fit(X_train, y_train) # Train the Model

    accuracy = genre_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Now, save the scaler and the model
    np.save('scaler.npy', scaler.scale_)
    np.save('scaler.mean_.npy', scaler.mean_)

    from joblib import dump, load

    dump(genre_model, 'genre_model.joblib')

    return genre_model, scaler
```
