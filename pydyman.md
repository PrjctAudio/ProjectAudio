# Demonstration on how the audio pipeline parts fit together

## 1. Audio Analysis and Feature Extraction

```
def analyze_audio(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    key = "Cmaj" # Placeholder: Implement a more accurate key detection
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased MFCCs for better genre classification
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    features = np.vstack([mfccs, chroma, contrast, zcr]).T
    return {"tempo": tempo, "key": key, "features": features, "sr": sr}
```

## 2. Genre Classification (Placeholder - uses dummy data)

```
def classify_genre(features, genre_model=None, scaler=None):
    # Load the genre model if not provided.  Trained separately to keep this self-contained.
    if genre_model is None or scaler is None:
        print("No trained model and scaler provided. Provide a trained model file, or use the train function to make one.")
        return "Unknown"
    X = scaler.transform(np.mean(features, axis=0).reshape(1, -1)) # Reshape and scale
    genre = genre_model.predict(X)[0]  # The model knows which number is which Genre.
    return genre
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

## Main function to orchestrate the pipeline

```
def main(file_path, acoustid_api_key, model_path=None, scaler_path=None):

    analysis_results = analyze_audio(file_path)
    tempo = analysis_results["tempo"]
    key = analysis_results["key"]
    features = analysis_results["features"]
    sr = analysis_results["sr"]

    from joblib import dump, load

    if model_path is None:
        # In this example we will create a fake model, but you need to train your model elsewhere.
        print("You didn't provide a trained Genre model, therefore we must exit")
        sys.exit(1)

    genre_model = load(model_path)

    scaler = StandardScaler()
    scaler.scale_ = np.load(scaler_path)
    scaler.mean_ = np.load(scaler_path.replace("npy","mean_.npy"))

    genre = classify_genre(features, genre_model, scaler) # Classify using new data

    fingerprint_results = fingerprint_audio(file_path, acoustid_api_key)
    artist = fingerprint_results["artist"]
    title = fingerprint_results["title"]

    print(f"Identified: {artist} - {title}, Genre: {genre}, Tempo: {tempo}, Key: {key}")

    # Example database search
    similar_songs = search_similar_songs(genre, (tempo - 5, tempo + 5)) # Search tempos a little faster and slower
    print("Similar Songs:")
    for song in similar_songs:
        print(f"- {song['title']}")

    # File Organization
    organize_file(file_path, genre, key, tempo)

if __name__ == "__main__":
    # CLI invocation example:
    # python your_script.py "path/to/audio.mp3" "your_acoustid_api_key" "path/to/genre_model.joblib" "path/to/scaler.npy"
    # This version takes the API key as a command line argument and the model path.
    # Example usage
    if len(sys.argv) != 5:
        print("Usage: python your_script.py <file_path> <acoustid_api_key> <model_path> <scaler_path>")
        sys.exit(1)

    file_path = sys.argv[1] # "path/to/audio.mp3"
    acoustid_api_key = sys.argv[2] # "your_acoustid_api_key"
    model_path = sys.argv[3] # "path/to/genre_model.joblib"
    scaler_path = sys.argv[4] # "path/to/scaler.npy"

    main(file_path, acoustid_api_key, model_path, scaler_path)
    ```

----
