# 🎵 ProjectAudio

## 🎯 The Goal

To create an intelligent audio library manager and transformation system with the following objectives:

- **Ingest** an entire instrumental audio library.
- **Analyze** each track for:
  - Key
  - BPM (Beats Per Minute)
  - Time Signature
  - Genre
- **Automatically write** analysis results to:
  - Metadata tags
  - Local storage files (e.g., JSON, CSV)
- **Organize** songs into a nested folder structure:
  - Top-level: `Genre`
  - Subfolder: `Key`
  - Sub-subfolder: `BPM`
  - Sub-sub-subfolder: `Time Signature`
  - *Folder sorting is optional and customizable.*
- **Search** the library using:
  - Key, BPM, Time Signature, Genre
- **Find related tracks** using audio fingerprinting and similarity detection.
- **Quantize** songs to a preferred tempo.
- **Transpose** songs by semitones to match a preferred key.
- **Split** songs into stems, with options for:
  - All instruments
  - A single instrument of choice (e.g., drums, bass)

---

## 🧠 System Capabilities

### 1. 📥 Ingest Audio Library

- Aggregate audio files from local drives or cloud sources.
- Normalize all tracks to a common format (e.g., WAV or MP3).
- Store audio in a central indexed location for processing.

### 2. 🧪 Audio Analysis Features

- **Key Detection**: Identify the musical key of a track.
- **Tempo/BPM Analysis**: Detect the speed of the song.
- **Time Signature**: Determine rhythmic grouping (e.g., 4/4, 6/8).
- **Genre Classification**: Categorize style using ML or metadata.
- **Audio Fingerprinting**: Create a unique hash for similarity matching.
- **Song Structure Detection**: Identify and label:
  - Intro
  - Verse
  - Pre-Chorus
  - Chorus
  - Bridge
  - Outro

### 3. 🗂️ Smart Organization

Folder hierarchy based on analysis:

```markdown
/LibraryRoot
  └── Genre
      └── Key
          └── BPM
              └── TimeSignature
                  └── SongName.ext
```

Optional automation with override settings (e.g., custom genre mapping, BPM rounding).

### 4. 🔍 Advanced Search

Query and filter by:

- 🎼 Key Signature
- 🕒 Tempo/BPM
- 🧬 Genre
- 🎛️ Time Signature
- 🤝 Similar Songs (based on audio fingerprint distance)

### 5. 🧩 Optional Advanced Features

#### 🎛️ Structural Extraction

Export individual sections from track structure:

- Intro
- Verse
- Pre-Chorus
- Chorus
- Bridge
- Outro

#### 🎚️ Stem Separation

Split tracks into isolated stems:

- 🎤 Vocals
- 🎸 Guitars
- 🧱 Bass
- 🎹 Keys
- 🥁 Drums

Built-in support or integration with Spleeter, Demucs, or Open-Unmix.

#### 🌀 Tempo Adjustment

Quantize any track to a new target BPM without pitch distortion.

#### 🔁 Key Transposition

Automatically shift pitch to a new key by semitone adjustment.

## ⚙️ Implementation Strategy

### Phase 1: Ingestion & Analysis

- Parse metadata
- Run analysis (Librosa, Essentia, Madmom, etc.)
- Store results (e.g., SQLite, JSON, or embedded tags)

### Phase 2: Organization Engine

- Scriptable folder creation
- Rules engine for custom sorting
- Logging and rollback features

### Phase 3: Search & Recommendation

- Build a searchable index
- Integrate fingerprinting (e.g., Chromaprint)
- Train a similarity model (e.g., cosine distance on MFCCs)

### Phase 4: Optional Processing Tools

- Interface with Spleeter or Demucs for stem separation
- Use `pydub`, `librosa`, or `rubberband` for pitch and tempo changes
- CLI and GUI interface options

## 🧰 Recommended Tech Stack

| Feature                | Suggested Tools/Libraries            |
|------------------------|--------------------------------------|
| Key & BPM Detection    | `librosa`, `Essentia`                |
| Time Signature         | `madmom`                             |
| Genre Classification   | `scikit-learn`, `TensorFlow`, `PyTorch` |
| Metadata Handling      | `mutagen`, `tinytag`                 |
| Stem Separation        | `Spleeter`, `Demucs`, `Open-Unmix`  |
| Pitch/Tempo Change     | `librosa`, `rubberband`, `pydub`    |
| Fingerprinting         | `chromaprint`, `dejavu`              |
| File I/O & Tagging     | `os`, `shutil`, `eyed3`, `mutagen`   |

---
