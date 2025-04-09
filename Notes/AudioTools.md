# **Project Goal:** ProjectAudio

**Core Features:**

1. **Content Ingestion & Library Management:**
    * **YouTube Downloading:** Can download videos from YouTube using a provided video ID, automatically extracting the audio track. (`yt_dlp`)
    * **Local File Handling:** Can process individual local audio files (`.wav`, `.mp3`) or entire directories containing such files.
    * **Format Conversion & Standardization:** Automatically converts `.mp3` files to `.wav` format. Ensures all ingested audio is standardized to a 44.1kHz sample rate. (`librosa`, `soundfile`)
    * **Library Persistence:** Maintains a persistent database (`library/database.p`) using `pickle` to store metadata about each track (ID, name, source URL/path, paths to audio/video files) and its extracted features.
    * **Unique Identification:** Generates unique IDs for local files based on filename and a hash of the file path to prevent duplicates and manage them within the library structure. (`hashlib`)
    * **Command-Line Management:** Allows adding new content (YouTube IDs, file paths, directories) and removing existing items from the library via CLI arguments (`-a`, `-r`).

2. **Deep Audio Analysis & Feature Extraction:**
    * **Automated Analysis:** Automatically analyzes audio files upon addition to the library.
    * **Feature Caching:** Saves extracted audio features to separate `.a` files (e.g., `library/some_id.a`) using `pickle` to avoid redundant computations on subsequent runs.
    * **Source Separation (Stems):** Uses the Demucs neural network (`htdemucs_6s` model via `subprocess`) to separate audio files into distinct instrumental/vocal stems (default: bass, drums, guitar, other, piano, vocals). Stems are saved in the `separated/htdemucs_6s/<id>/` directory.
    * **Musical Structure Analysis:** Identifies structural segments (like verse, chorus) and their boundaries within the audio using `sf_segmenter`.
    * **Pitch & Key Detection:**
        * Performs detailed pitch tracking (fundamental frequency over time) using the Crepe neural network (`crepe`).
        * Calculates the overall average frequency and estimates the musical key (e.g., C4, G#3).
    * **Tempo & Beat Detection:** Detects the tempo (BPM) and the timing of beats within the audio using `librosa`.
    * **Timbre Analysis:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) and their deltas (beat-synchronous) to represent the timbral quality of the sound using `librosa`. Calculates an average timbre value.
    * **Intensity/Loudness Analysis:**
        * Calculates beat-synchronous loudness/intensity using Constant-Q Transform (CQT) perceptual weighting (`librosa`). Calculates an average intensity value.
        * Calculates overall RMS volume and perceptual loudness after trimming silence (`librosa`, custom functions).
    * **Chroma Analysis:** Extracts beat-synchronous Chroma features representing the distribution of musical pitch classes (`librosa`). Calculates an average pitch class distribution value.

3. **Audio Processing & Manipulation:**
    * **Quantization:**
        * Aligns audio files (and their corresponding stems) to a regular beat grid.
        * Can quantize to a user-specified target BPM (`-t`) or keep the track's original detected tempo (`-k`).
        * Uses `librosa` for beat detection and `pyrubberband` for high-quality time-stretching based on a calculated time map aligning original beats to the target grid.
        * Saves the quantized full audio and all corresponding quantized stems to the `processed` directory with descriptive filenames including original/target BPM, key, etc.
    * **Audio-to-MIDI Conversion:**
        * Uses the Basic Pitch neural network (`basic_pitch`) to transcribe audio into MIDI files (`-m` flag).
        * This can be triggered during the initial analysis (`get_audio_features`) to convert the *original stems* or during quantization (`quantizeAudio`) to convert the *quantized full audio and quantized stems*.
        * MIDI files are saved alongside the audio they were generated from (either in `separated` or `processed` directories).

4. **Content-Based Search & Retrieval:**
    * **Similarity Search:** Allows searching the library for tracks musically similar to a specified query track (using its ID via `-s`).
    * **Feature-Based Comparison:** Similarity is primarily calculated based on the distance between the average `frequency` (pitch) of tracks.
    * **Tempo-Aware Search:** Optionally includes tempo difference as an additional search criterion (`-st` flag), prioritizing results that are similar in both pitch and tempo.
    * **Chained Results:** Can return a specified number (`-sa`) of related items, allowing for discovery chains (e.g., finding 10 tracks sequentially related to the initial query).
    * **Integrated Workflow:** Search results can be directly piped into the quantization process (`-q all` or specific IDs combined with `-s`), allowing users to find similar tracks and immediately quantize them to a common tempo.

5. **Technical Aspects:**
    * **Command-Line Interface:** All primary functions are controlled via command-line arguments (`argparse`).
    * **GPU Acceleration:** Leverages GPU capabilities (via `numba.cuda` and implicitly through `tensorflow` used by Crepe, Basic Pitch, and Demucs) for potentially faster processing, provided CUDA and TensorFlow with GPU support are set up correctly.
    * **Modularity:** Uses established external libraries for core ML and audio processing tasks.

In essence, ProjectAudio acts as an automated music analysis and processing pipeline, creating a rich, structured library from raw audio/video sources, enabling creative reuse and discovery based on musical content.

## My Machine Specs

**System Overview:**

* **Model:** Dell Precision 5570 (Mobile workstation-class laptop)
* **Chassis:** Standard notebook form factor
* **OS:** Windows 11 Pro 24H2
* **Date:** February 18th, 2025

**Key Components:**

1. **Processor (CPU):**
   * **Intel Core i7-12800H** (12th Gen "Alder Lake")
   * **Cores/Threads:** 14 (6 Performance + 8 Efficient) / 20 threads
   * **Clock Speed:** Up to 4.8 GHz with dynamic adjustment
   * **Cache:** Large L1, L2, L3 for faster data access
   * **Performance Score:** 9.4/9.9 (exceptional)
   * **Architecture:** Hybrid design for performance and efficiency

2. **Memory (RAM):**
   * **Capacity:** 32GB DDR5 (4800 MHz, quad-channel)
   * **Configuration:** Two 16GB SODIMM modules (Hyundai Electronics)
   * **Upgradable:** Up to 64GB

3. **Graphics (GPU):**
   * **Integrated:** Intel Iris Xe Graphics
     * **Performance Scores:** 8.4 (graphics), 9.9 (D3D)
     * **Video Memory:** 128MB dedicated, 15.83GB shared
   * **Discrete:** NVIDIA RTX A2000 8GB Laptop GPU
     * **Purpose:** Professional applications (CAD, 3D rendering, AI)
     * **Features:** Ray tracing, AI acceleration, 8GB GDDR6 memory
     * **Status:** Currently offline (likely due to NVIDIA Optimus for power-saving)

---

## My Installed Development Tools

A prompt theme engine for any shell 24.19.0
A prompt theme engine for any shell 25.6.0
AnythingLLM 1.7.4
AnythingLLM Desktop 1.7.4
Chocolatey (Install Only) 2.4.2.0
Chocolatey GUI 2.1.1.0
ColourCopy 4.18.23110.3
CUDNN Development 9.7
CUDNN Samples 9.7
Cursor (User) 0.46.11
Fabric 0.1.9
FFmpeg 7.1
Git 2.47.1.2
GitHub Desktop 3.4.18
glow 2.1.0
Go Programming Language amd64 1.24.1
Google Chrome Canary 136.0.7095.0
gsudo 2.6.0+Branch.tags-v2.6.0.Sha.8067a30b9a83dc3ba318c0007644d1878773abec
"HTTP(S) debugging, development & testing tool" 1.20.0
ICAT 0.6.17
LM Studio 0.3.14
Microsoft Visual Studio Code Insiders (User) 1.99.0
Microsoft Visual Studio Installer 3.13.2069.59209
mise-en-place 2025.3.10
Node.js 23.10.0
Node.js JavaScript Runtime 23.10.0
NVIDIA CUDA Toolkit 12.8
NVIDIA FrameView 1.6.10929.35462032
NVIDIA FrameView SDK 1.5.10819.35301613
NVIDIA Nsight Compute 2025.1.0.0 (build 35237751)
NVIDIA Nsight Compute CLI 2025.1.0.0 (build 35237751)
NVIDIA Nsight Systems 2024.6.2.225
NVIDIA Nsight Visual Studio Edition 25.1.0.25002
Oh My Posh 24.19.0
Oh My Posh 25.6.0
Ollama 0.5.11
Pandoc 3.6.3
PowerShell 7.5.0 SHA: 99dab561892364d82d4965068f7f8b175e768b1b+99dab561892364d82d4965068f7f8b175e768b1b
PowerShell 7-x64 7.5.0.0
PowerToys (Preview) x64 0.89.0
Python 3.12.9 (64-bit) 3.12.9150.0
Python 3.13.2 (64-bit) 3.13.2150.0
Replit 1.0.14
Rustup: the Rust toolchain installer 1.28.1
scoop-search 2.0.0
Scour the web for whatever you’re looking for 1.1.11
Scourhead 1.1.11
System activity monitor 15.15
System Informer 3.2.25082.2220
Tesseract-OCR - open source OCR engine 5.4.0.20240606
Timeline Explorer 2.1.0+0bc6c40071317884fa0510c1d3a6fb6f57b01b55
Trippy 0.12.2
UniGetUI 3.1.8
Visual Studio Build Tools 2019 16.11.45
Warp v0.2025.03.26.08.10.stable_02
Windows SDK AddOn 10.1.0.0
Windows Software Development Kit - Windows 10.0.19041.685 10.1.19041.685
Windsurf (User) 1.94.0

## Tools That Could Help

1. **PianoTransformers (<https://github.com/rlax59us/PianoTransformers>)**
   A project focused on generating piano music using Transformer models, a type of neural network commonly used in natural language processing and now adapted for music generation. It likely leverages deep learning to compose or transform piano sequences, aimed at musicians or AI researchers interested in generative music.

2. **dechorder (<https://github.com/YuriyGuts/dechorder>)**
   A tool designed to transcribe polyphonic audio into guitar chords. It uses signal processing and machine learning techniques to analyze audio files and output chord progressions, making it useful for musicians looking to learn songs by ear or automate transcription tasks.

3. **ZLHistogramAudioPlot (<https://github.com/zhxnlai/ZLHistogramAudioPlot>)**
   A hardware-accelerated audio visualization tool built with EZAudio, inspired by AudioCopy. It displays audio data as a histogram plot, with customizable frequency ranges and bin settings. This repo is geared toward developers needing real-time audio visualization for music or sound analysis applications.

4. **VisualMidi (<https://github.com/Wally869/VisualMidi>)**
   A project for visualizing MIDI files, likely rendering musical notes and sequences into graphical representations. It’s aimed at musicians, educators, or developers who want to analyze or present MIDI data in a more intuitive, visual format.

5. **spafe (<https://github.com/SuperKogito/spafe>)**
   A library for speech and audio feature extraction, offering tools like MFCCs, filter banks, and other acoustic features. While not exclusively music-focused, it’s highly relevant for music analysis, speech processing, or audio classification tasks, targeting researchers and engineers in audio signal processing.

6. **Music-Source-Separation (<https://github.com/s603122001/Music-Source-Separation>)**
   A repository implementing music source separation, likely using deep learning models to isolate individual instruments or vocals from mixed audio tracks. It’s valuable for music producers, remixers, or researchers working on audio decomposition and remixing.

7. **learn-an-effective-lip-reading-model-without-pains (<https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains>)**
   While primarily focused on lip-reading from video using audio-visual data, this repo bridges music and speech by leveraging audio cues. It provides a deep learning model for lip-reading, useful for applications like syncing lyrics or analyzing vocal performances in music videos.

8. **music-lrc-match (<https://github.com/RavelloH/music-lrc-match>)**
   A tool to match and synchronize lyrics (in LRC format) with music files. It automates the process of aligning timestamped lyrics to audio, benefiting karaoke enthusiasts, music app developers, or anyone needing precise lyric synchronization.

9. **LearningfromAudio (<https://github.com/theadamsabra/LearningfromAudio>)**
   A project exploring machine learning techniques applied to audio data, likely including music-related tasks such as classification, generation, or feature extraction. It’s aimed at learners or researchers experimenting with audio-based AI models, though specifics depend on the repo’s development stage.

1. **slidingHPSS (<https://github.com/tachi-hi/slidingHPSS>)**
   A project implementing sliding Harmonic-Percussive Source Separation (HPSS) and a two-stage HPSS for singing voice enhancement. It separates audio into harmonic, percussive, and vocal components, useful for karaoke generation or audio analysis, with a focus on real-time processing.

2. **transcribe (<https://github.com/sevagh/transcribe>)**
   A tool for automatic music transcription, converting audio into symbolic representations like MIDI. It likely uses signal processing or machine learning to identify notes and rhythms, aimed at musicians or researchers needing transcriptions from audio recordings.

3. **pyace (<https://github.com/tangkk/pyace>)**
   A Python implementation of the Audio Chord Estimation (ACE) algorithm. It detects chords from audio files, providing a lightweight solution for music analysis or educational tools for musicians learning chord progressions.

4. **Mel-Spectrum-Analyzer (<https://github.com/tabahi/Mel-Spectrum-Analyzer>)**
   A tool for visualizing audio through Mel spectrograms, which represent frequency content over time on a perceptually relevant scale. It’s designed for audio engineers or researchers analyzing music or speech patterns.

5. **CLMR (<https://github.com/Spijkervet/CLMR>)**
   A Contrastive Learning framework for Music Representations (CLMR), leveraging self-supervised learning to generate embeddings for music tracks. It’s aimed at MIR tasks like music similarity, classification, or recommendation systems.

6. **symbolic-music-datasets (<https://github.com/wayne391/symbolic-music-datasets>)**
   A collection of symbolic music datasets (e.g., MIDI files) for research and machine learning applications. It provides standardized data for tasks like music generation, analysis, or training models in MIR.

7. **jazznet (<https://github.com/tosiron/jazznet>)**
   A project focused on jazz music generation or analysis using neural networks. It likely involves training models on jazz-specific datasets to create improvisations or study jazz structures, targeting musicians and AI enthusiasts.

8. **HPSS (<https://github.com/tachi-hi/HPSS>)**
   An earlier implementation of Harmonic-Percussive Source Separation by the same author as slidingHPSS. It separates audio into harmonic and percussive elements, useful for music production, sound design, or academic research in audio processing.

9. **stable-audio-tools (<https://github.com/Stability-AI/stable-audio-tools>)**
   A toolkit from Stability AI for audio generation and manipulation, likely powered by diffusion models (similar to their image generation work). It’s designed for creating high-quality audio or music, appealing to creators and AI researchers.

10. **RT-MIR-OSC (<https://github.com/sideeffectdk/RT-MIR-OSC>)**
    A real-time Music Information Retrieval (MIR) system using Open Sound Control (OSC) for communication. It extracts features like tempo or pitch in real time, suitable for live music applications or interactive installations.

11. **KKNet (<https://github.com/SmoothKen/KKNet>)**
    A neural network project possibly related to music or audio (context is limited), potentially for tasks like key detection or karaoke processing. It’s likely experimental, aimed at developers exploring deep learning in audio.

12. **pypianoroll (<https://github.com/salu133445/pypianoroll>)**
    A Python library for handling piano roll representations of music (e.g., MIDI data). It provides tools for visualization, manipulation, and analysis, widely used in MIR research and music generation projects.

13. **inceptionkeynet (<https://github.com/stefan-baumann/inceptionkeynet>)**
    A deep learning model for musical key estimation based on the Inception architecture. It predicts the key of a piece of music from audio, useful for music tagging, analysis, or educational tools.

14. **arranger (<https://github.com/salu133445/arranger>)**
    A tool for automatic music arrangement, likely converting symbolic music (e.g., MIDI) into multi-track arrangements. It’s aimed at composers or producers seeking to automate orchestration or arrangement tasks.

15. **Vocal-Melody-Extraction (<https://github.com/s603122001/Vocal-Melody-Extraction>)**
    A project for extracting vocal melodies from mixed audio tracks using signal processing or machine learning. It’s valuable for music production, karaoke creation, or MIR research focused on melody analysis.

1. **Zero_Shot_Audio_Source_Separation (<https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation>)**
   The official code repository for "Zero-shot Audio Source Separation through Query-based Learning from Weakly-labeled Data" (AAAI 2022). It implements a method to separate audio sources (e.g., instruments, vocals) without prior training on specific examples, using weakly labeled data and query-based learning. It’s aimed at researchers and audio engineers exploring advanced separation techniques.

2. **HTS-Audio-Transformer (<https://github.com/RetroCirce/HTS-Audio-Transformer>)**
   The official code for "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection." This project uses a Transformer model to classify and detect sounds hierarchically, supporting datasets like AudioSet and ESC-50. It’s designed for audio researchers working on sound event detection or classification tasks.

3. **masters-thesis-music-autotagging (<https://github.com/renesemela/masters-thesis-music-autotagging>)**
   A master’s thesis project focused on music autotagging, likely using machine learning to automatically assign tags (e.g., genre, mood) to music tracks based on audio features. It’s geared toward academic researchers or students studying music information retrieval (MIR) and tagging systems.

4. **js-audio-recording (<https://github.com/ralzohairi/js-audio-recording>)**
   A JavaScript-based project for audio recording in the browser, utilizing Web Audio API or similar technologies. It’s a lightweight tool for developers building web applications that need to capture and process audio, such as music or voice recording interfaces.

5. **automatic-music-transcription (<https://github.com/rachhshruti/automatic-music-transcription>)**
   A project for automatic music transcription, converting audio recordings into symbolic representations like sheet music or MIDI. It likely employs signal processing or deep learning techniques, targeting musicians and MIR researchers who need automated transcription solutions.

6. **equivariant-self-supervision-tempo (<https://github.com/Quint-e/equivariant-self-supervision-tempo>)**
   The official implementation of "Equivariant Self-Supervision for Musical Tempo Estimation" (ISMIR 2022). This project uses self-supervised learning with equivariant constraints to estimate the tempo of music tracks, offering pretrained models and datasets. It’s aimed at MIR researchers studying tempo analysis.

7. **ccml (<https://github.com/pxaris/ccml>)**
   A project likely related to "Cross-Modal Learning" (CCML), possibly linking audio (music) with other modalities like visuals or text. While specifics are limited, it’s probably designed for researchers exploring multimodal machine learning applications in music or audio contexts.

8. **AudioOwl (<https://github.com/dodiku/AudioOwl>)**
   A tool or library for audio processing and analysis, potentially focused on music or sound exploration. Details are sparse, but it’s likely intended for developers or researchers needing a flexible framework for audio feature extraction, visualization, or manipulation.

1. **audioflux (<https://audioflux.top>)**
   A library for audio and music analysis, focusing on feature extraction. Implemented in C and Python, it offers systematic, multi-dimensional feature extraction for tasks like classification, separation, and music information retrieval (MIR). It’s optimized for high performance, supporting mobile platforms and real-time audio stream processing.

2. **audiomate (<https://audiomate.readthedocs.io/en/latest/>)**
   A framework for processing and managing audio data, particularly for batch-wise audio processing. It provides tools to handle corpora, utterances, and features, with a pipeline system for customizable audio processing steps like spectrogram or MFCC extraction. It’s designed for deep learning audio applications.

3. **babycat (<https://babycat.io>)**
   A library for audio manipulation and playback, written in Rust with Python bindings. It excels at decoding, resampling, and playing back audio files efficiently, supporting multiple formats. It’s useful for tasks requiring fast audio processing, like waveform generation or batch audio analysis.

4. **dsdtools (<https://dsdtools.readthedocs.io/en/latest/>)**
   A toolkit for working with Direct Stream Digital (DSD) audio, a high-resolution audio format. It provides utilities for reading, writing, and converting DSD files, aimed at audiophiles and researchers working with specialized audio formats.

5. **essentia (<https://essentia.upf.edu/>)**
   An open-source library for audio and music analysis, feature extraction, and MIR. It offers a wide range of algorithms for tasks like tempo detection, key estimation, and audio segmentation, with a focus on real-time and large-scale applications. It’s widely used in academic and industrial settings.

6. **librosa (<https://librosa.org>)**
   A popular Python library for music and audio analysis, providing tools for feature extraction (e.g., spectrograms, MFCCs), beat tracking, and tempo estimation. It’s user-friendly, well-documented, and integrates seamlessly with NumPy, making it a go-to for MIR and audio signal processing.

7. **mirdata (<https://mirdata.readthedocs.io/en/latest/>)**
   A library for working with Music Information Retrieval (MIR) datasets. It standardizes access to datasets (e.g., downloading, loading, validating), supporting research by providing a unified interface for audio, annotations, and metadata across various MIR collections.

8. **mutagen (<https://mutagen.readthedocs.io/en/latest/>)**
   A library for handling audio metadata (tags) and file formats. It supports reading and writing tags for formats like MP3, FLAC, and OGG, making it ideal for organizing music libraries or extracting metadata for further analysis.

9. **msaf (<https://pythonhosted.org/msaf/>)**
   The Music Structure Analysis Framework (MSAF) is a Python package for analyzing the structural segmentation of music. It includes algorithms, features, and evaluation metrics to identify sections (e.g., verses, choruses) in audio, aimed at MIR researchers.

10. **torchaudio (<https://pytorch.org/audio/stable/index.html>)**
    A PyTorch-based library for audio processing, offering tools for loading, transforming, and augmenting audio data. It integrates with PyTorch’s ecosystem, making it ideal for deep learning tasks like speech recognition or audio generation, with GPU acceleration support.

11. **pedalboard (<https://spotify.github.io/pedalboard/>)**
    A library from Spotify for audio effects processing, designed for music production and experimentation. It provides a simple API to apply effects like reverb, distortion, or EQ to audio signals, with a focus on performance and ease of use.

12. **spafe (<https://superkogito.github.io/spafe/>)**
    A library for speech and audio feature extraction, offering tools like MFCCs, filter banks, and other acoustic features. It’s lightweight and aimed at researchers working on speech processing, audio classification, or related signal analysis tasks.
