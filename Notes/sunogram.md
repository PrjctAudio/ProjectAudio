# Sunomaid

```mermaid
graph TD
    subgraph "Core Idea"
        A[/"Treating Audio like Language"/] --> B(Predict Next Audio Unit);
    end

    subgraph "Inputs & Foundational Resources"
        C[Massive & Diverse Dataset] --> C1;
        C1["Audio (Music w/ Vocals, A Cappella, Instrumentals, Speech)"];
        C2["Metadata (Lyrics<br/><i>(Time-Aligned)</i>, Tags<br/><i>(Genre, Mood, etc.)</i>,<br/>Tempo, Key, Structure)"];
        C3["Data Sources & Legality<br/><i>(Licensing, Public Domain?)</i>"];
        C --> C4[Data Prep & Representation];
        C4 --> C5{"Audio Representation Choices<br/>(e.g., Quantized Codes via Codec,<br/>Spectrograms, Raw Waveform?)"};

        P[User Prompt<br/>(Lyrics, Style, Mood)] --> P1[Text Embedding];
    end

    subgraph "Model & Training Pipeline"
        T[Large-Scale Training];
        T --> T1["Compute Resources<br/>(GPUs/TPUs)"];
        T --> T2["Training Objective<br/>(Predict Next Token/Segment)"];
        T --> T3["Optimization & Techniques<br/>(AdamW, Curriculum Learning)"];

        M[Sophisticated Model Architecture];
        M --> M1["Transformer Backbone<br/><i>(Self-Attention for Long Dependencies)</i>"];
        M --> M2["Audio Tokenization/Handling<br/><i>(Works with Representation C5)</i>"];
        M --> M3["Hierarchical Structure?<br/><i>(Multiple Timescales)</i>"];
        M --> M4["Conditioning Mechanisms<br/><i>(Cross-Attention for Prompts)</i>"];
        M --> M5["Unified or Separate Models?<br/><i>(e.g., Music + Vocal Synth)</i>"];

        C5 --> M2;  // Representation feeds into how the Model handles audio
        C --> T;     // Dataset is used FOR Training
        M --> T;     // Architecture is defined FOR Training
        T --> TrainedModel{Trained Model}; // Training PRODUCES the Trained Model
    end

    subgraph "Generation & Refinement Process"
        P1 --> TrainedModel; // Text Embeddings condition the model
        PrevAudio[Previously Generated Audio Units] --> TrainedModel; // Model uses context
        TrainedModel --> RawOutput[Raw Model Output<br/>(e.g., Audio Codes, Spectrograms)];

        RawOutput --> R[Refinement & Post-Processing];
        R --> R1["Vocoding / Synthesis<br/><i>(Intermediate Rep -> High-Fidelity Audio)</i><br/><b>Crucial for Quality!</b>"];
        R --> R2["Fine-tuning<br/><i>(On specific data)</i>"];
        R --> R3["RLHF / Instruction Tuning<br/><i>(Human Feedback)</i>"];
        R --> R4["Rule-Based Fixes / Heuristics"];

        R3 ---> T; // Feedback loop potentially influences further Training/Fine-tuning

        R --> FinalAudio[Final Audio Output];
    end

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#ccf,stroke:#333,stroke-width:2px
    style T fill:#ff9,stroke:#333,stroke-width:2px
    style R fill:#9cf,stroke:#333,stroke-width:2px
    style A fill:#bbf,stroke:#333,stroke-width:4px,font-weight:bold
    style P fill:#lightgrey,stroke:#333,stroke-width:1px
    style FinalAudio fill:#9f9,stroke:#333,stroke-width:2px,font-weight:bold
```

---
