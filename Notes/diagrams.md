# Diagrams

```mermaid
graph TD
    A[Three-Part Prompt] --> B1[Style Description]
    A --> B2[Lyrics Text (Optional)]
    A --> B3[Exclusions]

    B1 --> C1[Bark Model]
    B1 --> C2[Chirp Model]
    B2 --> C1
    B2 --> C2
    B3 --> C1
    B3 --> C2

    subgraph Bark [Bark: Vocal Generator]
        C1 --> D1[Transformer 1: Text to Semantic Tokens]
        D1 --> D2[Transformer 2: Semantic to Coarse Audio Tokens]
        D2 --> D3[Transformer 3: Fine Audio Tokens (Non-Causal)]
        D3 --> E1[Vocal Track Output]
    end

    subgraph Chirp [Chirp: Instrumental Generator]
        C2 --> F1[Transformer 1: Text to Semantic Tokens]
        F1 --> F2[Transformer 2: Semantic to Coarse Audio Tokens]
        F2 --> F3[Transformer 3: Fine Audio Tokens (Non-Causal)]
        F3 --> E2[Instrumental Track Output]
    end

    E1 --> G[Mixing & Alignment]
    E2 --> G

    G --> H[Final Song Output]

    %% Style
    classDef model fill:#f9f,stroke:#333,stroke-width:1px
    class C1,C2 model

    classDef transformer fill:#cff,stroke:#000,stroke-width:1px
    class D1,D2,D3,F1,F2,F3 transformer

    classDef prompt fill:#fcf,stroke:#000,stroke-width:1px
    class A,B1,B2,B3 prompt

    classDef output fill:#cfc,stroke:#000,stroke-width:1px
    class E1,E2,G,H output
```
