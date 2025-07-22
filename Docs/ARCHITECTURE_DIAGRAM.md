# System Architecture Diagrams

This document provides visual representations of the AI Bio Autocomplete system architecture.

## 🏗️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Browser                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Next.js Frontend (Port 3000)               │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │   │
│  │  │   Form     │  │ 5-Hook System │  │  Feature         │  │   │
│  │  │ Component  │──│ Architecture  │──│ Coordinators     │  │   │
│  │  └────────────┘  └──────────────┘  └────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend Services                             │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │   API Server     │  │   MLX Server     │  │    Ollama       │  │
│  │   (Port 8001)    │  │   (Port 8003)    │  │  (Port 11434)   │  │
│  │                  │  │                  │  │                 │  │
│  │  - Hybrid API    │  │  - Fine-tuned    │  │  - Gemma 3 12B │  │
│  │  - Vector Search │  │    Phi-3 Model   │  │  - Fallback    │  │
│  │  - Orchestration │  │  - Fast Inference│  │    Generation  │  │
│  └────────┬─────────┘  └──────────────────┘  └─────────────────┘  │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                              │
│  │    ChromaDB      │                                              │
│  │  Vector Store    │                                              │
│  │                  │                                              │
│  │ - 5000+ Bios     │                                              │
│  │ - Embeddings     │                                              │
│  │ - Similarity     │                                              │
│  └──────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow Diagram

```
User Types Text
      │
      ▼
[Debounce: 1.5s]
      │
      ▼
Check Conditions:
- 5+ words?
- Not typing?
- No active feature lock?
      │
      ├─── No ──→ Wait
      │
      Yes
      │
      ▼
Frontend Sends Request
      │
      ▼
┌─────────────────────┐
│  Check Cache First  │
│  (5 min TTL)        │
└──────┬──────────────┘
       │
       ├─── Hit ──→ Return Cached Result
       │
      Miss
       │
       ▼
┌─────────────────────────────┐
│  Route to Backend Service   │
└──────┬──────────────────────┘
       │
       ├─── MLX Model (if enabled) ──→ Port 8003
       │
       ├─── API Server ──→ Port 8001
       │                      │
       │                      ▼
       │              ┌──────────────┐
       │              │ Vector Search│
       │              │   ChromaDB   │
       │              └──────┬───────┘
       │                     │
       │                     ▼
       │              ┌──────────────┐
       │              │ Context +    │
       │              │ Ollama Gen   │
       │              └──────┬───────┘
       │                     │
       └─── Ollama Direct ───┴──→ Combine Results
                                         │
                                         ▼
                                  Filter & Rank
                                         │
                                         ▼
                                  Return Top 3
                                         │
                                         ▼
                                  Update Cache
                                         │
                                         ▼
                                  Display in UI
```

## 🧩 Frontend Hook Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    useTextFeatureCoordinator                 │
│                  (Central Control System)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Features: AUTOCOMPLETE, SPELLCHECK, CAPITALIZATION │   │
│  │  Active Feature Tracking & Conflict Prevention       │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────────┘
                    │ Coordinates
    ┌───────────────┼───────────────┬─────────────────────┐
    ▼               ▼               ▼                     ▼
┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│  Form   │  │   useForm   │  │ useSpellCheck│  │ useKick      │
│Component│──│Autocomplete │  │ +Debounced   │  │ Detection    │
│         │  │             │  │              │  │              │
│ Main UI │  │ AI Suggest  │  │ Red Lines    │  │ Warning      │
└─────────┘  └─────────────┘  └──────────────┘  └──────────────┘
```

## 🚂 Training Pipeline Flow

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  bio.json    │────▶│ Data Preparation│────▶│ JSONL Files  │
│ (5000 bios)  │     │ (Split & Clean) │     │ train/val/   │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Base Model   │────▶│  MLX Training   │────▶│ LoRA Adapter │
│   Phi-3      │     │  (Fine-tuning)  │     │   Weights    │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                                              ┌──────────────┐
                                              │ MLX Server   │
                                              │ Deployment   │
                                              └──────────────┘
```

## 💾 Data Flow in Hybrid Mode

```
                     User Input: "I am a fun loving person who"
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   API Server        │
                            │   (Port 8001)       │
                            └──────────┬──────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
        ┌───────────────┐                          ┌────────────────┐
        │ Vector Search │                          │ Ollama Context │
        │               │                          │                │
        │ Find Similar: │                          │ Examples:      │
        │ - Bio 1       │────────────────────────▶│ - Bio 1        │
        │ - Bio 2       │                          │ - Bio 2        │
        │ - Bio 3...    │                          │ - Bio 3...     │
        └───────────────┘                          │                │
                │                                  │ + User prompt  │
                │                                  └────────┬───────┘
                │                                           │
                │                                           ▼
                │                                  ┌────────────────┐
                │                                  │ Generate with  │
                │                                  │ Gemma 3 12B    │
                │                                  └────────┬───────┘
                │                                           │
                └─────────────────┬─────────────────────────┘
                                  ▼
                          ┌───────────────┐
                          │ Combine &     │
                          │ Filter Results│
                          │               │
                          │ - Exact match │
                          │ - AI Gen 1    │
                          │ - AI Gen 2    │
                          └───────┬───────┘
                                  │
                                  ▼
                            Return Top 3
```

## 🔧 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Form.tsx Component                      │
│                                                              │
│  State Management:                                           │
│  - inputValue (current text)                                │
│  - currentSuggestions (AI completions)                      │
│  - misspelledWords (spell check results)                    │
│  - kickDetected (content filter)                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Textarea   │  │  Suggestion  │  │  SpellCheck     │  │
│  │   (Input)    │  │   Overlay    │  │    Overlay      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                  │                    │            │
│         └──────────────────┴────────────────────┘           │
│                            │                                 │
│                            ▼                                 │
│                     Event Handlers:                          │
│                     - onChange                               │
│                     - onKeyDown (Tab)                        │
│                     - onClick (spell)                        │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ Uses
                             ▼
         ┌───────────────────────────────────────┐
         │          Custom Hooks Layer           │
         │                                       │
         │  - useFormAutocomplete               │
         │  - useDebouncedSpellCheck            │
         │  - useKickDetection                  │
         │  - useTextFeatureCoordinator         │
         └───────────────────────────────────────┘
                             │
                             │ Calls
                             ▼
         ┌───────────────────────────────────────┐
         │       Server Actions Layer            │
         │                                       │
         │  - getAITextSuggestions()            │
         │  - getAITextStream()                 │
         │  - AI response caching               │
         └───────────────────────────────────────┘
```

## 🚀 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                     │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Nginx/       │  │   PM2 Process  │  │  Systemd     │  │
│  │   Reverse      │  │   Manager      │  │  Services    │  │
│  │   Proxy        │  │                │  │              │  │
│  │                │  │  - Next.js     │  │  - Ollama    │  │
│  │  :80 → :3000   │  │  - API Server  │  │  - MLX       │  │
│  │  /api → :8001  │  │  - MLX Server  │  │              │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Persistent Storage                     │ │
│  │                                                         │ │
│  │  - ChromaDB: /var/lib/chromadb/                       │ │
│  │  - Models: /opt/models/                               │ │
│  │  - Logs: /var/log/ai-bio/                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Performance Metrics Flow

```
User Action → Frontend Timer Start
                    │
                    ▼
            API Request Sent
                    │
                    ├─── Cache Hit ──→ <50ms
                    │
                   Miss
                    │
                    ▼
            ┌───────────────┐
            │ Vector Search │ ~100ms
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ LLM Generation│ ~200-500ms
            └───────┬───────┘
                    │
                    ▼
            Response Processing
                    │
                    ▼
            Frontend Timer End
                    │
                    ▼
         Total Time: 300-600ms
```

---

These diagrams provide a comprehensive view of the system architecture, data flows, and component interactions. They can be used to understand how the system works and to identify optimization opportunities.