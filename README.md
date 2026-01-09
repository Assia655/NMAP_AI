# NMAP_AI - Technical Report

## Team
- Assia Haimeur
- Kaoutar Boudribila
- Douae El Hannach
- Fatima Ezzahraa GAROUD
- Intissar Raissouni

## 1) Summary
This project generates Nmap commands from natural language queries.
The pipeline combines comprehension, complexity classification, generation, and validation.

## 2) Global diagram
The main pipeline follows this chain:
1) Comprehension Agent
2) Complexity Agent
3) Generation (Easy / Medium / Hard)
4) Validation (Kali API)
5) Retry / Escalation

![Global diagram](images/Schema.jpeg)

## 3) Agents and how they work

### 3.1 Comprehension Agent (spaCy embeddings)
File: `backend/Agents/Agent_comprehension/nmap_agent_embeddings.py`

Role:
- Filter out-of-context queries (non-Nmap)

How it works:
- Loads spaCy `en_core_web_sm`
- Computes a domain embedding from `nmap_domain.txt`
- Uses cosine similarity between the query and domain
- Detects out-of-context keywords (pdf, video, music, etc.)

Outputs:
- `is_relevant`, `confidence`, `keywords`, `reason`

![Comprehension example](images/Exemple.png)

### 3.2 Complexity Agent (SLM + Word2Vec)
File: `backend/Agents/Agent_complexity/complexity_slm_word2vec.py`

Role:
- Classify the query into `easy`, `medium`, `hard`

How it works:
- Unigram language model (SLM) + Word2Vec
- Combines the scores (50% SLM, 50% W2V)
- Returns the class with the highest probability

Outputs:
- `level`, `confidence`, `probabilities`, `recommended_model`

### 3.3 Easy Agent (RAG + Neo4j)
File: `backend/Agents/Agent_easy/rag_engine.py`

Role:
- Generate reliable commands for simple cases

How it works:
- Neo4j Knowledge Graph: Scans, Options, Categories, Examples, Validation
- IntentClassifier: detects intent (host discovery, port scan, vuln, etc.)
- Extracts target (IP/hostname) and ports
- CommandGenerator: selects a scan and compatible options
- CommandValidator: checks conflicts and prerequisites

Strengths:
- Explainable
- Robust for simple queries

### 3.4 Medium Agent (T5 + LoRA)
File: `backend/APIs/api_medium.py`

Role:
- Generate more flexible text while keeping valid syntax

How it works:
- T5 fine-tuned with LoRA (PEFT)
- Autoregressive token-by-token generation
- Corrective rules (ex: ping scan -> -sn)
- CPU-only execution (stable)

![Medium example](images/medium.png)

### 3.5 Hard Agent (Diffusion token-level)
Files:
- `backend/Agents/Agent_hard/train_model.py`
- `backend/Agents/Agent_hard/token_level_tokenizer.py`
- `backend/Agents/Agent_hard/quick_test.py`

Role:
- Explore diffusion in embedding space to generate commands

Tokenization:
- Flags: `-sS`, `-sU`, `--script`, `-p`
- Placeholders: `<IP>`, `<PORT_RANGE>`, `<TARGET>`
- Typed unknowns: `<UNK_FLAG>`, `<UNK_ARG>`, `<UNK_TARGET>`

Model:
- Transformer encoder/decoder
- d_model=384, 6 layers, 8 heads
- Diffusion schedule: 200 timesteps (linear betas)

Loss:
- MSE on embeddings
- CrossEntropy on tokens
- Total loss = MSE + 0.5 * CE

Generation:
- Start from noise -> iterative denoising
- Project to vocab via embedding similarity
- Top-k sampling
- Post-processing: dedup + minimal command (nmap + one -p)

State:
- Not integrated in `backend/api.py`
- `backend/Agents/Agent_hard/__init__.py` references `agent_diffusion.py` (missing file)

## 4) Why diffusion does not work (and will not here)

Structural issue:
- Gaussian diffusion assumes a smooth continuous space.
- Nmap text is discrete, symbolic, and strongly constrained.

Observed effects:
- Incoherent tokens (incompatible flags, wrong options)
- Repetitions
- Useless fragments (ex: "open", "Ports:")

Root cause:
- The model optimizes distance in embedding space, not syntax validity.
- Generation is non-autoregressive: no explicit token dependencies.
- Post-processing fixes surface form but not global structure.

Conclusion:
In this setup (Gaussian diffusion on embeddings + non-autoregressive generation),
the model cannot reliably output correct Nmap commands. Even with more data,
it remains unstable because the continuous representation does not encode
strict discrete grammar.

![Diffusion error](images/Erreur%20diffusion.png)

## 5) Validation
File: `backend/api.py`

- Validation via an API on a Kali VM
- Verdicts: VALID / INVALID / REPAIRABLE / UNSAFE
- Automatic retry up to 5 attempts

![Validation Kali](images/Validation%20Kali.png)

## 6) Interface
Folder: `Interface/`

- UI for entering a query
- Displays generated command and validation

![Interface](images/interface.png)

## 7) Technologies used (outside diffusion)
- FastAPI (unified API)
- spaCy embeddings (comprehension)
- Word2Vec (complexity)
- Neo4j (RAG)
- Transformers + PEFT LoRA (T5)
- External validation API (Kali)
