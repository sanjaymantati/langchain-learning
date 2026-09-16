## Resource
1. Folder: ```runnable```

### 1. Historical Context & Motivation
* **Evolution of LangChain**: Following the release of ChatGPT and OpenAI APIs, LangChain was created to simplify building LLM applications by offering reusable components (e.g., Document Loaders, Text Splitters, Vector Stores, Retrievers, Output Parsers).
* **Creation of Chains**: To avoid writing manual glue code between components, LangChain introduced Chains (such as `LLMChain` and `RetrievalQAChain`) to automate pipeline execution.

---

### 2. The "Too Many Chains" Problem
* **Codebase Bloat**: Over time, LangChain created specialized, bespoke chains for every specific use case (e.g., `SQLChain`, `APIChain`, `LLMMathChain`).
* **Steep Learning Curve**: Having dozens of rigid chain classes made the framework heavy to maintain and difficult for developers to learn.
* **Root Cause — Non-Standardized Components**: Individual components used completely different interaction methods—`predict()` for LLMs, `format()` for PromptTemplates, `get_relevant_documents()` for Retrievers, and `parse()` for Parsers. Because they lacked a uniform interface, custom wrapper code was required to connect any two components.

---

### 3. What is a Runnable?
* **Core Definition**: A Runnable is a standardized, modular unit of work in LangChain that accepts an input, processes it, and returns an output.
* **Lego Block Analogy**: Like Lego blocks, every Runnable performs a specific job, features uniform connecting interfaces, can be linked to other Runnables, and when connected, the resulting pipeline is itself a Runnable.

---

### 4. Standard Interface & Key Methods
* **Uniform API**: All Runnable components inherit from the abstract `Runnable` class, ensuring every component implements identical method signatures.
* **Essential Runnable Methods**:
  * `invoke()`: Passes a single input through the Runnable to produce a single output.
  * `batch()`: Processes multiple inputs in parallel.
  * `stream()`: Streams the output back incrementally.
* **Automatic Output-to-Input Flow**: Piping Runnables together automatically feeds the output of step $N$ as the direct input to step $N+1$ without manual parameter mapping.

---

### 5. Summary Comparison: Traditional Chains vs. Runnable Architecture

| Metric / Aspect | Traditional Chains (Legacy) | Runnable / LCEL Architecture |
| :--- | :--- | :--- |
| **Interface** | Unstandardized (`predict()`, `format()`, `parse()`) | Standardized (`invoke()`, `batch()`, `stream()`) |
| **Composition** | Custom wrapper functions required per task | Modular pipe operator (`\|`) / `RunnableSequence` |
| **Maintainability** | Rigid, fragmented codebase with too many chains | Uniform, Lego-like composability |
| **Flexibility** | Difficult to nest or extend arbitrary steps | Seamlessly connect components and nested chains |
