## Resource
1. Folder: ```prompts```

### 1. Overview of the Model Component
* **Common Interface**: The Model component in LangChain acts as a standardized interface to connect and interact with diverse AI models without changing overall application code.
* **Two Main Categories**: LangChain categorizes models into **Language Models** and **Embedding Models**.

---

### 2. Language Models: LLMs vs. Chat Models

Language models process text inputs and return generated text.

#### A. LLMs (Legacy General-Purpose)
* **Function**: Accepts a single plain text string and returns a plain text string.
* **Use Cases**: General NLP tasks such as free-form text generation, summarization, and translation.
* **Current Status**: Lacks conversation history memory and role awareness; support for traditional LLM classes is being phased out in newer LangChain versions in favor of Chat Models.

#### B. Chat Models (Modern Conversational Standard)
* **Function**: Takes a structured list or sequence of messages as input and returns a chat message object containing content and metadata.
* **Features**:
  * Specialized for multi-turn conversations, chatbots, and AI agents.
  * Role-aware (`SystemMessage`, `HumanMessage`, `AIMessage`) and supports conversation history tracking.

---

### 3. Closed-Source vs. Open-Source Models

#### A. Closed-Source (Proprietary APIs)
* **Providers**: Commercial APIs such as OpenAI GPT models, Anthropic Claude, and Google Gemini.
* **Characteristics**: Hosted on provider infrastructure, priced on a per-token usage basis, and accessed via API keys.

#### B. Open-Source Models
* **Examples**: Llama, Mistral, Falcon, and TinyLlama.
* **Access Methods**: Hosted on repositories like Hugging Face, accessible either via the **Hugging Face Inference API** or downloaded locally.
* **Trade-offs**: Local execution offers complete data privacy, customization/fine-tuning, and zero API token costs; however, it requires heavy local computing resources (GPUs/RAM) and may run slower on standard hardware.

---

### 4. Key Model Parameters
* **Temperature**: Controls determinism versus creative randomness.
  * Low values (`0.0 - 0.3`): Deterministic and predictable responses for factual/code tasks.
  * High values (`1.5+`): Diverse and creative outputs for storytelling or brainstorming.
* **Max Completion Tokens**: Restricts the maximum length of generated output tokens to control API costs and response size.

---

### 5. Embedding Models & Semantic Search
* **Function**: Takes text input and converts it into a numerical vector representation (embeddings) that captures contextual meaning.
* **Core Methods**:
  * `embed_query()`: Generates a vector for a single query string.
  * `embed_documents()`: Generates vectors for a list/batch of document strings.
* **Primary Use Case**: Enables semantic search, document similarity scoring using metrics like cosine similarity, and Retrieval-Augmented Generation (RAG) applications.

---

### 6. Comparison Summary

| Model Type | Input Format | Output Format | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **LLMs** | Single Text String | Single Text String | Traditional text generation & completion |
| **Chat Models** | Sequence of Message Objects | Chat Message Object | Conversational AI, chatbots, and agents |
| **Embedding Models** | Text String or List of Texts | Numerical Vector(s) | Semantic search, vector similarity, & RAG |
