## Resource
1. Folder: ```chains```


### 1. Concept and Purpose of Chains
* **Definition**: Chains are components in LangChain used to build automated processing pipelines.
* **The Manual Execution Problem**: Building LLM applications manually requires calling `.invoke()` on individual components, manually extracting text contents, and manually passing outputs to the next stage.
* **Pipeline Automation**: Chains link individual steps so that the output of one step automatically becomes the input for the next step, requiring only a single input to trigger the entire sequence.
* **LangChain Expression Language (LCEL)**: Uses the pipe operator `|` to declaratively connect components into a unified chain.

---

### 2. Sequential Chains
* **Simple Sequential Chain**: Connects a `PromptTemplate`, a language model (`ChatOpenAI`), and an output parser (`StrOutputParser`) sequentially.
* **Multi-LLM Sequential Chain**: Connects multiple LLM calls in series.
  * *Example*: Topic $\rightarrow$ Prompt 1 $\rightarrow$ Model $\rightarrow$ Parser (Detailed Report) $\rightarrow$ Prompt 2 $\rightarrow$ Model $\rightarrow$ Parser (5-Pointer Summary).
* **Chain Visualization**: The execution graph of any chain can be rendered in ASCII format using `chain.get_graph().print_ascii()`.

---

### 3. Parallel Chains (`RunnableParallel`)
* **Function**: Executes multiple independent chains simultaneously on the same input using `RunnableParallel`.
* **Use Case**: Generating lecture notes and a quiz concurrently from a single document.
* **Merging Parallel Outputs**: Outputs from parallel branches are returned as a dictionary (e.g., `{'notes': notes_chain, 'quiz': quiz_chain}`), which can then be piped into a downstream merge chain.

---

### 4. Conditional Chains (`RunnableBranch`)
* **Function**: Implements branching logic (`if-elif-else`) to execute specific chains based on conditions evaluated on the input.
* **Structuring Classifier Output**: Uses `PydanticOutputParser` to ensure the classification step returns strict, predictable values (e.g., `Literal['positive', 'negative']`) rather than arbitrary text.
* **Branch Syntax**: `RunnableBranch` accepts tuples of `(condition_lambda, chain)` and ends with a default fallback chain.
* **`RunnableLambda`**: Converts standard Python lambda functions into Runnables to serve as valid chains or fallbacks.

---

### 5. Summary Comparison Table

| Chain Type | Core Runnable / Syntax | Primary Purpose | Key Advantage |
| :--- | :--- | :--- | :--- |
| **Sequential Chain** | Pipe Operator `\|` | Execute steps in linear series | Eliminates manual output-to-input mapping |
| **Parallel Chain** | `RunnableParallel` | Run multiple chains concurrently | Speeds up execution for independent tasks |
| **Conditional Chain** | `RunnableBranch` | Route execution based on conditions | Enables dynamic control flow and decision trees |
