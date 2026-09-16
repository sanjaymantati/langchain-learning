## Resource
1. Folder: ```structured_output```


### 1. Motivation & Core Concept
* **Unstructured Output Problem**: By default, LLMs return plain text responses, which are unstructured and cannot be passed directly into downstream systems like databases or APIs.
* **Structured Output**: Forcing an LLM to return responses in a well-defined data format (such as JSON) enables reliable programmatic integration.
* **3 Primary Use Cases**:
  1. **Data Extraction**: Extracting specific candidate fields (e.g. name, marks) from resumes into a database.
  2. **API Building**: Transforming raw text product reviews into structured JSON (summary, sentiment, pros/cons) for API endpoints.
  3. **AI Agents**: Extracting parameters (e.g. operation and operands) from prompt text to pass to agent tools like calculators.

---

### 2. Model Compatibility & `with_structured_output`
* **Model Differences**:
  * **Native Models**: Commercial models (e.g. OpenAI GPT models) are fine-tuned to natively support structured output.
  * **Non-Native Models**: Open-source models (e.g. TinyLlama) lack native structured output support and require Output Parsers.
* **`with_structured_output()`**: A built-in LangChain method called on compatible chat models to bind a target output schema before invocation.

---

### 3. The 3 Methods for Defining Schemas

#### A. TypedDict (`typing.TypedDict`)
* **Function**: Uses Python's native `TypedDict` to define expected key-value pairs and data types.
* **Features**: Provides type hints in code editors and supports `Annotated` for descriptions, `Optional` fields, and `Literal` choices.
* **Limitation**: Does **not** perform runtime data validation (e.g. passing a string for an integer field will not trigger an error).

#### B. Pydantic (`pydantic.BaseModel`)
* **Function**: Uses Pydantic `BaseModel` classes and `Field` objects to define schemas with strict type enforcement.
* **Features**:
  * Enforces runtime data type validation and raises errors for invalid inputs.
  * Performs automatic type coercion (e.g. converting a numeric string `"32"` to integer `32`).
  * Supports field constraints (`gt`, `lt`), default values, built-in types like `EmailStr`, and custom descriptions.
* **Best Use Case**: Primary go-to method for production Python applications requiring strict data validation.

#### C. JSON Schema
* **Function**: Defines output schemas using raw JSON objects specifying `title`, `type`, `properties`, and `required` fields.
* **Best Use Case**: Essential for cross-language compatibility (e.g. Python backend sharing schemas with a JavaScript frontend).

---

### 4. Comparison Summary

| Method | Primary Language | Runtime Validation? | Type Coercion? | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **TypedDict** | Python-native | No | No | Quick Python type-hinting without extra dependencies |
| **Pydantic** | Python | Yes | Yes | Production Python applications requiring strict type safety |
| **JSON Schema** | Universal | Dependent on consumer | No | Multi-language cross-platform projects |

---

### 5. Execution Modes
* The `with_structured_output()` method supports two underlying execution modes via its `method` parameter:
  * `json_mode`: Requests structured JSON output directly from the model.
  * `function_calling`: Uses model-level function/tool calling mechanisms to return structured data.
