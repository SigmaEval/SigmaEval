# Proposed API Design for SigmaEval

This document outlines a proposed API design that allows for comprehensive, mixed-modal testing of AI applications. The goal is to enable the evaluation of both subjective, LLM-judged behaviors and objective, quantitative metrics within a single, efficient test run, using a consistent and intuitive API.

## Core Concepts

The proposed design unifies all evaluations under a single, declarative pattern. It introduces two types of expectations that can be evaluated on the same set of simulated conversations:

1.  **`BehavioralExpectation`**: For subjective tests that require a Judge LLM to score the application's response against a qualitative rubric (e.g., helpfulness, politeness, factual accuracy).
2.  **`MetricExpectation`**: For objective tests that measure quantifiable aspects of the interaction, such as response latency, turn count, or message length.

Both expectation types use a **`criteria`** parameter to define the condition for a successful test, leveraging a developer's existing familiarity with testing frameworks.

## Example: The Unified Assertion-Based API

The following example demonstrates how this API can be used to test an AI bot for correctness, tone, latency, and conciseness simultaneously. The entire test is defined using a consistent structure, making it highly readable and easy to understand.

```python
from sigmaeval import (
    ScenarioTest,
    BehavioralExpectation,
    MetricExpectation,
    metrics,
    assertions # The single source for all statistical tests
)

# The final, unified API design
comprehensive_test = ScenarioTest(
    title="Bot provides fast, helpful, and polite support",
    given="A user needs to reset their password.",
    when="The user asks for help with the password reset process.",
    sample_size=100,
    then=[
        # --- Behavioral Expectations (use assertions.scores for LLM scores) ---
        BehavioralExpectation(
            expected_behavior="The bot provides accurate, step-by-step instructions for resetting a password.",
            # The context `assertions.scores` makes the operation unambiguous.
            criteria=assertions.scores.proportion_gte(
                min_score=6, proportion=0.95
            )
        ),
        BehavioralExpectation(
            expected_behavior="The bot maintains a patient and helpful tone throughout the interaction.",
            # Asserts that the median of the 1-10 scores is above a threshold.
            criteria=assertions.scores.median_gte(
                threshold=8.0
            )
        ),

        # --- Metric Expectations (use assertions.metrics for calculated data) ---
        MetricExpectation(
            metric=metrics.per_turn.response_latency,
            # `assertions.metrics` clearly separates this from score-based tests.
            criteria=assertions.metrics.proportion_lt(
                threshold=1.5, proportion=0.99
            )
        ),
        MetricExpectation(
            metric=metrics.per_conversation.turn_count,
            # Asserts that the average number of turns is less than 2.
            criteria=assertions.metrics.mean_lt(
                threshold=2.0
            )
        )
    ]
)
```

### A Note on `sample_size` and the Data Collection Lifecycle

The `sample_size` parameter is intentionally defined once at the top level of a `ScenarioTest`. This design choice formalizes a core principle of the framework: **collect data once, analyze it in multiple ways.**

When a test is executed, SigmaEval first performs the data collection phase, running `sample_size` number of simulated conversations. This single, rich dataset is then passed to every `BehavioralExpectation` and `MetricExpectation` within the `then` block for analysis.

This approach has two key benefits:

*   **Efficiency**: It is highly efficient, as the potentially time-consuming and expensive process of interacting with LLMs to generate conversations only happens once per test run.
*   **Consistency**: All assertions are evaluated against the exact same set of user interactions, ensuring that the results are consistent and comparable.

For this reason, `sample_size` is a property of the entire scenario and cannot be overridden on a per-expectation basis.

### Benefits of this Approach

*   **Unified and Consistent**: The API is perfectly symmetrical. Whether testing a subjective behavior or an objective metric, the pattern is the same: specify the expectation and define its `criteria`.
*   **Intuitive and Familiar**: The use of `assertions` leverages a developer's existing mental model of how tests work, drastically reducing the learning curve.
*   **Conceptual Simplicity**: The framework is simplified to its essence: collect data, then apply statistical assertions. This provides a clear and robust mental model.
*   **Highly Extensible**: The `metrics` and `assertions` modules can be easily expanded with custom, user-defined functions to support any evaluation scenario.
*   **Improved Discoverability**: In an IDE, a developer can type `assertions.` and be prompted with `scores` or `values`, immediately guiding them to the correct set of assertions for their context. The same applies to the namespaced `metrics.per_turn` and `metrics.per_conversation`.

### A Note on the `assertions` Naming Convention

To ensure maximum clarity, consistency, and discoverability, the `assertions` module is organized into two namespaces:

*   **`assertions.scores`**: This namespace contains all assertion functions that operate on the 1-10 subjective `scores` generated by a `BehavioralExpectation`.
*   **`assertions.metrics`**: This namespace contains all assertion functions that operate on the numerical `values` generated by a `MetricExpectation`.

The functions within these namespaces follow a clear and consistent naming scheme:

**`[aggregation] _ [comparison]`**

*   **`aggregation`**: The statistical method used (e.g., `proportion`, `median`, `mean`). To promote statistical best practices and avoid ambiguity, assertions are explicit about the aggregation method. The median is often preferred for its robustness to outliers, a common characteristic of LLM performance data.
*   **`comparison`**: The mathematical comparison being made (e.g., `gte` for "greater than or equal to," `lt` for "less than").

This namespaced convention makes every assertion explicit and readable, eliminating hidden logic and making the test suite a clear source of truth for the project's quality standards.

To further improve clarity, assertion parameters are named to be self-documenting. Instead of a generic `value`, parameters are descriptive (e.g., `min_score` for proportion-based assertions, `threshold` for mean/median assertions). This ensures that the purpose of each argument is immediately obvious.

### Endorsed Naming Conventions

To ensure the framework is as clear and self-documenting as possible, the following naming conventions are formally endorsed:

*   **`ScenarioTest`**: This class name is preferred over the previous `BehavioralTest`, as it more accurately reflects its expanded role in handling a mix of behavioral and metric-based expectations.
*   **Assertion Naming (`assertions.scores.[aggregation]_[comparison]`)**: This namespaced approach strikes the perfect balance between clarity and conciseness. It eliminates ambiguity and improves IDE discoverability, ensuring that the test suite serves as a transparent and reliable specification of the application's quality bar.
*   **Metric Naming (`metrics.[scope].[name]`)**: To make the scope of data collection explicit, all built-in metrics are namespaced under either `metrics.per_turn` or `metrics.per_conversation`. This makes the granularity of the collected data immediately obvious from the test definition itself.

### Handling `significance_level`

To reduce boilerplate and enforce a consistent standard of statistical rigor, the `significance_level` is configured globally on the `SigmaEval` object when it is initialized.

```python
# The default significance level for all tests is set once
sigma_eval = SigmaEval(
    judge_model="openai/gpt-5-nano",
    significance_level=0.05
)
```

This default will be used for all assertions. However, for specific critical tests that require a different level of confidence, the `significance_level` can be optionally overridden directly in the `assertion` call.

```python
# This assertion uses the default level from the SigmaEval object
criteria=assertions.metrics.mean_lt(threshold=4.0)

# This assertion overrides the default for a more stringent check
criteria=assertions.scores.proportion_gte(
    min_score=10,
    proportion=0.99,
    significance_level=0.01
)
```

This approach provides the best of both worlds: clean, simple tests for the common case, and the flexibility to handle exceptions when needed.

## Appendix: Common Metrics and Data Collection Lifecycle

To provide clarity on how `MetricExpectation` works, it is essential to understand the data collection lifecycle. Metrics can be collected at two different granularities, which affects how criteria are evaluated.

*   **Per-Turn Metrics**: These metrics are collected for *each response* from the application within a single simulated conversation. If a conversation has 5 turns, a per-turn metric will produce 5 data points for that single run. If `sample_size=100`, you could have `5 * 100 = 500` total data points for the statistical assertion to analyze. This is ideal for metrics where you need to ensure consistency in every interaction, such as latency or response length.
*   **Per-Conversation Metrics**: These metrics are collected once for the *entire* simulated conversation, producing a single data point per run. If `sample_size=100`, you will have exactly 100 data points to analyze. This is suitable for evaluating the overall efficiency or outcome of an interaction, such as the total number of turns to resolve an issue.

Here are the top 10 most common and useful metrics for evaluating AI applications:

### 1. Response Latency
*   **Description**: Measures the time (in seconds) between the user's message and the application's response.
*   **Scope**: Per-Turn
*   **Use Case**: Ensuring the application feels responsive and meets performance requirements (e.g., "99% of responses should be under 1.5 seconds").

### 2. Turn Count
*   **Description**: The total number of exchanges (user message + app response) in a conversation.
*   **Scope**: Per-Conversation
*   **Use Case**: Measuring the efficiency of the AI. A lower turn count to resolve an issue is often better (e.g., "The average conversation should be less than 4 turns").

### 3. Response Length (Tokens/Characters)
*   **Description**: The number of tokens or characters in the application's response.
*   **Scope**: Per-Turn
*   **Use Case**: Ensuring responses are concise or detailed as required (e.g., "The median response length should be under 200 characters").

### 4. Total Conversation Time
*   **Description**: The total time elapsed from the first user message to the final application response.
*   **Scope**: Per-Conversation
*   **Use Case**: Evaluating the overall time investment required from a user to complete their goal.

### Advanced Usage: Multiple Assertions

For more comprehensive validation, the `criteria` parameter can also accept a list of assertion objects. This allows you to check multiple statistical properties of the same data—for example, checking both the mean and the 99th percentile of response latency—without running the simulation again.

```python
MetricExpectation(
    metric=metrics.per_turn.response_latency,
    # The `criteria` parameter accepts a list for multiple checks.
    criteria=[
        assertions.metrics.mean_lt(threshold=1.0),
        assertions.metrics.proportion_lt(threshold=2.5, proportion=0.99)
    ]
)
```

This design requires a corresponding adjustment to the result object structure to ensure that all outcomes are reported clearly.

## Recommended Result Object Structure

To ensure the best possible developer experience, the `evaluate` method should return a rich, composite object that makes simple checks easy while enabling complex analysis. This is achieved by defining explicit `dataclass` objects for the results, providing type safety, IDE autocompletion, and a clean, readable interface.

### Proposed `dataclass` Structure

To provide strong types for developers programmatically inspecting test failures, the raw conversation data is also exposed via concrete `dataclass` objects. The result objects are structured to handle multiple assertions per expectation.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Turn:
    """Represents a single turn in a conversation."""
    user_message: str
    app_response: str
    latency: float
    # Other per-turn metric data can be stored here
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    """Represents a full conversation from a single simulation run."""
    turns: List[Turn]
    # Other per-conversation metric data can be stored here
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AssertionResult:
    """The result of a single assertion check."""
    about: str
    passed: bool
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        p_value_str = f", p-value: {self.p_value:.4f}" if self.p_value is not None else ""
        return f"[{status}] {self.about}{p_value_str}"

@dataclass
class ExpectationResult:
    """The result of a single Expectation, which may contain multiple assertions."""
    about: str
    assertion_results: List[AssertionResult]

    @property
    def passed(self) -> bool:
        """True only if ALL assertions for this expectation passed."""
        return all(r.passed for r in self.assertion_results)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        # The 'about' for the Expectation gives high-level context
        if len(self.assertion_results) == 1:
            # If there's only one assertion, condense the output to a single line
            res = self.assertion_results[0]
            res_status = "✅ PASSED" if res.passed else "❌ FAILED"
            p_value_str = f", p-value: {res.p_value:.4f}" if res.p_value is not None else ""
            return f"[{res_status}] {self.about}{p_value_str}"
        else:
            # For multiple assertions, use a detailed breakdown
            title_line = f"Expectation: '{self.about}' -> {status}"
            # Each assertion result is then listed
            results_breakdown = []
            for res in self.assertion_results:
                assertion_status = "✅" if res.passed else "❌"
                p_value_str = f", p-value: {res.p_value:.4f}" if res.p_value is not None else ""
                results_breakdown.append(f"    - [{assertion_status}] {res.about}{p_value_str}")
            
            breakdown_str = "\n".join(results_breakdown)
            return f"{title_line}\n{breakdown_str}"

@dataclass
class ScenarioTestResult:
    """The comprehensive result of a single ScenarioTest run."""
    title: str
    expectation_results: List[ExpectationResult]
    conversations: List[Conversation] # Use the strong Conversation type

    @property
    def passed(self) -> bool:
        """True only if ALL expectations passed."""
        return all(r.passed for r in self.expectation_results)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        title_line = f"--- Result for Scenario: '{self.title}' ---"
        status_line = f"Overall Status: {status}"
        passed_count = sum(1 for r in self.expectation_results if r.passed)
        total_count = len(self.expectation_results)
        summary_line = f"Summary: {passed_count}/{total_count} expectations passed."
        results_breakdown = "\n\n".join(f"  - {r}" for r in self.expectation_results)
        return f"{title_line}\n{status_line}\n{summary_line}\n\nBreakdown:\n{results_breakdown}"
```

### Benefits for Developer Experience

*   **Simple Things Simple**: The most common use case—checking if a test passed—is trivial: `if results.passed: ...`. Additionally, `print(results)` produces a clean, human-readable summary perfect for CI logs. The new structure provides an even clearer, two-level breakdown for readability.

    ```
    --- Result for Scenario: 'Bot provides fast, helpful, and polite support' ---
    Overall Status: ❌ FAILED
    Summary: 2/3 expectations passed.

    Breakdown:
      - [✅ PASSED] Correctness: The bot provides accurate, step-by-step instructions...

      - [✅ PASSED] Tone: The bot maintains a patient and helpful tone

      - Expectation: 'Performance' -> ❌ FAILED
          - [✅] mean latency is under 1.5s, p-value: 0.0200
          - [❌] P99 latency is under 2.5s, p-value: 0.1500
    ```

*   **Complex Things Possible**: For failures, developers have a structured, type-safe object to analyze. This enables programmatic deep dives into `p_value` or the `details` dictionary, and provides full access to the raw `conversations` that led to the result. By using concrete types like `List[Conversation]` instead of `List[Any]`, developers get full type-checking and IDE autocompletion when inspecting the data, making it much easier to debug failed tests.


## Implementation Checklist

Here is a step-by-step checklist to guide the implementation of the new API design.

### 1. Update Core Data Models (`models.py`)

- [ ] Rename `BehavioralTest` to `ScenarioTest`.
- [ ] Add `sample_size: int` to the `ScenarioTest` class.
- [ ] Create a new `BehavioralExpectation` class with `expected_behavior: str` and `criteria: Any`.
- [ ] Create a new `MetricExpectation` class with `metric: Any` and `criteria: Any`.
- [ ] Update the `then` parameter in `ScenarioTest` to be `then: List[Union[BehavioralExpectation, MetricExpectation]]`.
- [ ] Implement the new result object structure:
    - [ ] Create `Turn` dataclass.
    - [ ] Create `Conversation` dataclass.
    - [ ] Create `AssertionResult` dataclass.
    - [ ] Create `ExpectationResult` dataclass.
    - [ ] Rename `EvaluationResult` to `ScenarioTestResult` and update its structure.

### 2. Create `assertions` and `metrics` Modules

- [ ] Create a new file `sigmaeval/core/assertions.py`.
    - [ ] Implement the `scores` namespace with functions like `proportion_gte` and `median_gte`.
    - [ ] Implement the `metrics` namespace with functions like `proportion_lt` and `mean_lt`.
    - [ ] Ensure all assertion functions can optionally accept a `significance_level` to override the global default.
- [ ] Create a new file `sigmaeval/core/metrics.py`.
    - [ ] Implement the `per_turn` namespace with metrics like `response_latency`.
    - [ ] Implement the `per_conversation` namespace with metrics like `turn_count`.
    - [ ] Design a base `Metric` class to ensure a consistent interface for data collection.

### 3. Refactor the Main `SigmaEval` Class (`framework.py`)

- [ ] Update the `__init__` method to accept a global `significance_level`.
- [ ] Update the `evaluate` method to accept a `ScenarioTest` object (or a list of them).
- [ ] Refactor the `_evaluate_single` method to orchestrate the new evaluation flow:
    - [ ] **1. Collect Data**: Run `sample_size` simulations to generate a dataset of `Conversation` objects.
    - [ ] **2. Calculate Metrics**: Process the raw conversation data to calculate all metrics defined in the test's `MetricExpectation`s. A `MetricCollector` class could be useful here.
    - [ ] **3. Judge Behaviors**: For each `BehavioralExpectation`, generate a rubric and use a Judge LLM to score each of the collected conversations, producing a set of scores.
    - [ ] **4. Run Assertions**: For each `Expectation`, apply its `criteria` (the assertion function) to the corresponding dataset (either metric values or judge scores) to get a pass/fail result.
    - [ ] **5. Aggregate Results**: Combine the outcomes of all assertions into the new `ScenarioTestResult` object.

### 4. Update Evaluators (`evaluators.py`)

- [ ] The existing `SuccessRateEvaluator`, `RatingAverageEvaluator`, and `RatingProportionEvaluator` will be replaced by the functions in the new `assertions.py` module.
- [ ] Move the statistical logic from the evaluator classes into the new assertion functions.
- [ ] Delete the `evaluators.py` file after refactoring is complete.

### 5. Update Data Collection (`data_collection.py`)

- [ ] Modify `collect_evaluation_data` to collect and return the raw `Conversation` objects, including per-turn data like `latency`. This will likely involve modifying the `AppResponse` to include this information or wrapping the `app_handler`.
- [ ] Ensure that the data collection process is decoupled from the judging process, as judging will now be driven by `BehavioralExpectation`s after the conversations are collected.

### 6. Update Public API (`__init__.py`)

- [ ] Update the main `__init__.py` file to export the new classes: `ScenarioTest`, `BehavioralExpectation`, `MetricExpectation`, and the `assertions` and `metrics` modules.
- [ ] Remove the old classes (`BehavioralTest`, `Expectation`, evaluators) from the public API.

### 7. Documentation and Examples

- [ ] Update the `README.md` to reflect the new API, using the example from this document as the primary guide.
- [ ] Create new examples in the `tests/example_apps/` directory to showcase the new API's capabilities, including mixed behavioral and metric tests.
- [ ] Update all docstrings to reflect the new class and parameter names.