<div align="center">

# SigmaEval

The Gen AI App Evaluation Framework

[![PyPI version](https://badge.fury.io/py/sigmaeval-framework.svg)](https://badge.fury.io/py/sigmaeval-framework)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python versions](https://img.shields.io/pypi/pyversions/sigmaeval-framework.svg)](https://pypi.org/project/sigmaeval-framework/)

</div>

Tired of shipping Gen AI features based on gut feelings and vibes?

**SigmaEval** is a Python framework for the **statistical** evaluation of Gen AI apps, agents, and bots that helps you move from "it seems to work" to making **statistically** rigorous statements about your AI's quality. It allows you to set and enforce objective quality bars by making statements like:

> _"We are confident that at least 90% of user issues coming into our customer support chatbot will be resolved with a quality score of 8/10 or higher."_

> _"With a high degree of confidence, the median response time of our new AI-proposal generator will be lower than our 5-second SLO."_

> _"For our internal HR bot, we confirmed that it will likely succeed in answering benefits-related questions in fewer than 4 turns in a typical conversation."_

Testing Gen AI apps is challenging due to their non-deterministic outputs and the infinite space of possible user inputs. SigmaEval addresses this by replacing simple pass/fail checks with statistical evaluation. It uses an AI User Simulator to test your app against a wide variety of inputs, and then applies statistical methods to quantify your AI's performance with confidence. This is like a clinical drug trial: the goal isn't to guarantee a specific outcome for every individual but to ensure the treatment is effective for a significant portion of the population (within a certain risk tolerance).

This process transforms subjective assessments into quantitative, data-driven conclusions, giving you a reliable framework for building high-quality AI apps.

At its core, SigmaEval uses two AI agents to automate evaluation: an **AI User Simulator** that realistically tests your application, and an **AI Judge** that scores its performance. The process is as follows:

1.  **Define "Good"**: You start by defining a test scenario in plain language, including the user's goal and a clear description of the successful outcome you expect. This becomes your objective quality bar.

2.  **Simulate and Collect Data**: The **AI User Simulator** acts as a test user, interacting with your application based on your scenario. It runs these interactions many times to collect a robust dataset of conversations.

3.  **Judge and Analyze**: The **AI Judge** scores each conversation against your definition of success. SigmaEval then applies statistical methods to these scores to determine if your quality bar has been met with a specified level of confidence.

<div align="center">
    <img src="docs/images/sigmaeval-architecture.jpg" alt="SigmaEval Architecture Diagram" width="600">
</div>

## Installation

```bash
pip install sigmaeval-framework
```

Or install from source:

```bash
git clone https://github.com/Itura-AI/sigmaeval.git
cd sigmaeval
pip install -e .
```

## Hello World

Here is a minimal, complete example of how to use SigmaEval (run `pip install sigmaeval-framework` first):

```python
from sigmaeval import SigmaEval, ScenarioTest, assertions
import asyncio
from typing import List, Dict, Any

# 1. Define the ScenarioTest to describe the desired behavior
scenario = (
    ScenarioTest("Hello World Test")
    .given("A user says hello")
    .when("The user says 'Hello'")
    .expect_behavior(
        "The bot replies with 'Hello, world!'",
        # We are confident that at least 90% of responses will score an 8/10 or higher.
        criteria=assertions.scores.proportion_gte(min_score=8, proportion=0.90)
    )
)

# 2. Implement the app_handler to allow SigmaEval to communicate with your app
async def app_handler(messages: List[Dict[str, str]], state: Any) -> str:
    # In a real test, you will pass the messages to your app and return the response back to SigmaEval.
    # For this example, we are just returning a static response.
    return "Hello, world!"

# 3. Initialize SigmaEval and run the evaluation
async def main():
    # Make sure your environment is configured with an API key for your chosen model (e.g., OPENAI_API_KEY)
    # You can use any model that LiteLLM supports: https://docs.litellm.ai/docs/providers
    sigma_eval = SigmaEval(
        judge_model="openai/gpt-5-nano",
        sample_size=30,  # The number of times to run the test
        significance_level=0.05  # Corresponds to a 95% confidence level
    )
    result = await sigma_eval.evaluate(scenario, app_handler)
    assert result.passed

if __name__ == "__main__":
    asyncio.run(main())
```

When you run this script, SigmaEval will:

1.  **Generate a Rubric**: Based on the `expect_behavior`, it will create a 1-10 scoring rubric for the Judge LLM.
2.  **Simulate Conversations**: It will call your `app_handler` 30 times (`sample_size=30`), each time simulating a user saying "Hello".
3.  **Judge the Responses**: For each of the 30 conversations, the `judge_model` will score your app's response against the rubric.
4.  **Perform Statistical Analysis**: SigmaEval will then run a hypothesis test to determine if it can be concluded, with 95% confidence (`significance_level=0.05`), that at least 90% of the responses scored an 8 or higher.
5.  **Determine Pass/Fail**: The script will exit with a pass or fail status based on the final assertion.

## Table of Contents

- [Installation](#installation)
- [Hello World](#hello-world)
- [Core Concepts](#core-concepts)
- [Supported LLMs](#supported-llms)
- [API Reference](#api-reference)
  - [Assertions](#assertions)
  - [Metrics](#metrics)
- [Guides](#guides)
  - [Managing Cost](#managing-cost)
  - [User Simulation Writing Styles](#user-simulation-writing-styles)
  - [Logging](#logging)
  - [Retry Configuration](#retry-configuration)
  - [Evaluating a Test Suite](#evaluating-a-test-suite)
  - [Evaluating Multiple Conditions and Assertions](#evaluating-multiple-conditions-and-assertions)
  - [Accessing Evaluation Results](#accessing-evaluation-results)
  - [Compatibility with Testing Libraries](#compatibility-with-testing-libraries)
- [Appendix](#appendix)
  - [Sample Size and Statistical Significance](#sample-size-and-statistical-significance)
  - [Statistical Methods](#statistical-methods)
  - [An Example Rubric](#an-example-rubric)
- [Development](#development)
- [License](#license)
- [Contributing](#contributing)

## Core Concepts

SigmaEval combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge evaluation. This powerful combination allows you to move beyond simple pass/fail tests and gain statistical confidence in your AI's performance.

Each scenario is defined using a `ScenarioTest` object with a fluent builder API. The test has three main parts that follow the familiar Given-When-Then pattern:

- **`.given()`**: This method establishes the prerequisite state and context for the **User Simulator LLM**. This can include the persona of the user (e.g., a new user, an expert user), the context of the conversation (e.g., a customer's order number), or any other background information.
- **`.when()`**: This method describes the specific goal or action the **User Simulator LLM** will try to achieve. SigmaEval uses this to guide the simulation.
- **`.expect_behavior()` / `.expect_metric()`**: These methods (the "Then" part of the pattern) specify the expected outcomes. Use `.expect_behavior()` for qualitative checks evaluated by an LLM judge, or `.expect_metric()` for quantitative checks on objective metrics. Both methods accept `criteria` to perform the statistical analysis.

This approach allows for a robust, automated evaluation of the AI's behavior against clear, human-readable standards. The full evaluation process for a `ScenarioTest` unfolds in three main phases: Test Setup, Data Collection, and Statistical Analysis.

```python
from sigmaeval import (
    SigmaEval, 
    ScenarioTest, 
    ScenarioTestResult,
    assertions,
    metrics,
)
import asyncio
from typing import Dict, Any, List, Union, Tuple
import secrets

# --- Define the ScenarioTest ---
scenario = (
    ScenarioTest("Bot explains its capabilities")
    .given("A new user who has not interacted with the bot before")
    .when("The user asks a general question about the bot's capabilities")
    .expect_behavior(
        "Bot lists its main functions: tracking orders, initiating returns, answering product questions, and escalating to a human agent.",
        criteria=assertions.scores.proportion_gte(min_score=6, proportion=0.90)
    )
    .expect_metric(
        metrics.per_turn.response_latency,
        criteria=assertions.metrics.proportion_lt(threshold=1.0, proportion=0.90)
    )
)

# Define the callback to connect SigmaEval to your app
async def app_handler(messages: List[Dict[str, str]], state: Any) -> Tuple[str, Any]:
    """
    This function acts as a bridge between SigmaEval and your application.
    It allows SigmaEval to initialize a conversation and pass it to your application,
    and for your application to respond back to SigmaEval's User Simulator LLM.
    """
    new_user_message = messages[-1]["content"]
    print(f"  [App] Received message: '{new_user_message}'")
    
    # For stateful apps, you can use the `state` object to track information
    # across turns. It will be an empty dictionary on the first turn.
    # Stateless apps can return just the response string without a state.
    convo_id = state.get("convo_id", secrets.token_hex(4))

    await asyncio.sleep(0.1)  # Simulate your apps generation time
    response_message = (
        f"Response for convo_id '{convo_id}' to message: '{new_user_message}'"
    )
    
    # Return the response string and the updated state to SigmaEval.
    return response_message, {"convo_id": convo_id}

# Initialize SigmaEval and run the evaluation
async def main():
    # significance_level and sample_size can be provided here or in the scenario
    sigma_eval = SigmaEval(
        judge_model="openai/gpt-5-nano", 
        significance_level=0.05,
        sample_size=30
    )
    results: ScenarioTestResult = await sigma_eval.evaluate(scenario, app_handler)

    # The result object provides a comprehensive, human-readable summary
    print(results)
    
    # The `ScenarioTestResult` object contains all the data from the evaluation,
    # including the pass/fail status, all conversations, and judge's scores.
    
    # You can also programmatically access the results
    if results.passed:
        print("\n✅ Scenario passed!")
    else:
        print("\n❌ Scenario failed.")
    
    # For more detailed analysis, you can inspect individual expectation results
    # and the raw conversation data stored in the `results` object.

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### Assertions

SigmaEval provides different statistical criteria to evaluate your AI's performance based on the 1-10 scores from the Judge LLM or on objective metrics. You can choose the one that best fits your scenario. All assertions are available under the `assertions` object.

All statistical tests require a `significance_level` (alpha), which can be provided to the `SigmaEval` constructor as a default, or on a per-assertion basis. This value, typically set to 0.05, represents the probability of rejecting the null hypothesis when it is actually true (a Type I error).

#### `assertions.scores.proportion_gte(min_score, proportion, significance_level=None)`

This criterion helps you answer the question: **"Is my AI's performance good enough, most of the time?"**

It performs a one-sided hypothesis test to verify that a desired proportion of your app's responses meet a minimum quality bar.

Specifically, it checks if there is enough statistical evidence to conclude that the true proportion of scores _greater than or equal to_ your `min_score` is _at least_ your specified `proportion`.

This is useful for setting quality targets. For example, `assertions.scores.proportion_gte(min_score=8, proportion=0.75)` lets you test the hypothesis: "Are at least 75% of our responses scoring an 8 or higher?". The test passes if the collected data supports this claim with statistical confidence.

#### `assertions.scores.median_gte(threshold, significance_level=None)`

This criterion helps you answer the question: **"Is the _typical_ user experience good?"** The median represents the middle-of-the-road experience, so this test is robust to a few unusually bad outcomes.

It performs a one-sided bootstrap hypothesis test to determine if the true median score is statistically greater than or equal to your `threshold`. Because the median is the 50th percentile, passing this test means you can be confident that at least half of all responses will meet the quality bar.

This is particularly useful for subjective qualities like helpfulness or tone. For example, `assertions.scores.median_gte(threshold=8.0)` tests the hypothesis: "Is the typical score at least an 8?".

#### `assertions.metrics.proportion_lt(threshold, proportion, significance_level=None)`

This criterion is used for "lower is better" metrics like response latency. It performs a one-sided hypothesis test to verify that a desired proportion of your app's responses are fast enough.

Specifically, it checks if there is enough statistical evidence to conclude that the true proportion of metric values _less than_ your `threshold` is _at least_ your specified `proportion`.

This is useful for setting performance targets (e.g., Service Level Objectives). For example, `assertions.metrics.proportion_lt(threshold=1.5, proportion=0.95)` lets you test the hypothesis: "Are at least 95% of our responses faster than 1.5 seconds?". The test passes if the collected data supports this claim with statistical confidence.

#### `assertions.metrics.median_lt(threshold, significance_level=None)`

This criterion helps you answer the question: **"Is the _typical_ performance efficient?"** for "lower is better" metrics like latency or turn count. The median is robust to a few unusually slow or long-running outcomes.

It performs a one-sided bootstrap hypothesis test to determine if the true median of a metric is statistically lower than your `threshold`.

This is useful for evaluating the typical efficiency of your system. For example, when applied to turn count, `assertions.metrics.median_lt(threshold=3.0)` tests the hypothesis: "Does a typical conversation wrap up in fewer than 3 turns?".

### Metrics

SigmaEval provides several built-in metrics to measure objective, quantitative aspects of your AI's performance. All metrics are available under the `metrics` object and are namespaced by their scope: `per_turn` or `per_conversation`.

- **Per-Turn Metrics**: Collected for each assistant response within a conversation.
- **Per-Conversation Metrics**: Collected once for the entire conversation.

#### `metrics.per_turn.response_latency`

- **Description**: Measures the time (in seconds) between the application receiving a user's message and sending its response.
- **Scope**: Per-Turn
- **Use Case**: Ensuring the application feels responsive and meets performance requirements (e.g., "95% of responses should be under 1.5 seconds").

#### `metrics.per_turn.response_length_chars`

- **Description**: The number of characters in an assistant's response.
- **Scope**: Per-Turn
- **Use Case**: Enforcing conciseness in individual responses to prevent overly long messages (e.g., "90% of responses must be under 1000 characters").

#### `metrics.per_conversation.turn_count`

- **Description**: The total number of assistant responses in a conversation.
- **Scope**: Per-Conversation
- **Use Case**: Measuring the efficiency of the AI. A lower turn count to resolve an issue is often better (e.g., "The median conversation should be less than 4 turns").

#### `metrics.per_conversation.total_assistant_response_time`

- **Description**: The total time (in seconds) the assistant spent processing responses for the entire conversation. This is the sum of all response latencies.
- **Scope**: Per-Conversation
- **Use Case**: Evaluating the total computational effort of the assistant over a conversation, useful for monitoring cost and overall performance.

#### `metrics.per_conversation.total_assistant_response_chars`

- **Description**: The total number of characters in all of the assistant's responses in a conversation.
- **Scope**: Per-Conversation
- **Use Case**: Measuring the overall verbosity of the assistant. This is useful for ensuring that the total amount of text a user has to read is not excessive.


## Supported LLMs

SigmaEval is agnostic to the specific model/provider used by the application under test. For the LLM-as-a-Judge component, SigmaEval uses the [LiteLLM](https://github.com/BerriAI/litellm) library under the hood, which provides a unified interface to many providers and models (OpenAI, Anthropic, Google, Ollama, etc.).

## Guides

### Managing Cost

Each `SigmaEval` run performs multiple LLM calls for rubric generation, user simulation, and judging, which has direct cost implications based on the models and `sample_size` you choose. While thorough evaluation requires investment, you can manage costs effectively:

-   **Use Different Models for Different Roles**: The quality of the judge is critical for reliable scores, so it's best to use a relatively powerful model (e.g., `openai/gpt-5-mini` or `gemini/gemini-2.5-flash`) for the `judge_model`. The user simulation, however, is often less demanding. You can use a smaller, faster, and cheaper model (e.g., `openai/gpt-5-nano`, `gemini/gemini-2.5-flash-lite`, or a local model like `ollama/llama2`) for the `user_simulator_model` to significantly reduce costs without compromising the quality of the evaluation.

    ```python
    sigma_eval = SigmaEval(
        judge_model="gemini/gemini-2.5-flash",
        user_simulator_model="gemini/gemini-2.5-flash-lite",
        # ... other settings
    )
    ```

-   **Start with a Small `sample_size`**: During iterative development and debugging, use a small `sample_size` (e.g., 5-10) to get a quick signal on performance. This allows you to fail fast and fix issues without incurring high costs. Once you are ready for a final, statistically rigorous validation (e.g., before a release), you can increase the `sample_size` to a larger number (e.g., 30-100+) to achieve higher statistical confidence.

Ultimately, the cost of evaluation should be seen as a trade-off. A small investment in automated, statistical evaluation can prevent the much higher costs associated with shipping a low-quality, unreliable AI product.

### User Simulation Writing Styles

To better address the "infinite input space" problem, SigmaEval's user simulator can be configured to adopt a wide variety of writing styles. This feature helps ensure your application is robust to the many ways real users communicate.

By default, for each of the `sample_size` evaluation runs, the user simulator will randomly adopt a different writing style by combining four independent axes:

- **`proficiency`**: The user's grasp of grammar and vocabulary (e.g., "Middle-school level," "Flawless grammar and sophisticated vocabulary").
- **`tone`**: The user's emotional disposition (e.g., "Polite and friendly," "Impatient and slightly frustrated").
- **`verbosity`**: The length and detail of the user's messages (e.g., "Terse and to-the-point," "Verbose and descriptive").
- **`formality`**: The user's adherence to formal language conventions (e.g., "Formal and professional," "Casual with slang").

This behavior is on by default and can be configured or disabled via the `WritingStyleConfig` object passed to the `SigmaEval` constructor.

```python
from sigmaeval import SigmaEval, WritingStyleConfig, WritingStyleAxes

# Disable writing style variations completely
no_style_config = WritingStyleConfig(enabled=False)

# Customize the axes with your own values
custom_axes = WritingStyleAxes(
    proficiency=["writes perfectly", "makes some mistakes"],
    tone=["happy", "sad"],
    verbosity=["short", "long"],
    formality=["formal", "casual"]
)
custom_style_config = WritingStyleConfig(axes=custom_axes)

sigma_eval = SigmaEval(
    judge_model="openai/gpt-5-nano",
    significance_level=0.05,
    writing_style_config=custom_style_config
)
```

This system ensures that the `Given` (persona) and `When` (goal) clauses of your `ScenarioTest` are always prioritized. The writing style adds a layer of realistic, stylistic variation without overriding the core of the test scenario.

### Logging

SigmaEval uses Python's standard `logging` module to provide visibility into the evaluation process. You can control the verbosity by passing a `log_level` to the `SigmaEval` constructor.

- **`logging.INFO`** (default): Provides a high-level overview, including a progress bar for data collection.
- **`logging.DEBUG`**: Offers detailed output for troubleshooting, including LLM prompts, conversation transcripts, and judge's reasoning.

### Retry Configuration

To improve robustness against transient network or API issues, SigmaEval automatically retries failed LLM calls using an exponential backoff strategy (powered by the [Tenacity](https://tenacity.readthedocs.io/en/latest/) library). This also includes retries for malformed or unparsable LLM responses. This applies to rubric generation, user simulation, and judging calls.

The retry behavior can be customized by passing a `RetryConfig` object to the `SigmaEval` constructor. If no configuration is provided, default settings are used.

```python
from sigmaeval import SigmaEval, RetryConfig

# Example: Customize retry settings
custom_retry_config = RetryConfig(
    max_attempts=3,
    backoff_multiplier=1,
    max_backoff_seconds=10
)

# You can also disable retries completely
# no_retry_config = RetryConfig(enabled=False)

sigma_eval = SigmaEval(
    judge_model="openai/gpt-5-nano",
    # significance_level can be omitted here if provided in all assertions
    significance_level=0.05,
    retry_config=custom_retry_config
)
```

### Evaluating a Test Suite

You can also run a full suite of tests by passing a list of `ScenarioTest` objects to the `evaluate` method. The tests will be run concurrently.

```python
# Assume scenario_1 and scenario_2 are defined ScenarioTest objects
test_suite = [scenario_1, scenario_2]
all_results = await sigma_eval.evaluate(test_suite, app_handler)

# all_results will be a list of ScenarioTestResult objects
for result in all_results:
    print(result)
```

### Evaluating Multiple Conditions and Assertions

For more comprehensive validation, SigmaEval supports testing multiple conditions and assertions within a single `ScenarioTest`. This allows you to check for complex behaviors and verify multiple statistical properties in an efficient manner.

#### Multiple Conditions

You can call `.expect_behavior()` or `.expect_metric()` multiple times on a `ScenarioTest` to add multiple expectations. The test will only pass if all expectations are met. Each expectation is evaluated independently (behavioral expectations get their own rubric), but they all share the same `sample_size`. This is useful for testing complex behaviors that have multiple success criteria.

For efficiency, the user simulation is run only once to generate a single set of conversations. This same set of conversations is then judged against each expectation, making this approach ideal for evaluating multiple facets of a single interaction. When using multiple expectations, you can provide an optional `label` to each one to easily identify it in the results.

```python
multi_condition_scenario = (
    ScenarioTest("Bot handles a complex multi-part request")
    .given("A user needs to both track a package and ask a question about a different product")
    .when("The user asks to track their package and then asks a follow-up question about a product's warranty")
    .sample_size(30)
    .expect_behavior(
        "Bot successfully provides the tracking status for the user's package.",
        criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.90),
        label="Tracks Package"
    )
    .expect_behavior(
        "Bot accurately answers the user's question about the product warranty.",
        criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.90),
        label="Answers Warranty Question"
    )
)
```

#### Multiple Assertions

You can also specify a list of `criteria` in a single `.expect_behavior()` or `.expect_metric()` call. The test will only pass if all assertions are met. This is useful for checking multiple statistical properties of the same set of scores or metric values.

For efficiency, the user simulation and judging are run only once to generate a single set of scores. This same set of scores is then evaluated against each criterion.

```python
multi_assertion_scenario = (
    ScenarioTest("Bot gives a comprehensive and helpful answer")
    .given("A user is asking about the return policy for electronics.")
    .when("The user asks if they can return a laptop after 30 days.")
    .sample_size(50)
    .expect_behavior(
        "The bot correctly states that laptops must be returned within 30 days, but also helpfully suggests checking the manufacturer's warranty.",
        criteria=[
            assertions.scores.proportion_gte(min_score=7, proportion=0.90),
            assertions.scores.median_gte(threshold=8)
        ]
    )
)
```

### Accessing Evaluation Results

The `evaluate` method returns a `ScenarioTestResult` object (or a list of them) that contains all the information about the test run.

For a quick check, you can inspect the `passed` property:

```python
if results.passed:
    print("✅ Scenario passed!")
```

Printing the result object provides a comprehensive, human-readable summary of the outcomes, which is ideal for logs:

```python
print(results)
```

For more detailed programmatic analysis, the object gives you full access to the nested `expectation_results` (including scores and reasoning) and the complete `conversations` list.

### Compatibility with Testing Libraries

SigmaEval is designed to integrate seamlessly with standard Python testing libraries like `pytest` and `unittest`. Since the `evaluate` method returns a result object with a simple `.passed` boolean property, you can easily use it within your existing test suites.

Here's an example of how to use SigmaEval with `pytest`:

```python
import pytest
from sigmaeval import SigmaEval, ScenarioTest, assertions
from typing import List, Dict, Any

# A stateless app_handler that returns a static response.
async def app_handler(messages: List[Dict[str, str]], state: Any) -> str:
    return "I can track orders, initiate returns, and answer product questions."

@pytest.mark.asyncio
async def test_bot_capabilities_scenario():
    """
    This test will pass if the SigmaEval scenario passes.
    """
    sigma_eval = SigmaEval(
        judge_model="openai/gpt-5-nano",
        sample_size=30,
        significance_level=0.05
    )
    
    scenario = (
        ScenarioTest("Bot explains its capabilities")
        .given("A new user asks about what the bot can do")
        .when("The user asks 'what can you do?'")
        .expect_behavior(
            "Bot lists its main functions.",
            criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.90)
        )
    )
    
    # The app_handler is the callback to your application
    result = await sigma_eval.evaluate(scenario, app_handler)
    
    # Print the detailed summary for logs
    print(result)
    
    # Use a standard pytest assertion
    assert result.passed, "The bot capabilities scenario failed."

```

This allows you to incorporate rigorous, statistical evaluation of your AI's behavior directly into your CI/CD pipelines.

## Appendix

### Sample Size and Statistical Significance

The `sample_size` determines the number of conversations to simulate for each `ScenarioTest`. It can be set globally in the `SigmaEval` constructor or on a per-scenario basis using the `.sample_size()` method. The scenario-specific value takes precedence.

It is important to note that the `sample_size` plays a crucial role in the outcome of the hypothesis tests used in criteria like `assertions.scores.proportion_gte`. A larger sample size provides more statistical evidence, making it easier to detect a true effect. With very small sample sizes (e.g., less than 10), a test might fail to achieve statistical significance (i.e., pass) even if the observed success rate in the sample is 100%. This is the expected and correct behavior, as there isn't enough data to confidently conclude that the _true_ success rate for the entire user population is above the minimum threshold.

### Statistical Methods

To ensure robust and reliable conclusions, SigmaEval uses established statistical hypothesis tests tailored to the type of evaluation being performed.

- **For Proportion-Based Criteria** (e.g., `proportion_gte`): The framework employs a **one-sided binomial test**. This test is ideal for scenarios where each data point can be classified as a binary outcome (e.g., "success" or "failure," like a score being above or below a threshold). It directly evaluates whether the observed proportion of successes in your sample provides enough statistical evidence to conclude that the true proportion for all possible interactions meets your specified minimum target.

- **For Median-Based Criteria** (e.g., `median_gte`): The framework uses a **bootstrap hypothesis test**. The median is a robust measure of central tendency, but its theoretical sampling distribution can be complex. Bootstrapping is a powerful, non-parametric resampling method that avoids making assumptions about the underlying distribution of the scores or metric values. By repeatedly resampling the collected data, it constructs an empirical distribution of the median, which is then used to determine if the observed median provides statistically significant evidence for the hypothesis.

This approach ensures that the framework's conclusions are statistically sound without imposing rigid assumptions on the nature of your AI's performance data.

### An Example Rubric

For the `ScenarioTest` defined in the Python snippet:

```python
scenario = (
    ScenarioTest("Bot explains its capabilities")
    .given("A new user who has not interacted with the bot before")
    .when("The user asks a general question about the bot's capabilities")
    .sample_size(30)
    .expect_behavior(
        "Bot lists its main functions: tracking orders, initiating returns, answering product questions, and escalating to a human agent.",
        criteria=assertions.scores.proportion_gte(min_score=6, proportion=0.90)
    )
)
```

SigmaEval might generate the following 1-10 rubric for the Judge LLM:

**1:** Bot gives no answer or ignores the question.

**2:** Bot answers irrelevantly, with no mention of its functions.

**3:** Bot gives vague or incomplete information, missing most functions.

**4:** Bot names one correct function but misses the rest.

**5:** Bot names some functions but omits key ones or adds irrelevant ones.

**6:** Bot names most functions but in unclear or confusing language.

**7:** Bot names all required functions but with weak clarity or order.

**8:** Bot names all required functions clearly but without polish or flow.

**9:** Bot names all required functions clearly, concisely, and in a logical order.

**10:** Bot names all required functions clearly, concisely, in order, and with natural, helpful phrasing.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black sigmaeval tests
ruff check sigmaeval tests
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
