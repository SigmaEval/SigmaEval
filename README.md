# SigmaEval

A Python library for evaluating Generative AI agents and apps.

## Installation

```bash
pip install sigmaeval
```

Or install from source:

```bash
git clone https://github.com/Itura-AI/sigmaeval.git
cd sigmaeval
pip install -e .
```

## SigmaEval: A Statistical Framework for AI App Evaluation

### The Problem

Ensuring the quality of LLM-based apps is critical, but their inherent nature presents unique challenges that traditional software evaluation frameworks are not equipped to handle. Two fundamental properties of these apps make reliable evaluation a difficult task:

1.  **Non-deterministic Outputs:** Unlike traditional software, where a given input consistently produces the same output, LLM-based systems can produce a variety of different, often equally valid, responses to the same prompt.
2.  **Infinite Input Space:** The sheer variety of possible inputs—from subtle changes in prompt phrasing to the vast context windows these models can handle—makes comprehensive testing an impossibility. Even the slightest change in the punctuation or phrasing of a prompt can produce a different output.

Consider a customer support bot designed to handle returns. A user might say, "I want to return this," "This isn't what I wanted," or "How do I start a return?" A good bot could respond with, "I can help with that. What's your order number?" or "Certainly, let's get that return started for you. Could you provide the order number?" All of these are valid and helpful responses. A traditional testing mindset that checks for a single, exact-match "correct" answer would incorrectly fail these perfectly good interactions.

This departure from deterministic behavior calls for a new approach to evaluation—one that embraces variability and uncertainty rather than trying to eliminate them.

### The Stakes: Why Traditional QA Fails

Without a reliable evaluation framework, development becomes a guessing game. Teams ship features based on gut feelings and anecdotal evidence, leading to unreliable products, eroded user trust, and wasted engineering cycles. In high-stakes domains, the consequences can be even more severe, ranging from reputational damage to significant safety concerns. Sticking with testing paradigms built for deterministic systems is not just ineffective; it's a risk.

### A New Paradigm: From Determinism to Statistical Confidence

Instead of asking, "Is this output correct?" we need to shift our mindset to ask, "How likely is this system to produce a high-quality output?" This is where statistical methods become essential. Before we can measure quality, however, we must define it. "Quality" is not a universal metric; it is a composite of factors tailored to a specific application, such as factual accuracy, task-completion, helpfulness, relevance, appropriate tone, and the absence of harmful content.

The core tenets of this new approach are:

*   **Accept Irreducible Variability:** We must accept that some level of variability in outputs is inherent to these systems.
*   **Quantify Uncertainty Statistically:** The goal is not to achieve certainty for every single output, but to quantify the system's performance and uncertainty using established statistical tools like significance level, p-value, minimum proportion of successes, and hypothesis testing.
*   **Focus on Average Effectiveness:** Like in complex biological systems, a single failure may not point to a simple cause but rather a complex interaction of factors. The primary goal is to ensure that the system is effective *on average* and that its benefits outweigh its risks.

An apt analogy comes from clinical drug trials. The objective isn't to guarantee a specific outcome for every patient but to ensure the treatment is of a consistently high quality. We don't ask, "Does this drug work for everyone?" but rather, "For what percentage of the target population does this drug produce a statistically significant positive effect with an acceptable risk profile?"

Similarly, for LLM-based applications, the key is to ensure the system is reliable, effective, and safe enough for its intended purpose, allowing us to make informed, data-driven decisions based on known probabilities of success and failure. SigmaEval provides the tools to do just that.

### How SigmaEval Works

SigmaEval combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge evaluation within a Behavior-Driven Development (BDD) framework. This powerful combination allows you to move beyond simple pass/fail tests and gain statistical confidence in your AI's performance.

The evaluation process for a single `ScenarioTest` unfolds in three main phases:

**Phase 1: Test Setup**

1.  **Defining Behavior with BDD:** You start by defining a test scenario using a `ScenarioTest` with its `Given`, `When`, and `Then` clauses. This sets the stage for the entire evaluation.
2.  **Creating the Rubric:** Based on the `expected_behavior` you specified in the `Then` clause, SigmaEval generates a detailed 1-10 scoring rubric. This rubric is created once per test case and ensures that every interaction is evaluated against the same consistent criteria (see Appendix A for an example).

**Phase 2: Data Collection (Repeated for `sample_size`)**

To gather a statistically meaningful sample, the following steps are repeated multiple times (as defined by `sample_size`):

3.  **Simulating the User:** For each repetition, SigmaEval uses the `Given` (user's persona) and `When` (user's goal) clauses to prompt a **User Simulator LLM**. This LLM realistically simulates a user interacting with your application. The interaction can span multiple turns.
4.  **Recording the Interaction:** The entire conversation between the User Simulator LLM and your AI application is recorded for judgment.
5.  **Judging the Outcome:** A separate **Judge LLM** analyzes the recorded conversation against the pre-defined rubric. It assigns a score from 1-10 based on how well the AI application's behavior met the desired `expected_behavior`.

**Phase 3: Statistical Analysis**

6.  **Drawing a Conclusion:** After all repetitions are complete, the collection of scores (the sample) is passed to the statistical evaluator you defined (`SuccessRateEvaluator`, `RatingAverageEvaluator`, etc.). This evaluator performs the appropriate statistical tests to determine if the application's performance meets your quality bar, providing a final pass/fail result with statistical confidence.

Each scenario is defined using a `ScenarioTest` object with a fluent builder API. The test has three main parts that mirror the Behavior-Driven Development (BDD) pattern:

*   **`.given()`**: This method establishes the prerequisite state and context for the **User Simulator LLM**. This can include the persona of the user (e.g., a new user, an expert user), the context of the conversation (e.g., a customer's order number), or any other background information.
*   **`.when()`**: This method describes the specific goal or action the **User Simulator LLM** will try to achieve. SigmaEval uses this to guide the simulation.
*   **`.expect_behavior()` / `.expect_metric()`**: These methods specify the expected outcomes. Use `.expect_behavior()` for qualitative checks evaluated by an LLM judge, or `.expect_metric()` for quantitative checks on objective metrics. Both methods accept `criteria` to perform the statistical analysis.

This approach allows for a robust, automated evaluation of the AI's behavior against clear, human-readable standards.

```python
from sigmaeval import (
    SigmaEval, 
    ScenarioTest, 
    AppResponse,
    ScenarioTestResult,
    assertions,
    metrics,
)
import asyncio
from typing import Dict, Any

# --- Define the ScenarioTest ---
scenario = (
    ScenarioTest("Bot explains its capabilities")
    .given("A new user who has not interacted with the bot before")
    .when("The user asks a general question about the bot's capabilities")
    .sample_size(30)
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
async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
    """
    This function acts as a bridge between SigmaEval and your application.
    It takes a message and a state dictionary, and returns an AppResponse.
    The 'message' is the message from SigmaEval's User Simulator LLM.
    The 'state' dictionary is empty on the first turn of a conversation.
    The 'AppResponse' is the response from your application, containing the response string and the updated state of the conversation.
    """
    print(f"  [App] Received message: '{message}'")

    # In a real test, you would call your actual application logic here.
    # For this example, we'll manage a simple history in the state.
    history = state.get("history", [])
    history.append({"role": "user", "content": message})
    
    await asyncio.sleep(0.1)  # Simulate async work
    response_message = f"Hello! This is turn #{len(history)}. You said: '{message}'."
    history.append({"role": "assistant", "content": response_message})
    
    print(f"  [App] Sending response: '{response_message}'")
    
    # Return the response and the updated state
    return AppResponse(response=response_message, state={"history": history})

# Initialize SigmaEval and run the evaluation
async def main():
    sigma_eval = SigmaEval(judge_model="openai/gpt-5-nano", significance_level=0.05)
    results: ScenarioTestResult = await sigma_eval.evaluate(scenario, app_handler)

    # The result object provides a comprehensive, human-readable summary
    print(results)
    
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

### Available Criteria

SigmaEval provides different statistical criteria to evaluate your AI's performance based on the 1-10 scores from the Judge LLM. You can choose the one that best fits your scenario. All criteria are available under the `assertions.scores` object.

All statistical tests require a `significance_level` (alpha), which can be provided to the `SigmaEval` constructor as a default, or on a per-assertion basis. This value, typically set to 0.05, represents the probability of rejecting the null hypothesis when it is actually true (a Type I error).

#### `assertions.scores.proportion_gte(min_score, proportion, significance_level=None)`
This criterion performs a one-sided hypothesis test to determine if the true proportion of high-quality outcomes for the entire user population is greater than a specified minimum. A score at or above `min_score` is considered a "high-quality" outcome. The test passes if there is statistical evidence that the system's performance exceeds the `min_proportion`.

This is useful when you have a clear minimum standard that a certain percentage of responses should meet. For example, `assertions.scores.proportion_gte(min_score=8, proportion=0.75)` checks if at least 75% of responses have a score of 8 or higher.


#### `assertions.scores.median_gte(threshold, significance_level=None)`
This criterion is particularly useful for subjective qualities like helpfulness or tone. It performs a one-sided **bootstrap hypothesis test** to determine if the true median rating for a response across the entire user population is statistically higher than the specified `threshold`.

The bootstrap method is a modern, non-parametric method that is robust to the underlying distribution of the data. By testing the **median**, it ensures that at least 50% of responses meet a certain quality bar. For example, `assertions.scores.median_gte(threshold=8.0)` checks if there is statistical evidence that the median score is greater than 8.

#### `assertions.metrics.proportion_lt(threshold, proportion, significance_level=None)`
This criterion performs a one-sided hypothesis test to determine if the true proportion of metric values below a certain `threshold` is statistically significant. It checks if the proportion of observations under the threshold is less than the expected `proportion`.

This is useful for metrics like latency, where a lower value is better. For example, `assertions.metrics.proportion_lt(threshold=1.5, proportion=0.95)` checks if it is statistically likely that 95% of responses have a latency of less than 1.5 seconds.

#### `assertions.metrics.median_lt(threshold, significance_level=None)`
This criterion performs a one-sided bootstrap hypothesis test to determine if the true median of a metric is statistically lower than a specified `threshold`. This non-parametric test is robust to outliers and does not assume the data is normally distributed, making it ideal for skewed metrics like latency or turn count.

This is useful for evaluating the typical performance of a system. For example, `assertions.metrics.median_lt(threshold=2.0)` could be used to test if the median number of turns in a conversation is less than 2.

### Available Metrics

SigmaEval provides several built-in metrics to measure objective, quantitative aspects of your AI's performance. All metrics are available under the `metrics` object and are namespaced by their scope: `per_turn` or `per_conversation`.

*   **Per-Turn Metrics**: Collected for each assistant response within a conversation.
*   **Per-Conversation Metrics**: Collected once for the entire conversation.

#### `metrics.per_turn.response_latency`
*   **Description**: Measures the time (in seconds) between the application receiving a user's message and sending its response.
*   **Scope**: Per-Turn
*   **Use Case**: Ensuring the application feels responsive and meets performance requirements (e.g., "95% of responses should be under 1.5 seconds").

#### `metrics.per_turn.response_length_chars`
*   **Description**: The number of characters in the assistant's response.
*   **Scope**: Per-Turn
*   **Use Case**: Enforcing conciseness in individual responses to prevent overly long messages (e.g., "90% of responses must be under 1000 characters").

#### `metrics.per_conversation.turn_count`
*   **Description**: The total number of assistant responses in a conversation.
*   **Scope**: Per-Conversation
*   **Use Case**: Measuring the efficiency of the AI. A lower turn count to resolve an issue is often better (e.g., "The average conversation should be less than 4 turns").

#### `metrics.per_conversation.total_assistant_response_time`
*   **Description**: The total time (in seconds) the assistant spent processing responses for the entire conversation. This is the sum of all response latencies.
*   **Scope**: Per-Conversation
*   **Use Case**: Evaluating the total computational effort of the assistant over a conversation, useful for monitoring cost and overall performance.

#### `metrics.per_conversation.total_assistant_response_chars`
*   **Description**: The total number of characters in all of the assistant's responses in a conversation.
*   **Scope**: Per-Conversation
*   **Use Case**: Measuring the overall verbosity of the assistant. This is useful for ensuring that the total amount of text a user has to read is not excessive.

### A Note on Sample Size and Statistical Significance

It is important to note that the `sample_size` plays a crucial role in the outcome of the hypothesis tests used by `SuccessRateEvaluator` and `RatingProportionEvaluator`. A larger sample size provides more statistical evidence, making it easier to detect a true effect. With very small sample sizes (e.g., less than 10), a test might fail to achieve statistical significance (i.e., pass) even if the observed success rate in the sample is 100%. This is the expected and correct behavior, as there isn't enough data to confidently conclude that the *true* success rate for the entire user population is above the minimum threshold.


### Supported LLMs

SigmaEval is agnostic to the specific model/provider used by the application under test. For the LLM-as-a-Judge component, SigmaEval uses the [LiteLLM](https://github.com/BerriAI/litellm) library under the hood, which provides a unified interface to many providers and models (OpenAI, Anthropic, Google, etc.).

### Logging

SigmaEval uses Python's standard `logging` module to provide visibility into the evaluation process. You can control the verbosity by passing a `log_level` to the `SigmaEval` constructor.
*   **`logging.INFO`** (default): Provides a high-level overview, including a progress bar for data collection.
*   **`logging.DEBUG`**: Offers detailed output for troubleshooting, including LLM prompts, conversation transcripts, and judge's reasoning.

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
    significance_level=0.05,
    retry_config=custom_retry_config
)
```

### User Simulation Writing Styles

To better address the "infinite input space" problem, SigmaEval's user simulator can be configured to adopt a wide variety of writing styles. This feature helps ensure your application is robust to the many ways real users communicate.

By default, for each of the `sample_size` evaluation runs, the user simulator will randomly adopt a different writing style by combining four independent axes:
*   **`proficiency`**: The user's grasp of grammar and vocabulary (e.g., "Middle-school level," "Flawless grammar and sophisticated vocabulary").
*   **`tone`**: The user's emotional disposition (e.g., "Polite and friendly," "Impatient and slightly frustrated").
*   **`verbosity`**: The length and detail of the user's messages (e.g., "Terse and to-the-point," "Verbose and descriptive").
*   **`formality`**: The user's adherence to formal language conventions (e.g., "Formal and professional," "Casual with slang").

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

For efficiency, the user simulation is run only once to generate a single set of conversations. This same set of conversations is then judged against each expectation, making this approach ideal for evaluating multiple facets of a single interaction.

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

### Appendix A: Example Rubric

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

**7.5:** Bot names all required functions but with weak clarity or order.

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

