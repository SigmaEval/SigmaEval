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

The evaluation process for a single `BehavioralTest` unfolds in three main phases:

**Phase 1: Test Setup**

1.  **Defining Behavior with BDD:** You start by defining a test scenario using a `BehavioralTest` with its `Given`, `When`, and `Then` clauses. This sets the stage for the entire evaluation.
2.  **Creating the Rubric:** Based on the `expected_behavior` you specified in the `Then` clause, SigmaEval generates a detailed 1-10 scoring rubric. This rubric is created once per test case and ensures that every interaction is evaluated against the same consistent criteria (see Appendix A for an example).

**Phase 2: Data Collection (Repeated for `sample_size`)**

To gather a statistically meaningful sample, the following steps are repeated multiple times (as defined by `sample_size`):

3.  **Simulating the User:** For each repetition, SigmaEval uses the `Given` (user's persona) and `When` (user's goal) clauses to prompt a **User Simulator LLM**. This LLM realistically simulates a user interacting with your application. The interaction can span multiple turns.
4.  **Recording the Interaction:** The entire conversation between the User Simulator LLM and your AI application is recorded for judgment.
5.  **Judging the Outcome:** A separate **Judge LLM** analyzes the recorded conversation against the pre-defined rubric. It assigns a score from 1-10 based on how well the AI application's behavior met the desired `expected_behavior`.

**Phase 3: Statistical Analysis**

6.  **Drawing a Conclusion:** After all repetitions are complete, the collection of scores (the sample) is passed to the statistical evaluator you defined (`SuccessRateEvaluator`, `RatingMeanEvaluator`, etc.). This evaluator performs the appropriate statistical tests to determine if the application's performance meets your quality bar, providing a final pass/fail result with statistical confidence.

Each scenario is defined using a `BehavioralTest` object with three main parts:

*   **`Given`**: This section establishes the prerequisite state and context for the **User Simulator LLM**. This can include the persona of the user (e.g., a new user, an expert user), the context of the conversation (e.g., a customer's order number), or any other background information.
*   **`When`**: This describes the specific goal or action the **User Simulator LLM** will try to achieve. SigmaEval uses this to guide the simulation.
*   **`Then`**: This section specifies the expected outcome. It is an `Expectation` object containing an `expected_behavior` description (which is passed to the **Judge LLM**) and an `evaluator` to perform the statistical analysis.

This approach allows for a robust, automated evaluation of the AI's behavior against clear, human-readable standards.

```python
from sigmaeval import (
    SigmaEval, 
    BehavioralTest, 
    Expectation, 
    SuccessRateEvaluator, 
    AppResponse,
    EvaluationResult,
)
import asyncio
from typing import Dict, Any

# --- Define the BehavioralTest ---
scenario = BehavioralTest(
    title="Bot explains its capabilities",
    given="A new user who has not interacted with the bot before",
    when="The user asks a general question about the bot's capabilities",
    then=Expectation(
        expected_behavior="Bot lists its main functions: tracking orders, initiating returns, answering product questions, and escalating to a human agent.",
        evaluator=SuccessRateEvaluator(
            significance_level=0.05,
            min_proportion=0.90,
            sample_size=30
        )
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
    sigma_eval = SigmaEval(judge_model="openai/gpt-4o")
    results: EvaluationResult = await sigma_eval.evaluate(scenario, app_handler)

    # Print the results
    print(f"--- Results for BehavioralTest: {scenario.title} ---")
    print(f"Passed: {results.results['passed']}")
    print(f"P-value: {results.results['p_value']:.4f}")
    print(f"Average Score: {sum(results.scores) / len(results.scores):.2f}")
    
    # You can also inspect individual conversations
    # print("--- Example Conversation ---")
    # print(results.conversations[0])

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Evaluators

SigmaEval provides different methods to evaluate your AI's performance. You can choose the one that best fits your scenario.

#### SuccessRateEvaluator
This evaluator performs a one-sided hypothesis test to determine if the true proportion of successful outcomes for the entire user population is greater than a specified minimum. For each run, the Judge LLM provides a rating on a 1-10 scale based on the rubric. A rating of 6 or higher is considered a "success," and a rating of 5 or lower is considered a "failure." The test passes if the observed success rate is high enough to reject the null hypothesis, providing statistical evidence that the system's performance exceeds the `min_proportion` at the specified `significance_level`.

```python
from sigmaeval import SuccessRateEvaluator

binary_evaluator = SuccessRateEvaluator(
    significance_level=0.05,
    min_proportion=0.90,
    sample_size=30
)
```

#### RatingMeanEvaluator
This evaluator is particularly useful for subjective qualities like helpfulness or tone. The Judge LLM provides a rating on a 1-10 scale based on the rubric, and the evaluator performs a one-sided t-test to determine if the true average rating for a response across the entire user population is significantly higher than a specified baseline. It is useful for ensuring a high average quality standard.

```python
from sigmaeval import RatingMeanEvaluator

# This would be used inside an `Expectation` object
mean_rating_evaluator = RatingMeanEvaluator(
    significance_level=0.05,
    min_mean_rating=7.0, # The minimum mean rating to test against
    sample_size=50
)
```

#### RatingProportionEvaluator
For subjective qualities like helpfulness or tone, this evaluator tests if the true proportion of users who would rate a a response at or above a certain level exceeds a specified minimum. The Judge LLM grades each response on the 1-10 numerical scale from the rubric, and the evaluator then performs a one-sided hypothesis test, similar to the `SuccessRateEvaluator`, to make an inference about the entire user population.

```python
from sigmaeval import RatingProportionEvaluator

# This would be used inside an `Expectation` object, just like SuccessRateEvaluator
rating_evaluator = RatingProportionEvaluator(
    significance_level=0.05,
    min_rating=8, # The minimum acceptable rating on a 1-10 scale
    min_proportion=0.75, # We want at least 75% of responses to have a rating of 8 or higher
    sample_size=50
)
```

### Key Differences: `RatingMeanEvaluator` vs. `RatingProportionEvaluator`

While both evaluators measure subjective quality, they answer different questions:

*   **`RatingMeanEvaluator`** asks: "Is the *average* quality of the responses high enough?" It's useful when you want to ensure a generally high standard, but can be skewed by a few very high or very low scores. For example, if your `min_mean_rating` is 7, you might pass with an average score of 7.5, which could be achieved with half your responses rated a mediocre 5 and half rated a perfect 10, masking the fact that half of your users had a poor experience.
*   **`RatingProportionEvaluator`** asks: "Do *enough* of our responses meet a specific quality bar?" This is better when you have a clear minimum standard that every response should ideally meet. It ensures a consistent user experience by minimizing the number of poor-quality responses, even if the average is high. For example, you can ensure that at least 75% of users rate the response an 8 or higher.

Choosing between them depends on your specific quality goals. Are you aiming for a high average performance, or do you need to guarantee a consistent minimum level of quality for most users?


### A Note on `SuccessRateEvaluator`

You may have noticed that the functionality of `SuccessRateEvaluator` is a specific use case of `RatingProportionEvaluator`. That is correct. `SuccessRateEvaluator` is provided as a convenience API for the common scenario where any score of 6 or higher on a 1-10 scale is considered a "success."

Internally, `SuccessRateEvaluator(...)` is equivalent to `RatingProportionEvaluator(min_rating=6, ...)`. It simplifies test definition when you only need a simple pass/fail judgment based on a fixed threshold.


### Supported LLMs

SigmaEval is agnostic to the specific model/provider used by the application under test. For the LLM-as-a-Judge component, SigmaEval uses the [LiteLLM](https://github.com/BerriAI/litellm) library under the hood, which provides a unified interface to many providers and models (OpenAI, Anthropic, Google, etc.).

### Logging

SigmaEval uses Python's standard `logging` module to provide visibility into the evaluation process. You can control the verbosity by passing a `log_level` to the `SigmaEval` constructor.
*   **`logging.INFO`** (default): Provides a high-level overview, including a progress bar for data collection.
*   **`logging.DEBUG`**: Offers detailed output for troubleshooting, including LLM prompts, conversation transcripts, and judge's reasoning.

### Retry Configuration

To improve robustness against transient network or API issues, SigmaEval automatically retries failed LLM calls using an exponential backoff strategy (powered by the [Tenacity](https://tenacity.readthedocs.io/en/latest/) library). This applies to rubric generation, user simulation, and judging calls.

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
    judge_model="openai/gpt-4o",
    retry_config=custom_retry_config
)
```

### Appendix A: Example Rubric

For the `BehavioralTest` defined in the Python snippet:

```python
scenario = BehavioralTest(
    title="Bot explains its capabilities",
    given="A new user who has not interacted with the bot before",
    when="The user asks a general question about the bot's capabilities",
    then=Expectation(
        expected_behavior="Bot lists its main functions: tracking orders, initiating returns, answering product questions, and escalating to a human agent.",
        # ... evaluator details
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
