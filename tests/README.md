### Testing instructions

1. Create a `.env` file at the project root with the required model identifiers and provider API keys. The exact environment variables depend on your chosen provider; see the LiteLLM providers documentation for details: [LiteLLM Providers Docs](https://docs.litellm.ai/docs/providers).

   Example `.env`:

   ```env
   TEST_APP_MODEL=gemini/gemini-2.5-flash-lite
   TEST_EVAL_MODEL=gemini/gemini-2.5-flash
   GEMINI_API_KEY=<api-key>
   ```

2. Run the test suite:

   ```bash
   poetry run pytest
   ```


