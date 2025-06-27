# Gemini CLI (OpenAI compatible)

This is a fork of the Gemini CLI that allows you to use any LLM API, as long as it's OpenAI compatible.

To see how to use Gemini CLI, see the official [Gemini CLI documentation](https://github.com/google-gemini/gemini-cli/blob/main/README.md).

## Usage

Clone the repo and run the following command:

```bash
npm install
npm run build
npm install -g .
```

In the project root where you run Gemini CLI, create a `.env` file and set the following variables:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL= # optional, defaults to https://api.openai.com/v1
MAIN_MODEL=gpt-4o # your choice of model
FLASH_MODEL=gpt-4o-mini # your choice of fast/mini model for simple tasks
```

Then run `gemini` to start the CLI. You'll choose "OpenAI" when prompted. If you used Gemini CLI before, you'll need to type `/auth` to reset the auth method.

## Limitations

- Tokens are not counted accurately.
