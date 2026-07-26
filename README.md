# Mistral + Gemini + LM Studio Prompt Extension

Extension for Forge/Stable Diffusion WebUI that generates prompts from images using Mistral (`mistral-large-2512`), current Gemini Flash/Flash-Lite/Pro models, or local LM Studio models through its OpenAI-compatible API.

## What It Does

- Uploads multiple images via drag-and-drop or file picker
- Supports direct paste from clipboard
- Shows a custom preview gallery with per-image delete, remove-last, and clear-all actions
- Sends text and optional images plus your prompt template to the selected model
- Supports Mistral, Gemini, and LM Studio model selection from the same UI
- Can refresh LM Studio models from the local server without restarting WebUI
- Can send a reusable instruction plus a separate source prompt for local text-model prompt improvement
- Lets you append extra text to the generated prompt
- Inserts the result directly into `txt2img` or `img2img` prompt field
- Includes editable prompt presets stored in WebUI settings

## Requirements

- Forge / Stable Diffusion WebUI with extension support
- Python dependencies available in your WebUI environment:
  - `requests`
  - `Pillow`
- **Mistral API key is required for Mistral models**
- **Gemini API key is required for Gemini models**
- Supported Gemini models are limited to stable models with a standard API free tier at the time of release. Actual quotas depend on your Google AI Studio project, account, and region. Google states that free-tier submissions may be used to improve its products.
- **LM Studio must be running with its local server enabled for LM Studio models**

## Setup

1. Place this extension in your Forge extensions directory.
2. Restart WebUI.
3. Open Settings -> `Mistral++`.
4. Set:
   - `Mistral API Key` (required)
   - `Gemini API Key` (required for Gemini models)
   - `LM Studio API Base` (default: `http://127.0.0.1:1234/v1`)
   - `LM Studio API Key` (optional, only if your local server requires one)
   - Optional image limits:
     - `Max image size sent to model (longest side, px)`
     - `Max JPEG size sent to model (KB)`
5. Apply settings and reload UI if needed.

## How To Use

1. Open the `Mistral++` accordion in `txt2img` or `img2img`.
2. Add images (drag-and-drop, click to select, or `Paste from clipboard`).
3. Choose or edit an initial preset prompt.
4. Optionally paste an existing prompt into `Prompt to improve`.
5. Choose a model.
   - If LM Studio was started after WebUI, click `Refresh LM Studio models`.
6. Adjust sampling options (`Temperature`, `Max tokens`, `Top P`) if needed.
7. Click `Get Prompt`.
8. Click `Insert into Prompt` to send text into the main prompt field.

## Notes

- Maximum number of images per request is limited in code (`MAX_IMAGES = 30`).
- Images are automatically downscaled/compressed before upload according to settings.
- If API key is missing, the extension returns an explicit error in output.
- Gemini choices: `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`, and `gemini-2.5-flash-lite`.
- LM Studio models are loaded from `/v1/models` and requests are sent to `/v1/chat/completions`.
- Image support depends on the selected LM Studio model. Text-only local models may reject image input.
- `Prompt to improve` is appended to the selected preset/instruction before the request is sent.

## License

See [LICENSE.md](LICENSE.md).
