# Mistral + Gemini Prompt Extension

Extension for Forge/Stable Diffusion WebUI that generates prompts from images using Mistral (`pixtral-large-latest`) or Gemini (`gemini-2.5-flash`, `gemini-2.5-pro`).

## What It Does

- Uploads multiple images via drag-and-drop or file picker
- Supports direct paste from clipboard
- Shows a custom preview gallery with per-image delete, remove-last, and clear-all actions
- Sends text and optional images plus your prompt template to the selected model
- Supports Mistral and Gemini model selection from the same UI
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

## Setup

1. Place this extension in your Forge extensions directory.
2. Restart WebUI.
3. Open Settings -> `Mistral++`.
4. Set:
   - `Mistral API Key` (required)
   - `Gemini API Key` (required for Gemini models)
   - Optional image limits:
     - `Max image size sent to Mistral (longest side, px)`
     - `Max JPEG size sent to Mistral (KB)`
5. Apply settings and reload UI if needed.

## How To Use

1. Open the `Mistral++` accordion in `txt2img` or `img2img`.
2. Add images (drag-and-drop, click to select, or `Paste from clipboard`).
3. Choose or edit an initial preset prompt.
4. Choose a model.
5. Adjust sampling options (`Temperature`, `Max tokens`, `Top P`) if needed.
6. Click `Get Prompt`.
7. Click `Insert into Prompt` to send text into the main prompt field.

## Notes

- Maximum number of images per request is limited in code (`MAX_IMAGES = 30`).
- Images are automatically downscaled/compressed before upload according to settings.
- If API key is missing, the extension returns an explicit error in output.

## License

See [LICENSE.md](LICENSE.md).
