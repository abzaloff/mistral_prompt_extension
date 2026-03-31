# Mistral Prompt Extension

Extension for Forge/Stable Diffusion WebUI that generates prompts from images using the Mistral API (`pixtral-large-latest`).

## What It Does

- Uploads multiple images via drag-and-drop or file picker
- Supports direct paste from clipboard
- Shows a custom preview gallery with per-image delete, remove-last, and clear-all actions
- Sends images plus your prompt template to Mistral and returns a generated prompt
- Lets you append extra text to the generated prompt
- Inserts the result directly into `txt2img` or `img2img` prompt field
- Includes editable prompt presets stored in WebUI settings

## Requirements

- Forge / Stable Diffusion WebUI with extension support
- Python dependencies available in your WebUI environment:
  - `requests`
  - `Pillow`
- **Mistral API key is required**

## Setup

1. Place this extension in your Forge extensions directory.
2. Restart WebUI.
3. Open Settings -> `Mistral Prompt Generator`.
4. Set:
   - `Mistral API Key` (required)
   - Optional image limits:
     - `Max image size sent to Mistral (longest side, px)`
     - `Max JPEG size sent to Mistral (KB)`
5. Apply settings and reload UI if needed.

## How To Use

1. Open the `Mistral Prompt` accordion in `txt2img` or `img2img`.
2. Add images (drag-and-drop, click to select, or `Paste from clipboard`).
3. Choose or edit an initial preset prompt.
4. Adjust sampling options (`Temperature`, `Max tokens`, `Top P`) if needed.
5. Click `Get Prompt from Mistral`.
6. Click `Insert into Prompt` to send text into the main prompt field.

## Notes

- Maximum number of images per request is limited in code (`MAX_IMAGES = 30`).
- Images are automatically downscaled/compressed before upload according to settings.
- If API key is missing, the extension returns an explicit error in output.
