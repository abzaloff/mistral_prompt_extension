const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class TextArea {
    constructor() {
        this.events = [];
        this.value = "";
    }

    dispatchEvent(event) {
        this.events.push(event.type);
    }

    focus() {}
}

const fields = {
    txt2img: new TextArea(),
    img2img: new TextArea(),
};
const clicks = {txt2img: 0, img2img: 0};
const buttons = Object.fromEntries(
    Object.keys(clicks).map((tab) => [tab, {
        disabled: false,
        click() { clicks[tab] += 1; },
    }]),
);
const app = {
    querySelector(selector) {
        for (const tab of Object.keys(fields)) {
            if (selector === `#${tab}_prompt textarea`) return fields[tab];
            if (selector === `#${tab}_generate`) return buttons[tab];
        }
        return null;
    },
};
const frames = [];
const window = {gradioApp: () => app};
const context = {
    window,
    document: app,
    gradioApp: () => app,
    HTMLTextAreaElement: TextArea,
    HTMLInputElement: TextArea,
    Event: class Event { constructor(type) { this.type = type; } },
    requestAnimationFrame: (callback) => frames.push(callback),
    onUiLoaded() {},
    onAfterUiUpdate() {},
    console,
};
const source = fs.readFileSync(path.join(__dirname, "../javascript/mistral_prompt.js"), "utf8");
vm.runInNewContext(source, context);

function runFrames() {
    while (frames.length) frames.shift()();
}

assert.equal(window.mistralPromptQuickGenerate("Ignored", false, "txt2img"), "Ignored");
assert.equal(fields.txt2img.value, "");
assert.equal(clicks.txt2img, 0);

assert.equal(window.mistralPromptQuickGenerate("Error: provider failed", true, "txt2img"), "Error: provider failed");
assert.equal(fields.txt2img.value, "");
assert.equal(clicks.txt2img, 0);

assert.equal(window.mistralPromptQuickGenerate("  ", true, "txt2img"), "  ");
assert.equal(fields.txt2img.value, "");
assert.equal(clicks.txt2img, 0);

assert.equal(window.mistralPromptQuickGenerate("A red fox", true, "txt2img"), "A red fox");
assert.equal(fields.txt2img.value, "A red fox");
assert.deepEqual(fields.txt2img.events, ["input", "change"]);
assert.equal(clicks.txt2img, 0);
runFrames();
assert.equal(clicks.txt2img, 1);

assert.equal(window.mistralPromptQuickGenerate("A blue bird", true, "img2img"), "A blue bird");
runFrames();
assert.equal(fields.img2img.value, "A blue bird");
assert.equal(clicks.img2img, 1);
assert.equal(clicks.txt2img, 1);

assert.equal(window.mistralPromptInsert("Manual text", "txt2img"), "Manual text");
runFrames();
assert.equal(fields.txt2img.value, "Manual text");
assert.equal(clicks.txt2img, 1);

console.log("Quick mode behavior passed");
