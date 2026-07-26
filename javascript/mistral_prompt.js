(function () {
    "use strict";

    function appRoot() {
        try {
            return window.gradioApp ? gradioApp() : document;
        } catch (_) {
            return document;
        }
    }

    function setGradioValue(field, value) {
        const prototype = field instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;

        if (setter) {
            setter.call(field, value);
        } else {
            field.value = value;
        }
        field.dispatchEvent(new Event("input", {bubbles: true}));
    }

    function handleGalleryDelete(event) {
        const target = event.target;
        const button = target instanceof Element
            ? target.closest(".mp-delete-btn")
            : null;
        if (!button) {
            return;
        }

        const pipeId = button.dataset.mpDeletePipeId;
        const index = button.dataset.mpDeleteIndex;
        if (!pipeId || index === undefined) {
            return;
        }

        const app = appRoot();
        const pipeContainer = app.getElementById
            ? app.getElementById(pipeId)
            : document.getElementById(pipeId);
        const field = pipeContainer?.querySelector("textarea, input");
        if (!field) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        setGradioValue(field, index);
    }

    function setupDeleteHandler() {
        if (window.__mistralPromptDeleteReady) {
            return;
        }

        appRoot().addEventListener("click", handleGalleryDelete, true);
        window.__mistralPromptDeleteReady = true;
    }

    function setupDrop(drop) {
        if (drop.dataset.mpDragReady === "true") {
            return;
        }
        drop.dataset.mpDragReady = "true";

        const prevent = (event) => {
            event.preventDefault();
            event.stopPropagation();
        };
        ["dragenter", "dragover"].forEach((eventName) => {
            drop.addEventListener(eventName, (event) => {
                prevent(event);
                drop.classList.add("dragover");
            });
        });
        ["dragleave", "drop"].forEach((eventName) => {
            drop.addEventListener(eventName, (event) => {
                prevent(event);
                drop.classList.remove("dragover");
            });
        });

        const ensureHeights = () => {
            const boxes = drop.querySelectorAll(".wrap, .file-wrap, .border, .container");
            boxes.forEach((box) => {
                box.style.height = "100%";
                box.style.minHeight = "100%";
            });
        };
        ensureHeights();
        new MutationObserver(ensureHeights).observe(drop, {
            subtree: true,
            childList: true,
            attributes: true,
        });
    }

    function setupDropZones() {
        appRoot().querySelectorAll(".mp-drop").forEach(setupDrop);
    }

    onUiLoaded(() => {
        setupDeleteHandler();
        setupDropZones();
    });
    onAfterUiUpdate(setupDropZones);
})();
