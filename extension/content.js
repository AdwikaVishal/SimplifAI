// content.js
let tooltip = null;

function createTooltip() {
    tooltip = document.createElement("div");
    tooltip.id = "aiTooltip";
    tooltip.style.position = "absolute";
    tooltip.style.backgroundColor = "#fff";
    tooltip.style.border = "1px solid #ccc";
    tooltip.style.padding = "10px";
    tooltip.style.borderRadius = "8px";
    tooltip.style.boxShadow = "0px 2px 10px rgba(0,0,0,0.2)";
    tooltip.style.zIndex = "9999";
    tooltip.style.maxWidth = "350px";
    tooltip.style.maxHeight = "300px";
    tooltip.style.overflowY = "auto";
    tooltip.style.fontFamily = "Arial, sans-serif";
    tooltip.style.fontSize = "14px";
    tooltip.style.lineHeight = "1.5";
    tooltip.style.color = "#000";
    tooltip.style.whiteSpace = "normal";
    tooltip.style.display = "none";
    document.body.appendChild(tooltip);
}
createTooltip();

function markdownToHtml(md) {
    if (!md) return "";
    // Basic escape
    const escaped = md
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Bold: **text**
    const withBold = escaped.replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>");

    // Convert lines to paragraphs and lists
    const lines = withBold.split(/\r?\n/);
    let html = "";
    let inList = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith("- ")) {
            if (!inList) {
                html += "<ul style=\"margin:0 0 6px 18px; padding:0;\">";
                inList = true;
            }
            const item = line.slice(2).trim();
            html += `<li>${item}</li>`;
        } else if (line.length === 0) {
            if (inList) {
                html += "</ul>";
                inList = false;
            }
        } else {
            if (inList) {
                html += "</ul>";
                inList = false;
            }
            html += `<p style=\"margin:0 0 8px 0;\">${line}</p>`;
        }
    }
    if (inList) html += "</ul>";
    return html || "";
}

function showTooltip(text, x, y) {
    tooltip.innerHTML = markdownToHtml(text);
    tooltip.style.left = x + "px";
    tooltip.style.top = y + 20 + "px";
    tooltip.style.display = "block";
}

function hideTooltip() {
    tooltip.style.display = "none";
}

async function fetchAI(text, mode = "simplify") {
    try {
        const response = await fetch("http://localhost:8000/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, mode })
        });
        if (!response.ok) throw new Error("HTTP " + response.status);
        const data = await response.json();
        return data.output || "No explanation available.";
    } catch (err) {
        console.error("AI fetch error:", err);
        return "Error fetching AI explanation.";
    }
}

document.addEventListener("mouseup", async (event) => {
    const selectedText = window.getSelection().toString().trim();
    hideTooltip();
    if (!selectedText) return;

    const x = event.pageX;
    const y = event.pageY;

    showTooltip("AI is processing...", x, y);

    const explanation = await fetchAI(selectedText, "simplify");
    showTooltip(explanation, x, y);
});

document.addEventListener("mousedown", () => hideTooltip());
