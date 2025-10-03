// SimplifierAgent.js
const fetch = require("node-fetch");

module.exports = async function SimplifierAgent(text, mode = "simplify") {
  try {
    const endpoint = mode === "paraphrase" ? "paraphrase" : "simplify";
    const response = await fetch(`http://localhost:8000/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) throw new Error(`HTTP error! ${response.status}`);

    const data = await response.json();
    return data.output || "Could not process text.";
  } catch (err) {
    console.error("SimplifierAgent error:", err);
    return "Could not process text.";
  }
};
