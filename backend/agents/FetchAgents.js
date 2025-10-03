// FetchAgents.js
const fetch = require("node-fetch");

async function callLocal(endpoint, text) {
  try {
    const res = await fetch(`http://localhost:8000/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.output || "No info.";
  } catch (err) {
    console.error(`Local agent ${endpoint} error:`, err);
    return "Could not fetch data.";
  }
}

module.exports = async function fetchAgents(query) {
  const [research, trials, patents, market, nlp] = await Promise.all([
    callLocal("research", query),
    callLocal("trials", query),
    callLocal("patents", query),
    callLocal("market", query),
    callLocal("nlp", query),
  ]);

  return {
    research,
    trials,
    patents,
    market,
    nlp,
  };
};
