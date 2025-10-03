// MasterAgent.js
const SimplifierAgent = require("./SimplifierAgent");
const fetchAgents = require("./FetchAgents"); // All external data-fetching functions in one file

module.exports = async function MasterAgent(query, mode = "simplify") {
  try {
    // Fetch raw data from all sources
    const rawResults = await fetchAgents(query);

    // Combine everything into one text
    const combinedText = Object.entries(rawResults)
      .map(([k, v]) => `${k.toUpperCase()}:\n${v}`)
      .join("\n\n");

    // Run through ML model for simplification/paraphrasing
    const processed = await SimplifierAgent(combinedText, mode);
    return processed;
  } catch (err) {
    console.error("MasterAgent error:", err);
    return "Could not process query.";
  }
};
