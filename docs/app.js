const form = document.getElementById("predictionForm");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const urlInput = document.getElementById("urlInput");
const textInput = document.getElementById("textInput");
const htmlInput = document.getElementById("htmlInput");
const sampleButtons = document.querySelectorAll("[data-sample]");

const samples = {
  legitimate: {
    url: "https://accounts.google.com/signin",
    text: "Welcome back. Sign in to continue securely to your account dashboard. Review your saved devices and manage privacy settings.",
    html: "<div><form action='/signin'><input type='email'><input type='password'></form><a href='/support'>Help Center</a></div>",
  },
  phishing: {
    url: "http://paypal-secure-login-account-update.verify-user-alert.com/login",
    text: "Urgent account verification required. Confirm your bank password immediately to avoid suspension. Secure your wallet and verify now.",
    html: "<iframe src='http://unknown.example'></iframe><form action='mailto:fake@alert-mail.com'><input type='password'></form><script>window.open('promo')</script>",
  },
  mixed: {
    url: "https://offers-example-shop.xyz/redirect/login",
    text: "Special promotional access. Sign in to confirm your profile and continue to the exclusive redirect page with limited-time rewards.",
    html: "<div onmouseover='status=true'><a href='http://redirect.example'>Continue</a><script>window.open('offer')</script></div>",
  },
};

function titleize(value) {
  return value.replaceAll("_", " ");
}

function defaultValueForField(field) {
  return "";
}

async function loadSchema() {
  const response = await fetch("/api/schema");
  const data = await response.json();

  for (const field of data.fields) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.htmlFor = field;
    label.textContent = titleize(field);

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = field;
    input.name = field;
    input.value = defaultValueForField(field);
    input.placeholder = "auto";

    wrapper.append(label, input);
    form.appendChild(wrapper);
  }
}

function applySample(sampleKey) {
  const sample = samples[sampleKey];
  if (!sample) {
    return;
  }

  urlInput.value = sample.url;
  textInput.value = sample.text;
  htmlInput.value = sample.html;
}

async function loadSummary() {
  const response = await fetch("/api/summary");
  const data = await response.json();

  document.getElementById("accuracyMetric").textContent =
    data.classification_metrics?.accuracy?.toFixed(3) ?? "--";
  document.getElementById("r2Metric").textContent =
    data.regression_metrics?.r2?.toFixed(3) ?? "--";
  document.getElementById("recordsMetric").textContent =
    data.dataset_records?.toLocaleString() ?? "--";
}

function renderRecommendations(items) {
  const container = document.getElementById("recommendations");
  container.innerHTML = "";

  if (!items?.length) {
    container.textContent = "No recommendations available.";
    return;
  }

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    card.innerHTML = `
      <span>Match ${item.rank}</span>
      <strong>${item.label} webpage</strong>
      <p>Dataset row ${item.dataset_index} · Similarity ${item.similarity.toFixed(3)} · Traffic ${item.web_traffic.toFixed(2)} · ${item.match_signal}</p>
    `;
    container.appendChild(card);
  }
}

function renderFindings(items) {
  const container = document.getElementById("findings");
  container.innerHTML = "";

  if (!items?.length) {
    container.innerHTML = `<article class="finding-card medium"><strong>No major suspicious components detected</strong><p>The extracted URL and content signals did not surface strong phishing-specific components.</p></article>`;
    return;
  }

  for (const item of items) {
    const card = document.createElement("article");
    card.className = `finding-card ${item.severity}`;
    card.innerHTML = `
      <strong>${item.description}</strong>
      <p>Feature: ${titleize(item.feature)} · Severity: ${item.severity}</p>
    `;
    container.appendChild(card);
  }
}

function renderNotes(items, sourceMeta) {
  const container = document.getElementById("notes");
  container.innerHTML = "";

  const notes = [...(items || [])];
  if (sourceMeta?.page_text_length) {
    notes.push(`Processed content length: ${sourceMeta.page_text_length.toLocaleString()} characters.`);
  }

  if (!notes.length) {
    container.innerHTML = `<article class="note-card"><strong>No additional notes</strong><p>The analysis completed without extra extraction notes.</p></article>`;
    return;
  }

  for (const note of notes) {
    const card = document.createElement("article");
    card.className = "note-card";
    card.innerHTML = `<strong>Observation</strong><p>${note}</p>`;
    container.appendChild(card);
  }
}

async function runPrediction() {
  const inputs = form.querySelectorAll("input");
  const overrides = {};

  for (const input of inputs) {
    if (input.value !== "") {
      overrides[input.name] = Number(input.value);
    }
  }

  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url: urlInput.value.trim(),
      text: textInput.value.trim(),
      html: htmlInput.value.trim(),
      overrides,
    }),
  });
  const data = await response.json();

  const badge = document.getElementById("labelBadge");
  badge.textContent = data.classification_label;
  badge.className = `status-badge ${data.classification_label === "Legitimate" ? "safe" : "risk"}`;

  document.getElementById("probabilityText").textContent =
    `Probability: ${(data.classification_probability * 100).toFixed(2)}%`;
  document.getElementById("trafficText").textContent =
    data.web_traffic_prediction.toFixed(3);
  document.getElementById("clusterText").textContent =
    `Cluster ${data.cluster_id}`;

  renderFindings(data.findings);
  renderNotes(data.notes, data.source_meta);
  renderRecommendations(data.similar_webpages);
}

function clearInputs() {
  urlInput.value = "";
  textInput.value = "";
  htmlInput.value = "";

  for (const input of form.querySelectorAll("input")) {
    input.value = "";
  }
}

predictButton.addEventListener("click", runPrediction);
clearButton.addEventListener("click", clearInputs);
for (const button of sampleButtons) {
  button.addEventListener("click", () => applySample(button.dataset.sample));
}

Promise.all([loadSchema(), loadSummary()]);
