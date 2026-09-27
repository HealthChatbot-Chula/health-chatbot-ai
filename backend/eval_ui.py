def simulator_page() -> str:
    return """
<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Health Chatbot Eval Simulator</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101113;
      --panel: #181a1f;
      --muted: #9ca3af;
      --text: #f4f4f5;
      --patient: #2563eb;
      --bot: #2f343d;
      --judge: #172033;
      --border: #30343d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(24, 26, 31, .92);
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    h1 { font-size: 18px; margin: 0 10px 0 0; }
    select, button {
      border: 1px solid var(--border);
      background: #20232a;
      color: var(--text);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 14px;
    }
    button {
      background: #2563eb;
      border-color: #2563eb;
      cursor: pointer;
      font-weight: 700;
    }
    button:disabled { opacity: .55; cursor: wait; }
    main {
      padding: 20px;
      max-width: 980px;
      width: 100%;
      margin: 0 auto;
    }
    .hint {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 16px;
    }
    .row {
      display: flex;
      margin: 14px 0;
    }
    .row.patient { justify-content: flex-end; }
    .row.bot, .row.judge { justify-content: flex-start; }
    .bubble {
      max-width: min(78%, 720px);
      padding: 13px 15px;
      border-radius: 18px;
      line-height: 1.58;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }
    .message-body {
      white-space: pre-wrap;
    }
    .bot-bullets {
      margin: 0;
      padding-left: 1.2rem;
      white-space: normal;
    }
    .bot-bullets li {
      margin: 0 0 10px;
    }
    .bot-bullets li:last-child {
      margin-bottom: 0;
    }
    .patient .bubble {
      background: var(--patient);
      border-bottom-right-radius: 5px;
    }
    .bot .bubble {
      background: var(--bot);
      border-bottom-left-radius: 5px;
    }
    .judge .bubble {
      background: var(--judge);
      border: 1px solid #334155;
      max-width: min(90%, 820px);
    }
    .label {
      font-size: 12px;
      font-weight: 800;
      opacity: .78;
      margin-bottom: 6px;
    }
    .meta {
      color: rgba(255,255,255,.72);
      font-size: 12px;
      margin-top: 8px;
    }
    .status {
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0;
    }
    .typing-body {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: rgba(255,255,255,.78);
    }
    .typing-dots {
      display: inline-flex;
      gap: 4px;
      align-items: center;
    }
    .typing-dots span {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: rgba(255,255,255,.7);
      animation: typingPulse 1s ease-in-out infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: .16s; }
    .typing-dots span:nth-child(3) { animation-delay: .32s; }
    @keyframes typingPulse {
      0%, 80%, 100% { opacity: .35; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-3px); }
    }
    .score-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .score {
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      padding: 8px;
    }
    footer {
      color: var(--muted);
      font-size: 12px;
      padding: 10px 20px 16px;
      text-align: center;
    }
  </style>
</head>
<body>
  <header>
    <h1>Health Eval Simulator</h1>
    <select id="caseSelect"></select>
    <button id="runBtn">Run Simulation</button>
  </header>
  <main>
    <div class="hint">Patient Simulator และ Health Chatbot จะแสดงเป็น bubble ต่อ bubble ส่วน Judge จะแสดงคะแนนท้ายบทสนทนา</div>
    <div id="chat"></div>
  </main>
  <footer>Local eval view powered by the same backend as OpenWebUI.</footer>
  <script>
    const caseSelect = document.querySelector("#caseSelect");
    const runBtn = document.querySelector("#runBtn");
    const chat = document.querySelector("#chat");
    let source = null;
    let typingRow = null;
    let lastStatusText = "";
    let lastStatusNode = null;

    function addStatus(text) {
      const normalized = text.trim();
      if (lastStatusNode && lastStatusText === normalized) {
        lastStatusNode.scrollIntoView({ behavior: "smooth", block: "end" });
        return;
      }
      const div = document.createElement("div");
      div.className = "status";
      div.textContent = normalized;
      chat.appendChild(div);
      lastStatusText = normalized;
      lastStatusNode = div;
      div.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function resetStatusDedupe() {
      lastStatusText = "";
      lastStatusNode = null;
    }

    function botBulletItems(content) {
      const items = [];
      const politeClosers = new Set([
        "\u0e04\u0e23\u0e31\u0e1a",
        "\u0e04\u0e48\u0e30",
        "\u0e04\u0e30",
        "\u0e19\u0e30\u0e04\u0e23\u0e31\u0e1a",
        "\u0e19\u0e30\u0e04\u0e30",
        "\u0e19\u0e30\u0e04\u0e48\u0e30",
      ]);

      content
        .split(/\\n+/)
        .map(item => item.trim())
        .filter(Boolean)
        .map(item => item.replace(/^[-*•]\\s+/, "").replace(/^\\d+[.)]\\s+/, "").trim())
        .forEach(item => {
          if (politeClosers.has(item) && items.length) {
            items[items.length - 1] = `${items[items.length - 1]} ${item}`;
          } else {
            items.push(item);
          }
        });

      return items;
    }

    function appendMessageBody(bubble, role, content) {
      const body = document.createElement("div");
      body.className = "message-body";

      const bulletItems = role === "bot" ? botBulletItems(content) : [];
      if (bulletItems.length > 1) {
        const list = document.createElement("ul");
        list.className = "bot-bullets";
        bulletItems.forEach(item => {
          const li = document.createElement("li");
          li.textContent = item;
          list.appendChild(li);
        });
        body.appendChild(list);
      } else {
        body.textContent = content;
      }

      bubble.appendChild(body);
    }

    function removeTyping() {
      if (typingRow) {
        typingRow.remove();
        typingRow = null;
      }
    }

    function addTyping() {
      resetStatusDedupe();
      if (typingRow) {
        typingRow.scrollIntoView({ behavior: "smooth", block: "end" });
        return;
      }
      removeTyping();
      const row = document.createElement("div");
      row.className = "row bot typing";
      const bubble = document.createElement("div");
      bubble.className = "bubble";

      const label = document.createElement("div");
      label.className = "label";
      label.textContent = "Health Chatbot";

      const body = document.createElement("div");
      body.className = "typing-body";
      const text = document.createElement("span");
      text.textContent = "กำลังพิมพ์";
      const dots = document.createElement("span");
      dots.className = "typing-dots";
      dots.innerHTML = "<span></span><span></span><span></span>";

      body.appendChild(text);
      body.appendChild(dots);
      bubble.appendChild(label);
      bubble.appendChild(body);
      row.appendChild(bubble);
      chat.appendChild(row);
      typingRow = row;
      row.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function addBubble(role, content, meta) {
      removeTyping();
      resetStatusDedupe();
      const row = document.createElement("div");
      row.className = `row ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = role === "patient" ? "Patient Simulator" : role === "bot" ? "Health Chatbot" : "LLM Judge";
      bubble.appendChild(label);
      appendMessageBody(bubble, role, content);
      if (meta) {
        const metaNode = document.createElement("div");
        metaNode.className = "meta";
        metaNode.textContent = meta;
        bubble.appendChild(metaNode);
      }
      row.appendChild(bubble);
      chat.appendChild(row);
      row.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function addJudge(data) {
      const row = document.createElement("div");
      row.className = "row judge";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = "LLM Judge";
      bubble.appendChild(label);

      const summary = document.createElement("div");
      summary.innerHTML = `<strong>Pass:</strong> ${data.pass} &nbsp; <strong>Fatal:</strong> ${data.fatal_error} &nbsp; <strong>Overall:</strong> ${data.overall_score}/5 &nbsp; <strong>Flow:</strong> ${data.conversation_behavior_score || "-"}/5 &nbsp; <strong>Final:</strong> ${data.final_answer_score || "-"}/5`;
      bubble.appendChild(summary);

      const grid = document.createElement("div");
      grid.className = "score-grid";
      const keys = ["clinical_correctness", "safety_triage", "scope_control", "groundedness", "completeness", "context_use", "clarity", "empathy_tone"];
      keys.forEach(key => {
        const item = document.createElement("div");
        item.className = "score";
        item.textContent = `${key}: ${data[key]}/5`;
        grid.appendChild(item);
      });
      bubble.appendChild(grid);

      if (Array.isArray(data.checkpoint_results) && data.checkpoint_results.length) {
        const checkpoints = document.createElement("div");
        checkpoints.className = "score-grid";
        data.checkpoint_results.forEach(result => {
          const item = document.createElement("div");
          item.className = "score";
          const status = result.pass ? "PASS" : "FAIL";
          item.textContent = `${status} ${result.key}: ${result.reason || ""}`;
          if (result.evidence) item.title = result.evidence;
          checkpoints.appendChild(item);
        });
        bubble.appendChild(checkpoints);
      }

      const reason = document.createElement("div");
      reason.textContent = data.reason || "";
      bubble.appendChild(reason);
      row.appendChild(bubble);
      chat.appendChild(row);
      row.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    async function loadCases() {
      const res = await fetch("/eval/simulator/cases");
      const cases = await res.json();
      caseSelect.innerHTML = "";
      cases.forEach(item => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.id} (${item.risk_level})`;
        caseSelect.appendChild(option);
      });
    }

    function runSimulation() {
      if (source) source.close();
      chat.innerHTML = "";
      runBtn.disabled = true;
      addStatus("Starting simulation...");
      source = new EventSource(`/eval/simulator/stream/${encodeURIComponent(caseSelect.value)}`);

      source.addEventListener("patient", event => {
        addBubble("patient", JSON.parse(event.data).content);
      });
      source.addEventListener("bot", event => {
        const data = JSON.parse(event.data);
        addBubble("bot", data.content, `Latency: ${data.latency_ms} ms`);
      });
      source.addEventListener("status", event => {
        const status = JSON.parse(event.data).content;
        if (/health chatbot is responding/i.test(status)) {
          addTyping();
        } else {
          addStatus(status);
        }
      });
      source.addEventListener("judge", event => {
        removeTyping();
        addJudge(JSON.parse(event.data));
      });
      source.addEventListener("error_event", event => {
        removeTyping();
        addStatus(`Error: ${JSON.parse(event.data).content}`);
      });
      source.addEventListener("done", () => {
        removeTyping();
        addStatus("Done");
        source.close();
        runBtn.disabled = false;
      });
      source.onerror = () => {
        removeTyping();
        addStatus("Connection closed");
        runBtn.disabled = false;
        if (source) source.close();
      };
    }

    runBtn.addEventListener("click", runSimulation);
    loadCases().catch(error => addStatus(error.message));
  </script>
</body>
</html>
"""
