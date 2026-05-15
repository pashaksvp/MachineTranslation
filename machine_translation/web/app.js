const sourceText = document.querySelector("#sourceText");
const targetText = document.querySelector("#targetText");
const statusBox = document.querySelector("#status");
const clearButton = document.querySelector("#clearButton");
const copyButton = document.querySelector("#copyButton");

let timer = null;
let requestId = 0;

function setStatus(text, active = false) {
  statusBox.textContent = text;
  statusBox.classList.toggle("active", active);
}

async function translate() {
  const text = sourceText.value.trim();
  const currentRequest = ++requestId;
  targetText.textContent = "";

  if (!text) {
    setStatus("Ready");
    return;
  }

  setStatus("Streaming", true);
  const response = await fetch("/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done || currentRequest !== requestId) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      if (!event.startsWith("data: ")) continue;
      const payload = JSON.parse(event.slice(6));
      if (payload.error) {
        targetText.textContent = payload.error;
        setStatus("Error");
      } else if (payload.done) {
        setStatus("Done");
      } else if (payload.token) {
        targetText.textContent += payload.token;
      }
    }
  }
}

sourceText.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(translate, 500);
});

clearButton.addEventListener("click", () => {
  sourceText.value = "";
  targetText.textContent = "";
  setStatus("Ready");
  requestId += 1;
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(targetText.textContent);
  setStatus("Copied");
});
