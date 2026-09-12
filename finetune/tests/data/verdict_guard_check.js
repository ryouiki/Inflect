// Exercise the page's own verdict-export script under a minimal DOM stub.
// Three claims: a blank field blocks the first click, a second click with no
// edit in between goes through, and any edit re-arms the guard.
const fs = require("fs");
const vm = require("vm");

const page = fs.readFileSync(process.argv[2], "utf8");
const script = page.split("<script>")[1].split("</script>")[0];

let clicks = 0;

function field(row, fieldName, letter, value) {
  return {
    dataset: letter ? { row, field: fieldName, letter } : { row, field: fieldName },
    value,
    classes: new Set(),
    classList: {
      toggle(name, on) {
        if (on) this.owner.classes.add(name);
        else this.owner.classes.delete(name);
      },
    },
    scrollIntoView() {},
  };
}

function makeField(row, fieldName, letter, value) {
  const f = field(row, fieldName, letter, value);
  f.classList.owner = f;
  return f;
}

function buildDom() {
  const rows = ["0001", "0002"].map(id => {
    const fields = [
      makeField(id, "quality", "A", "3"),
      makeField(id, "defect", "A", "1"),
      makeField(id, "language", "A", "예"),
      makeField(id, "most_natural", null, "A"),
      makeField(id, "most_blurred", null, "A"),
      makeField(id, "comment", null, "말이 된다"),
    ];
    return { id: "row-" + id, fields, querySelectorAll: () => fields };
  });
  const all = rows.flatMap(r => r.fields);
  const status = { textContent: "" };
  const listeners = { input: [], change: [] };
  const document = {
    querySelectorAll(selector) {
      if (selector === "section.row") return rows;
      if (selector === "[data-field]") return all;
      return [];
    },
    querySelector(selector) {
      if (selector === ".blank") return all.find(f => f.classes.has("blank")) || null;
      return null;
    },
    getElementById(id) {
      if (id === "status") return status;
      if (id === "save") return saveButton;
      return null;
    },
    createElement() {
      return { href: "", download: "", click() { clicks += 1; } };
    },
    addEventListener(type, handler) {
      if (listeners[type]) listeners[type].push(handler);
    },
  };
  const saveButton = { handler: null, addEventListener(_type, handler) { this.handler = handler; } };
  return { document, rows, all, status, listeners, click: () => saveButton.handler(), saveButton };
}

function run(dom) {
  const store = {};
  const context = {
    document: dom.document,
    localStorage: {
      getItem: key => (key in store ? store[key] : null),
      setItem: (key, value) => { store[key] = value; },
    },
    Blob: class { constructor() {} },
    URL: { createObjectURL: () => "blob:stub" },
    JSON,
    console,
  };
  vm.createContext(context);
  vm.runInContext(script, context);
  return context;
}

function check(name, condition, detail) {
  const mark = condition ? "ok  " : "FAIL";
  console.log("  " + mark + " " + name + ": " + detail);
  if (!condition) process.exitCode = 1;
}

// --- case 1: every field filled -> the first click downloads ---
let dom = buildDom();
run(dom);
clicks = 0;
dom.click();
check("a complete page downloads on the first click", clicks === 1, "clicks=" + clicks + ", status=" + JSON.stringify(dom.status.textContent));

// --- case 2: one blank axis cell -> first click refused, second goes through ---
dom = buildDom();
dom.all.find(f => f.dataset.field === "language").value = "";
run(dom);
clicks = 0;
dom.click();
const refused = clicks === 0;
const marked = dom.all.filter(f => f.classes.has("blank")).length;
check("a blank cell blocks the first click", refused, "clicks=" + clicks + ", status=" + JSON.stringify(dom.status.textContent));
check("the blank cell is marked", marked === 1, marked + " field(s) carry .blank");
dom.click();
check("a second click with no edit downloads", clicks === 1, "clicks=" + clicks);

// --- case 3: an edit between the two clicks re-arms the guard ---
dom = buildDom();
dom.all.find(f => f.dataset.field === "comment").value = "";
run(dom);
clicks = 0;
dom.click();
check("blank free text blocks the first click too", clicks === 0, "clicks=" + clicks);
dom.listeners.input.forEach(handler => handler());
dom.click();
check("an edit re-arms the guard", clicks === 0, "clicks=" + clicks + " after the edit");
dom.click();
check("and the click after that goes through", clicks === 1, "clicks=" + clicks);
