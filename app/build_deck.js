const pptxgen = require("pptxgenjs");

// Palette — content-informed, tied to the dashboard's own identity and to the
// risk/value subject itself (deep violet = authority/brand, red = risk, green = safe)
const INK = "17151F";
const INK_SOFT = "55506B";
const VIOLET = "392C85";
const VIOLET_SOFT = "4A3AA7";
const LAVENDER = "F1EFF6";
const WHITE = "FFFFFF";
const RED = "D03B3B";
const RED_SOFT = "FBE6E6";
const GREEN = "0CA30C";
const GREEN_SOFT = "E3F6E3";
const AMBER = "FAB219";
const AMBER_SOFT = "FDF0D9";
const ORANGE = "EB6834";
const BLUE = "2A78D6";

const HEAD_FONT = "Cambria";
const BODY_FONT = "Calibri";

function newPres() {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  return p;
}

function darkSlide(p) {
  const s = p.addSlide();
  s.background = { color: VIOLET };
  return s;
}
function lightSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  return s;
}

function kicker(s, text, color) {
  s.addText(text.toUpperCase(), {
    x: 0.6, y: 0.5, w: 8, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: color || VIOLET_SOFT,
    charSpacing: 2,
  });
}
function title(s, text, opts) {
  opts = opts || {};
  s.addText(text, {
    x: 0.6, y: opts.y || 0.85, w: opts.w || 10.6, h: opts.h || 1.0, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: opts.size || 32, bold: true, color: opts.color || INK,
    valign: "top",
  });
}
function pageNum(s, n, color) {
  s.addText(String(n).padStart(2, "0"), {
    x: 12.5, y: 7.05, w: 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 10, color: color || INK_SOFT, align: "right",
  });
}
function statTile(s, x, y, w, h, value, label, opts) {
  opts = opts || {};
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || LAVENDER }, line: { type: "none" } });
  s.addText(value, { x: x + 0.2, y: y + 0.14, w: w - 0.4, h: h * 0.6, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: opts.valueSize || 26, bold: true, color: opts.valueColor || VIOLET });
  s.addText(label, { x: x + 0.2, y: y + h - 0.42, w: w - 0.4, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 11, color: INK_SOFT });
}

const pres = newPres();

// ---------------- Slide 1: Title ----------------
(() => {
  const s = darkSlide(pres);
  s.addText("RETENTION SIGNAL", {
    x: 0.8, y: 2.15, w: 11.7, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: "CABFFF", charSpacing: 3,
  });
  s.addText("Crossing churn risk with subscription value", {
    x: 0.8, y: 2.7, w: 11.7, h: 1.6, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 44, bold: true, color: WHITE, valign: "top",
  });
  s.addText("Who's actually worth spending a retention budget on — a real subscriber base, real transactions, real dollars.", {
    x: 0.8, y: 4.45, w: 9.5, h: 0.7, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 16, color: "D8D2F0", italic: true,
  });
  s.addText("268,828 subscribers scored  ·  music-streaming subscription base  ·  held-out validation snapshot", {
    x: 0.8, y: 6.6, w: 11, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, color: "9686E6",
  });
})();

// ---------------- Slide 2: The problem ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "The problem");
  title(s, "A churn score alone isn't a spending plan");
  s.addText(
    "Retention budget usually gets spent reactively — on whoever looks about to leave, regardless of whether they were ever paying much. A churn model tells you who's leaving. It doesn't tell you who's worth paying to keep.",
    { x: 0.6, y: 1.95, w: 6.6, h: 1.9, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, color: INK_SOFT, valign: "top" }
  );
  s.addText(
    "The real output of this project is churn risk crossed with subscription value — so spend goes where it returns.",
    { x: 0.6, y: 4.0, w: 6.6, h: 1.0, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 18, bold: true, color: VIOLET, italic: true, valign: "top" }
  );
  statTile(s, 7.7, 2.0, 5.0, 1.7, "39.7%", "of subscribers in this base actually cancel (held-out validation set)", { valueSize: 40 });
  statTile(s, 7.7, 3.9, 5.0, 1.7, "2.7%", "of subscribers account for a wildly outsized share of spend — and churn at 81.9%", { valueSize: 40, fill: RED_SOFT, valueColor: RED });
  pageNum(s, 2);
})();

// ---------------- Slide 3: Data & method ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "The data & method");
  title(s, "A churn definition earned from the data, not a rule of thumb");
  s.addText(
    "21.5M real transactions from a music-streaming subscriber base (2015–2017). Churn = no renewal within 30 days of a subscription's expiry — chosen because that's where this data's own renewal-gap curve flattens out, not copied from a blog post.",
    { x: 0.6, y: 1.95, w: 5.9, h: 2.0, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT, valign: "top" }
  );
  s.addText([
    { text: "Out-of-time validation, not a random split", options: { bold: true, color: INK, breakLine: true } },
    { text: "Train on an earlier period, test on a genuinely later one — same discipline a random split would flatter past.", options: { color: INK_SOFT, breakLine: true, paraSpaceAfter: 12 } },
    { text: "Every data-quality issue investigated, not assumed away", options: { bold: true, color: INK, breakLine: true } },
    { text: "Billing-correction duplicates, pre-registration transactions, and cancellation-linked anomalies were each found, quantified, and handled deliberately.", options: { color: INK_SOFT } },
  ], { x: 0.6, y: 4.15, w: 5.9, h: 2.3, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13.5, valign: "top", lineSpacingMultiple: 1.15 });

  s.addChart(pres.ChartType.line, [{
    name: "Renewals within N days of expiry",
    labels: ["0", "1", "3", "5", "7", "10", "14", "21", "30"],
    values: [77.7, 90.1, 92.9, 93.9, 94.5, 95.0, 95.5, 96.0, 96.8],
  }], {
    x: 6.9, y: 1.95, w: 5.8, h: 4.6,
    showTitle: true, title: "Cumulative % of renewals, by days after expiry", titleFontSize: 13, titleColor: INK,
    showLegend: false, lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    chartColors: [VIOLET_SOFT],
    catAxisLabelColor: INK_SOFT, catAxisLabelFontSize: 10,
    valAxisLabelColor: INK_SOFT, valAxisLabelFontSize: 10, valAxisTitle: "% of renewals", showValAxisTitle: true,
    valAxisMinVal: 70, valAxisMaxVal: 100,
    valGridLine: { color: "E1E0D9", size: 1 }, catGridLine: { style: "none" },
    showDataLabels: false,
  });
  s.addText("30 days is where the curve is essentially flat — the churn window", {
    x: 6.9, y: 6.55, w: 5.8, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 11, italic: true, color: INK_SOFT,
  });
  pageNum(s, 3);
})();

// ---------------- Slide 4: The model, honestly ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "The model");
  title(s, "Three approaches, judged on the same held-out data");
  s.addChart(pres.ChartType.bar, [{
    name: "PR-AUC",
    labels: ["Recency rule", "LightGBM (deployed)", "Logistic regression"],
    values: [0.618, 0.647, 0.925],
  }], {
    x: 0.6, y: 1.95, w: 7.0, h: 4.5,
    showTitle: true, title: "PR-AUC on real, out-of-time validation", titleFontSize: 13, titleColor: INK,
    showLegend: false, barDir: "bar",
    chartColors: [INK_SOFT, VIOLET_SOFT, GREEN],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 12, dataLabelFontBold: true,
    dataLabelFormatCode: "0.000",
    catAxisLabelColor: INK, catAxisLabelFontSize: 13,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 1,
  });
  s.addText([
    { text: "Logistic regression wins on raw accuracy.", options: { bold: true, color: INK, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Reported honestly — not hidden in favour of the flashier model.", options: { color: INK_SOFT, breakLine: true, paraSpaceAfter: 16 } },
    { text: "LightGBM is deployed anyway.", options: { bold: true, color: INK, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Its native SHAP support is what lets us tell a retention team why someone's at risk — not just that they are. That explainability is what powers every reason shown in the live dashboard.", options: { color: INK_SOFT } },
  ], { x: 7.9, y: 2.1, w: 4.9, h: 4.2, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 14.5, valign: "top", lineSpacingMultiple: 1.2 });
  pageNum(s, 4);
})();

// ---------------- Slide 5: Segments ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "Who they are");
  title(s, "Three kinds of subscriber — before we even look at risk");
  s.addText("Grouped by behaviour alone (recency, frequency, spend, tenure) — never by whether they actually churned.", {
    x: 0.6, y: 1.7, w: 11, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13, italic: true, color: INK_SOFT,
  });

  const cards = [
    { name: "Drifting Standard Subscribers", share: "53.8% of base", churn: "55.8%", spend: "$129 avg. spend", tenure: "~2 yr tenure", color: VIOLET_SOFT, fill: LAVENDER },
    { name: "Loyal Frequent Renewers", share: "43.4% of base", churn: "17.0%", spend: "$144 avg. spend", tenure: "~4.6 yr tenure", color: BLUE, fill: "EAF1FB" },
    { name: "High-Value Subscribers Going Dark", share: "2.7% of base", churn: "81.9%", spend: "$977 avg. spend", tenure: "~3.9 yr tenure", color: ORANGE, fill: "FDEEE6" },
  ];
  const cardW = 3.75, gap = 0.35, startX = 0.6, y = 2.35, h = 4.1;
  cards.forEach((c, i) => {
    const x = startX + i * (cardW + gap);
    s.addShape("roundRect", { x, y, w: cardW, h, rectRadius: 0.08, fill: { color: c.fill }, line: { type: "none" } });
    s.addShape("ellipse", { x: x + 0.3, y: y + 0.3, w: 0.22, h: 0.22, fill: { color: c.color }, line: { type: "none" } });
    s.addText(c.name, { x: x + 0.3, y: y + 0.65, w: cardW - 0.6, h: 0.9, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 18, bold: true, color: INK, valign: "top" });
    s.addText(c.share, { x: x + 0.3, y: y + 1.55, w: cardW - 0.6, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 11.5, color: INK_SOFT });
    s.addText(c.churn, { x: x + 0.3, y: y + 2.05, w: cardW - 0.6, h: 0.65, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 34, bold: true, color: c.color });
    s.addText("churn rate", { x: x + 0.3, y: y + 2.65, w: cardW - 0.6, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 10.5, color: INK_SOFT });
    s.addText(c.spend, { x: x + 0.3, y: y + 3.1, w: cardW - 0.6, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13, bold: true, color: INK });
    s.addText(c.tenure, { x: x + 0.3, y: y + 3.45, w: cardW - 0.6, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 11.5, color: INK_SOFT });
  });
  pageNum(s, 5);
})();

// ---------------- Slide 6: The four quadrants ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "Risk × value");
  title(s, "The priority order for retention spend");

  const quads = [
    { name: "Priority Save", size: "120,422", share: "44.8%", churn: "47.6%", spend: "$190", fill: RED_SOFT, accent: RED, badge: "ACT NOW" },
    { name: "Low-Cost Nudge", size: "75,240", share: "28.0%", churn: "43.1%", spend: "$88", fill: "FDEEE6", accent: ORANGE, badge: "LOW-COST" },
    { name: "Quiet Value — Monitor", size: "32,857", share: "12.2%", churn: "39.6%", spend: "$263", fill: AMBER_SOFT, accent: "B8860B", badge: "MONITOR" },
    { name: "Stable, No Action", size: "40,309", share: "15.0%", churn: "9.5%", spend: "$111", fill: GREEN_SOFT, accent: GREEN, badge: "NO ACTION" },
  ];
  const w = 5.55, h = 2.05, gx = 0.3, gy = 0.3, startX = 0.6, startY = 1.95;
  const positions = [[0, 0], [1, 0], [0, 1], [1, 1]];
  quads.forEach((q, i) => {
    const [col, row] = positions[i];
    const x = startX + col * (w + gx), y = startY + row * (h + gy);
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.07, fill: { color: q.fill }, line: { type: "none" } });
    s.addText(q.name, { x: x + 0.25, y: y + 0.18, w: w - 2.0, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 17, bold: true, color: INK });
    s.addShape("roundRect", { x: x + w - 1.55, y: y + 0.2, w: 1.3, h: 0.34, rectRadius: 0.17, fill: { color: q.accent }, line: { type: "none" } });
    s.addText(q.badge, { x: x + w - 1.55, y: y + 0.2, w: 1.3, h: 0.34, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 9.5, bold: true, color: WHITE, align: "center", valign: "middle" });
    s.addText([
      { text: q.size + "  ", options: { fontFace: HEAD_FONT, fontSize: 20, bold: true, color: INK } },
      { text: "(" + q.share + ")   ", options: { fontFace: BODY_FONT, fontSize: 11, color: INK_SOFT } },
      { text: q.churn + " churn   ", options: { fontFace: HEAD_FONT, fontSize: 20, bold: true, color: q.accent } },
      { text: q.spend + " avg. spend", options: { fontFace: BODY_FONT, fontSize: 13, color: INK_SOFT } },
    ], { x: x + 0.25, y: y + 0.85, w: w - 0.5, h: 1.0, isTextBox: true, margin: 0, valign: "top" });
  });
  pageNum(s, 6);
})();

// ---------------- Slide 7: Recommendation per quadrant ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "The recommendation");
  title(s, "What to actually do, Monday morning");

  const rows = [
    { name: "Priority Save", accent: RED, text: "Proactive outreach with a real offer — highest revenue at stake per member saved." },
    { name: "Low-Cost Nudge", accent: ORANGE, text: "Automated, passive retention only — the assumed offer cost isn't justified by the value here." },
    { name: "Quiet Value — Monitor", accent: "B8860B", text: "Not “safe” — 39.6% still actually churn. A light-touch check-in, short of full priority-save spend." },
    { name: "Stable, No Action", accent: GREEN, text: "Genuinely low risk and low value — leave alone." },
  ];
  let y = 2.0;
  rows.forEach((r) => {
    s.addShape("ellipse", { x: 0.6, y: y + 0.42, w: 0.2, h: 0.2, fill: { color: r.accent }, line: { type: "none" } });
    s.addText(r.name, { x: 1.0, y: y, w: 3.4, h: 1.05, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 18, bold: true, color: INK, valign: "middle" });
    s.addText(r.text, { x: 4.6, y: y, w: 8.1, h: 1.05, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT, valign: "middle" });
    y += 1.2;
  });
  pageNum(s, 7);
})();

// ---------------- Slide 8: Cost and return ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "What it would cost and return");
  title(s, "The economics, under stated assumptions");
  s.addText("These are ASSUMED figures, clearly labeled — not observed real dollars. Sensitivity to these assumptions is exactly why they're stated, not buried.", {
    x: 0.6, y: 1.7, w: 11.4, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 12.5, italic: true, color: INK_SOFT,
  });

  s.addText("Assumed inputs", { x: 0.6, y: 2.25, w: 4, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 15, bold: true, color: VIOLET });
  const assumptions = [
    ["Retention offer cost", "$75  (half the modal $149 plan)"],
    ["Value of a retained subscriber", "$894  (6× modal plan price — also a real plan price point)"],
    ["Offer success rate", "30%  (a commonly-cited industry figure)"],
  ];
  let ay = 2.75;
  assumptions.forEach(([k, v]) => {
    s.addText(k, { x: 0.6, y: ay, w: 4.5, h: 0.55, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13, bold: true, color: INK });
    s.addText(v, { x: 0.6, y: ay + 0.32, w: 4.5, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 12, color: INK_SOFT });
    ay += 1.0;
  });

  s.addText("Deployed threshold (0.88) — chosen to maximise total expected net benefit", {
    x: 5.6, y: 2.25, w: 7.1, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 15, bold: true, color: VIOLET,
  });
  statTile(s, 5.6, 2.75, 3.4, 1.7, "195,662", "subscribers flagged (72.8% of base)");
  statTile(s, 9.3, 2.75, 3.4, 1.7, "45.9%", "precision — of those flagged, actually cancel");
  statTile(s, 5.6, 4.6, 3.4, 1.7, "84.2%", "recall — of all cancellations, caught");
  statTile(s, 9.3, 4.6, 3.4, 1.7, "$9.4M", "estimated net benefit, under the assumptions above", { fill: "EAF1FB", valueColor: VIOLET });
  pageNum(s, 8);
})();

// ---------------- Slide 9: Limitations ----------------
(() => {
  const s = lightSlide(pres);
  kicker(s, "Limitations — draft, for review");
  title(s, "What this can't tell you");
  const items = [
    ["Listening-activity data is thin", "Only March 2017 (one month) of real listening intensity exists — used for live scoring only, never backtested against a real future outcome."],
    ["The 30% offer success rate is an assumption", "Not measured from this data. No A/B test of an actual retention offer exists yet — that's the single highest-value next step."],
    ["This churn rate isn't the industry-famous one", "39.7% here vs. the well-known 9.0% WSDM benchmark — different methodology (every member with enough history, not just a snapshot of currently-active ones), not a miscalibrated model."],
  ];
  let y = 2.1;
  items.forEach(([h, b]) => {
    s.addShape("roundRect", { x: 0.6, y, w: 11.7, h: 1.35, rectRadius: 0.07, fill: { color: LAVENDER }, line: { type: "none" } });
    s.addText(h, { x: 0.95, y: y + 0.15, w: 11.0, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 15.5, bold: true, color: INK });
    s.addText(b, { x: 0.95, y: y + 0.58, w: 11.0, h: 0.7, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 12.5, color: INK_SOFT });
    y += 1.55;
  });
  pageNum(s, 9);
})();

// ---------------- Slide 10: Next steps + closing ----------------
(() => {
  const s = darkSlide(pres);
  kicker(s, "Next steps", "CABFFF");
  s.addText("Where this goes next", {
    x: 0.6, y: 0.85, w: 10.6, h: 1.0, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 32, bold: true, color: WHITE,
  });
  const steps = [
    "A/B test the retention offers against a real holdout — replaces the 30% assumption with a measured number.",
    "Extend listening-activity history beyond one month, so intensity features can be backtested too.",
    "Establish a retrain cadence and drift monitoring — subscriber behaviour and pricing both move over time.",
  ];
  let y = 2.2;
  steps.forEach((t, i) => {
    s.addShape("ellipse", { x: 0.6, y: y + 0.03, w: 0.42, h: 0.42, fill: { color: "4A3AA7" }, line: { color: "9686E6", width: 1 } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.03, w: 0.42, h: 0.42, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
    s.addText(t, { x: 1.25, y: y - 0.08, w: 10.5, h: 0.7, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 15.5, color: "E5E1F5", valign: "top" });
    y += 0.95;
  });
  s.addText("Try it live: the risk×value dashboard, the segments, the subscriber lookup — all built on these same numbers.", {
    x: 0.6, y: 6.2, w: 11.5, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 14, italic: true, color: "CABFFF",
  });
  pageNum(s, 10, "9686E6");
})();

pres.writeFile({ fileName: "retention-signal-presentation.pptx" }).then(() => {
  console.log("written retention-signal-presentation.pptx");
});
