const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, PageBreak, Header, Footer,
  PageNumber, NumberFormat,
} = require("docx");

const VIOLET = "392C85";
const VIOLET_SOFT = "4A3AA7";
const INK = "17151F";
const INK_SOFT = "55506B";
const RED = "D03B3B";
const GREEN = "0CA30C";
const AMBER = "B8860B";
const ORANGE = "C1531F";
const LAVENDER = "F1EFF6";
const RED_SOFT = "FBE6E6";
const GREEN_SOFT = "E3F6E3";
const AMBER_SOFT = "FDF0D9";

const FONT = "Calibri";
const HEAD_FONT = "Cambria";

function h1(text) {
  return new Paragraph({
    text, heading: HeadingLevel.HEADING_1,
    spacing: { before: 180, after: 130 },
    keepNext: true,
  });
}
function h2(text) {
  return new Paragraph({
    text, heading: HeadingLevel.HEADING_2,
    spacing: { before: 170, after: 90 },
    keepNext: true,
  });
}
function body(text, opts) {
  opts = opts || {};
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22, color: opts.color || INK, italics: opts.italics, bold: opts.bold })],
    spacing: { after: 140, line: 288 },
    alignment: AlignmentType.LEFT,
    keepLines: true,
  });
}
function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22, color: INK })],
    bullet: { level: 0 },
    keepLines: true,
    spacing: { after: 70, line: 268 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 18, color: INK_SOFT, italics: true })],
    spacing: { before: 60, after: 200 },
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ---- Page 1: title + exec summary key-stat band ----
const titleBlock = [
  new Paragraph({
    children: [new TextRun({ text: "RETENTION SIGNAL", font: FONT, size: 20, bold: true, color: VIOLET_SOFT, characterSpacing: 30 })],
    spacing: { after: 40 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Subscriber Churn Risk × Value", font: HEAD_FONT, size: 52, bold: true, color: INK })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Which music-streaming subscribers are about to cancel, and which of them are actually worth spending a retention budget on.", font: FONT, size: 24, italics: true, color: INK_SOFT })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Written report · September 2026 · Real KKBox subscription transaction data (2015–2017)", font: FONT, size: 18, color: INK_SOFT })],
    spacing: { after: 300 },
  }),
];

function statCell(value, label, opts) {
  opts = opts || {};
  return new TableCell({
    width: { size: 25, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill: opts.fill || LAVENDER },
    margins: { top: 160, bottom: 160, left: 160, right: 160 },
    borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
    children: [
      new Paragraph({ children: [new TextRun({ text: value, font: HEAD_FONT, size: 34, bold: true, color: opts.color || VIOLET })], spacing: { after: 60 } }),
      new Paragraph({ children: [new TextRun({ text: label, font: FONT, size: 16, color: INK_SOFT })] }),
    ],
  });
}

const statBand = new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [2500, 2500, 2500, 2500],
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE }, insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } },
  rows: [
    new TableRow({ children: [
      statCell("268,828", "subscribers scored"),
      statCell("39.7%", "actually cancel"),
      statCell("45.9%", "precision at the deployed threshold"),
      statCell("84.2%", "of cancellations caught", { fill: "EAF1FB", color: VIOLET_SOFT }),
    ]}),
  ],
});

const execSummary = [
  h1("Executive Summary"),
  body("A churn score alone tells a retention team who is about to leave. It doesn't tell them who is worth spending money to keep. This project builds both halves: a churn risk model on real subscription transaction data, and a real-dollar value axis crossed against it — so retention spend goes where it actually returns."),
  statBand,
  new Paragraph({ text: "", spacing: { after: 200 } }),
  body("The recommendation, in one sentence: focus proactive, costed retention offers on the 120,422 subscribers (44.8% of the base) in the Priority Save quadrant — high predicted risk and above-median spend, where 47.6% actually cancel and the value at stake is real. Everyone else gets a cheaper or no intervention, not because they don't matter, but because the assumed economics ($75 offer cost against a $894 potential retained value, at a 30% success rate) don't justify the same spend on lower-value or lower-risk subscribers."),
  body("This is built on 21.5 million real transactions from a real music-streaming subscription service (KKBox, WSDM Cup 2018) — not a synthetic or engagement-proxy dataset. The churn definition (no renewal within 30 days of a subscription's expiry) was derived from this data's own renewal-gap curve, not copied from industry convention. Three real methodological bugs were found and fixed during development — not hidden — and the deployed model (LightGBM) is honestly reported as less accurate than a logistic regression baseline on this data; it is deployed anyway for the plain-language, per-subscriber explanations its SHAP support enables.", { }),
  body("Read only this page and you can act correctly: spend on Priority Save first, understand that the population overall churns at 39.7%, and know that every dollar figure past this page is built from real data under stated, adjustable assumptions — never invented.", { bold: true }),
];

// ---- Findings ----
function metricTable(rows, headers) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
      shading: { type: ShadingType.CLEAR, fill: VIOLET },
      margins: { top: 100, bottom: 100, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: h, font: FONT, size: 18, bold: true, color: "FFFFFF" })] })],
    })),
  });
  const dataRows = rows.map((r, ri) => new TableRow({
    children: r.map((cell, ci) => new TableCell({
      width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? "FFFFFF" : LAVENDER },
      margins: { top: 90, bottom: 90, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), font: FONT, size: 20, color: INK, bold: ci === 0 })] })],
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "C3C2B7" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "C3C2B7" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E1E0D9" }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...dataRows],
  });
}

const findings = [
  h1("Findings"),
  h2("Data & method"),
  body("21,547,746 real transactions from a music-streaming subscriber base, January 2015 through February 2017. Churn is defined as no new transaction within 30 days of a subscription's expiry — chosen because that is where this data's own renewal-gap curve flattens (96.8% of all renewals that happen occur within 30 days of expiry; the rest is a long, thin tail). Validation is out-of-time: the model trains on an earlier period and is tested on a genuinely later, disjoint set of subscribers — never a random split, which would flatter the result."),
  body("Every data-quality issue found was investigated and handled deliberately, not assumed away: same-day billing-correction transactions (2.5% of rows), pre-registration transactions (153 rows), and cancellation-linked expiry anomalies (which turned out to be 95.8% legitimate cancellation records, not defects) were each quantified and resolved on their own terms."),
  h2("The model, reported honestly"),
  metricTable(
    [["Recency rule", "0.618", "0.567"], ["LightGBM (deployed)", "0.647", "0.676"], ["Logistic regression", "0.925", "0.936"]],
    ["Approach", "PR-AUC", "ROC-AUC"]
  ),
  caption("PR-AUC on real, out-of-time validation. Higher is better."),
  body("Logistic regression is the most accurate of the three approaches on this data — reported plainly, not buried in favour of the more sophisticated model. LightGBM is deployed anyway, because its native SHAP support is what lets the dashboard tell a retention team why a specific subscriber is flagged, not just that they are. LightGBM does still clear the bar the spec set: it beats the simple recency-rule baseline on the headline metric."),
  h2("Three subscriber segments"),
  body("Subscribers group into three behavioural segments — built from recency, frequency, spend, and tenure alone, never from whether they actually churned:"),
  metricTable(
    [
      ["Drifting Standard Subscribers", "144,725 (53.8%)", "55.8%", "$129"],
      ["Loyal Frequent Renewers", "116,789 (43.4%)", "17.0%", "$144"],
      ["High-Value Subscribers Going Dark", "7,314 (2.7%)", "81.9%", "$977"],
    ],
    ["Segment", "Size", "Churn rate", "Avg. spend"]
  ),
  caption("K-Means, k=3 (chosen by elbow + silhouette sweep, k=2..8). The third segment is small but pays roughly 7x the average per transaction — a disproportionate revenue exposure hiding in a 2.7% slice of the base."),
];

// ---- Risk-Value Quadrant + Recommendations ----
function quadCell(name, badge, badgeColor, size, share, churn, spend, fill) {
  return new TableCell({
    width: { size: 50, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill },
    margins: { top: 160, bottom: 160, left: 180, right: 180 },
    borders: { top: { style: BorderStyle.SINGLE, size: 2, color: "FFFFFF" }, bottom: { style: BorderStyle.SINGLE, size: 2, color: "FFFFFF" }, left: { style: BorderStyle.SINGLE, size: 2, color: "FFFFFF" }, right: { style: BorderStyle.SINGLE, size: 2, color: "FFFFFF" } },
    children: [
      new Paragraph({ children: [
        new TextRun({ text: name, font: HEAD_FONT, size: 24, bold: true, color: INK }),
      ], spacing: { after: 80 } }),
      new Paragraph({ children: [new TextRun({ text: badge, font: FONT, size: 16, bold: true, color: badgeColor })], spacing: { after: 120 } }),
      new Paragraph({ children: [
        new TextRun({ text: size + "  ", font: HEAD_FONT, size: 26, bold: true, color: INK }),
        new TextRun({ text: "(" + share + ")", font: FONT, size: 18, color: INK_SOFT }),
      ], spacing: { after: 60 } }),
      new Paragraph({ children: [
        new TextRun({ text: churn + " churn", font: HEAD_FONT, size: 22, bold: true, color: badgeColor }),
        new TextRun({ text: "   " + spend + " avg. spend", font: FONT, size: 18, color: INK_SOFT }),
      ] }),
    ],
  });
}

const quadrantTable = new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [5000, 5000],
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE }, insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } },
  rows: [
    new TableRow({ children: [
      quadCell("Priority Save", "ACT NOW", RED, "120,422", "44.8%", "47.6%", "$190", RED_SOFT),
      quadCell("Low-Cost Nudge", "LOW-COST", ORANGE, "75,240", "28.0%", "43.1%", "$88", "FDEEE6"),
    ]}),
    new TableRow({ children: [
      quadCell("Quiet Value — Monitor", "MONITOR", AMBER, "32,857", "12.2%", "39.6%", "$263", AMBER_SOFT),
      quadCell("Stable, No Action", "NO ACTION", GREEN, "40,309", "15.0%", "9.5%", "$111", GREEN_SOFT),
    ]}),
  ],
});

const quadrantSection = [
  h1("Risk × Value: the Priority Order for Retention Spend"),
  body("Every subscriber sits in exactly one of four cells — churn risk from the deployed model (threshold 0.88) crossed against real spend (above/below the median $149, which also happens to be the modal single-plan price in the data)."),
  quadrantTable,
  new Paragraph({ text: "", spacing: { after: 200 } }),
  h2("Recommendation, per quadrant"),
  bullet("Priority Save — proactive outreach with a real offer. Highest revenue at stake per member saved."),
  bullet("Low-Cost Nudge — automated, passive retention only. The assumed offer cost isn't justified by the value here."),
  bullet("Quiet Value — Monitor — not “safe”: 39.6% of this group still actually cancels. A light-touch periodic check-in, short of full Priority Save spend."),
  bullet("Stable, No Action — genuinely low risk and low value. Leave alone."),
];

// ---- Cost/return + limitations ----
const costSection = [
  h1("What It Would Cost and Return"),
  body("The figures below follow directly from real precision/recall numbers on real validation data, multiplied by three ASSUMED unit-economics inputs — stated explicitly here because the whole recommendation is sensitive to them, not because they were measured from this dataset."),
  metricTable(
    [
      ["Retention offer cost", "$75", "Half the modal $149 monthly plan — a token discount"],
      ["Value of a retained subscriber", "$894", "6× modal plan price (149 × 6) — also itself a real observed plan price"],
      ["Offer success rate", "30%", "A commonly-cited industry figure, not derived from this data"],
    ],
    ["Assumption", "Value", "Basis"]
  ),
  new Paragraph({ text: "", spacing: { after: 160 } }),
  body("At the deployed threshold (0.88, chosen to maximise total expected net benefit under these assumptions): 195,662 subscribers flagged (72.8% of the base), 45.9% precision, 84.2% recall, an estimated $9.4M in net benefit. The threshold flags a wide majority of the base deliberately — a cheap offer against a high potential payoff makes almost any positive-precision threshold profitable under this cost model, so the optimum favours reach over restraint. Different assumptions would move this threshold; that sensitivity is the point of stating them.", { bold: false }),
  h1("Limitations"),
  caption("Drafted from findings already established in this project; final judgment on scope and framing is Eileen's, per this project's own working discipline."),
  bullet("Listening-activity data is thin. Only March 2017 (one month) of real listening intensity exists in this dataset — used for live scoring only, never backtested against a real future outcome."),
  bullet("The 30% offer success rate is an assumption, not a measurement. No A/B test of an actual retention offer exists yet against this population — that is the single highest-value next step."),
  bullet("This churn rate (39.7%) is not the well-known WSDM benchmark (9.0%). The difference is methodology, not miscalibration: this analysis evaluates every subscriber with enough history as of a reference date, including many who lapsed long ago and never returned, while the published benchmark snapshots only currently-active subscribers at one point in time."),
  h2("Next steps"),
  bullet("A/B test the retention offers against a real holdout — replaces the 30% assumption with a measured number."),
  bullet("Extend listening-activity history beyond one month, so intensity features can be backtested rather than used for live scoring only."),
  bullet("Establish a retrain cadence and drift monitoring — subscriber behaviour and pricing both move over time."),
];

// ---- Appendix ----
const appendix = [
  pageBreak(),
  h1("Appendix: Method"),
  h2("Churn definition and windowing"),
  body("A member is counted as churned if no new transaction occurs within 30 days of a subscription's membership_expire_date. This threshold was chosen by examining the cumulative distribution of renewal gaps in this data directly: 77.7% of renewals happen on or before expiry (auto-renew), 90.1% within 1 day, and 96.8% within 30 days — the point where the curve is essentially flat. Features for each labeled subscriber are built from their full transaction history strictly before the labeling cutoff — never a fixed rolling window, and never anything on or after the cutoff itself."),
  h2("Leakage tests"),
  body("Four tests guard the feature/label boundary and are run on every change: no feature uses a transaction on or after its own cutoff date; labels derive only from the prediction window; train and validation subscriber IDs are disjoint; every feature column has either zero nulls or an explicitly documented reason for any it has. All four pass."),
  h2("Three bugs, found and fixed"),
  body("Real methodological problems surfaced during development and were corrected before any result was trusted — documented here because catching them is part of the actual work, not a footnote."),
  bullet("Cutoff-date-as-churn-proxy: an early design selected each subscriber's own last eligible transaction as their one labeled example, then split train/validation by that date. On real data this produced 69% churn in train vs. 7% in validation from the same population — the cutoff date itself was a stand-in for churn status, since a churned subscriber's history simply stops early. Fixed by evaluating every subscriber against a shared external reference date per split."),
  bullet("A missing feature, not a leak: a tuned LightGBM model showed near-perfect accuracy on an internal split of the training data but collapsed on real validation. The cause was a genuinely absent feature — the gap between a subscriber's last activity and the actual scoring date — which the model had been forced to reconstruct indirectly instead of being given directly. Adding it directly fixed the collapse."),
  bullet("A population bias disguised as a time split: guaranteeing no subscriber appeared in both train and validation by excluding “already-used” subscribers from validation silently restricted validation to only newer, shorter-tenured subscribers. This broke the LightGBM model so badly its accuracy fell below random chance. Fixed by randomly assigning subscribers to disjoint pools before any time-based logic runs, confirmed by matching tenure distributions across the corrected splits."),
  h2("Feature engineering, model, and segmentation"),
  body("Twenty features across recency (days since last transaction, days since the scoring reference date), frequency (transaction count, active days, longest gap, first-half vs. second-half trend), monetisation (total and average amount paid, most recent plan price), tenure (days since registration, registration channel), and trajectory — all built from data strictly before each subscriber's labeling cutoff. LightGBM (300 trees, learning rate 0.05, 31 leaves) is the deployed model; a hyperparameter-tuning pass did not beat this simple configuration on real out-of-time validation, so the simple configuration is production. SHAP (TreeExplainer) provides both global feature importance and the plain-language, per-subscriber explanations. K-Means (k=3, chosen via an elbow-plus-silhouette sweep across k=2..8) segments subscribers on behavioural features alone."),
];

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 22, color: INK } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 30, bold: true, color: VIOLET }, paragraph: { spacing: { before: 260, after: 140 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 24, bold: true, color: INK }, paragraph: { spacing: { before: 200, after: 100 } } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        children: [new TextRun({ text: "RETENTION SIGNAL", font: FONT, size: 14, color: INK_SOFT, characterSpacing: 20 })],
        alignment: AlignmentType.RIGHT,
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        children: [new TextRun({ text: "Subscriber Churn Risk × Value — Written Report   |   Page ", font: FONT, size: 16, color: INK_SOFT }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: INK_SOFT })],
        alignment: AlignmentType.RIGHT,
      })] }),
    },
    children: [
      ...titleBlock, ...execSummary,
      pageBreak(), ...findings,
      ...quadrantSection,
      ...costSection,
      ...appendix,
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  require("fs").writeFileSync("retention-signal-report.docx", buffer);
  console.log("written retention-signal-report.docx");
});
