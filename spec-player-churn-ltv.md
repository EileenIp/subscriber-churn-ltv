# Project spec — Player Churn Early-Warning & LTV Segmentation

**For:** Eileen Ip · portfolio project 1 of 3
**Theme:** Gaming
**Pattern:** deliberately mirrors the E-commerce Purchase-Prediction project (event logs → session features → gradient boosting → SHAP → segmentation → recommendations), so the methodology is already defensible in interview.
**Agent:** one builder, Sonnet, in Claude Code. No checker agent — tests do the mechanical checking, Eileen does the judgement checking.

---

## How to use this document

Hand this whole file to the builder agent. It works through the phases **in order** and **stops at each checkpoint** for Eileen to review before continuing.

Two kinds of instruction appear below:

- **AGENT:** the agent does this.
- **EILEEN DECIDES:** the agent must stop, state the options it sees, and wait. It must not pick for her. These are the questions an interviewer will ask, and the answer has to be hers.

Every stop is a natural session boundary. Run one phase per session rather than leaving the agent going open-ended — that is where the money goes.

---

## The business question

> Which players are about to stop playing, how early can we see it coming, and are the at-risk players actually worth spending money to retain?

The second half is what makes this more than a churn tutorial. A churn model that flags 10,000 players is useless if 9,000 of them never spent anything. The project's real output is **churn risk crossed with player value**, so a studio can spend retention budget where it returns.

**Who cares:** live-ops and player-retention teams at a F2P studio; UA teams deciding acquisition spend by segment.

**What a good answer changes:** which players get a retention offer, and how much is spent on each.

---

## Phase −1 — Readiness gate

Run Appendix B before any code is written. It takes 20 minutes and it is the cheapest phase in the project. If the idea fails the gate, the right move is to change it now rather than after a week of building.

**EILEEN DECIDES:** go or pivot. **Checkpoint −1.**

---

## Phase 0 — Data selection

**Setup, done by Eileen once before the agent starts:** `pip install kaggle`, then download `kaggle.json` from Kaggle → Account → Create New API Token and place it at `~/.kaggle/kaggle.json` (`chmod 600`). Without this the agent cannot list or download anything.

**AGENT:**

1. Search the Kaggle CLI across several phrasings, since one query will miss most of the field:

```bash
kaggle datasets list -s "game player churn"
kaggle datasets list -s "mobile game analytics"
kaggle datasets list -s "player sessions"
kaggle datasets list -s "game telemetry"
kaggle datasets list -s "steam player behaviour"
kaggle datasets list -s "in-app purchase game"
```

2. For the ~10 most promising, download and inspect. Report for each: row count, every column name with dtype, date span, null rates, and file size.

3. **Apply the hard filter.** The project needs **one row per event or per session, with a player identifier and a timestamp.** Reject any dataset that is one row per player with pre-computed totals (`total_sessions`, `total_spend`, `churned` as a supplied column). Such a dataset makes observation/prediction windowing impossible, which removes out-of-time validation, the leakage tests, and the entire early-warning premise. State the rejection reason for each one rejected — this list is useful material for the write-up.

4. **Flag likely synthetic data.** Tells: suspiciously round row counts, uniform distributions, no missing values anywhere, no duplicate players, timestamps evenly spaced, a recent upload date with a generic title. Say which candidates look generated and why.

5. Rank the survivors on: event-level structure, presence of spend data, realistic messiness, size, and date span. **Report the shortlist and stop. Do not choose.**

**Fallback if nothing survives the filter:** the AWS `players-behaviors-dataset-generator` (pip: `players-behaviors-dataset-generator`), which emits session events with `player_id`, `player_type`, `cohort_id`, `session_id`, `event_type`, `timestamp`, treats a player as churned after a configurable inactivity period (default 5 days), and supports acquisition cohorts. Structurally realistic, openly synthetic.

**Context worth knowing, and worth putting in the README either way:** almost all published game-churn research runs on private studio data — four years of records from a European developer, 2.1M players of a mobile card game, six titles from one publisher. There is no public equivalent. A portfolio project acknowledging that constraint and handling it deliberately reads better than one that pretends otherwise.

**EILEEN DECIDES:** which dataset, and whether synthetic is acceptable. If synthetic, the README says so in its first paragraph — not a footnote. Note the Creator Content dashboard already used synthetic data; two in a row weakens the portfolio, so prefer real even if smaller.

**Checkpoint 0.** Do not proceed until the dataset is chosen.

---

## Phase 1 — Repo scaffold and ingestion

**AGENT:**

- Create the repo structure:

```
player-churn-ltv/
├── README.md            # SPIDER write-up, written last
├── NOTES.md             # decision log — Eileen writes this, agent never edits it
├── data/
│   ├── raw/             # gitignored
│   └── processed/       # gitignored
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── model.py
│   ├── segment.py
│   └── config.py        # all windows, thresholds, seeds in one place
├── notebooks/
│   └── 01_exploration.ipynb
├── tests/
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── .gitignore
```

- `ingest.py`: load raw data, validate schema, report row count, date span, null rates per column, and duplicate rows. If the file is large, process in chunks — reuse the chunked-processing approach from the e-commerce project.
- Every parameter (window lengths, churn threshold, random seed) lives in `config.py`. No magic numbers scattered through the code.
- Commit after this phase.

**Acceptance:** `python -m src.ingest` runs clean and prints a data quality summary. No hardcoded file paths outside `config.py`.

---

## Phase 2 — Data quality investigation

**AGENT:** Profile the data and report, without fixing anything yet:

- Missing or malformed session boundaries. In real game logs, session-end events are commonly misreported — logged after the next session's start, or missing entirely, because players switch apps rather than quitting through a menu. Quantify how often this happens here.
- Players whose events precede their recorded install time.
- Players with an implausible event volume (bots, test accounts, QA rigs).
- Gaps in the date range — a missing day of logging looks exactly like a day of mass churn.

**EILEEN DECIDES:** what to do about each issue. Drop, impute, or cap — and why. Write the reasoning in `NOTES.md` as you go. This is the "handling ambiguity" criterion straight out of the hiring-manager checklist, and it is the single most common thing that separates a real project from a tutorial clone.

**Checkpoint 2.**

---

## Phase 3 — Churn definition and windowing

This is the heart of the project. Get it wrong and everything downstream is meaningless.

**EILEEN DECIDES, before the agent writes any code:**

1. **What counts as churn?** N days of inactivity is standard; the AWS generator defaults to 5, published game studies often use 14 or 30. The right answer depends on the game's rhythm — a daily-login mobile game and a weekend-session PC game have different natural gaps. Look at the distribution of gaps between sessions in *this* data and pick a threshold you can justify from that distribution, not from a blog post.
2. **Observation window vs prediction window.** Features are built only from the observation window; the label is measured only in the prediction window that follows it. E.g. 30 days of behaviour → predict churn in the next 14 days.
3. **Which players are eligible?** Players who installed mid-window have less history than players present throughout. Include them with a `tenure_days` feature, or exclude them — either is defensible, but pick one and say why.

**AGENT:** implement exactly what she decides, in `config.py`. Write a short docstring at the top of `features.py` restating the definition in plain English.

**Checkpoint 3.**

---

## Phase 4 — Feature engineering

**AGENT:** Build session-level then player-level features from the observation window only. Suggested families:

- **Recency** — days since last session, days since last purchase.
- **Frequency** — sessions per week, active days, longest gap, trend in sessions across the window (first half vs second half).
- **Intensity** — mean and median session duration, events per session.
- **Monetisation** — total spend, purchase count, days since first purchase, spend trend (only if the dataset has spend).
- **Tenure** — days since install, acquisition cohort.
- **Trajectory** — the deltas matter more than the levels. A player at 5 sessions/week trending down from 15 is a very different risk from one who has always played 5.

**Leakage is the failure mode here.** Any feature that peeks into the prediction window invalidates the model, and it will look *great* in validation, which is what makes it dangerous.

**AGENT: write these tests and make them pass:**

- `test_no_future_events`: assert the max event timestamp used in any feature is ≤ the observation window cutoff.
- `test_label_from_prediction_window_only`: assert labels derive only from events after the cutoff.
- `test_no_player_overlap`: assert train/validation player IDs are disjoint.
- `test_feature_nulls`: assert every feature column's null rate is either 0 or explicitly documented in `config.py`.

**Acceptance:** `pytest` green, feature matrix shape and churn base rate printed.

---

## Phase 5 — Model

**AGENT:**

- Baseline first: logistic regression, or an even simpler recency rule ("no session in 7 days → will churn"). Report its scores. **A model that cannot beat a recency heuristic is not worth deploying, and saying so is a strength, not a failure.**
- Then LightGBM (matches the e-commerce project's stack).
- Report ROC-AUC, PR-AUC, precision and recall at several thresholds, and a confusion matrix. With imbalanced churn data, PR-AUC is the more honest headline number.
- **Out-of-time validation**: train on the earlier period, validate on a later unseen period — same discipline as the e-commerce project's November hold-out. A random split on time-series data flatters the model and any interviewer worth working for will ask about it.
- SHAP: global importance plus 2–3 individual player explanations.

**EILEEN DECIDES:**

- The operating threshold, and the reasoning. This is a cost question, not a statistical one: what does a wasted retention offer cost versus a lost player? State the assumed costs explicitly in the README even though they're assumed.
- Whether the SHAP story is *believable*. If the top feature is something odd, that is usually leakage or a data artefact, not an insight. Investigate before you write it up.

**Checkpoint 5.**

---

## Phase 6 — Segmentation and value

**AGENT:**

- K-Means on behavioural features (scaled). Choose k with elbow plus silhouette, and report both.
- For each segment: size, churn rate, mean spend, mean tenure, and predicted churn risk.
- Cross-tabulate risk against value: the deliverable is a 2×2 — high risk/high value, high risk/low value, low risk/high value, low risk/low value — with the retention recommendation for each quadrant.
- If the dataset has no spend data, use an engagement proxy and label it honestly as a proxy. Do not call it LTV if it is not LTV.

**EILEEN DECIDES:** the segment names and what each one means in plain language. "Cluster 3" is not an insight; "lapsing weekend spenders" is.

---

## Phase 7 — Deliverables for a non-technical audience

Four outputs, all aimed at someone who will never open the notebook: a live-ops lead, a marketing manager, a hiring manager with 90 seconds.

**7a — HTML dashboard.** Self-contained, no server, opens in a browser and deploys to GitHub Pages. Contents: the risk-vs-value quadrant as the hero, segment cards with size and churn rate, a player lookup with a plain-language explanation of *why* that player is at risk, and a threshold slider showing how the flagged population and its value change. No metric jargon on the surface — "8 in 10 of the players we flag do leave" rather than "precision 0.81". Technical detail lives behind a toggle.

**7b — Presentation.** 8–10 slides for a stakeholder audience: the problem, what the data showed, the four quadrants, the recommendation per quadrant, what it would cost and return, limitations, next steps. One idea per slide, no equations. Built with the pptx skill.

**7c — Written report.** 4–6 pages: executive summary on page one that stands alone, then findings, recommendations, method in an appendix. Someone should be able to read only page one and act correctly.

**7d — Website case study.** Matches the structure and depth of the e-commerce case study on the portfolio site. Same section order, same level of methodological detail, real numbers only.

**Acceptance:** the HTML dashboard loads from a logged-out browser at a public URL; a non-technical reader can state the recommendation after two minutes with any one of the four.

---

## Phase 8 — Write-up

**AGENT:** draft the README following the SPIDER structure in Appendix A, which is partly filled in already — the agent completes the sections marked *fill from the run*, and leaves the rest for Eileen. Include the live link, a screenshot, and setup instructions someone else could follow.

**EILEEN WRITES, not the agent:**

- **"What didn't work"** — the dead ends, the features that added nothing, the first churn definition that turned out wrong. This section is what distinguishes the project from a tutorial.
- **Limitations** — what the data can't tell you, what you'd need to do this properly.
- **The recommendation paragraph** — what a live-ops team should actually do on Monday morning.

**Acceptance:** a reader who has never seen the repo understands the business question within 30 seconds, and can reproduce the analysis from the instructions.

---

## Guardrails

**Never invent a number.** Not in the README, not on a slide, not on a dashboard card, not as a placeholder to be swapped later. If a figure isn't in the output yet, the text says `[from run]` and stays that way until it is. Placeholder impact figures that survive into a published page are the single worst failure mode available here — they turn a good project into a credibility problem. This applies to plausible-sounding business impact most of all: unless the analysis computed a dollar figure from real inputs, there is no dollar figure.

**The agent does not write `NOTES.md`.** It is Eileen's decision log, in her words, and it is the evidence of independent work.

**The agent does not choose** at any point marked **EILEEN DECIDES**. If it needs an answer to keep going, it stops and asks. A decision made to avoid stopping is a decision that can't be defended later.

**Stop and ask, rather than assume**, whenever: the data doesn't match what the spec expected, a test fails in a way that suggests the design is wrong, or a result looks too good. A validation score that jumps unexpectedly is a leakage signal, not a success.

## Resuming between sessions

The agent has no memory across sessions. Every new session starts by reading, in order: this spec, `NOTES.md`, and the last commit message. Every session ends by committing with a message naming the phase and what changed, and appending a one-line status to the bottom of the spec's checklist.

Keep all parameters in `config.py` and pin `requirements.txt` with exact versions. Set and record a random seed for every model and clustering step — a portfolio project whose numbers move between runs cannot be written up honestly.

## How the repo reads to a stranger

A recruiter reaches the repo before they reach the analysis. Ten minutes of work:

- Repo description and GitHub topics filled in (`churn-prediction`, `lightgbm`, `shap`, `game-analytics`).
- README opens with a screenshot of the dashboard and the live link, above the fold.
- Pinned on the GitHub profile.
- Commit history that reads as a project rather than one "initial commit" dump — the phase-per-commit rule handles this by itself.

## Cost discipline

- One phase per session. Sonnet throughout.
- If the agent starts looping on an error, stop it and hand the error here rather than letting it retry — retries are where credits vanish.
- Log every run in the tracker immediately after it finishes.
- Estimated total: A$25–40 across all phases. If it passes A$50, stop and reassess.

## Definition of done

- [ ] HTML dashboard live, loads for a stranger
- [ ] Presentation, report, and website case study all written from the same numbers
- [ ] `pytest` green, including all four leakage tests
- [ ] README in SPIDER order with a "what didn't work" section
- [ ] `NOTES.md` with a dated decision log
- [ ] Out-of-time validation, not a random split
- [ ] Answers ready for: why this churn window, why LightGBM over the baseline, why this threshold, what breaks at 100× scale
- [ ] Card added to the portfolio site with a real number from the analysis — never a placeholder figure

---

# Appendix A — SPIDER framework

This is the project's planning document and, later, the skeleton of the README. Sections marked **[pre-filled]** are decided already. Sections marked **[fill from the run]** the agent completes from actual results. Sections marked **[Eileen]** must be written by hand — they are the interview answers.

## Project overview

**Title:** Player Churn Early-Warning & LTV Segmentation **[pre-filled]**

**One sentence:** Predicts which F2P players are about to lapse and crosses that risk against player value, so retention spend goes where it returns. **[pre-filled]**

**Target roles:** Data Analyst / Data Scientist, gaming and media. **[pre-filled]**

---

## S — Summary impact

*3–5 sentences: problem context, what you did, data source, methodology, key finding. Write this last, once the numbers exist.*

- **Problem context:** F2P studios lose most players in the first weeks, and retention budget is usually spent on whoever churns loudest rather than whoever is worth keeping. **[pre-filled]**
- **What you did:** built a player-level churn risk score from behavioural event logs and crossed it with player value to produce a retention priority matrix. **[pre-filled]**
- **Data source:** **[fill from the run — name, provenance, row count, date span]**
- **Methodology:** **[fill from the run — baseline vs LightGBM, out-of-time validation, SHAP, K-Means]**
- **Key finding:** **[fill from the run — the one sentence a live-ops lead would repeat to their manager]**

---

## P — Problem space

**Why this problem matters:** retention is cheaper than acquisition, and a player flagged a week before they lapse can still be reached. Once they've gone, they've gone. **[pre-filled]**

**Who would care:** live-ops and retention teams; UA teams allocating acquisition spend by segment. **[pre-filled]**

**Scope:** one game's player base over a fixed window; predicting lapse, not lifetime revenue forecasting. **[pre-filled]**

**Why this is a business problem, not a data exercise:** the output is a spending decision — which players get an offer and how much it's worth paying to keep them. A model with no value dimension can't answer that. **[pre-filled]**

---

## I — Inputs

- **Primary source:** **[fill from the run — name, URL, rows × columns, format]**
- **Real or synthetic:** **[Eileen — state plainly, first paragraph of the README if synthetic]**
- **Collection method:** downloaded public dataset / generated. **[fill from the run]**
- **Target variable:** churned within the prediction window, defined as **[Eileen — Phase 3 decision]**
- **Key predictors:** recency, frequency trend, session intensity, monetisation, tenure. **[pre-filled, confirm after Phase 4]**

---

## D — Discovery

**Data quality issues found:** **[fill from the run — Phase 2 profiling output]** — expect missing or misordered session-end events, pre-install events, bot-like accounts, and logging gaps.

**How you addressed them:** **[Eileen — the Phase 2 decisions and the reasoning behind each]**

**Surprising patterns:** **[fill from the run]**

**Initial hypotheses:** recency dominates; spend history predicts retention; trajectory beats level. **[pre-filled — check whether each held]**

**What actually happened:** **[Eileen]**

**Dead ends and methods that didn't work:** **[Eileen — this is the section hiring managers actually read; do not let the agent write it]**

---

## E — Execution

**Primary method:** LightGBM binary classification, benchmarked against a logistic regression and a plain recency rule. **[pre-filled]**

**Why this method:** **[Eileen — must cover interpretability via SHAP, handling of mixed feature types, and why not something simpler or more complex]**

**Alternatives considered and rejected:** survival analysis, a sequence model on raw events. **[Eileen — why not]**

**Key technical decisions:**
- Observation and prediction windows: **[Eileen — Phase 3]**
- Validation approach: out-of-time hold-out, not a random split. **[pre-filled — explain why in one line]**
- Evaluation metric: PR-AUC as headline given class imbalance. **[pre-filled]**
- Operating threshold: **[Eileen — Phase 5, with the assumed costs stated]**

**Code highlights to showcase:** the chunked ingestion, the leakage tests, the risk-vs-value cross-tab. Two or three snippets, not the whole notebook. **[pre-filled]**

---

## R — Results & recommendations

**Key results:** **[fill from the run — baseline vs model, with the numbers]**

**Business interpretation:** **[Eileen — in plain language, no metric names]**

**Recommendations:** **[Eileen — what a live-ops team does differently on Monday, per quadrant of the risk/value matrix]**

**Limitations and assumptions:** **[Eileen — what the data can't tell you; if synthetic, what that invalidates]**

**Next steps:** A/B test the retention offers against a holdout; add social-graph features; retrain cadence and drift monitoring. **[pre-filled — trim to what you'd actually do]**

---

## Deliverables

- [x] GitHub repo with clean README
- [ ] Deployed Streamlit app
- [ ] `NOTES.md` decision log
- [ ] Portfolio site card
- [ ] LinkedIn post

## Success criteria

**Minimum viable:** a deployed app where a stranger can see the risk/value matrix and look up a player, with a README that survives the "why this method" question.

**Stretch:** a cost-sensitive threshold analysis showing the retention budget implied by each operating point.

## Final checks

☐ Follows SPIDER completely
☐ Every technical decision explainable in interview
☐ Clear business value, not just a technical exercise
☐ Dead ends and challenges documented
☐ A hiring manager understands it in 90 seconds
☐ Demonstrates 3+ technical skills
☐ Code clean and commented
☐ Visualisations tell a story
☐ Reproducible from the documentation
☐ Proud to show it

---

# Appendix B — Readiness gate (run before Phase 0)

From the Project Readiness Checklist. Answered in advance where the answer is already settled; the open ones are for Eileen.

## Pitfall 1 — Visualisation-only?

**No.** The project goes past "here's what the data shows" on four counts: it investigates *why* players lapse (SHAP), predicts *what happens next* (churn risk), compares segments, and ends in a spending recommendation. Required: at least two layers beyond charts. This has four.

## Pitfall 2 — Overused dataset with no new angle?

Player churn is a common portfolio topic, so the angle has to carry it. **The angle here is risk crossed with value.** Most churn portfolios stop at "this player will leave"; this one answers "and is that player worth paying to keep", which is the question a studio actually has. Keep that framing visible in the title, the README opening, and the dashboard hero.

☐ **Eileen:** once the dataset is chosen, search "[dataset name] churn analysis" and see how many near-identical notebooks exist. If there are many, the risk/value framing and the cost-based threshold must be foregrounded harder.

## Pitfall 3 — Missing business context?

- **What problem exists?** Retention budget is spent on the wrong players.
- **Who has it?** Live-ops and retention teams at F2P studios.
- **Why it matters to them?** Retention is cheaper than acquisition, and a lapsed player is unrecoverable.
- **What changes if it works?** Offers go to high-risk high-value players instead of everyone.

Would a company pay for this analysis? Yes — it is a standing function at every F2P studio. Three companies that would care: any mobile studio with live ops, a publisher's central analytics team, a games-analytics vendor.

## Pitfall 4 — Tutorial clone?

The risks are real, so guard them deliberately:

- The churn window is derived from this data's own gap distribution, not copied from a blog post.
- `NOTES.md` records dead ends as they happen, so the write-up is not retrospectively tidy.
- The baseline-first rule means a result of "the simple recency rule was nearly as good" gets reported rather than buried.
- Rejected datasets and their reasons are documented from Phase 0.

☐ **Eileen:** if you consult a tutorial or paper, cite it in the README — "inspired by X, extended to Y".

## Final validation

**Data quality** — decided at Checkpoint 0. Needs event-level rows, real-world messiness, several usable variables.

**Business relevance** — cleared above.

**Technical showcase** — feature engineering, gradient boosting, model interpretability, clustering, and a front end. Five skill areas, comfortably past the three-skill bar.

**Differentiation** — rests on the risk/value framing and the cost-based threshold. This is the weakest axis; protect it.

**Scope and feasibility** — designed for 2–3 weeks part-time, using tools already in the CV, on public data.

**Presentation potential** — four audience-facing deliverables in Phase 7.

**Verdict:** proceed. The one axis to watch is differentiation.

---

# Appendix C — Screening audit (run before Phase 7 ships)

Score the finished project 1–5 on each criterion. Below 4 on any row, fix it before the case study goes on the site.

| # | Criterion | What it means here | Score |
|---|---|---|---|
| 1 | Industry relevance | Does a games person recognise their own problem? Churn window, cohorts, F2P monetisation language used correctly. | /5 |
| 2 | Depth of analysis | Does it explain *why* players lapse, not just who? SHAP read and interpreted, not just plotted. | /5 |
| 3 | Thought process | Are the churn window, model choice, and threshold each justified, with the alternatives named? | /5 |
| 4 | Defendability | Can you answer all six questions below without notes? | /5 |
| 5 | Trade-offs | Is the precision/recall trade-off framed as a cost decision, with the assumed costs stated? | /5 |
| 6 | Independence | Is it visible that you set the scope and made the calls? `NOTES.md` is the evidence. | /5 |
| 7 | Handling ambiguity | Are the data-quality problems documented with the reasoning, including what you couldn't fix? | /5 |
| 8 | Implementation thinking | Retraining cadence, drift, what breaks at 100× — addressed, not ignored. | /5 |

**Total: /40.** 35+ ready to showcase · 28–34 minor fixes · 21–27 significant gaps · under 21 do not ship.

## The six questions to rehearse

Say each answer out loud before the case study goes live. Written fluency is not the same as spoken fluency, and these are the ones that get asked.

1. Why that churn window and not a longer or shorter one?
2. Why LightGBM when the recency baseline was already close?
3. How do you know the model isn't leaking future information?
4. Where did the threshold come from, and what did you assume about costs?
5. What would you do differently with real studio data?
6. What breaks when this runs on 100× the players, every day, in production?

## Interview-day one-liner

One sentence, ready to deliver: *"I built a churn early-warning model for free-to-play players and crossed the risk score against player value, so a studio can spend retention budget on the players actually worth keeping — the model flags [X]% of lapsing players [N] days early."*

☐ Fill the bracketed numbers from the real results. Never a placeholder.
