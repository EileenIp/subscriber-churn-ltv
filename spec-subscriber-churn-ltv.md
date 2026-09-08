# Project spec — Subscriber Churn Early-Warning & LTV Segmentation

**For:** Eileen Ip · portfolio project 1 of 3
**Theme:** Subscription media (music streaming)
**Pattern:** deliberately mirrors the E-commerce Purchase-Prediction project (event logs → subscriber features → gradient boosting → SHAP → segmentation → recommendations), so the methodology is already defensible in interview.
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

> Which subscribers are about to cancel, how early can we see it coming, and are the at-risk subscribers actually worth spending money to retain?

The second half is what makes this more than a churn tutorial. A churn model that flags 10,000 subscribers is useless if most of them are on the cheapest plan and were never going to renew profitably anyway. The project's real output is **churn risk crossed with subscription value**, so a streaming service can spend retention budget where it returns.

**Who cares:** subscriber-retention and lifecycle-marketing teams at a subscription media company; finance teams sizing retention budget by segment.

**What a good answer changes:** which subscribers get a retention offer, and how much is spent on each.

---

## Phase −1 — Readiness gate

Run Appendix B before any code is written. It takes 20 minutes and it is the cheapest phase in the project. If the idea fails the gate, the right move is to change it now rather than after a week of building.

**EILEEN DECIDES:** go or pivot. **Checkpoint −1.**

**Resolved (2026-09-08):** go — see session log.

---

## Phase 0 — Data selection

**Status: resolved.** This phase originally targeted gaming (F2P player churn) and ran an extensive Kaggle/Hugging Face search under that theme. It's recorded here because a future session has no memory of this conversation and needs the full reasoning, not just the answer.

**What was tried under the gaming theme, and why it didn't hold up:**

- Searched Kaggle across `game player churn`, `mobile game analytics`, `player sessions`, `game telemetry`, `steam player behaviour`, `in-app purchase game`, and several follow-up phrasings; checked Hugging Face too. Roughly 20 candidates inspected.
- Nearly everything real and popular (`rabieelkharoua/predict-online-gaming-behavior-dataset`, `cookie_cats`, `pratyushpuri/mobile-game-in-app-purchases-dataset-2025`, etc.) turned out to be one row per player with pre-computed totals — fails the hard filter below, since there's no timestamp to window on.
- The one dataset that looked real, large, and event-level (`debs2x/gamelytics-mobile-analytics-challenge` — 1M users, 9.6M login events, real-looking revenue) turned out to be synthetic on closer inspection **during Phase 2**: every one of 8.6M consecutive-login gaps fell in a suspiciously exact [1.0, 7.04]-day window (a bounded random walk), and `reg_ts` exactly equalled the first `auth_ts` for all 1,000,000 users with zero variance — not something real registration/login systems produce. This is now a standing check (see "synthetic tells" below) for any future dataset, gaming or otherwise.
- `troykueh/dragon-fantasy-mmorpg-economy-simulation` was structurally rich (real session-shaped data, quests, guilds, an economy) but openly synthetic (a labelled "economy simulation"), and Eileen's stated preference was to prefer real data — the portfolio already has one synthetic project (Creator Content dashboard) and a second in a row would weaken it.
- `mylesoneill/warcraft-avatar-history` (WoWAH) was found to be genuinely real (scraped from WoW's public Armory API, ~10.8M rows, 37,354 characters, a full year of 2008 data, realistic long-tailed gap distribution) but (a) already has a published churn-prediction paper and GitHub projects built on it, and (b) is a subscription MMORPG, not F2P — using it would have meant reframing most of the gaming-specific language below anyway.
- Given that reframing was needed regardless, Eileen chose to drop the gaming theme entirely rather than force a fit.

**Pivot decision (2026-09-08):** switch to subscription media / music streaming, specifically because real, public, **event-level** subscription data with genuine transaction/pricing records exists for this domain in a way it largely doesn't for gaming (most studios keep raw logs private).

**Dataset chosen: KKBox Churn Prediction Challenge (WSDM Cup 2018)** — `kaggle.com/c/kkbox-churn-prediction-challenge`. Real production data from KKBox, a real Asian music-streaming subscription service, released for a Kaggle/WSDM competition. Requires accepting the competition rules on Kaggle's site before download (done).

Files used (all verified — see below):

| File | Rows | Span | What it gives us |
|---|---|---|---|
| `transactions.csv` | 21,547,746 | 2015-01-01 to 2017-02-28 | Real subscription transactions: plan price, amount actually paid, auto-renew flag, cancellation flag, transaction date, membership expiry date |
| `members_v3.csv` | 6,769,473 | registration dates from 2004 onward | One row per member: city, age (`bd`), gender, registration channel, registration date |
| `user_logs_v2.csv` | 18,396,362 | March 2017 (one month) | Real daily listening activity: songs played by completion-percentage bucket, unique songs, total seconds listened |
| `train_v2.csv` | 970,960 | — | The competition's own `is_churn` label (9.0% churn rate) — kept as a **reference/validation check only**, not our source of truth. Our own churn definition is derived in Phase 3 from this data's own distribution, per the hard filter's whole point. |

**Verification performed (the checks that should be run on any future candidate too):**

1. **Hard filter** — event-level, not pre-aggregated. `transactions.csv` and `user_logs_v2.csv` both have multiple real, dated rows per member. Passes cleanly.
2. **Synthetic tells** — checked the gap distribution between a member's consecutive transactions: 83.3% fall in the 30–31 day monthly-renewal window, but the full range is 0–709 days (real early/late/lapsed-and-returned behaviour), nothing like a bounded random walk. `plan_list_price` and `payment_plan_days` show a realistic discrete set of real-world values (149, 99, 129, 180, …; 30, 31, 7, 90, 180, 195, 410 days), not round synthetic numbers. 158 exact duplicate rows out of 21.5M (present but rare — realistic, not suspiciously zero or suspiciously common). Passes.
3. **Overuse check (Pitfall 2)** — this is the one weak spot, flagged honestly rather than buried: KKBox/WSDM Cup 2018 is one of the most-used real churn datasets that exists, with thousands of public notebooks. See Appendix B, Pitfall 2 for the differentiation plan.

**Scope decision:** the full `user_logs.csv` (7.1GB compressed, ~2 years) is out of scope — `transactions.csv` already covers long-history recency/frequency/monetisation/tenure, and `user_logs_v2.csv` (one month, March 2017) is enough for listening-intensity features in the observation window, at a fraction of the processing cost. Revisit only if Phase 2/4 finds March 2017 isn't representative enough.

**Checkpoint 0.** Resolved — see above.

---

## Phase 1 — Repo scaffold and ingestion

**AGENT:**

- Create the repo structure:

```
subscriber-churn-ltv/
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

- `ingest.py`: load raw data, validate schema, report row count, date span, null rates per column, and duplicate rows. If the file is large, process in chunks — `transactions.csv` (21.5M rows) and `user_logs_v2.csv` (18.4M rows) both need this; `members_v3.csv` (6.8M rows) can be loaded in one pass.
- Every parameter (window lengths, churn threshold, random seed) lives in `config.py`. No magic numbers scattered through the code.
- Commit after this phase.

**Acceptance:** `python -m src.ingest` runs clean and prints a data quality summary. No hardcoded file paths outside `config.py`.

---

## Phase 2 — Data quality investigation

**AGENT:** Profile the data and report, without fixing anything yet:

- Members with transactions before their recorded `registration_init_time`.
- Members with an implausible transaction volume (test accounts, promotional/free-tier abuse, duplicate billing).
- Gaps or inconsistencies between `transaction_date` and `membership_expire_date` (e.g. an expiry date before its own transaction date).
- Whether `user_logs_v2.csv`'s single month (March 2017) is representative — compare its members against the broader `transactions.csv` population, and flag if it skews toward a particular cohort (e.g. only currently-active subscribers).
- Gaps in the date range — a missing day of transaction logging looks exactly like a day of mass cancellation.

**EILEEN DECIDES:** what to do about each issue. Drop, impute, or cap — and why. Write the reasoning in `NOTES.md` as you go. This is the "handling ambiguity" criterion straight out of the hiring-manager checklist, and it is the single most common thing that separates a real project from a tutorial clone.

**Checkpoint 2.**

---

## Phase 3 — Churn definition and windowing

This is the heart of the project. Get it wrong and everything downstream is meaningless.

**Context available but not binding:** KKBox's own competition labeller (`WSDMChurnLabeller.scala`, included in the raw data) defines churn as no valid new subscription within 30 days of a membership's expiry date. That's a legitimate industry reference point — but the spec's discipline still applies: justify the choice from this data's own distribution, and decide independently whether to adopt, adapt, or diverge from it. Matching the official rule without checking it against the data would be the same mistake as copying a threshold from a blog post.

**EILEEN DECIDES, before the agent writes any code:**

1. **What counts as churn?** Look at the distribution of gaps between a member's subscription expiry and their next renewal transaction in *this* data, and pick a threshold you can justify from that distribution.
2. **Observation window vs prediction window.** Features are built only from the observation window; the label is measured only in the prediction window that follows it. E.g. subscription/listening behaviour through a cutoff month → predict cancellation in the following month.
3. **Which members are eligible?** Members who registered mid-window have less history than members present throughout. Include them with a `tenure_days` feature, or exclude them — either is defensible, but pick one and say why.

**AGENT:** implement exactly what she decides, in `config.py`. Write a short docstring at the top of `features.py` restating the definition in plain English.

**Checkpoint 3.**

---

## Phase 4 — Feature engineering

**AGENT:** Build daily/transaction-level then member-level features from the observation window only. Suggested families:

- **Recency** — days since last transaction, days since last active listening day.
- **Frequency** — transactions per period, active listening days, longest gap, trend in listening days (first half vs second half of the window).
- **Intensity** — mean/median seconds listened per active day, songs played per day, song-completion depth (share of plays reaching ≥98.5% of the track) — from `user_logs_v2.csv`.
- **Monetisation** — total amount actually paid, transaction count, days since first payment, spend trend, plan price/tier. This is **real revenue**, not a proxy — no engagement-proxy fallback needed here.
- **Tenure** — days since registration, registration channel (`registered_via`).
- **Trajectory** — the deltas matter more than the levels. A member listening 5 days/week trending down from 15 is a very different risk from one who has always listened 5.

**Leakage is the failure mode here.** Any feature that peeks into the prediction window invalidates the model, and it will look *great* in validation, which is what makes it dangerous.

**AGENT: write these tests and make them pass:**

- `test_no_future_events`: assert the max event timestamp used in any feature is ≤ the observation window cutoff.
- `test_label_from_prediction_window_only`: assert labels derive only from events after the cutoff.
- `test_no_member_overlap`: assert train/validation member IDs (`msno`) are disjoint.
- `test_feature_nulls`: assert every feature column's null rate is either 0 or explicitly documented in `config.py`.

**Acceptance:** `pytest` green, feature matrix shape and churn base rate printed.

---

## Phase 5 — Model

**AGENT:**

- Baseline first: logistic regression, or an even simpler recency rule ("no renewal within N days of expiry → will churn"). Report its scores. **A model that cannot beat a recency heuristic is not worth deploying, and saying so is a strength, not a failure.**
- Then LightGBM (matches the e-commerce project's stack).
- Report ROC-AUC, PR-AUC, precision and recall at several thresholds, and a confusion matrix. With imbalanced churn data, PR-AUC is the more honest headline number.
- **Out-of-time validation**: train on the earlier period, validate on a later unseen period — same discipline as the e-commerce project's November hold-out. A random split on time-series data flatters the model and any interviewer worth working for will ask about it.
- SHAP: global importance plus 2–3 individual subscriber explanations.

**EILEEN DECIDES:**

- The operating threshold, and the reasoning. This is a cost question, not a statistical one: what does a wasted retention offer cost versus a lost subscriber? State the assumed costs explicitly in the README even though they're assumed.
- Whether the SHAP story is *believable*. If the top feature is something odd, that is usually leakage or a data artefact, not an insight. Investigate before you write it up.

**Checkpoint 5.**

---

## Phase 6 — Segmentation and value

**AGENT:**

- K-Means on behavioural features (scaled). Choose k with elbow plus silhouette, and report both.
- For each segment: size, churn rate, mean amount paid, mean tenure, and predicted churn risk.
- Cross-tabulate risk against value: the deliverable is a 2×2 — high risk/high value, high risk/low value, low risk/high value, low risk/low value — with the retention recommendation for each quadrant.
- Value here is real payment data (`actual_amount_paid`), not a proxy — say so plainly in the write-up as a point of strength over a typical engagement-proxy churn project.

**EILEEN DECIDES:** the segment names and what each one means in plain language. "Cluster 3" is not an insight; "lapsing premium-plan subscribers" is.

---

## Phase 7 — Deliverables for a non-technical audience

Four outputs, all aimed at someone who will never open the notebook: a retention lead, a marketing manager, a hiring manager with 90 seconds.

**7a — HTML dashboard.** Self-contained, no server, opens in a browser and deploys to GitHub Pages. Contents: the risk-vs-value quadrant as the hero, segment cards with size and churn rate, a subscriber lookup with a plain-language explanation of *why* that subscriber is at risk, and a threshold slider showing how the flagged population and its value change. No metric jargon on the surface — "8 in 10 of the subscribers we flag do cancel" rather than "precision 0.81". Technical detail lives behind a toggle.

**7b — Presentation.** 8–10 slides for a stakeholder audience: the problem, what the data showed, the four quadrants, the recommendation per quadrant, what it would cost and return, limitations, next steps. One idea per slide, no equations. Built with the pptx skill.

**7c — Written report.** 4–6 pages: executive summary on page one that stands alone, then findings, recommendations, method in an appendix. Someone should be able to read only page one and act correctly.

**7d — Website case study.** Matches the structure and depth of the e-commerce case study on the portfolio site. Same section order, same level of methodological detail, real numbers only.

**Acceptance:** the HTML dashboard loads from a logged-out browser at a public URL; a non-technical reader can state the recommendation after two minutes with any one of the four.

---

## Phase 8 — Write-up

**AGENT:** draft the README following the SPIDER structure in Appendix A, which is partly filled in already — the agent completes the sections marked *fill from the run*, and leaves the rest for Eileen. Include the live link, a screenshot, and setup instructions someone else could follow.

**EILEEN WRITES, not the agent:**

- **"What didn't work"** — the dead ends, the features that added nothing, the first churn definition that turned out wrong, the gaming-to-streaming pivot itself (why the original theme didn't hold up is a legitimate, interesting dead end). This section is what distinguishes the project from a tutorial.
- **Limitations** — what the data can't tell you, what you'd need to do this properly.
- **The recommendation paragraph** — what a retention team should actually do on Monday morning.

**Acceptance:** a reader who has never seen the repo understands the business question within 30 seconds, and can reproduce the analysis from the instructions.

---

## Guardrails

**Never invent a number.** Not in the README, not on a slide, not on a dashboard card, not as a placeholder to be swapped later. If a figure isn't in the output yet, the text says `[from run]` and stays that way until it is. Placeholder impact figures that survive into a published page are the single worst failure mode available here — they turn a good project into a credibility problem. This applies to plausible-sounding business impact most of all: unless the analysis computed a dollar figure from real inputs, there is no dollar figure.

**The agent does not write `NOTES.md`.** It is Eileen's decision log, in her words, and it is the evidence of independent work.

**The agent does not choose** at any point marked **EILEEN DECIDES**. If it needs an answer to keep going, it stops and asks. A decision made to avoid stopping is a decision that can't be defended later.

**Stop and ask, rather than assume**, whenever: the data doesn't match what the spec expected, a test fails in a way that suggests the design is wrong, or a result looks too good. A validation score that jumps unexpectedly is a leakage signal, not a success. **Concretely: before locking in any dataset, check its consecutive-event gap distribution for suspiciously exact bounds, and check whether any timestamp field exactly equals another with zero variance across many rows** — both are what caught the `debs2x` gaming dataset being synthetic in this project's own history, one phase too late. Do that check in Phase 0, not Phase 2.

## Resuming between sessions

The agent has no memory across sessions. Every new session starts by reading, in order: this spec, `NOTES.md`, and the last commit message. Every session ends by committing with a message naming the phase and what changed, and appending a one-line status to the bottom of the spec's checklist.

Keep all parameters in `config.py` and pin `requirements.txt` with exact versions. Set and record a random seed for every model and clustering step — a portfolio project whose numbers move between runs cannot be written up honestly.

## How the repo reads to a stranger

A recruiter reaches the repo before they reach the analysis. Ten minutes of work:

- Repo description and GitHub topics filled in (`churn-prediction`, `lightgbm`, `shap`, `subscription-analytics`).
- README opens with a screenshot of the dashboard and the live link, above the fold.
- Pinned on the GitHub profile.
- Commit history that reads as a project rather than one "initial commit" dump — the phase-per-commit rule handles this by itself.

## Cost discipline

- One phase per session where practical. Sonnet throughout.
- If the agent starts looping on an error, stop it and hand the error here rather than letting it retry — retries are where credits vanish.
- Log every run in the tracker immediately after it finishes.
- Estimated total: A$25–40 across all phases. If it passes A$50, stop and reassess. (Note: the gaming-to-streaming pivot cost real Phase 0/1/2 rework — factor that into any total, and don't repeat a domain pivot lightly a second time.)

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

**Title:** Subscriber Churn Early-Warning & LTV Segmentation **[pre-filled]**

**One sentence:** Predicts which music-streaming subscribers are about to cancel and crosses that risk against subscription value, so retention spend goes where it returns. **[pre-filled]**

**Target roles:** Data Analyst / Data Scientist, subscription & media/streaming. **[pre-filled]**

---

## S — Summary impact

*3–5 sentences: problem context, what you did, data source, methodology, key finding. Write this last, once the numbers exist.*

- **Problem context:** Subscription streaming services lose revenue when members cancel, and retention budget is often spent reactively rather than on the members whose subscriptions are actually worth saving. **[pre-filled]**
- **What you did:** built a subscriber-level churn risk score from real transaction and listening-activity logs and crossed it with subscription value to produce a retention priority matrix. **[pre-filled]**
- **Data source:** **[fill from the run — name, provenance, row count, date span]**
- **Methodology:** **[fill from the run — baseline vs LightGBM, out-of-time validation, SHAP, K-Means]**
- **Key finding:** **[fill from the run — the one sentence a retention lead would repeat to their manager]**

---

## P — Problem space

**Why this problem matters:** retention is cheaper than acquisition, and a subscriber flagged before they cancel can still be reached. Once they've gone, they've gone. **[pre-filled]**

**Who would care:** subscriber-retention and lifecycle-marketing teams; finance teams allocating retention budget by segment. **[pre-filled]**

**Scope:** one streaming service's subscriber base over a fixed window; predicting cancellation, not lifetime revenue forecasting. **[pre-filled]**

**Why this is a business problem, not a data exercise:** the output is a spending decision — which subscribers get an offer and how much it's worth paying to keep them. A model with no value dimension can't answer that. **[pre-filled]**

---

## I — Inputs

- **Primary source:** KKBox Churn Prediction Challenge (WSDM Cup 2018), Kaggle. `transactions.csv` (21,547,746 rows), `members_v3.csv` (6,769,473 rows), `user_logs_v2.csv` (18,396,362 rows). **[pre-filled, confirm exact figures used after Phase 1]**
- **Real or synthetic:** Real — production data from KKBox, a real subscription music-streaming service, released for a public competition. **[pre-filled]**
- **Collection method:** downloaded from Kaggle (competition data, rules accepted). **[pre-filled]**
- **Target variable:** cancelled within the prediction window, defined as **[Eileen — Phase 3 decision]**
- **Key predictors:** recency, frequency trend, listening intensity, real payment amount, tenure. **[pre-filled, confirm after Phase 4]**

---

## D — Discovery

**Data quality issues found:** **[fill from the run — Phase 2 profiling output]** — expect pre-registration transactions, promotional/duplicate billing accounts, and a possible representativeness gap between the one-month listening log and the multi-year transaction history.

**How you addressed them:** **[Eileen — the Phase 2 decisions and the reasoning behind each]**

**Surprising patterns:** **[fill from the run]**

**Initial hypotheses:** recency dominates; payment history predicts retention; trajectory beats level. **[pre-filled — check whether each held]**

**What actually happened:** **[Eileen]**

**Dead ends and methods that didn't work:** **[Eileen — this is the section hiring managers actually read; do not let the agent write it. The gaming-to-streaming domain pivot belongs here, in outline — why the original theme didn't survive contact with real data availability.]**

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

**Code highlights to showcase:** the chunked ingestion of a 21.5M-row transaction log and an 18.4M-row listening log, the leakage tests, the risk-vs-value cross-tab. Two or three snippets, not the whole notebook. **[pre-filled]**

---

## R — Results & recommendations

**Key results:** **[fill from the run — baseline vs model, with the numbers]**

**Business interpretation:** **[Eileen — in plain language, no metric names]**

**Recommendations:** **[Eileen — what a retention team does differently on Monday, per quadrant of the risk/value matrix]**

**Limitations and assumptions:** **[Eileen — what the data can't tell you; note the listening-activity window is one month while transactions span two years]**

**Next steps:** A/B test the retention offers against a holdout; extend listening-activity history beyond one month; retrain cadence and drift monitoring. **[pre-filled — trim to what you'd actually do]**

---

## Deliverables

- [x] GitHub repo with clean README
- [ ] Deployed Streamlit app
- [ ] `NOTES.md` decision log
- [ ] Portfolio site card
- [ ] LinkedIn post

## Success criteria

**Minimum viable:** a deployed app where a stranger can see the risk/value matrix and look up a subscriber, with a README that survives the "why this method" question.

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

**No.** The project goes past "here's what the data shows" on four counts: it investigates *why* subscribers cancel (SHAP), predicts *what happens next* (churn risk), compares segments, and ends in a spending recommendation. Required: at least two layers beyond charts. This has four.

## Pitfall 2 — Overused dataset with no new angle?

**This is the weakest axis for this project, and it should be named plainly rather than downplayed.** KKBox / WSDM Cup 2018 is one of the most-used real churn datasets in existence — a flagship Kaggle competition with thousands of public notebooks, mostly chasing leaderboard AUC on a pure classification task.

The differentiation has to come from what those notebooks don't do:
- Almost none of them cross churn risk against subscription value to produce a spending decision — they stop at "predict `is_churn`". The risk/value quadrant and the cost-based operating threshold are not something a leaderboard submission needs, so they're largely absent from the existing body of work.
- Almost none derive their own churn definition from the data — they import `train_v2.csv`'s label directly. This project derives its own from the gap distribution (Phase 3), which is both more defensible and a visibly different process.
- The out-of-time validation, leakage tests, and "what didn't work" narrative are standard portfolio discipline, not standard competition-notebook discipline.

Keep the risk/value framing visible in the title, the README opening, and the dashboard hero — harder than it would need to be for a less-picked-over dataset.

☑ **Eileen:** confirmed via web search (2026-09-08) — KKBox has an academic paper trail and heavy Kaggle-kernel presence, but essentially none of it uses the risk × value framing. Differentiation plan above stands.

## Pitfall 3 — Missing business context?

- **What problem exists?** Retention budget is spent on the wrong subscribers.
- **Who has it?** Subscriber-retention and lifecycle-marketing teams at subscription media companies.
- **Why it matters to them?** Retention is cheaper than acquisition, and a cancelled subscriber is unrecoverable without a fresh acquisition cost.
- **What changes if it works?** Offers go to high-risk high-value subscribers instead of everyone.

Would a company pay for this analysis? Yes — it is a standing function at every subscription media company. Three companies that would care: any music or video streaming service with a retention team, a telco bundling media subscriptions, a subscription-analytics vendor.

## Pitfall 4 — Tutorial clone?

The risks are real, so guard them deliberately:

- The churn window is derived from this data's own gap distribution, not copied from the official competition labeller or a blog post (even though that labeller is available as a reference — see Phase 3).
- `NOTES.md` records dead ends as they happen, so the write-up is not retrospectively tidy.
- The baseline-first rule means a result of "the simple recency rule was nearly as good" gets reported rather than buried.
- The domain pivot itself (gaming → subscription streaming, and why) is documented from Phase 0 and becomes part of the "what didn't work" narrative — genuinely unusual material most portfolio projects don't have.

☐ **Eileen:** if you consult a tutorial or paper, cite it in the README — "inspired by X, extended to Y".

## Final validation

**Data quality** — decided at Checkpoint 0. Event-level rows, real-world messiness, several usable variables, all confirmed by direct inspection (see Phase 0 record above).

**Business relevance** — cleared above.

**Technical showcase** — feature engineering, gradient boosting, model interpretability, clustering, and a front end. Five skill areas, comfortably past the three-skill bar.

**Differentiation** — rests on the risk/value framing and the cost-based threshold, working against a genuinely overused dataset. This is the weakest axis by some margin; protect it hardest of any project in the portfolio.

**Scope and feasibility** — designed for 2–3 weeks part-time, using tools already in the CV, on public data. Full `user_logs.csv` explicitly descoped to keep this true.

**Presentation potential** — four audience-facing deliverables in Phase 7.

**Verdict:** proceed. The one axis to watch, more than usual, is differentiation.

---

# Appendix C — Screening audit (run before Phase 7 ships)

Score the finished project 1–5 on each criterion. Below 4 on any row, fix it before the case study goes on the site.

| # | Criterion | What it means here | Score |
|---|---|---|---|
| 1 | Industry relevance | Does a subscription/media person recognise their own problem? Renewal cadence, cohorts, subscription-pricing language used correctly. | /5 |
| 2 | Depth of analysis | Does it explain *why* subscribers cancel, not just who? SHAP read and interpreted, not just plotted. | /5 |
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
5. This is a famously overused dataset — what did you do differently from the thousands of existing notebooks?
6. What breaks when this runs on 100× the subscribers, every day, in production?

## Interview-day one-liner

One sentence, ready to deliver: *"I built a churn early-warning model for a real music-streaming subscriber base and crossed the risk score against subscription value, so a retention team can spend budget on the subscribers actually worth keeping — the model flags [X]% of cancelling subscribers [N] days early."*

☐ Fill the bracketed numbers from the real results. Never a placeholder.

---

## Session log

- 2026-09-08 — Checkpoint −1 passed (go). Checkpoint 0 passed: dataset locked to `debs2x/gamelytics-mobile-analytics-challenge` (Kaggle) — real per-player revenue, 1M registered players, 9.6M login events. Phase 1 complete: repo scaffold created, `src/ingest.py` validates schema and prints a data quality summary (chunked for the 9.6M-row auth log), 3 tests passing, committed (`ff64558`). Next: Phase 2 (data quality investigation).
- 2026-09-08 — Phase 2 investigation on `debs2x` found it to be synthetic (bounded [1,7.04]-day random-walk login gaps; `reg_ts` exactly equals first `auth_ts` for all 1M users). Reverted the Checkpoint 0 decision. Explored real event-level alternatives in gaming (Dota 2 matches — too short a date span; LoL ranked matches — no persistent player ID across matches; WoWAH — real and workable but already used for published churn research and not F2P) and ultimately pivoted the whole project theme from gaming to subscription media / music streaming, since real public event-level data is much more available there. Project and folder renamed `player-churn-ltv` → `subscriber-churn-ltv`. New Checkpoint 0: **KKBox Churn Prediction Challenge (WSDM Cup 2018)** — verified real via gap-distribution and duplicate-rate checks (see Phase 0 record above). Scope includes listening-activity intensity features from `user_logs_v2.csv` (one month, March 2017) per Eileen's explicit request, alongside `transactions.csv` (21.5M rows, 2015–2017) and `members_v3.csv` (6.8M rows). Spec fully rewritten for the new domain. Next: rebuild Phase 1 (repo scaffold/ingestion already exists but is `debs2x`-shaped and needs to be redone for the KKBox schema).
- 2026-09-08 — Phase 1 rebuilt for KKBox: `src/config.py`/`src/ingest.py` target `members_v3.csv` (6,769,473 rows), `transactions.csv` (21,547,746 rows, chunked), `user_logs_v2.csv` (18,396,362 rows, chunked); `train_v2.csv` kept as reference-only. 3 tests passing. Committed (`6f51ce9`). Phase 2 investigation found: 12.3% of transaction rows have no matching member in `members_v3.csv` (join gap); 153 genuinely pre-registration transactions (69 members); 2.52% of rows (542,407, affecting 8.92% of members) are same-day repeat transactions — a billing-correction pattern, not real cadence; 153,660 rows (0.71%) have `membership_expire_date` before `transaction_date`, but 95.8% of those are `is_cancel=1` (immediate-cancellation is the legitimate explanation — not a defect), leaving only ~6,450 rows as genuine anomalies; zero missing calendar days in either the transaction span or `user_logs_v2`'s March 2017; `user_logs_v2` skews toward higher-value members (76.2% coverage of plausibly-active members, price distribution shifted up vs. absent members) — a real sampling characteristic, not fixable by dropping rows. Eileen's Checkpoint 2 decisions: drop unmatched-member transactions from member-joined analysis; drop the 153 pre-registration rows; agent's proposed handling adopted for the rest (dedupe same-day transactions to the last row for recency/frequency features; keep cancel-linked negative-coverage rows as real signal, flag/drop only the non-cancel ~6,450-row anomalous subset; treat `user_logs_v2` absence as a legitimate zero-listening signal, not something to impute, and document the skew as a named limitation).
- 2026-09-08 — Phase 3 (Checkpoint 3) resolved. Renewal-gap distribution computed across 19.2M transactions with a known next renewal: 77.7% renew on/before expiry, 90.1% within 1 day, 96.8% within 30 days, long thin tail beyond that. Eileen's decisions: **churn threshold = 30 days** no-renewal-after-expiry (matches the official WSDM rule and the point the curve flattens); **observation window = full transaction history to cutoff** per member (not a fixed rolling window); **eligibility = exclude members with no prior transaction** before the labeled expiry event (the full-history equivalent of excluding mid-window registrants); **windowing scope = option (a)** — build all training/validation windows entirely inside `transactions.csv` (2015-01-01 to 2017-02-28), treat `user_logs_v2` (March 2017, non-overlapping with transactions.csv) as live-scoring-only enrichment, never backtested. Implemented in `src/config.py` (`CHURN_WINDOW_DAYS`, `OBSERVATION_WINDOW`, `MIN_PRIOR_TRANSACTIONS`, `DATA_MAX_DATE`) and documented in plain English at the top of `src/features.py`. Exact train/validation calendar cutoffs deferred to Phase 4/5. Next: Phase 4 (feature engineering).
- 2026-09-08 — Phase 4 (feature engineering) built and, after catching a real bug, fixed. `src/features.py` implements cleaning (per Checkpoint 2), labeled-event selection, member-level feature building (recency/frequency/monetisation/tenure/trajectory), and 4 leakage tests (`tests/test_features.py`, all passing on synthetic fixtures) — see spec Phase 4 acceptance criteria. **Bug caught before shipping**: the first design picked each member's own last eligible transaction as their one labeled row and split train/validation by that date. On the real data this produced 69% churn in train vs 7% in validation from the same population — the cutoff date itself was a proxy for churn status, since a churned member's transaction history simply stops early while an active member's keeps getting pushed later. This is exactly the "a result looks too good/off, don't push forward" signal the spec's guardrails call out. Fixed by evaluating every member as of a shared external reference date per split (`identify_labeled_events(max_transaction_date=..., exclude_msno=...)`, orchestrated by `build_train_validation_matrices`), guaranteeing disjoint membership by construction and removing the date-as-churn-proxy leak. Real run with `TRAIN_CUTOFF_DATE=2016-06-30` / `VALIDATION_CUTOFF_DATE=2017-01-29` (the latest date a labeled expiry can still resolve within the data): train 1,125,164 rows / 41.2% churn, validation 217,977 rows / 30.6% churn, zero member overlap, zero nulls in any feature column. The ~10-point churn-rate gap between splits is normal out-of-time drift, not a leakage signal. Both churn rates are well above the well-known WSDM competition label's 9.0% — expected, since that label snapshots only currently-active subscribers at one point in time, while ours evaluates every member with enough history as of a reference date, including many who lapsed long ago and never returned; this needs to be addressed head-on in the README, not left looking like an error. Feature matrices saved to `data/processed/train_features.parquet` and `validation_features.parquet` (gitignored). Committed. Next: Phase 5 (baseline + LightGBM model).
- 2026-09-09 — Phase 5 (model). Built recency-rule baseline, logistic regression, LightGBM, and SHAP per the spec — then caught and fixed two more real bugs before trusting any of it, on top of the Phase 4 one. **Bug 2**: a tuned LightGBM showed near-perfect PR-AUC (~0.97-0.99) on an internal split of TRAIN but collapsed to PR-AUC=0.54 (worse than the untuned model) on real validation — traced to a genuinely *missing* feature, not leakage: `days_since_last_transaction` measured the gap between two of a member's own past transactions, but never the gap from their last transaction to the actual scoring/reference date, which is the single most fundamental recency signal in churn modeling (non-churned members' labeled transaction sits a median 14 days before the reference date; churned members' sits a median 151 days before it). The model was forced to reconstruct that signal indirectly through feature interactions, which memorized the specific snapshot it was tuned on instead of learning something transferable. Fixed by adding `days_since_cutoff` (reference_date - cutoff_date) directly to `build_member_features` — fully known at scoring time, not leaky. **Bug 3**: guaranteeing train/validation member-disjointness via `exclude_msno` (excluding anyone already used in train) turned out to silently restrict validation to only newer/shorter-tenured members, since anyone long-tenured enough to be eligible by the train cutoff was always claimed by train first. This wasn't a time split, it was a population-composition split wearing a time split's clothes — it broke LightGBM catastrophically (ROC-AUC fell *below* 0.5, worse than random) while logistic regression partially masked it through feature scaling, which is what exposed the gap between the two models and led to the investigation. Fixed by partitioning members into random disjoint pools *before* any time-based eligibility logic runs (`build_train_validation_matrices`), confirmed by comparing `tenure_days` distributions across splits (now closely matched: train median 893 days, validation median 898). Also discovered along the way: naively splitting TRAIN by `cutoff_date` for internal hyperparameter-tuning purposes reproduces the exact same churn-rate-vs-date confound as bug 3 (one internal split attempt produced a 94.8%-churn population vs a 17.6%-churn population) — random stratified splitting is the correct approach for internal tuning here, specifically because `cutoff_date` is inherently entangled with the label by construction of this whole labeling methodology. Final real validation numbers, all on the corrected split: recency rule ROC-AUC=0.567/PR-AUC=0.618; **LightGBM ROC-AUC=0.676/PR-AUC=0.647 (beats the baseline)**; logistic regression ROC-AUC=0.936/PR-AUC=0.925 (the strongest of the three — reported honestly rather than favouring the "primary" model). A hyperparameter-tuning attempt didn't beat the simple untuned LightGBM config (n_estimators=300, lr=0.05, num_leaves=31, is_unbalance=True) on real validation, so the simple config is the production one. SHAP confirms `days_since_cutoff` as the dominant feature (mean|SHAP|=3.72, dwarfing everything else) and the previously-flagged sharp-threshold pattern in `days_since_last_transaction` resolved once the real recency signal had its own feature. **Operating threshold, per Eileen's explicit delegation** ("can you pick the threshold"): swept 0.05-0.95 maximizing expected net benefit under assumed costs (retention offer cost=75, retained-subscriber value=894 [= 6 x the modal plan price of 149, and itself a real observed plan price], offer success rate=30%, all clearly marked ASSUMED not observed) — chosen threshold **0.88** (precision=0.459, recall=0.842, flags 72.8% of validation, a wide net that's the correct consequence of a cheap offer against a high potential payoff, not a bug). All in `src/config.py`, `src/model.py` (`select_operating_threshold`), 8/8 tests passing. Next: Phase 6 (segmentation and value).
- 2026-09-09 — Phase 6 (segmentation and value). `src/segment.py` built: K-Means on behavioural features only (recency/frequency/monetisation/tenure/trajectory — never on `is_churn` or the predicted risk score, which would make "high-risk segment" a tautology), scored population = validation set (genuine held-out LightGBM predictions). k chosen via elbow + silhouette sweep (k=2..8): silhouette peaks at k=3 (0.302), k=7 a close second (0.294); went with k=3 for both statistical support and dashboard interpretability. Segment profiles (unnamed) presented to Eileen; she delegated both naming and the value-threshold decision ("your choice"). Named, grounded directly in the stats: **segment 0 "Drifting Standard Subscribers"** (144,725 / 53.8%, 55.8% churn, moderate tenure, infrequent, silent ~7mo); **segment 2 "Loyal Frequent Renewers"** (116,789 / 43.4%, 17.0% churn, long-tenured, frequent, recently active — the stable core); **segment 1 "High-Value Subscribers Going Dark"** (7,314 / 2.7%, 81.9% churn, ~7x average spend/transaction, long-tenured but infrequent, silent ~8.7mo). Risk-value quadrant (risk threshold = `OPERATING_THRESHOLD` 0.88, value threshold = median `avg_amount_paid_prior` = 149, which happens to be the real modal single-plan price — kept for that reason): **Priority Save** (high risk/high value, 44.8%, 47.6% churn), **Low-Cost Nudge** (high risk/low value, 28.0%, 43.1% churn), **Stable, No Action** (low risk/low value, 15.0%, 9.5% churn), **Quiet Value — Monitor** (low risk/high value, 12.2%, **39.6% churn — deliberately not called "safe"**, since OPERATING_THRESHOLD is tuned for net-benefit-maximizing recall, not for cleanly separating safe from at-risk). Quadrant names and recommendations are the agent's deliverable per spec Phase 6 (only segment naming was explicitly Eileen's call, though she delegated that too). 11/11 tests passing (`tests/test_segment.py` added, synthetic fixtures — no dependency on real data being present). Next: Phase 7 (dashboard, presentation, report, website case study).
- 2026-09-09 — Phase 7a (HTML dashboard) built. `docs/index.html` — a single self-contained file (225KB, embedded data, no server, no fetch calls, works opened directly as a local file or served via GitHub Pages from /docs). Contents per spec: risk-value quadrant as the hero (true 2x2 spatial layout, not four disconnected cards), 3 segment cards, an interactive threshold slider (live precision/recall/flagged-count/net-benefit readout against the real threshold sweep), a subscriber lookup (400 real validation-set subscribers, sampled across risk deciles, each with plain-language SHAP-derived reasons — e.g. "Hasn't renewed in about 7 months" rather than a feature name and a number), and a "Show the numbers" toggle revealing the full model comparison table (honestly showing logistic regression beats the deployed LightGBM on PR-AUC) and the assumed-cost disclosure. No metric jargon on the surface. Design: Fraunces/IBM Plex Sans/IBM Plex Mono type system, a violet-indigo accent distinct from generic AI-dashboard defaults, full light/dark theming, dataviz-skill-validated categorical/status colour palette. Caught and fixed one bug before shipping: the plain-language explanation generator for `auto_renew_rate` branched on the raw feature value instead of the actual SHAP direction, occasionally producing reassuring-sounding text labeled as risk-increasing — fixed to branch on SHAP sign first. Data generation (headline stats, threshold sweep, segment/quadrant profiles, subscriber SHAP sample) consolidated into `app/build_dashboard.py` (reproducible: `python -m app.build_dashboard`, verified to regenerate a byte-identical file) with `app/dashboard_template.html` as the template — not left stranded in scratch scripts. Note: the spec's old Deliverables checklist mentions a "Streamlit app" (`app/streamlit_app.py`, left as an unused placeholder) but Phase 7's actual detailed instructions call for a static HTML dashboard deployed via GitHub Pages, which is what got built — the checklist item is stale boilerplate, not a second deliverable. Published live as a Claude Artifact for immediate review; GitHub Pages deployment itself needs Eileen to enable Pages (serve from /docs) in the repo's GitHub settings, which only she can do. 11/11 tests still passing. Next: Phase 7b (presentation).
- 2026-09-09 — Phase 7b (presentation) built. `deliverables/retention-signal-presentation.pptx` — 10 slides, built with the pptx skill (pptxgenjs from scratch, not a template): title, the problem, data & method (with a native line chart of the real renewal-gap curve), the model (native bar chart, honestly showing logistic regression beating the deployed LightGBM on PR-AUC — same finding as the dashboard, not softened here), three segments, the risk-value quadrant as a 2x2 grid, recommendation per quadrant, cost/return under clearly-labeled assumed economics, limitations (marked "draft, for review" since qualitative judgment on this belongs to Eileen per spec Phase 8, though the specific facts drafted are all already-established findings, not new opinion), and next steps (verbatim from the spec's pre-filled Appendix A list). Custom content-informed palette (deep violet + risk-red + status green, tied to the dashboard's own identity) and Cambria/Calibri type pairing, both from the skill's safe-font list. `python scripts/office/validate.py` passed clean; visual QA via PowerPoint COM automation (no LibreOffice on this machine, so the skill's usual soffice.py path didn't work — worked around it by rendering slides through installed PowerPoint instead) caught and fixed 4 real issues before shipping: a title-slide text overlap, chart data labels rounding 0.618/0.647/0.925 all down to "1" (missing a decimal format code), vertical accent bars on the recommendation slide (the skill explicitly flags these as an AI-generated-slide hallmark — replaced with colored dots matching the segment cards' own established language), and a page number invisible against a dark slide background. Generator script kept at `app/build_deck.js` for reproducibility. Next: Phase 7c (written report).
