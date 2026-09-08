"""Feature engineering — transaction/listening-level then member-level, built from the observation window only.

Churn, in plain English: a member has churned if they let a subscription expire
and don't come back with a new transaction within 30 days of that expiry date.
Most renewals happen same-day (auto-renew); the ones that don't mostly resolve
within the first few days, and by 30 days almost everyone who was coming back
already has. A member who's silent past that point is counted as churned.

Each member's features are built from their entire transaction history up to
whatever cutoff date is being used to label them — not a fixed rolling window.
A member has to have at least one prior transaction before the expiry event
being labeled, so there's real history to build recency/frequency/trajectory
features from. Listening-intensity features (from user_logs_v2, March 2017)
are for scoring current members only — they fall outside the range the
transaction data can independently confirm outcomes for, so they're never used
to train or validate the churn model itself.

See src/config.py for the exact parameters (CHURN_WINDOW_DAYS, OBSERVATION_WINDOW,
MIN_PRIOR_TRANSACTIONS, DATA_MAX_DATE) and spec Phase 3 for the full reasoning.

Feature building itself is implemented in Phase 4, once this definition is locked.
"""
