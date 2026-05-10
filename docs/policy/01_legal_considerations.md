# Legal Considerations

**Document:** `docs/policy/01_legal_considerations.md`
**Version:** 1.0
**Created:** 2026-05-03
**Status:** Active — requires legal review before any operational deployment

---

## 1. Purpose of This Document

This document flags legal considerations identified during development. It does not constitute legal advice. Legal review by qualified counsel is required before this system is used in any operational, enforcement, or policy-making capacity.

## 2. 23 U.S.C. § 409 — Highway Safety Data Protection

### 2.1 What § 409 Says

Section 409 of Title 23, United States Code, provides that crash data reported to the federal government for highway safety purposes is exempt from discovery or admission in federal or state court proceedings. This protection exists to encourage honest and complete crash reporting by law enforcement agencies.

### 2.2 How This Project Uses § 409 Protected Data

FTFLFD uses crash data from the Vermont Public Crash Query Tool, which is publicly available. The VTrans Public Query Tool disclaimers explicitly state that the data is "exempt from discovery or admission under 23 U.S.C. 409."

### 2.3 Potential Tension

Using § 409 protected data in a predictive model that could inform resource allocation, enforcement deployment, or infrastructure investment creates a tension:

- The data is protected from legal scrutiny precisely because it may contain errors and omissions
- The model treats the same data as ground truth for prediction
- If predictions inform operational decisions and adverse outcomes occur, the § 409 protection of the underlying data may face legal challenge

### 2.4 Project Position

This project is:

- An ad honorem research and proof-of-concept effort
- Built entirely with publicly available, free data
- Intended as an observational and predictive tool, not an enforcement directive
- Designed to demonstrate that proactive crash risk prediction is technically feasible

What higher management, legislature, or enforcement agencies choose to do with a validated tool is an advocacy and policy decision beyond the scope of this project. The project creates solutions. Operational adoption requires institutional decision-making that includes legal review.

### 2.5 Required Before Operational Deployment

Before this system is used to direct any operational activity (patrol deployment, infrastructure prioritization, budget allocation), the following must occur:

- Formal legal review of § 409 implications by qualified counsel
- Institutional approval from the deploying agency
- Documentation of how model outputs are used in decision-making
- Liability framework for model errors (false negatives, false positives)

## 3. Data Privacy

- Only non-personal data is used (per VTrans Public Query Tool disclaimers)
- No personally identifiable information exists in the dataset
- Raw crash data is never committed to the public GitHub repository
- Aggregated outputs only appear in `data/public/`

## 4. Open Source License

This project uses the Apache 2.0 license, which includes an explicit patent grant. This was chosen specifically because it provides clearer legal footing for institutional adoption compared to more permissive licenses like MIT.

---

**This document does not resolve the legal questions it raises. It documents them for future resolution by qualified parties.**