# 🏛️ LexVerify CRAG — Demo Showcase

> **Corrective Retrieval-Augmented Generation for Legal Citation Integrity**
>
> Every answer is verified. Every citation is real. Every claim is grounded.

---

## Quick Start

```bash
# Install & configure
pip install -e .
cp .env.example .env  # Add your API keys

# Ingest legal documents
python -m scripts.ingest_data

# Run your first query
python -m src.main query "What is the statute of limitations for PI in Florida?" -j Florida
```

---

## Demo Cases

### 📋 Case 1: Statute of Limitations — Accurate, Current Law

> **Capability shown:** Vector retrieval → CRAG evaluation → Cited generation → Hallucination grading

Most AI systems still say Florida's PI statute of limitations is **4 years** (pre-2023 law). LexVerify gets it right.

```bash
python -m src.main query \
  "What is the statute of limitations for personal injury claims in Florida?" \
  -j Florida
```

**Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In Florida, the statute of limitations for personal injury claims based on
negligence is two years from the date the cause of action accrues. This change
was enacted by HB 837 and became effective on March 24, 2023. Prior to this
amendment, the statute of limitations was four years. The new two-year limitation
applies to causes of action accruing after March 24, 2023, while the previous
four-year limitation applies to causes of action that accrued before that date
[Source 1].

📚 Citations:
  [1] https://www.flsenate.gov/Laws/Statutes/2023/95.11

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** ChatGPT and other LLMs still report the outdated 4-year limit. LexVerify pulls from verified, current statute text and cites the source.

---

### ⚖️ Case 2: Comparative Negligence — Practical Legal Advice

> **Capability shown:** GraphRAG enrichment → HB 837 amendment tracking → Contextual legal reasoning

```bash
python -m src.main query \
  "What happens if I'm 60% at fault in a car accident in Florida?" \
  -j Florida
```

**Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In Florida, if you are found to be 60% at fault in a car accident, you would
be BARRED FROM RECOVERING ANY DAMAGES due to the state's modified comparative
negligence rule. As per Florida Statute § 768.81, amended by HB 837 in 2023,
a claimant who is more than 50% at fault for their own harm cannot recover any
damages [Source 1]. This change from the previous pure comparative negligence
system applies to negligence actions filed after March 24, 2023 [Source 2].

📚 Citations:
  [1] https://www.flsenate.gov/Laws/Statutes/2023/768.81
  [2] https://www.flsenate.gov/Session/Bill/2023/837

🔗 Knowledge Graph Context:
  • fl-stat-768-81: Amended by HB 837 — Tort Reform Act of 2023
  • fl-hb837-overview: Modifies §§ 95.11, 768.81

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** The answer depends on *when* the accident happened (before/after March 2023). LexVerify surfaces the temporal context from the knowledge graph.

---

### 🔬 Case 3: Medical Malpractice Pre-Suit — Procedural Precision

> **Capability shown:** Precise retrieval → Multi-source citation → Step-by-step procedural breakdown

```bash
python -m src.main query \
  "What are the pre-suit requirements for a medical malpractice case in Florida?" \
  -j Florida
```

**Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before filing a medical malpractice action in Florida, a claimant must:

1. Medical Expert Opinion: Obtain a verified written opinion from a medical
   expert confirming reasonable grounds to believe the defendant was
   negligent [Source 1].

2. Notice of Intent: Serve notice on each prospective defendant at least
   90 days before filing suit [Source 1, Source 4].

3. 90-Day Pre-suit Period: Exchange preliminary information and engage in
   informal discovery. This allows for settlement before formal litigation
   [Source 1].

4. Statute of Limitations: Two years from discovery of injury, with an
   absolute four-year statute of repose. Exceptions for fraud, concealment,
   or intentional misrepresentation [Source 1].

📚 Citations:
  [1] https://www.flsenate.gov/Laws/Statutes/2023/766.106
  [4] https://codes.findlaw.com/fl/title-xlv-torts/fl-st-sect-766-106/

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** A missed pre-suit requirement can get a case dismissed. LexVerify pulls the exact procedural steps from the statute.

---

### 🔗 Case 4: Good Law Detection — GraphRAG in Action

> **Capability shown:** Knowledge graph traversal → Overturned statute detection → Case law relationships

```bash
python -m src.main query \
  "Are there caps on non-economic damages in Florida medical malpractice?" \
  -j Florida
```

**Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In Florida, there are currently NO CAPS on non-economic damages in medical
malpractice cases. This is due to two Florida Supreme Court decisions:

• Estate of McCall v. United States (2014) — struck down caps for wrongful
  death med mal as violating equal protection [Source 2].

• North Broward Hospital District v. Kalitan (2017) — extended McCall to
  personal injury med mal, invalidating § 766.118 entirely [Source 1].

The Legislature's justification for the cap was not supported by evidence
and the alleged insurance crisis had dissipated.

📚 Citations:
  [1] https://casetext.com/case/n-broward-hosp-dist-v-kalitan
  [2] https://casetext.com/case/estate-of-mccall-v-united-states-1

🔗 Knowledge Graph Context:
  • Former § 766.118 (INVALIDATED) ──overturned_by──→ McCall (2014)
  • Former § 766.118 (INVALIDATED) ──overturned_by──→ Kalitan (2017)
  • Kalitan ──cites──→ McCall

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** The statute is still "on the books" but **no longer enforceable**. The knowledge graph tracks that McCall and Kalitan overturned it — information a pure vector search would miss.

---

### 🌐 Case 5: Web Search Fallback — Beyond the Index

> **Capability shown:** CRAG evaluator detects insufficient retrieval → Tavily web search fallback → Grounded response from live sources

```bash
python -m src.main query \
  "What are the requirements for a class action lawsuit in Florida?" \
  -j Florida
```

**Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Response synthesized from web search results when the topic
is not in the Pinecone index]

📚 Citations:
  [1] https://www.findlaw.com/... — Florida class action requirements
  [2] https://www.floridabar.org/... — Rule 1.220 class certification

⚠️ Grounding Score: 75% of claims verified

⚠️ Ungrounded Claims:
  • [Specific claim flagged for manual review]
```

> 💡 **Why this matters:** When retrieved documents don't cover the topic, the CRAG evaluator triggers web search automatically. The hallucination grader then flags which claims still need verification.

---

### 🗺️ Case 6: Cross-Jurisdiction Comparison — Multi-Step Reasoning

> **Capability shown:** Query decomposition → Parallel CRAG pipelines → Comparative synthesis

```bash
python -m src.main compare \
  "Compare the statute of limitations for personal injury claims across Florida, California, and Federal law"
```

**Result:**

```
⚖️ LexVerify Multi-Step Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decomposed into 3 sub-queries:
  1. [Florida] What is the statute of limitations for PI in Florida?
  2. [California] What is the statute of limitations for PI in California?
  3. [Federal] What is the statute of limitations for PI under Federal law?

📋 Comparative Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Jurisdiction │ Limitation │ Key Statute      │ Notes
  ─────────────┼────────────┼──────────────────┼──────────────────────
  Florida      │ 2 years    │ § 95.11(3)(a)    │ Changed from 4yr by
               │            │                  │ HB 837 (Mar 2023)
  California   │ 2 years    │ CCP § 335.1      │ Discovery rule may
               │            │                  │ toll the period
  Federal      │ Varies     │ Borrows state    │ FTCA: 2yr admin
               │            │ SOL              │ filing deadline

Key Differences:
  • Florida had a 4-year period before HB 837 (March 2023)
  • California explicitly allows tolling under the discovery rule
  • Federal law borrows the local state's SOL for § 1983 claims

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** A single question about "three jurisdictions" becomes three verified sub-queries, each running through the full CRAG pipeline, then synthesized into a lawyer-ready comparison.

---

### 🏛️ Case 7: Sovereign Immunity — Government Liability

> **Capability shown:** Jurisdiction routing → Specific statute retrieval → Practical damage cap information

```bash
python -m src.main query \
  "Can I sue the state of Florida for a car accident caused by a government vehicle?" \
  -j Florida
```

**Expected Result:**

```
📋 Legal Response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Yes, but with significant limitations. Under Florida Statute § 768.28,
Florida waives sovereign immunity for tort claims, but liability is capped:

  • $300,000 per claim
  • $500,000 per incident

Claims exceeding these caps require a special legislative claims bill.

You must provide written notice to the appropriate agency and allow 180 days
for investigation before filing suit.

📚 Citations:
  [1] https://www.flsenate.gov/Laws/Statutes/2023/768.28

✅ Grounding Score: 100% of claims verified
```

> 💡 **Why this matters:** The answer isn't just "yes" or "no" — it identifies the specific statutory waiver, the dollar caps, and the mandatory pre-suit notice period.

---

## Architecture at Work

Each demo case flows through the same verified pipeline:

```
Query → Route → Retrieve → GraphRAG Enrich → CRAG Evaluate → Generate → Grade
                                   │                │
                                   │           ┌────┴────┐
                                   │           │ AUGMENT  │
                                   │           │ (Tavily) │
                                   │           └─────────┘
                                   │
                              Citation chains
                              Good Law status
                              Amendment history
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `python -m src.main query "..." -j Florida` | Standard CRAG pipeline with jurisdiction hint |
| `python -m src.main query "..." --fast` | Use distilled critic for faster evaluation |
| `python -m src.main query "..." --multi-step` | Force multi-step reasoning |
| `python -m src.main compare "..."` | Cross-jurisdiction comparative analysis |
| `python -m src.main info` | Display current configuration |

## Why LexVerify?

| Feature | ChatGPT / Generic RAG | LexVerify CRAG |
|---------|----------------------|----------------|
| **Citation accuracy** | Hallucinates citations | Every citation links to a real source |
| **Current law** | Trained on stale data | Retrieves from verified, current corpus |
| **Good Law status** | No awareness | GraphRAG tracks overturned/amended status |
| **Confidence scoring** | No indication | Grounding score on every response |
| **Ungrounded claims** | Silent failures | Flagged explicitly for review |
| **Cross-jurisdiction** | Inconsistent | Structured decomposition + parallel verification |
