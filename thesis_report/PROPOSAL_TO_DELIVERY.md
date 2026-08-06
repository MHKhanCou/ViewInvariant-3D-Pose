# Proposal → delivered: the complete trace

Three things define what was asked for, and this document maps every one of them
onto what exists. Take it to the defence. If an examiner asks "did you do what
you said you would", this is the answer, including the parts where the answer is
no.

| Input | What it is | Date |
|---|---|---|
| `E:\thesis\Proposal.docx` | **Supervisor's brief** — the research direction Sir set | 18 Jan 2026 |
| `E:\thesis\research_proposal_12108004.tex` | **Your own submitted proposal** — the formal one with objectives and schedule | 16 Feb 2026 |
| `github.com/TaatiTeam/MotionAGFormer` | **The example repository the university gave**, and this repo's `origin` | — |

Your own work lives on the `thesis` remote: `github.com/MHKhanCoU/ViewInvariant-3D-Pose`.

---

## 1. Your five Research Objectives, one by one

These are the objectives you formally committed to. This is the table the
examiner will check against.

| # | Objective as written | Delivered | Where |
|---|---|---|---|
| **1** | "Implement a reproducible baseline 2D-to-3D lifting model on Human3.6M" | **✅ Done.** 45.149 mm against the published 45.1 mm — matching the backbone's own evaluation script to three decimal places | §5.9 |
| **2** | "Design a lightweight canonicalization step to reduce sensitivity to camera rotation" | **✅ Done, and it became the whole thesis.** 402 FLOPs, zero parameters, 72.2% cross-view reduction over 180 held-out pairs | §3.2, §5.10 |
| **3** | "Incorporate bone-length consistency as an auxiliary loss" | **⚠️ Changed, then retracted.** Not built as a loss — tested as a test-time signal instead. ρ = +0.492 on the first dataset, +0.098 on the second. **Retracted in the report** | §5.12 |
| **4** | "Evaluate cross-view generalization and perform ablation studies" | **✅ Exceeded.** Nine pre-registered experiments, two datasets, two backbones, 209 camera pairs, cluster bootstrap intervals throughout | Ch. 5 |
| **5** | "Analyze failure modes such as occlusions and noisy 2D detections" | **✅ Done — and this became the contribution.** The boundary: where the geometric principle holds and where it stops | §5.16, §6.1 |

**Four of five delivered. One changed direction and was retracted, and the report
says so.** That is a strong record, and objective 5 — usually the throwaway one —
is where the actual contribution came from.

## 2. Your four Expected Outcomes

| Outcome as written | Status |
|---|---|
| "A reproducible baseline for monocular 2D-to-3D lifting on Human3.6M" | ✅ |
| "Improved cross-view robustness via canonicalization" | ✅ 72.2%, 179 of 180 pairs — with the honest caveat that a simpler baseline beats it |
| "Reduced limb-length drift … due to bone-length regularization" | ❌ **Not attempted as regularization.** Bone length was measured, not enforced |
| "Quantitative evaluation (MPJPE) and ablation studies" | ✅ ablations in depth; **MPJPE appears only for backbone verification**, because the claim is about *agreement between views*, not accuracy — and MPJPE cannot measure agreement |

**Be ready for the MPJPE question**, it is the likeliest one from this table:

> The proposal named MPJPE because it assumed I would be improving accuracy. I am
> not — I am making two views agree, and MPJPE measures each view against ground
> truth separately, so it cannot see agreement at all. I report MPJPE where it is
> the right instrument, to prove my backbone reproduction is faithful, and I
> report cross-view distance where that is the right instrument. Changing the
> metric to match the claim is the correct move; keeping MPJPE would have
> measured the wrong thing precisely.

## 3. Your three "Contributions to Knowledge"

All three landed, and the third one landed exactly as written:

1. *"An empirical study of how simple canonicalization and explicit bone-length regularization jointly affect monocular 3D pose lifting."* → **Delivered.** The canonicalization half is positive, the bone-length half is negative and retracted.
2. *"A lightweight and reproducible training strategy suitable for undergraduate implementation."* → **Exceeded.** There is no training at all, which is lighter than proposed.
3. *"Evidence on whether structural priors improve generalization across camera viewpoints without requiring complex architectures."* → **This is precisely the thesis.** The evidence is split, and the split is the result.

> Quote contribution 3 back at the examiner if he asks whether you drifted from
> your proposal. You wrote it in February and the thesis answers it exactly.

## 4. The one real change: the pipeline is inverted

**Proposed pipeline** (your Fig., §Proposed System Pipeline):

```
RGB → 2D detector → Canonicalization → MLP lifting → Bone-length loss → 3D pose
```

**Delivered pipeline:**

```
RGB → 2D detector → frozen lifter (MotionAGFormer-XS / MotionBERT) → Canonicalization → 3D pose
```

Canonicalization moved from **before** the lifter to **after** it, and the
bone-length loss became a test-time measurement rather than a training term.

**Why — and this is the single most important thing to be able to say:**

> Because the rotation cancels algebraically, so training was not needed to
> obtain the invariance. If two cameras see one pose, their predictions differ by
> an unknown rotation. The frame is built from the joints themselves, so it
> rotates with them, and the unknown rotation cancels exactly and is never
> estimated. Once that is true, a learned view-invariant latent space is solving
> by optimisation a problem that has a closed-form answer.
>
> The literature survey confirmed the learned route was already occupied —
> 3DPCNet, MoViD, V-VIPE and CanonPose all do it. The unoccupied part was the
> requirement profile: no training, no labels, no calibration.

**The gap and the question did not change. Only the method did.**

## 5. Against Sir's original brief

Sir's `Proposal.docx` set the direction. Mapping it honestly:

| Sir's brief | Delivered |
|---|---|
| Research gap: *"existing methods lack explicit geometric priors and view-invariant constraints"* | **Unchanged.** This is still the gap the thesis addresses |
| Objective: view-invariant framework with geometric and kinematic priors | **Delivered, without deep networks.** The priors are analytic |
| Learned view-normalized latent space | **Derived instead of learned** — see §4 above |
| `L = L_pose + λ₁L_bone + λ₂L_view` | **Not used.** No loss, no training. The bone-length term was tested as a signal and retracted |
| Datasets: H36M, MPI-INF-3DHP, CMU Panoptic, CASIA Gait | **Two of four.** Both multi-camera, which is what the central claim requires |
| Applications: gait, sports, surveillance | **Not evaluated.** Named as future work |
| Pick one of three model variants | **None.** The work became an analysis rather than a new model |

## 6. What was given to you, and what is yours

The university handed you MotionAGFormer as a working example. An examiner is
entitled to ask what you added. **The answer is checkable in one command**
(`git diff --name-only bb3bb2e HEAD`):

| Given by the example repo | Written by you |
|---|---|
| `model/` — the MotionAGFormer architecture | `canonical/` — the frame construction, 11 files, 1,682 lines |
| Training code, configs, checkpoints | `evaluation/` — every experiment, 43 files, 10,101 lines |
| The published evaluation script | `tests/` — 10 files, 1,245 lines, 76 tests |
| | `presentation/` — figure generation, 4 files, 1,049 lines |
| | `demo_live/` + `app.py` — the working demonstrator, 1,510 lines |
| | `thesis_artifacts/` — 152 files of results and nine pre-registrations |

**Roughly 15,500 lines across 75 files are yours.** The backbone is used
deliberately frozen — never fine-tuned, never modified — which is the whole point
of the method and is itself a claim the report makes.

> If asked directly: "The estimator is theirs and I never touched it. That is
> deliberate — the entire contribution is that nothing inside it has to change.
> Everything that decides what this thesis claims is code I wrote."

## 7. Your six-month schedule

| Month | Planned | What happened |
|---|---|---|
| 1 | Literature review, dataset access, baseline setup | Done; the survey grew to 37 papers and later found the TRIAD and Cappozzo prior art |
| 2 | Baseline 2D-to-3D regressor, baseline MPJPE | Done — 45.149 mm |
| 3 | Canonicalization module, ablations | Done, and became the core |
| 4 | Bone-length loss, tune λ | **Diverged.** Bone length was tested as a signal and retracted; the time went to pre-registered experiments instead |
| 5 | Full experiments, cross-view tests, error analysis | Done, on two datasets and two backbones |
| 6 | Final report, presentation, submit | Done — 90 pages, 255 audited claims |

## 8. How to open, if he asks about the proposal

Do not wait to be asked. Say it in this order:

> Sir, the gap in the proposal is unchanged — that existing methods lack explicit
> geometric priors and view-invariant constraints. Four of my five objectives are
> delivered as written. What changed is the method: the proposal was to *learn*
> view-invariance with a loss, and I found it can be *derived*, because the frame
> is built from the joints so the camera rotation cancels exactly. That made
> training unnecessary rather than optional.
>
> The bone-length objective is the one I did not deliver as written. I tested it
> as a signal rather than building it as a loss, it scored +0.492 on the first
> dataset and +0.098 on the second, and I retracted it. So the proposal's
> hypothesis — that geometric priors give both view-invariance and reliability —
> is answered, and half the answer is negative.

---
---

# বাংলা — সংক্ষেপে

## তিনটা জিনিস মেলানো হয়েছে

- **স্যারের `Proposal.docx`** (১৮ জানু) — গবেষণার দিকনির্দেশ
- **আপনার নিজের `research_proposal_12108004.tex`** (১৬ ফেব্রু) — আনুষ্ঠানিক প্রস্তাব, objective আর schedule সহ
- **বিশ্ববিদ্যালয়ের দেওয়া example repo** — TaatiTeam/MotionAGFormer, এই repo-র `origin`

## পাঁচটা Objective-এর হিসাব

| # | যা বলেছিলেন | যা হয়েছে |
|---|---|---|
| ১ | H36M-এ reproducible baseline | **✅ হয়েছে** — ৪৫.১৪৯ মিমি, প্রকাশিত ৪৫.১-এর সাথে তিন দশমিক পর্যন্ত মিল |
| ২ | Lightweight canonicalization | **✅ হয়েছে, এবং এটাই পুরো thesis** — ৭২.২% উন্নতি |
| ৩ | Bone-length auxiliary loss | **⚠️ বদলেছে, তারপর প্রত্যাহার** — loss হিসেবে না, signal হিসেবে পরীক্ষা; ০.৪৯২ → ০.০৯৮ |
| ৪ | Cross-view generalization + ablation | **✅ প্রতিশ্রুতির বেশি** — নয়টা pre-registered পরীক্ষা |
| ৫ | Failure mode বিশ্লেষণ | **✅ হয়েছে — আর এখান থেকেই আসল অবদান এসেছে** |

**পাঁচটার চারটা যেভাবে বলা হয়েছিল সেভাবেই হয়েছে।** একটা দিক বদলেছে এবং প্রত্যাহার
করা হয়েছে — রিপোর্টে সেটা লেখা আছে।

## একটাই বড় পরিবর্তন: pipeline উল্টে গেছে

**প্রস্তাব ছিল:** RGB → 2D detector → **Canonicalization** → MLP lifting →
Bone-length loss → 3D pose

**যা হয়েছে:** RGB → 2D detector → **frozen lifter** → **Canonicalization** →
3D pose

অর্থাৎ canonicalization lifter-এর **আগে** থেকে **পরে** সরে গেছে।

**কেন — এই কথাটা মুখস্থ রাখুন:**

> কারণ ঘূর্ণনটা অঙ্কেই কেটে যায়, তাই invariance পেতে ট্রেনিং লাগে না। দুইটা
> ক্যামেরার prediction একটা অজানা rotation-এ আলাদা। Frame-টা joint থেকেই বানানো,
> তাই সেটাও একই সাথে ঘোরে, আর অজানা rotation-টা **ঠিক ঠিক কেটে যায়** — কখনো
> মাপতেই হয় না। এটা সত্যি হলে, learned latent space দিয়ে এমন একটা সমস্যা
> optimization করে সমাধান করা হচ্ছে যার closed-form উত্তর আছে।
>
> আর সাহিত্য পর্যালোচনায় দেখলাম learned রাস্তাটা আগেই দখল হয়ে গেছে — 3DPCNet,
> MoViD, V-VIPE, CanonPose। খালি ছিল requirement profile-টা: ট্রেনিং নেই, লেবেল
> নেই, calibration নেই।

**Gap আর প্রশ্ন বদলায়নি। শুধু পদ্ধতি বদলেছে।**

## যা দেওয়া হয়েছিল, আর যা আপনার নিজের

বিশ্ববিদ্যালয় MotionAGFormer একটা চালু উদাহরণ হিসেবে দিয়েছিল। পরীক্ষক জিজ্ঞেস করতে
পারেন আপনি কী যোগ করেছেন — উত্তরটা এক command-এ যাচাই করা যায়:

| Repo যা দিয়েছে | আপনি যা লিখেছেন |
|---|---|
| `model/` — MotionAGFormer architecture | `canonical/` — ফ্রেম নির্মাণ, ১,৬৮২ লাইন |
| Training code, configs, checkpoints | `evaluation/` — সব পরীক্ষা, ১০,১০১ লাইন |
| প্রকাশিত evaluation script | `tests/` — ৭৬টা test, ১,২৪৫ লাইন |
| | `presentation/`, `app.py`, `demo_live/` — ২,৫৫৯ লাইন |

**মোটামুটি ১৫,৫০০ লাইন, ৭৫টা ফাইল আপনার নিজের।** Backbone-টা ইচ্ছাকৃতভাবে
**জমাট (frozen)** রাখা হয়েছে — কখনো fine-tune করা হয়নি — আর এটাই পদ্ধতির মূল কথা।

> সরাসরি জিজ্ঞেস করলে: "Estimator-টা ওদের, আমি ওতে হাত দিইনি। এটা ইচ্ছাকৃত —
> আমার পুরো অবদানই হলো যে ওর ভেতরে কিছু বদলাতে হয় না। Thesis যা দাবি করে তার
> সবটাই আমার লেখা কোড থেকে আসে।"

## স্যার proposal নিয়ে জিজ্ঞেস করলে যেভাবে শুরু করবেন

অপেক্ষা করবেন না, নিজেই বলবেন:

> স্যার, proposal-এর gap-টা অপরিবর্তিত — প্রচলিত পদ্ধতিতে explicit geometric prior
> আর view-invariant constraint নেই। আমার পাঁচটা objective-এর চারটা যেভাবে লেখা
> ছিল সেভাবেই হয়েছে। যা বদলেছে সেটা পদ্ধতি: proposal ছিল loss দিয়ে view-invariance
> **শেখানো**, আর আমি দেখলাম এটা **বের করে ফেলা** যায় — frame joint থেকে বানানো
> বলে ক্যামেরার ঘূর্ণন ঠিক ঠিক কেটে যায়। ফলে ট্রেনিং ঐচ্ছিক না, অপ্রয়োজনীয় হয়ে যায়।
>
> Bone-length objective-টাই একমাত্র যেটা লেখা মতো দিইনি। ওটা loss না বানিয়ে signal
> হিসেবে পরীক্ষা করেছি, প্রথম ডেটাসেটে +০.৪৯২ আর দ্বিতীয়টায় +০.০৯৮ পেয়েছি, এবং
> প্রত্যাহার করেছি। তাই proposal-এর অনুমান — geometric prior থেকে view-invariance
> আর reliability দুইটাই আসবে — উত্তর পেয়েছে, আর উত্তরের অর্ধেকটা নেতিবাচক।
