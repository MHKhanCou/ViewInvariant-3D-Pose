# How the system works, end to end — and what improved

One page for Sir. English first, then Bengali. Every number here is recomputed
from stored result files by `evaluation/audit_numbers.py`.

---

# ENGLISH

## The pipeline, six steps

```
  RGB video (any camera, uncalibrated)
        │
  1 ▼   YOLOv8 pose detector                    FROZEN, off-the-shelf
        17 COCO-format 2D keypoints per frame
        │
  2 ▼   COCO → Human3.6M conversion + normalize  no parameters
        17 joints; pelvis, spine, thorax, head-top are derived
        │
  3 ▼   MotionAGFormer-XS                       FROZEN, 2.2 M params
        27-frame window → (17, 3) 3D joints
        │                 IN THE CAMERA'S OWN COORDINATE FRAME
        │
  4 ▼   Body-frame canonicalization             0 learned parameters, 402 FLOPs
        y = pelvis → thorax,  x = hip → hip,  Gram-Schmidt
        same 17 joints, re-expressed in a body-fixed frame
        │
  5 ▼   Two cameras, same instant
        mean per-joint distance between the two predictions
        │
  6 ▼   Cross-view comparable 3D pose
```

| Step | Trained by me? | What changes |
|---|---|---|
| 1 YOLOv8 | No — frozen | — |
| 2 conversion | No parameters | — |
| 3 MotionAGFormer | **No — frozen, never touched** | — |
| 4 canonicalization | **No — zero parameters** | coordinate frame only |
| 5–6 comparison | Measurement, not a model | — |

**Nothing in this pipeline was trained by me.** Step 4 is the whole thesis, and it
is arithmetic: a rotation built from the person's own joints.

## Why step 4 removes the viewpoint

Both axes are built *from the joints*, so they rotate with the body. If camera B
sees the same pose rotated by some unknown rotation **Q**, the frame is rotated
by **Q** too — and the two cancel exactly. The unknown camera rotation disappears
**without ever being estimated**. That is why no calibration is needed.

This construction is the **TRIAD algorithm** (Black, 1964, spacecraft attitude
determination). I did not invent it, and the report says so.

## Where the numbers move

Human3.6M, which took no part in developing the method. 180 held-out camera
pairs, MotionAGFormer-XS.

| Quantity | Before step 4 | After step 4 |
|---|---|---|
| **Accuracy** vs ground truth (MPJPE) | 45.149 mm | **45.149 mm — identical** |
| **Cross-view joint distance**, 13 joints | 372.7 mm | **93.4 mm  (−72.2 %)** |
| Cross-view joint distance, all 17 joints | 320.4 mm | 75.3 mm  (−74.1 %) |

Read the two rows together. **Accuracy does not change and cannot change** — step
3 is frozen and step 4 adds no parameters. What changes is *agreement between two
cameras looking at the same person*.

Three details, so no one has to dig them out:

1. **179 of the 180 pairs improve.** MotionBERT, as a second backbone: 75.8 %.
2. **Quote 72.2 %, not 74.1 %.** The frame is *built from* joints
   {pelvis, hips, spine, thorax}, so it pins them by construction — thorax
   disagrees by 22.1 mm against 197.5 mm for articulated joints. Averaging all
   seventeen flatters the method. The 13-joint figure excludes the four the frame
   touches.
3. **The floor is 56.2 mm** — a per-frame Procrustes oracle that is allowed to see
   both views. Canonicalization closes **87.0 %** of the gap to it. (Over all
   seventeen joints the oracle is 51.3 mm and the gap closed is 90.5 %.)
4. **One caveat I state myself:** MotionAGFormer-XS was trained on Human3.6M, so
   the *backbone* is in-domain here. Held out is the *method* — no H36M frame was
   used to design or tune the canonicalization. The report says this too.

## Say this before anyone asks

A simpler baseline beats this method. Kabsch-align every pose to one fixed
reference skeleton: training-free, label-free, calibration-free, single-view — it
meets *every* requirement I claim as my profile. It scores **57.5 mm against my
93.4 mm**, and it wins on **180 of 180 pairs, both backbones, all fifteen
actions**. It is in the abstract, §5.6.1 ("A Single-View Baseline, and It Wins"), Limitations and the conclusion, and its
criterion was committed to git before the experiment ran.

The work still stands because **that baseline cannot run the experiment this
thesis is about**. It has no anatomical axis, so there is no axis to hold fixed
and vary, and the question *what governs whether a body frame is consistent
across viewpoints* cannot be posed inside it. My frame is the instrument that
makes the boundary measurable. As a way of reducing cross-view distance on this
data, the simpler method is better — I say that plainly rather than argue it away.

## The two answers, word for word

**"What is your model actually doing?"**

> Sir, I do not generate a new pose. The exact same MotionAGFormer prediction is
> re-expressed in a coordinate system built from the person's own hips and torso.
> Then I measure how closely two cameras agree. Mean cross-view joint distance
> drops from 372.7 mm to 93.4 mm across 180 held-out pairs.

**"How much better is the output than base MotionAGFormer?"**

> Its accuracy is not better, Sir — it is identical, 45.149 millimetres either
> way, because the estimator is frozen and I add zero parameters. What improves
> is cross-view consistency: two cameras that disagreed by 372.7 mm now agree to
> 93.4 mm. And I should add that a simpler Kabsch baseline reaches 57.5 mm, so on
> that metric it beats mine; the report says so in the abstract.

## What I did not do

1. **No retraining.** No weight of MotionAGFormer or YOLOv8 was updated.
2. **No camera calibration.** No intrinsics, no extrinsics, at any stage.
3. **No accuracy claim.** The thesis claims cross-view *comparability*, never
   improved estimation.

---
---

# বাংলা

## পুরো pipeline, ছয় ধাপে

```
  RGB ভিডিও (যেকোনো ক্যামেরা, calibration ছাড়াই)
        │
  ১ ▼   YOLOv8 pose detector                    FROZEN, রেডিমেড
        প্রতি frame-এ ১৭টা COCO 2D keypoint
        │
  ২ ▼   COCO → Human3.6M রূপান্তর + normalize    কোনো parameter নেই
        ১৭টা joint; pelvis, spine, thorax, head-top হিসাব করে বানানো
        │
  ৩ ▼   MotionAGFormer-XS                       FROZEN, ২.২ মিলিয়ন parameter
        ২৭-frame window → (17, 3) 3D joint
        │                 ক্যামেরার নিজের coordinate frame-এ
        │
  ৪ ▼   Body-frame canonicalization             ০টা শেখা parameter, ৪০২ FLOPs
        y = pelvis → thorax,  x = hip → hip,  Gram-Schmidt
        একই ১৭টা joint, শরীর-নির্ভর frame-এ প্রকাশ করা
        │
  ৫ ▼   দুইটা ক্যামেরা, একই মুহূর্ত
        দুই prediction-এর মধ্যে গড় joint দূরত্ব
        │
  ৬ ▼   Cross-view তুলনাযোগ্য 3D pose
```

| ধাপ | আমি train করেছি? | কী বদলায় |
|---|---|---|
| ১ YOLOv8 | না — frozen | — |
| ২ রূপান্তর | কোনো parameter নেই | — |
| ৩ MotionAGFormer | **না — frozen, হাতই দেওয়া হয়নি** | — |
| ৪ canonicalization | **না — শূন্য parameter** | শুধু coordinate frame |
| ৫–৬ তুলনা | পরিমাপ, model না | — |

**এই pipeline-এর কোনো অংশ আমি train করিনি।** ৪ নম্বর ধাপটাই পুরো thesis, আর
সেটা নিছক গণিত — মানুষের নিজের joint থেকে বানানো একটা rotation।

## ৪ নম্বর ধাপ কেন viewpoint মুছে দেয়

দুইটা অক্ষই *joint থেকে* বানানো, তাই অক্ষ দুইটা শরীরের সাথেই ঘোরে। ক্যামেরা B
যদি একই pose-কে অজানা rotation **Q** দিয়ে ঘোরানো দেখে, frame-ও **Q** দিয়ে ঘোরে —
**আর দুইটা ঠিক ঠিক কাটাকাটি হয়ে যায়**। অজানা ক্যামেরা rotation **estimate না
করেই** মুছে যায়। এজন্যই calibration লাগে না।

এই construction-টা **TRIAD algorithm** (Black, ১৯৬৪, মহাকাশযানের attitude
determination)। এটা আমার আবিষ্কার নয়, report-এ সেটা লেখা আছে।

## সংখ্যাগুলো কোথায় বদলায়

Human3.6M — method তৈরির সময় একদমই ব্যবহার হয়নি। ১৮০টা held-out ক্যামেরা জোড়া,
MotionAGFormer-XS।

| কী মাপা হচ্ছে | ৪ নম্বর ধাপের আগে | ৪ নম্বর ধাপের পরে |
|---|---|---|
| **নির্ভুলতা** ground truth-এর সাপেক্ষে (MPJPE) | ৪৫.১৪৯ mm | **৪৫.১৪৯ mm — অপরিবর্তিত** |
| **Cross-view joint দূরত্ব**, ১৩টা joint | ৩৭২.৭ mm | **৯৩.৪ mm  (−৭২.২%)** |
| Cross-view joint দূরত্ব, সবগুলো ১৭ joint | ৩২০.৪ mm | ৭৫.৩ mm  (−৭৪.১%) |

দুইটা সারি একসাথে পড়তে হবে। **নির্ভুলতা বদলায় না, বদলাতে পারেও না** — ৩ নম্বর ধাপ
frozen আর ৪ নম্বর ধাপে কোনো parameter নেই। যা বদলায় তা হলো *একই মানুষকে দেখা দুইটা
ক্যামেরার মধ্যে মিল*।

তিনটা খুঁটিনাটি, যাতে কাউকে খুঁজতে না হয়:

১. **১৮০টার মধ্যে ১৭৯টা জোড়াতেই উন্নতি।** দ্বিতীয় backbone MotionBERT-এ ৭৫.৮%।
২. **৭২.২% বলবেন, ৭৪.১% না।** Frame টা {pelvis, দুই hip, spine, thorax} joint
   *দিয়ে বানানো*, তাই construction ওগুলোকে আটকে রাখে — thorax-এর অমিল ২২.১ mm,
   অথচ articulated joint-গুলোর ১৯৭.৫ mm। সতেরোটার গড় নিলে ফল আমাদের পক্ষে ফুলে
   যায়। ১৩-joint সংখ্যাটা ওই চারটা বাদ দিয়ে।
৩. **মেঝে ৫৬.২ mm** — per-frame Procrustes oracle, যেটা দুইটা view-ই দেখতে পায়।
   Canonicalization ওই ব্যবধানের **৮৭.০%** পূরণ করে। (সতেরোটা joint ধরলে oracle
   ৫১.৩ mm, আর পূরণ হয় ৯০.৫%।)
৪. **একটা সতর্কতা আমি নিজেই বলি:** MotionAGFormer-XS নিজে Human3.6M-এ train করা,
   তাই *backbone*-টা এখানে in-domain। Held out হলো *method* — canonicalization
   ডিজাইন বা tune করতে H36M-এর একটা frame-ও ব্যবহার হয়নি। Report-এও এটা লেখা আছে।

## কেউ জিজ্ঞেস করার আগেই এটা বলবেন

একটা সহজ baseline আমার method-কে হারায়। প্রত্যেকটা pose-কে একটা নির্দিষ্ট
reference কঙ্কালের সাথে Kabsch দিয়ে align করা: training-free, label-free,
calibration-free, single-view — আমি যেসব requirement নিজের বৈশিষ্ট্য বলে দাবি
করি, তার *প্রত্যেকটা* এটাও পূরণ করে। ওটার ফল **৫৭.৫ mm, আমার ৯৩.৪ mm-এর
বিপরীতে**, এবং **১৮০টার সবগুলো জোড়ায়, দুইটা backbone-এ, পনেরোটা action-এ** জেতে।
এটা abstract-এ, §5.6.1 ("A Single-View Baseline, and It Wins")-এ, Limitations-এ আর conclusion-এ আছে, আর এর criterion
পরীক্ষা চালানোর আগেই git-এ commit করা ছিল।

তারপরও কাজটা টেকে, কারণ **ওই baseline দিয়ে এই thesis-এর পরীক্ষাটাই চালানো যায়
না**। ওর anatomical অক্ষ নেই, তাই স্থির রেখে পরিবর্তন করার মতো অক্ষ নেই, এবং
*একটা body frame বিভিন্ন viewpoint-এ consistent থাকবে কি না তা কী নিয়ন্ত্রণ করে*
— এই প্রশ্নটা ওর ভেতরে করাই যায় না। আমার frame হলো সেই যন্ত্র যেটা সীমাটা
পরিমাপযোগ্য করে। এই data-তে cross-view দূরত্ব কমানোর উপায় হিসেবে সহজ method-টাই
ভালো — এটা এড়িয়ে না গিয়ে স্পষ্ট করেই বলি।

## দুইটা উত্তর, হুবহু

**"তোমার model আসলে কী করছে?"**

> স্যার, আমি নতুন কোনো pose তৈরি করি না। MotionAGFormer-এর ঠিক একই prediction-কে
> মানুষটার নিজের hip আর ধড় দিয়ে বানানো একটা coordinate system-এ প্রকাশ করি।
> তারপর মাপি দুইটা ক্যামেরার ফল কতটা মেলে। ১৮০টা held-out জোড়ায় গড় cross-view
> joint দূরত্ব ৩৭২.৭ mm থেকে ৯৩.৪ mm-এ নামে।

**"Base MotionAGFormer-এর চেয়ে output কতটা ভালো হলো?"**

> স্যার, নির্ভুলতা ভালো হয়নি — হুবহু এক, দুই ক্ষেত্রেই ৪৫.১৪৯ মিলিমিটার, কারণ
> estimator frozen আর আমি শূন্যটা parameter যোগ করি। যা উন্নত হয় তা হলো cross-view
> consistency: যে দুইটা ক্যামেরা ৩৭২.৭ mm অমিল দেখাত, তারা এখন ৯৩.৪ mm-এ মেলে।
> আর এটাও বলা দরকার, একটা সহজ Kabsch baseline ৫৭.৫ mm পায় — ওই metric-এ ওটা
> আমারটাকে হারায়, এবং report-এর abstract-এই সেটা লেখা আছে।

## যা আমি করিনি

১. **কোনো retraining না।** MotionAGFormer বা YOLOv8-এর একটা weight-ও বদলাইনি।
২. **কোনো ক্যামেরা calibration না।** কোনো ধাপে intrinsics বা extrinsics লাগেনি।
৩. **নির্ভুলতার কোনো দাবি না।** Thesis দাবি করে cross-view *তুলনাযোগ্যতা*, উন্নত
   estimation কখনোই না।
