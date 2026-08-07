# What this project actually is, and how to say it out loud

Written for you, not for the examiners. English first, then the same thing in
Bengali. If you understand this page you can defend the thesis, because every
hard question is a variation on something here.

---

# ENGLISH

## In one sentence

A 3D pose estimator gives you a skeleton in *the camera's* coordinate system, so
filming the same movement from two cameras gives two different sets of numbers
for the same pose — and we fix that after the fact, by rotating the skeleton into
a frame built from the body itself, without retraining anything.

## The problem, concretely

Put two cameras around a person walking. Run a pose estimator on both videos.
Both are *correct* — but camera A says the left wrist is at (0.3, 1.1, 2.4) and
camera B says (1.9, 1.1, 0.8), because each reports coordinates relative to
itself. You cannot compare them, average them, or search one recording against
another.

This matters for gait analysis in a clinic and for sports analysis, where the
whole point is comparing one recording to another.

**The existing fixes all cost something:** retrain the estimator, calibrate the
cameras, or train a separate network to canonicalize. If you cannot train and
cannot calibrate — which is the normal situation for a small clinic — none of
them is available.

## What we built

Take the predicted skeleton. Build two directions from the joints themselves:

- **y** = pelvis → thorax (the torso axis)
- **x_raw** = one hip → the other hip (the hip axis)

Cross them with Gram-Schmidt to get a clean orthonormal frame, then express the
whole skeleton in that frame.

Because both axes are built *from the joints*, they rotate with the body. So if
camera B sees the same pose rotated by some unknown rotation Q, the frame is also
rotated by Q — **and the two cancel exactly**. The unknown camera rotation
disappears without ever being estimated. That is why no calibration is needed.

It adds zero trained parameters, runs in 402 FLOPs, and the estimator is never
touched.

**Be honest about this part:** this construction is the **TRIAD algorithm**, from
spacecraft attitude determination, published in 1964. We did not invent it. The
thesis says so.

## The actual research question

Not "does our method work" — that is engineering. The question is:

> **What determines whether such a body frame is consistent across viewpoints?**

There is a classical answer from biomechanics: a direction read from two joints a
distance L apart, each with noise σ, is uncertain by about **2σ/L**. So *longer
axis → more stable frame*. We tested how far that principle carries. Three
levels, each pre-registered before running.

## What we found — and this is the contribution

| Level | Question | Result |
|---|---|---|
| **1. Between constructions** | Does a longer axis make a better frame? | **YES.** Swapping the hip axis for the longer shoulder axis improves the frame on both backbones, intervals excluding zero |
| **2. Within one construction** | Can it tell which *frame instance* to trust? | **NO.** Within one construction the axis is an anatomical near-constant — it varies by only 1.29× from the 1st to the 99th percentile |
| **3. Between joints** | Can it predict which *joint* will disagree? | **NO, and backwards.** Joints rigid with the torso sit at a *larger* radius than articulated joints and yet disagree **2.5× less** |

**Level 3 is the interesting one.** The rigid-body theory predicts error grows
with distance from the frame origin. On a real body it does the opposite —
because a body is not a rigid body. A joint past a hinge (elbow, knee) carries
the estimator's error *in that hinge angle*, and that term dominates the geometry
completely.

So the principle is real but **narrow**: it decides what to build the frame
*from*, and nothing finer. A principle with its scope established is more useful
than one asserted broadly — and the two failures are what establish the scope.

## The headline number

On **Human3.6M**, which played no part in developing the method, canonicalization
cuts cross-view distance by **72.2%** across 180 held-out pairs, and **179 of 180
improve**.

That 72.2% excludes the four joints the frame is *built* from, because the
construction pins them and including them flatters us. Over all seventeen joints
it is 74.1%. **Quote 72.2 and explain why** — that choice is a point in your
favour, not against you.

## The result that goes against us — say this before anyone asks

I ran the comparison a reviewer would demand: align each pose to a **single fixed
reference skeleton** using the Kabsch algorithm. It is training-free, label-free,
calibration-free and single-view — it meets *every* requirement we claim as our
profile.

**It beats our method on all 180 pairs, both backbones, and all fifteen
actions.**

This is in the abstract, the contributions list, the section "A Single-View Baseline, and It Wins", Limitations, and the
conclusion. It was pre-registered — the criterion and all three possible outcomes
were written down and committed to git *before* the experiment ran.

## Why the thesis still stands

This is the argument the whole defence rests on. Learn it properly:

> Kabsch wins on the metric, but **it cannot run the experiment this thesis is
> about**. It has no anatomical axis — so there is no axis to hold fixed and
> vary, and the question "what governs frame consistency" cannot even be *posed*
> inside it. The anatomical frame is the **instrument** that makes the boundary
> measurable, not the result being claimed.

And the honest half: **as a way of reducing cross-view distance on this data, the
simpler method is better.** Say that. Do not argue it away.

## How honest the process was — this is your strongest asset

Most undergraduate theses cannot say any of this:

- **Nine experiments pre-registered**, each criterion committed to git *before*
  the run. Timestamps prove it.
- **Five failed their own criteria. A sixth returned a competing method as
  better.** All reported as failures.
- **255 numerical claims** re-derived from stored data files by an automated
  audit that fails if any number drifts. It passes.
- **76 unit tests.** Every figure generated by script from the same data.
- Claims **withdrawn** when they did not hold: a bone-length signal (ρ=0.492 →
  0.098 on the second dataset), a reliability score falsified five ways, a 55.1%
  multi-scale figure found to be circular.

If anyone questions whether the work is yours, this is the answer: open the git
log. Nobody fakes seventeen pre-registered experiments with timestamps preceding their results.

---

## How to say it to Sir

**What Sir actually knows, and nothing more:** his `Proposal.docx`, your
submitted proposal, that he asked for output, that he gave you
`KelvinHong/pose-estimation-3d`, and the output you showed him from it. He has
not seen MotionAGFormer, the method, or any result. Start from that output, not
from the thesis.

### 30 seconds

> Sir, the output I showed you from the repository you gave me — like every
> pose estimator's output — is in the camera's own coordinate frame, so two cameras
> filming the same motion disagree. I build a body-fixed frame from the torso and
> hip axes and apply it after prediction — no retraining, no calibration. On
> Human3.6M it cuts cross-view disagreement by 72 percent over 180 held-out
> pairs. But the real question I studied is *what governs* whether such a frame
> works, and the answer is that the classical geometric principle is much
> narrower than it looks.

### 2 minutes

Add:

> The principle from biomechanics says a direction from two joints distance L
> apart is uncertain by about 2σ/L, so a longer axis gives a more stable frame. I
> tested that at three levels, pre-registering each criterion. It holds between
> frame constructions — the longer shoulder axis beats the hip axis on both
> backbones. It fails within a construction, because there the axis is an
> anatomical near-constant. And it fails between joints, in the wrong direction:
> joints rigid with the torso are at a *larger* radius yet disagree two and a
> half times *less*. A body is not a rigid body, and past a hinge the estimator's
> error in the joint angle dominates the geometry entirely.
>
> I should also say the frame construction is not mine — it is the TRIAD
> algorithm from spacecraft attitude determination, 1964, and the error
> propagation is from Cappozzo's biomechanics work. The report credits both. My
> contribution is the boundary, not the geometry.

### 5 minutes — add the hard part yourself

> One more thing, Sir, and I would rather raise it than be asked. I tested a
> simpler baseline — Kabsch alignment to a single fixed skeleton — which meets
> every requirement my framework claims. It beats my method on all 180 pairs,
> both backbones, all fifteen actions. It is in the abstract.
>
> I kept the work because the baseline cannot run the experiment. It has no
> anatomical axis, so there is no variable to hold fixed and vary, and the
> question of what governs frame consistency cannot be asked inside it. As a way
> of reducing cross-view distance, the simpler method is better on this data, and
> I say that plainly. What I claim is the boundary and the instrument that makes
> it measurable.

## The three questions he will ask

**1. "So what is actually yours?"**

> The boundary. Everyone knew the frame construction and everyone knew the error
> propagation. Nobody had tested how far that reasoning carries on an articulated
> body reconstructed by a network — and the answer is: less far than it looks. It
> decides what to build the frame from, and nothing finer.

**2. "If the simpler method wins, why keep yours?"**

> Because the simpler method cannot run the experiment. No anatomical axis means
> no variable to vary. Mine is the instrument, not the result.

**3. "Why report something that damages your own result?"**

> Because the criterion was committed to version control before the run. If I
> only reported the pre-registrations that came out well, none of the others
> would mean anything either.

---
---

# বাংলা

## এক লাইনে

3D pose estimator একটা মানুষের কঙ্কাল (skeleton) বের করে দেয় **ক্যামেরার নিজের
coordinate system-এ**। তাই একই নড়াচড়া দুইটা ক্যামেরা দিয়ে ধরলে একই pose-এর জন্য
দুই রকম সংখ্যা আসে। আমরা এই সমস্যাটা prediction-এর *পরে* ঠিক করি — কঙ্কালটাকে
শরীরের নিজের অক্ষ থেকে বানানো একটা frame-এ ঘুরিয়ে নিয়ে, **কোনো retraining
ছাড়াই**।

## সমস্যাটা আসলে কী

একজন মানুষ হাঁটছে, চারপাশে দুইটা ক্যামেরা। দুইটা ভিডিওতেই pose estimator চালান।
**দুইটাই সঠিক** — কিন্তু ক্যামেরা A বলে বাম কব্জি (0.3, 1.1, 2.4)-তে, আর ক্যামেরা
B বলে (1.9, 1.1, 0.8)-তে। কারণ প্রত্যেকে নিজের সাপেক্ষে coordinate দেয়। ফলে
আপনি দুইটাকে তুলনা করতে পারবেন না, গড় করতে পারবেন না, একটা রেকর্ডিং দিয়ে আরেকটা
খুঁজতেও পারবেন না।

ক্লিনিকে **gait analysis** আর **sports analysis**-এ এটাই মূল কাজ — এক রেকর্ডিংয়ের
সাথে আরেকটার তুলনা।

**প্রচলিত সমাধানগুলোর প্রত্যেকটার একটা খরচ আছে:** estimator আবার train করা,
ক্যামেরা calibrate করা, অথবা canonicalization-এর জন্য আলাদা network train করা।
যদি train-ও করা না যায়, calibrate-ও করা না যায় — যেটা একটা ছোট ক্লিনিকের
স্বাভাবিক অবস্থা — তাহলে এর একটাও কাজে লাগে না।

## আমরা যা বানিয়েছি

Predicted skeleton নিয়ে, joint গুলো থেকেই দুইটা direction বানাই:

- **y** = pelvis → thorax (শরীরের উল্লম্ব অক্ষ)
- **x_raw** = এক hip → আরেক hip (নিতম্বের অক্ষ)

Gram-Schmidt দিয়ে cross করে একটা orthonormal frame পাই, তারপর পুরো কঙ্কালটাকে ওই
frame-এ প্রকাশ করি।

**মূল কথাটা এখানে:** যেহেতু দুইটা অক্ষই *joint থেকে* বানানো, তাই অক্ষ দুইটা
শরীরের সাথে সাথেই ঘোরে। ক্যামেরা B যদি একই pose-কে অজানা কোনো rotation Q দিয়ে
ঘোরানো অবস্থায় দেখে, frame-ও একই Q দিয়ে ঘুরে যায় — **আর দুইটা ঠিক ঠিক কাটাকাটি
হয়ে যায়**। অজানা ক্যামেরা rotation-টা estimate না করেই মুছে যায়। এজন্যই কোনো
calibration লাগে না।

কোনো trained parameter যোগ হয় না, 402 FLOPs-এ চলে, আর estimator-এ হাত দেওয়া হয় না।

**এই জায়গায় সৎ থাকতে হবে:** এই construction-টা আসলে **TRIAD algorithm** —
মহাকাশযানের attitude determination-এর জন্য ১৯৬৪ সালে প্রকাশিত। এটা আমাদের
আবিষ্কার নয়, এবং thesis-এ সেটা পরিষ্কার করে লেখা আছে।

## আসল গবেষণা প্রশ্নটা

"আমাদের method কাজ করে কি না" — এটা প্রশ্ন না, এটা engineering। আসল প্রশ্ন:

> **কী নির্ধারণ করে যে এরকম একটা body frame বিভিন্ন viewpoint-এ consistent থাকবে
> কি না?**

Biomechanics থেকে একটা ধ্রুপদী উত্তর আছে: L দূরত্বে থাকা দুইটা joint থেকে পাওয়া
direction-এর অনিশ্চয়তা প্রায় **2σ/L**। অর্থাৎ *অক্ষ যত লম্বা, frame তত স্থিতিশীল*।
আমরা পরীক্ষা করেছি এই নীতিটা **কতদূর পর্যন্ত কাজ করে**। তিনটা স্তরে, প্রত্যেকটার
criterion আগে থেকে ঘোষণা করে (pre-register করে)।

## যা পেয়েছি — এটাই আসল অবদান

| স্তর | প্রশ্ন | ফলাফল |
|---|---|---|
| **১. দুই construction-এর মধ্যে** | লম্বা অক্ষ কি ভালো frame দেয়? | **হ্যাঁ।** Hip অক্ষের বদলে লম্বা shoulder অক্ষ নিলে দুইটা backbone-এই উন্নতি হয় |
| **২. এক construction-এর ভেতরে** | কোন frame-টা বিশ্বাস করব, বলতে পারে? | **না।** এক construction-এর ভেতরে অক্ষটা প্রায় ধ্রুবক — ১ম থেকে ৯৯তম percentile-এ মাত্র ১.২৯ গুণ পার্থক্য |
| **৩. Joint-দের মধ্যে** | কোন joint বেশি অমিল দেখাবে, বলতে পারে? | **না, বরং উল্টো।** ধড়ের সাথে শক্তভাবে যুক্ত joint গুলো *বেশি* দূরত্বে থেকেও **২.৫ গুণ কম** অমিল দেখায় |

**তৃতীয় স্তরটাই সবচেয়ে আকর্ষণীয়।** Rigid-body তত্ত্ব বলে frame-এর কেন্দ্র থেকে
দূরত্ব বাড়লে error বাড়বে। বাস্তব শরীরে ঠিক উল্টোটা হয় — **কারণ শরীর rigid body
না**। কনুই বা হাঁটুর ওপারের joint-এ estimator-এর *ওই কোণের* error-ও যোগ হয়, আর
সেই term জ্যামিতিকে সম্পূর্ণ ছাপিয়ে যায়।

তাই নীতিটা সত্যি, কিন্তু **সংকীর্ণ**: এটা ঠিক করে দেয় frame *কী দিয়ে* বানাবেন,
এর বেশি কিছু না। **যে নীতির সীমা জানা আছে, সেটা ঢালাওভাবে দাবি করা নীতির চেয়ে
বেশি কাজের** — আর ওই দুইটা ব্যর্থতাই সীমাটা প্রতিষ্ঠা করে।

## মূল সংখ্যা

**Human3.6M**-এ — যেটা method তৈরির সময় একদমই ব্যবহার হয়নি — canonicalization
cross-view দূরত্ব **৭২.২%** কমায়, ১৮০টা held-out জোড়ার মধ্যে **১৭৯টাতেই উন্নতি**।

এই ৭২.২% থেকে frame যে চারটা joint দিয়ে *বানানো* সেগুলো বাদ দেওয়া হয়েছে, কারণ
construction ওগুলোকে আটকে রাখে, আর ওগুলো রাখলে ফলাফল আমাদের পক্ষে ফুলিয়ে দেখায়।
সতেরোটা joint ধরলে সংখ্যাটা ৭৪.১%। **৭২.২ বলবেন এবং কেন বাদ দিয়েছেন সেটা
ব্যাখ্যা করবেন** — এই সিদ্ধান্তটা আপনার পক্ষে যায়, বিপক্ষে না।

## যে ফলাফল আমাদের বিপক্ষে যায় — কেউ জিজ্ঞেস করার আগেই বলবেন

আমি সেই তুলনাটা চালিয়েছি যেটা একজন reviewer দাবি করবেই: প্রত্যেকটা pose-কে
**একটা নির্দিষ্ট reference কঙ্কালের** সাথে Kabsch algorithm দিয়ে align করা। এটাও
training-free, label-free, calibration-free এবং single-view — অর্থাৎ আমরা যেসব
requirement নিজেদের বৈশিষ্ট্য বলে দাবি করি, **তার প্রত্যেকটা এটাও পূরণ করে**।

**এটা আমাদের method-কে হারিয়ে দেয় — ১৮০টা জোড়ার সবগুলোতে, দুইটা backbone-এই,
এবং পনেরোটা action-এর সবগুলোতে।**

এটা abstract-এ আছে, contribution তালিকায় আছে, the section "A Single-View Baseline, and It Wins"-এ আছে, Limitations-এ আছে,
conclusion-এও আছে। এবং এটা pre-registered ছিল — criterion এবং তিনটা সম্ভাব্য
ফলাফলই পরীক্ষা চালানোর **আগে** লিখে git-এ commit করা হয়েছিল।

## তারপরও thesis কেন টিকে যায়

পুরো defence এই যুক্তির ওপর দাঁড়ানো। এটা ভালো করে বুঝে নিন:

> Kabsch metric-এ জেতে, কিন্তু **এই thesis-এর পরীক্ষাটাই ও চালাতে পারে না**। ওর
> কোনো anatomical অক্ষ নেই — তাই স্থির রেখে পরিবর্তন করার মতো কোনো variable-ই
> নেই, এবং "frame consistency কী নিয়ন্ত্রণ করে" প্রশ্নটা ওর ভেতরে **করাই যায়
> না**। Anatomical frame হলো সেই **যন্ত্র** যেটা সীমাটা পরিমাপযোগ্য করে তোলে —
> এটাই দাবি, সংখ্যাটা না।

আর সৎ অর্ধেকটা: **এই data-তে cross-view দূরত্ব কমানোর উপায় হিসেবে সহজ
method-টাই ভালো।** এটা বলবেন। এড়িয়ে যাওয়ার চেষ্টা করবেন না।

## প্রক্রিয়াটা কতটা সৎ ছিল — এটাই আপনার সবচেয়ে বড় শক্তি

বেশিরভাগ undergraduate thesis এর কিছুই বলতে পারে না:

- **নয়টা পরীক্ষা pre-registered**, প্রত্যেকটার criterion চালানোর **আগে** git-এ
  commit করা। Timestamp দিয়ে প্রমাণ হয়।
- **পাঁচটা নিজেদের criterion-এ ব্যর্থ। ষষ্ঠটায় প্রতিদ্বন্দ্বী method ভালো
  প্রমাণিত।** সবগুলো ব্যর্থতা হিসেবেই রিপোর্ট করা।
- **২৫৫টা সংখ্যাগত দাবি** সংরক্ষিত data file থেকে স্বয়ংক্রিয় audit দিয়ে আবার
  যাচাই হয়; কোনো সংখ্যা সরে গেলে audit fail করে। এটা pass করে।
- **৭৬টা unit test।** প্রত্যেকটা চিত্র একই data থেকে script দিয়ে তৈরি।
- যেসব দাবি টেকেনি, **প্রত্যাহার করা হয়েছে**: bone-length signal (ρ=0.492 →
  দ্বিতীয় dataset-এ 0.098), reliability score পাঁচভাবে মিথ্যা প্রমাণিত, ৫৫.১%
  multi-scale সংখ্যাটা circular বলে ধরা পড়েছে।

কেউ যদি প্রশ্ন তোলে কাজটা সত্যিই আপনার কি না — উত্তর হলো git log খুলে দেখানো।
ফলাফলের আগের timestamp সহ নয়টা pre-registration কেউ বানিয়ে বানিয়ে তৈরি করে না।

---

## স্যারকে কীভাবে বলবেন

**স্যার ঠিক এতটুকুই জানেন:** ওনার `Proposal.docx`, আপনার জমা দেওয়া proposal,
উনি output দেখতে চেয়েছিলেন, উনি `KelvinHong/pose-estimation-3d` repo-টা
দিয়েছিলেন, আর ওটা থেকে আপনি যে output দেখিয়েছেন। MotionAGFormer, method, বা
কোনো ফলাফল — কিছুই উনি দেখেননি। তাই ওই output থেকে শুরু করবেন, thesis থেকে না।

### ৩০ সেকেন্ড

> স্যার, আপনার দেওয়া repo থেকে যে output দেখিয়েছিলাম, সেটা — সব pose
> estimator-এর মতোই — ক্যামেরার নিজের frame-এ coordinate দেয়, তাই একই নড়াচড়ায়
> দুইটা ক্যামেরা দুই রকম ফল দেয়। আমি ধড় আর নিতম্বের অক্ষ থেকে একটা body-fixed
> frame বানিয়ে prediction-এর পরে সেটা প্রয়োগ করি — কোনো retraining নেই,
> calibration নেই। Human3.6M-এ ১৮০টা held-out জোড়ায় cross-view অমিল ৭২ শতাংশ
> কমে। তবে আমার আসল গবেষণা হলো এরকম একটা frame **কেন এবং কতদূর** কাজ করে, আর
> উত্তর হলো — ধ্রুপদী জ্যামিতিক নীতিটা যতটা মনে হয় তার চেয়ে অনেক সংকীর্ণ।

### ২ মিনিট

উপরেরটার সাথে যোগ করুন:

> Biomechanics-এর নীতি বলে L দূরত্বের দুইটা joint থেকে পাওয়া direction-এর
> অনিশ্চয়তা প্রায় 2σ/L, অর্থাৎ লম্বা অক্ষ মানে স্থিতিশীল frame। আমি এটা তিনটা
> স্তরে পরীক্ষা করেছি, প্রত্যেকটার criterion আগে ঘোষণা করে। দুই construction-এর
> মধ্যে এটা টেকে — লম্বা shoulder অক্ষ hip অক্ষকে হারায়, দুইটা backbone-এই। এক
> construction-এর ভেতরে টেকে না, কারণ সেখানে অক্ষটা প্রায় ধ্রুবক। আর joint-দের
> মধ্যে উল্টো দিকে ব্যর্থ হয়: ধড়ের সাথে যুক্ত joint গুলো *বেশি* দূরত্বে থেকেও
> আড়াই গুণ *কম* অমিল দেখায়। শরীর rigid body না, আর কব্জা পেরোলে estimator-এর
> কোণের error জ্যামিতিকে ছাপিয়ে যায়।
>
> আরেকটা কথা বলা দরকার, স্যার — frame construction-টা আমার নিজের না, এটা TRIAD
> algorithm, ১৯৬৪ সালের মহাকাশযান attitude determination থেকে। Error propagation
> টাও Cappozzo-র biomechanics কাজ থেকে। Report-এ দুইটারই কৃতিত্ব দেওয়া আছে। আমার
> অবদান সীমাটা, জ্যামিতিটা না।

### ৫ মিনিট — কঠিন অংশটা নিজে থেকেই বলুন

> আরেকটা বিষয় স্যার, কেউ জিজ্ঞেস করার চেয়ে আমি নিজেই বলি। আমি একটা সহজ baseline
> পরীক্ষা করেছি — একটা নির্দিষ্ট কঙ্কালের সাথে Kabsch alignment — যেটা আমার
> framework-এর প্রত্যেকটা requirement পূরণ করে। এটা আমার method-কে হারিয়ে দেয়,
> ১৮০টা জোড়ার সবগুলোতে, দুইটা backbone-এ, পনেরোটা action-এ। এটা abstract-এ
> লেখা আছে।
>
> তারপরও কাজটা রেখেছি, কারণ ওই baseline দিয়ে এই পরীক্ষাটাই চালানো যায় না। ওর
> anatomical অক্ষ নেই, তাই স্থির রেখে পরিবর্তন করার মতো variable নেই, এবং frame
> consistency কী নিয়ন্ত্রণ করে সেই প্রশ্নটা ওর ভেতরে করা যায় না। Cross-view
> দূরত্ব কমানোর উপায় হিসেবে এই data-তে সহজ method-টাই ভালো, এটা আমি স্পষ্ট করেই
> বলি। আমার দাবি হলো সীমাটা, আর সেটা মাপার যন্ত্রটা।

## যে তিনটা প্রশ্ন উনি করবেনই

**১. "তাহলে তোমার নিজের কাজটা কী?"**

> সীমাটা, স্যার। Frame construction সবাই জানত, error propagation-ও সবাই জানত।
> কিন্তু একটা network দিয়ে পুনর্নির্মিত articulated শরীরে ওই যুক্তি কতদূর টেকে,
> সেটা কেউ পরীক্ষা করেনি — আর উত্তর হলো, যতটা মনে হয় তার চেয়ে অনেক কম দূর। এটা
> ঠিক করে frame কী দিয়ে বানাবেন, এর বেশি কিছু না।

**২. "সহজ method যদি জেতে, তোমারটা রাখলে কেন?"**

> কারণ সহজ method দিয়ে পরীক্ষাটাই চালানো যায় না। Anatomical অক্ষ না থাকলে
> পরিবর্তন করার মতো variable থাকে না। আমারটা যন্ত্র, ফলাফল না।

**৩. "নিজের ফলাফলের ক্ষতি করে এমন জিনিস রিপোর্ট করলে কেন?"**

> কারণ criterion টা পরীক্ষা চালানোর আগেই version control-এ commit করা ছিল। আমি
> যদি শুধু যেসব pre-registration ভালো ফল দিয়েছে সেগুলোই রিপোর্ট করতাম, তাহলে
> বাকিগুলোরও কোনো মানে থাকত না।
