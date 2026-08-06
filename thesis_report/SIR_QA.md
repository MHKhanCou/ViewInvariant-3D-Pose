# Sir's questions, answered

Every number here traces to a stored artifact and is checked by
`evaluation/audit_numbers.py`. English first, Bengali below.

**Two rules before you open your mouth.** Never call the cross-view number
MPJPE — MPJPE is error against ground truth and it does not move. And never say
the pose became "more accurate"; say two cameras came to *agree*.

---

# ENGLISH

## 1. What did you actually do?

I took a pose estimator that was already trained, **froze it completely**, and
added one arithmetic step after it. That step rewrites the predicted skeleton in
a coordinate system built from the person's own body — the torso axis
(pelvis → thorax) and the hip axis (hip → hip), made orthonormal by
Gram-Schmidt. Nothing is trained, nothing is calibrated, and the estimator never
knows it happened.

Then I spent most of the project on the harder question: **what determines
whether such a body frame stays consistent across viewpoints?** That question,
not the method, is the thesis.

## 2. How much did the output improve over base MotionAGFormer?

Two different answers, and both must be said in the same breath.

- **Accuracy: not at all. It is identical.** Action-balanced MPJPE on Human3.6M
  is **45.149 mm before and after** — matching the backbone's own published 45.1.
  It cannot change: the estimator is frozen and my step adds zero parameters.
- **Cross-view agreement: 372.7 mm → 93.4 mm, a 72.2 % reduction** over 180
  held-out camera pairs, 179 of 180 improving.

If you only remember one sentence: *I did not make the prediction better, I made
two predictions of the same pose comparable.*

## 3. If it improved, what actually changed? How?

The joint coordinates changed; the pose did not. Same skeleton, same shape, new
axes.

The reason it works is one line of algebra. Both axes are built **from the
joints themselves**, so they rotate with the body. If a second camera sees the
same pose rotated by some unknown rotation **Q**, the frame is rotated by **Q**
as well — and in the transformation the two cancel exactly. The unknown camera
rotation disappears *without ever being estimated*. That is precisely why no
calibration is needed.

## 4. How did you prove it?

Five things, in ascending order of how hard they are to fake.

1. **180 held-out camera pairs** on a dataset that played no part in developing
   the method.
2. **Cluster bootstrap over the thirty subject-action groups**, not over the 180
   pairs. Six pairs come from the same four camera streams of the same motion, so
   treating them as independent would give a falsely narrow interval. CI
   [+67.9, +75.5] %.
3. **Two backbones.** MotionAGFormer-XS and MotionBERT, 72.2 % and 75.8 %. A
   result on one network is a property of that network.
4. **Pre-registration.** Criteria committed to git *before* each run, timestamps
   visible in the log. Nine in the report, and a tenth run after the freeze.
   Six failed their own criteria; one returned a competing method as better.
5. **An automated audit of 255 numerical claims** against the stored result
   files, plus 76 unit tests. Both pass. If any number in the report drifts from
   its artifact, the audit fails.

## 5. Against which dataset?

**Human3.6M** for the headline. Four synchronized cameras, the field's standard
benchmark, and it took no part in designing or tuning the method.

One caveat I state myself rather than wait to be asked: MotionAGFormer-XS was
*trained* on Human3.6M, so the backbone is in-domain there. What is held out is
the **method** — no Human3.6M frame was used to design or tune the
canonicalization.

## 6. Which datasets did you use, and why two?

| Dataset | Role | Why |
|---|---|---|
| **MPI-INF-3DHP** | development | Eight synchronized cameras, so many cross-view pairs; large domain shift from the backbone's training data, which makes failures visible |
| **Human3.6M** | validation | The standard benchmark; disjoint from development; 180 held-out pairs |

**Why two and not one:** a result on a single dataset is an observation about
that dataset. I learned this the expensive way. A bone-length error signal
reached ρ = +0.492 on MPI-INF-3DHP and looked like a genuine finding; on
Human3.6M it fell to **ρ = +0.098** and failed every criterion. I retracted it in
§4.9. Had I used one dataset, that retraction would instead be a claim in my
abstract.

## 7. How did you approach it from the beginning?

1. **Jan–Feb 2026.** Your proposal set the direction; I submitted my own research
   proposal committing to it. You asked to see output and gave me
   `KelvinHong/pose-estimation-3d`. The output I showed you came from there.
2. **Literature.** A survey of 37 papers, to find what was already occupied.
3. **Backbone.** I chose MotionAGFormer myself as the frozen lifter, and later
   added MotionBERT to check that nothing I saw was a property of one network.
4. **First evaluation, then its collapse.** Early results on MPI-INF-3DHP were
   invalidated by review: display post-processing had been applied before
   comparison, and one frame had been repeated 27 times. I rebuilt the protocol
   with true 27-frame windows and ten extracted cameras. **Those old numbers do
   not appear anywhere in the report.**
5. **Pre-registration from then on.** After being wrong once through my own
   sloppiness, I wrote every criterion down before each run.

## 8. Why did your own initial novelty fail?

I claimed four novelties. **All four collapsed, and I found every one of them
myself.**

1. **Training-free geometric canonicalization** — it is the **TRIAD algorithm**
   (Black, 1964, spacecraft attitude determination). Not a failure of results, a
   failure of my literature search.
2. **A geometric reliability score predicting pose error** — falsified five
   independent ways: correlation ≈ 0 across simultaneous cameras; view selection
   straddling chance (4.78 of 8 on one sequence, 3.67 on another); a
   **cross-backbone sign flip** (−0.707 on MotionBERT, +0.375 on a plain MLP);
   reliability-weighted fusion worse than a plain mean (92.6 vs 88.5 mm); and a
   pre-registered test-time-augmentation test that failed all three criteria.
3. **Bone-length consistency as an error predictor** — ρ = +0.492 on the first
   dataset, **+0.098 on the second.** Retracted.
4. **Integrating all four** — the multi-scale part builds each limb frame from
   exactly the three joints it is then scored on. **Circular by construction.**
   Demoted to an exploratory measurement.

**They failed for one shared reason, and this is the interesting part.**
Geometric plausibility is invariant to a coherent depth error. If the network
gets the depth wrong but wrongly in a *self-consistent* way, the skeleton still
looks anatomically perfect — correct bone lengths, plausible angles — while being
badly wrong in space. Geometry cannot see that error, because geometry is exactly
what the error preserves. Every one of the four was a geometric quantity being
asked to predict a non-geometric failure.

## 9. Why did the novelty in your proposal not work out?

**It did not fail, Sir, and I want to be exact about this.** The gap your
proposal named — that existing methods lack explicit geometric priors and
view-invariant constraints — is real and still open. Two things happened
underneath it.

1. **It was published while I was working.** The contrastive-plus-kinematic
   direction became **MoViD** (2026), which disentangles motion from viewpoint
   with a learned orthogonal projection, and **3DPCNet** (Sept 2025), which does
   training-free post-hoc canonicalization of a frozen estimator — the same
   problem statement as mine. Implementing your proposal in 2026 would have
   reproduced published work, not created novelty. That the field went there
   within months is evidence the direction was right.
2. **It requires training, and I could not do it credibly.** Multi-view training
   at that scale needs compute and labels I did not have. What I *could* offer
   was the one thing those methods do not have: **no training, no labels, no
   camera parameters at any stage.** That requirement profile is the seam I
   worked in.

The honest ending: 3DPCNet benchmarks a hand-built anatomical-landmark baseline
of exactly my family, **and beats it**. I cite that rather than hide it.

---

## Follow-ups he is likely to ask next

### "If accuracy did not improve, what is the point?"

> The point is comparison, Sir. In a clinic you film a patient's gait in March
> and again in June, from whatever camera angle the room allows. The estimator is
> equally accurate both times and the numbers are still incomparable, because
> each is expressed in its own camera's frame. Accuracy was never the blocker.
> Comparability was.

### "Why freeze the model instead of training your own?"

> Three reasons. Retraining needs multi-view labelled data and compute I do not
> have. Anything I trained would be weaker than a published backbone, so a gain
> would be unmeasurable. And freezing is what makes the result transferable — it
> works on MotionAGFormer and MotionBERT unchanged, because it never touches
> either.

### "Why MotionAGFormer, and why MotionBERT as well?"

> MotionAGFormer because it is current, strong on Human3.6M, and small enough at
> 2.2 million parameters to run on my hardware. MotionBERT second because a
> finding on one network is a property of that network. That second backbone is
> how I caught the reliability score flipping sign — same code, opposite
> correlation.

### "What if the hips or the thorax are detected badly?"

> Then my frame degrades badly, and that is its real weakness — it reads only
> four joints, so a bad thorax rotates everything. But it also means the reverse:
> it never reads a wrist or an ankle, while the Kabsch baseline fits all
> seventeen. I pre-registered that as a place my method might win and tested it
> two days ago: corrupt the eight joints past a hinge and see which alignment
> survives.
>
> **It failed the criterion I set.** I required the crossover at 80 mm of noise
> or below on *both* backbones. MotionBERT crossed at 40 mm; MotionAGFormer only
> at 160 mm, which I had ruled out in advance. One backbone is not two — that is
> exactly the rule by which I rejected an earlier conditioning result, and it
> would be worthless if I only applied it when it was free. So: the tenth
> pre-registration, failed, and nothing in the report changes.
>
> What I will say descriptively is that my frame is *exactly* flat across every
> corruption level, 53.45 mm at every one, because the corrupted joints are ones
> it never reads — and the baseline degrades from 43.3 to 92.3 mm. There is
> probably a regime there. I did not establish it.

### "A simpler method beats yours. Why is this still a thesis?"

> Because the simpler method cannot run the experiment. Kabsch alignment has no
> anatomical axis, so there is no axis to hold fixed and vary, and the question
> *what governs whether a body frame is consistent across viewpoints* cannot be
> posed inside it. My frame is the instrument that makes the boundary measurable.
> And I say plainly that as a way of reducing cross-view distance on this data,
> the simpler method is better. It is in the abstract.

### "How do I know these numbers are real?"

> Open the git log, Sir. Nine pre-registrations with timestamps preceding their
> own results, five of which failed. Nobody fabricates failures. Then run
> `audit_numbers.py`: it recomputes all 255 claims from the stored result files
> and fails if a single one has drifted.

### "What is the single biggest weakness?"

> That a simpler baseline beats it on the headline metric, and I found no pose
> regime where my construction is preferable. Second: every result is on two
> datasets and two backbones, which is more than most but is still not a
> guarantee of generality — as my own retraction demonstrates.

### "What would you do with six more months?"

> Test the frame where the baseline's assumption breaks rather than where mine
> holds — non-standard body proportions, children, wheelchair users, anyone the
> fixed template does not describe. The template baseline needs a skeleton that
> resembles the subject; mine needs only that the subject *has* hips and a torso.
> That is the honest place to look for an advantage, and I did not get to it.

---
---

# বাংলা

## ১. আপনি আসলে কী করেছেন?

আগে থেকে train করা একটা pose estimator নিয়ে সেটাকে **সম্পূর্ণ frozen** রেখে তার
পরে একটা গাণিতিক ধাপ যোগ করেছি। ওই ধাপ predicted কঙ্কালটাকে মানুষটার নিজের শরীর
থেকে বানানো একটা coordinate system-এ লেখে — ধড়ের অক্ষ (pelvis → thorax) আর
নিতম্বের অক্ষ (hip → hip), Gram-Schmidt দিয়ে orthonormal করা। কিছু train হয় না,
কিছু calibrate হয় না।

তারপর প্রকল্পের বেশিরভাগ সময় দিয়েছি কঠিন প্রশ্নটায়: **কী নির্ধারণ করে যে এরকম
একটা body frame বিভিন্ন viewpoint-এ consistent থাকবে?** এই প্রশ্নটাই thesis,
method না।

## ২. Base MotionAGFormer-এর চেয়ে output কতটা ভালো হলো?

দুইটা আলাদা উত্তর, দুইটাই একসাথে বলতে হবে।

- **নির্ভুলতা: একটুও না, হুবহু এক।** Human3.6M-এ action-balanced MPJPE **আগে-পরে
  ৪৫.১৪৯ mm**। বদলাতে পারেও না — estimator frozen, আর আমার ধাপে শূন্যটা parameter।
- **Cross-view মিল: ৩৭২.৭ mm → ৯৩.৪ mm, ৭২.২% কম**, ১৮০টা held-out জোড়ায়,
  ১৭৯টাতেই উন্নতি।

একটা বাক্য মনে রাখলে এটাই: *আমি prediction ভালো করিনি, একই pose-এর দুইটা
prediction-কে তুলনাযোগ্য করেছি।*

## ৩. উন্নতি হলে আসলে কী বদলাল, কীভাবে?

Joint-এর coordinate বদলেছে, pose বদলায়নি। একই কঙ্কাল, একই আকৃতি, নতুন অক্ষ।

কারণটা এক লাইনের বীজগণিত। দুইটা অক্ষই **joint থেকেই** বানানো, তাই শরীরের সাথে
ঘোরে। দ্বিতীয় ক্যামেরা যদি একই pose-কে অজানা rotation **Q** দিয়ে ঘোরানো দেখে,
frame-ও **Q** দিয়ে ঘোরে — রূপান্তরে দুইটা **ঠিক ঠিক কাটাকাটি** হয়ে যায়। অজানা
ক্যামেরা rotation **estimate না করেই** মুছে যায়। এজন্যই calibration লাগে না।

## ৪. প্রমাণ করলেন কীভাবে?

১. **১৮০টা held-out ক্যামেরা জোড়া**, এমন dataset-এ যেটা method তৈরিতে ব্যবহার হয়নি।
২. **ত্রিশটা subject-action গ্রুপে cluster bootstrap** — ১৮০টা জোড়ায় না। একই
   গতির একই চারটা stream থেকে ছয়টা জোড়া আসে, তাই স্বাধীন ধরলে interval মিথ্যা
   সরু হয়। CI [+৬৭.৯, +৭৫.৫]%।
৩. **দুইটা backbone।** MotionAGFormer-XS ৭২.২%, MotionBERT ৭৫.৮%।
৪. **Pre-registration।** প্রত্যেক run-এর **আগে** criterion git-এ commit করা,
   timestamp log-এ দেখা যায়। Report-এ নয়টা, freeze-এর পরে দশমটা। ছয়টা নিজেদের
   criterion-এ ব্যর্থ, একটায় প্রতিদ্বন্দ্বী method ভালো প্রমাণিত।
৫. **২৫৫টা সংখ্যাগত দাবির স্বয়ংক্রিয় audit** সংরক্ষিত ফাইলের বিপরীতে, সাথে ৭৬টা
   test। দুইটাই pass করে।

## ৫. কোন dataset-এ?

মূল ফল **Human3.6M**-এ — চারটা synchronized ক্যামেরা, standard benchmark, এবং
method ডিজাইন বা tune করতে এর কোনো ভূমিকা ছিল না।

একটা সতর্কতা আমি নিজেই বলি: MotionAGFormer-XS নিজে Human3.6M-এ **train করা**, তাই
backbone-টা এখানে in-domain। Held out হলো **method** — canonicalization বানাতে
H36M-এর একটা frame-ও লাগেনি।

## ৬. কোন কোন dataset, আর দুইটা কেন?

| Dataset | ভূমিকা | কেন |
|---|---|---|
| **MPI-INF-3DHP** | development | আটটা synchronized ক্যামেরা, অনেক cross-view জোড়া; domain shift বড়, তাই ব্যর্থতা চোখে পড়ে |
| **Human3.6M** | validation | Standard benchmark; development থেকে সম্পূর্ণ আলাদা; ১৮০টা held-out জোড়া |

**দুইটা কেন:** এক dataset-এর ফল ওই dataset সম্পর্কে একটা পর্যবেক্ষণ মাত্র। এটা
আমি দামি অভিজ্ঞতা দিয়ে শিখেছি — একটা bone-length signal MPI-তে ρ = +০.৪৯২ পেয়ে
সত্যিকারের আবিষ্কার মনে হয়েছিল; H36M-এ সেটা **+০.০৯৮**-এ নেমে সব criterion-এ
ব্যর্থ হয়। §৪.৯-এ প্রত্যাহার করেছি। এক dataset ব্যবহার করলে ওটাই আজ আমার
abstract-এ দাবি হিসেবে থাকত।

## ৭. শুরু থেকে কীভাবে এগিয়েছেন?

১. **জানু–ফেব্রু ২০২৬।** আপনার proposal দিক ঠিক করে দেয়; আমি নিজের research
   proposal জমা দিই। আপনি output দেখতে চান এবং `KelvinHong/pose-estimation-3d`
   দেন — যে output আপনাকে দেখিয়েছি সেটা ওখান থেকেই।
২. **সাহিত্য পর্যালোচনা।** ৩৭টা পেপার, কোন জায়গা আগেই দখল হয়ে আছে দেখতে।
৩. **Backbone।** MotionAGFormer আমি নিজে বেছেছি, পরে MotionBERT যোগ করেছি যাতে
   কোনো পর্যবেক্ষণ একটা network-এর বৈশিষ্ট্য না হয়ে যায়।
৪. **প্রথম মূল্যায়ন, তারপর সেটার পতন।** MPI-তে প্রথম ফলাফল পর্যালোচনায় বাতিল
   হয়: তুলনার আগে display post-processing লেগেছিল, আর একটা frame ২৭ বার
   পুনরাবৃত্তি হয়েছিল। সত্যিকারের ২৭-frame window আর দশটা ক্যামেরা দিয়ে protocol
   আবার বানাই। **ওই পুরোনো সংখ্যা report-এর কোথাও নেই।**
৫. **তারপর থেকে pre-registration।** নিজের অসাবধানতায় একবার ভুল হওয়ার পরে
   প্রত্যেক run-এর আগে criterion লিখে রেখেছি।

## ৮. আপনার নিজের প্রাথমিক novelty কেন ব্যর্থ হলো?

চারটা novelty দাবি করেছিলাম। **চারটাই ভেঙে পড়েছে, এবং প্রত্যেকটা আমি নিজেই ধরেছি।**

১. **Training-free geometric canonicalization** — এটা **TRIAD algorithm** (Black,
   ১৯৬৪)। ফলাফলের ব্যর্থতা না, আমার literature search-এর ব্যর্থতা।
২. **Pose error অনুমান করার geometric reliability score** — পাঁচভাবে মিথ্যা
   প্রমাণিত: একসাথের ক্যামেরাগুলোর মধ্যে correlation ≈ ০; view selection
   সম্ভাবনার দুই পাশে (৮-এ ৪.৭৮ বনাম ৩.৬৭); **backbone বদলালে চিহ্ন উল্টে যায়**
   (MotionBERT-এ −০.৭০৭, সাধারণ MLP-তে +০.৩৭৫); reliability-weighted fusion
   সাধারণ গড়ের চেয়ে খারাপ (৯২.৬ বনাম ৮৮.৫ mm); এবং pre-registered TTA পরীক্ষা
   তিনটা criterion-এই ব্যর্থ।
৩. **Bone-length consistency দিয়ে error অনুমান** — প্রথম dataset-এ ρ = +০.৪৯২,
   দ্বিতীয়টায় **+০.০৯৮**। প্রত্যাহার করেছি।
৪. **চারটার সমন্বয়** — multi-scale অংশে প্রত্যেক limb frame ঠিক সেই তিনটা joint
   দিয়ে বানানো যেগুলোর ওপরেই score হয়। **গঠনগতভাবেই circular।** Exploratory
   পরিমাপে নামিয়ে দেওয়া হয়েছে।

**চারটাই এক কারণে ব্যর্থ, আর এখানেই আসল কথা।** Coherent depth error-এর সাপেক্ষে
geometric plausibility অপরিবর্তিত থাকে। Network যদি depth ভুল করে কিন্তু
*সামঞ্জস্যপূর্ণভাবে* ভুল করে, কঙ্কালটা দেখতে নিখুঁতই থাকে — bone length ঠিক,
কোণ যুক্তিসঙ্গত — অথচ স্থানিকভাবে মারাত্মক ভুল। জ্যামিতি ওই error দেখতে পায় না,
কারণ error টা ঠিক জ্যামিতিটাকেই অক্ষত রাখে।

## ৯. আপনার proposal-এর novelty কেন কাজে এলো না?

**ব্যর্থ হয়নি স্যার, এই জায়গায় আমি সুনির্দিষ্ট থাকতে চাই।** আপনার proposal যে
gap চিহ্নিত করেছিল — প্রচলিত পদ্ধতিতে explicit geometric prior আর view-invariant
constraint নেই — সেটা বাস্তব এবং এখনো খোলা। নিচে দুইটা ঘটনা ঘটেছে।

১. **আমি কাজ করতে করতেই সেটা প্রকাশিত হয়ে গেছে।** Contrastive + kinematic
   দিকটা হয়ে উঠেছে **MoViD** (২০২৬) — learned orthogonal projection দিয়ে
   motion আর viewpoint আলাদা করে; আর **3DPCNet** (সেপ্টেম্বর ২০২৫) — frozen
   estimator-এর training-free post-hoc canonicalization, হুবহু আমার সমস্যাটাই।
   ২০২৬-এ ওটা বানালে প্রকাশিত কাজের পুনরাবৃত্তি হতো, novelty হতো না। কয়েক মাসেই
   ক্ষেত্রটা ওদিকে গেছে — এটাই প্রমাণ যে দিকটা ঠিক ছিল।
২. **ওটার জন্য training লাগে, যেটা আমি বিশ্বাসযোগ্যভাবে করতে পারতাম না।**
   ওই মাপের multi-view training-এর compute আর label আমার ছিল না। আমি যা দিতে
   পারতাম সেটাই ওই পদ্ধতিগুলোর নেই: **কোনো ধাপে training নেই, label নেই, ক্যামেরা
   parameter নেই।**

সৎ শেষটা: 3DPCNet ঠিক আমার ঘরানার একটা anatomical-landmark baseline-এর সাথে
তুলনা করে **এবং সেটাকে হারায়**। এটা লুকাইনি, উদ্ধৃত করেছি।

## যে প্রশ্নগুলো এরপর আসবে

**"নির্ভুলতা না বাড়লে লাভ কী?"**

> লাভটা তুলনায়, স্যার। ক্লিনিকে মার্চে একজন রোগীর হাঁটা রেকর্ড করলেন, আবার জুনে —
> ঘরে যে কোণ থেকে সম্ভব। দুইবারই estimator সমান নির্ভুল, তবু সংখ্যা দুইটা
> তুলনাযোগ্য না, কারণ প্রত্যেকটা নিজের ক্যামেরার frame-এ। বাধাটা কখনোই
> নির্ভুলতা ছিল না, তুলনাযোগ্যতা ছিল।

**"নিজে train না করে model freeze করলেন কেন?"**

> তিনটা কারণ। Retraining-এর জন্য multi-view labelled data আর compute দরকার, যা
> আমার নেই। আমি যা train করতাম তা প্রকাশিত backbone-এর চেয়ে দুর্বল হতো, ফলে
> উন্নতি মাপাই যেত না। আর freeze করাই ফলটাকে হস্তান্তরযোগ্য করে — MotionAGFormer
> আর MotionBERT দুইটাতেই অপরিবর্তিতভাবে চলে, কারণ কোনোটাতেই হাত পড়ে না।

**"সহজ method জিতলে এটা thesis কেন?"**

> কারণ সহজ method দিয়ে পরীক্ষাটাই চালানো যায় না। Kabsch-এর anatomical অক্ষ নেই,
> তাই স্থির রেখে পরিবর্তন করার মতো অক্ষ নেই, আর "body frame-এর consistency কী
> নিয়ন্ত্রণ করে" প্রশ্নটা ওর ভেতরে করাই যায় না। আমার frame হলো সেই যন্ত্র যা
> সীমাটা পরিমাপযোগ্য করে। আর এই data-তে cross-view দূরত্ব কমানোর উপায় হিসেবে
> সহজ method-টাই ভালো — এটা abstract-এই লেখা আছে।

**"সংখ্যাগুলো সত্যি, বুঝব কীভাবে?"**

> Git log খুলুন স্যার। নয়টা pre-registration, প্রত্যেকটার timestamp তার নিজের
> ফলাফলের আগে, তার মধ্যে পাঁচটা ব্যর্থ। কেউ ব্যর্থতা বানিয়ে লেখে না। তারপর
> `audit_numbers.py` চালান — ২৫৫টা দাবিই সংরক্ষিত ফাইল থেকে আবার হিসাব করে, একটাও
> সরে গেলে fail করে।

**"সবচেয়ে বড় দুর্বলতা কী?"**

> মূল metric-এ একটা সহজ baseline এটাকে হারায়, আর এমন কোনো pose regime আমি পাইনি
> যেখানে আমার construction ভালো। দ্বিতীয়ত, সব ফল দুইটা dataset আর দুইটা
> backbone-এ — বেশিরভাগের চেয়ে বেশি, তবু সাধারণীকরণের নিশ্চয়তা না; আমার নিজের
> প্রত্যাহারই তার প্রমাণ।
