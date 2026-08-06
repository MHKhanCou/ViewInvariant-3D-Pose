# The really simple version

`EXPLAIN.md` is the full explanation. This one is shorter and uses no jargon.
Read this first. English, then Bengali.

---

# ENGLISH

## The problem, in one picture

Imagine a person standing in a room with two cameras — one in front, one to the
side. Both cameras film the same moment.

The computer looks at each video and works out where the body's joints are in 3D.
**Both answers are correct.** But the front camera describes the body *from where
it is standing*, and the side camera describes it *from where it is standing*. So
the two describe the same pose with completely different numbers.

It's like two people describing where a chair is. One says "three steps to my
left." The other says "five steps in front of me." Both are telling the truth.
Neither can be compared with the other, because **each one is measuring from
themselves.**

## Why that's a real problem

A doctor wants to compare how a patient walks today with how they walked six
months ago. A coach wants to compare an athlete's throw with last week's. Both
need the numbers to mean the same thing across recordings.

Right now they don't. And the usual fixes all need something a small clinic
doesn't have — money to retrain the AI, or a careful camera setup measured in
advance.

## What I did

Instead of measuring from the camera, **measure from the body itself.**

Take the skeleton the computer just predicted. Draw two lines *on the body*:

- one up the middle of the trunk (hips to chest)
- one across the hips (left hip to right hip)

Those two lines belong to the body, so **they turn when the body turns.** Use them
to build a set of directions, and describe every joint relative to *those*
instead of relative to the camera.

Now both cameras give the same answer. Not approximately — the camera's own
angle cancels out completely in the mathematics, and it cancels without ever
having to be measured. That's why no camera setup is needed.

**Best analogy:** instead of saying "the wrist is 3 steps to *my* left", say "the
wrist is in front of the chest and to the right of the spine." That sentence is
true from anywhere in the room. Everyone agrees, because it's described using the
body, not the viewer.

And it's cheap — I don't retrain the AI at all. I take whatever it produces and
turn it, afterwards. It costs about as much computing as a fraction of a
percent of what the AI itself costs.

## The honest part: I didn't invent this

The way I build those directions is a method from **1964, used to work out which
way a spacecraft is pointing** (the TRIAD algorithm). The maths about how joint
errors spread into direction errors comes from **medical movement science**. I
credit both in the report.

So the real question I studied is not "did I invent something." It's:

> **When does this trick work, and when does it stop working?**

## What I found — three questions, one yes and two noes

There's an old rule: **a longer line gives a more reliable direction.** If you
draw a direction between two points and both points are a bit wrong, the further
apart they are, the less that error matters. (Think of aiming: a long rifle
barrel points more precisely than a short one.)

I tested that rule at three levels. I wrote down what would count as success
**before** running each test, so I couldn't move the goalposts afterwards.

**1. Does a longer line make a better frame? → YES.**
The shoulders are further apart than the hips. Swapping the hip line for the
shoulder line makes it work better, on two different AI models.

**2. Can it tell me which particular recording to trust? → NO.**
Inside one method, everybody's hips are about the same width. There's nothing to
distinguish one from another.

**3. Can it tell me which joint will be worst? → NO — and it's backwards.**
The rule predicts joints further from the centre should be worse. In reality,
joints locked to the trunk sit *further out* and are **2.5 times better** than the
knees and elbows.

**Why?** Because a body is not a solid object. It bends. Once you go past a joint
that bends — an elbow, a knee — you also inherit the AI's mistake about *how much
that joint is bent*, and that mistake is much bigger than anything to do with
distance.

**That third result is my actual finding.** The old rule is real, but it only
answers one question: *what should I build the frame out of.* Nothing more.

## The main number

On a big standard dataset I never used while building the method, my approach
makes the two cameras agree **72% better**, and it improves **179 out of 180**
camera pairs.

## The bad news, which I put in my own abstract

I tested something simpler: just rotate every pose to match one single fixed
reference skeleton. It needs no training, no labels, no camera setup — exactly
the same advantages I claim for mine.

**It beat mine. Every single one of the 180 pairs. Both AI models. All 15
activities.**

## So why is my work still worth something?

> The simple method wins at the *task*. But **the simple method cannot ask my
> question.** It has no line drawn on the body, so there is nothing to lengthen
> or shorten, and you cannot study "does a longer line help" inside a method that
> has no line. My method is the **measuring instrument** that made the question
> answerable.

Think of a thermometer. A thermometer doesn't make the room warmer. It's still
worth having, because it's the thing that lets you find out what does.

And I say the honest half out loud too: **if you just want the cameras to agree
on this data, use the simpler method.**

## The thing I'm actually proud of

- I decided what would count as success **before** each of my nine experiments,
  and saved it with a timestamp so nobody can claim I changed my mind afterwards.
- **Five of the nine failed.** I reported all five.
- A sixth showed a rival method was better. I put that in the abstract.
- A computer program re-checks all **252 numbers** in my report against the raw
  data files, and fails if even one of them drifts.
- When I found one of my own results was measuring the wrong thing, I said so and
  took it back.

**Most projects report only what worked. Mine reports what didn't, and shows
you the timestamps.**

---
---

# বাংলা — একদম সহজ ভাষায়

## সমস্যাটা, একটা ছবির মতো করে

ভাবুন একজন মানুষ একটা ঘরে দাঁড়িয়ে আছে, দুইটা ক্যামেরা — একটা সামনে, একটা পাশে।
দুইটাই একই মুহূর্তের ভিডিও করছে।

কম্পিউটার দুইটা ভিডিও দেখে বের করে শরীরের জোড়াগুলো ত্রিমাত্রিক জায়গায় কোথায়।
**দুইটা উত্তরই ঠিক।** কিন্তু সামনের ক্যামেরা বর্ণনা করে *সে যেখানে দাঁড়িয়ে সেখান
থেকে*, আর পাশের ক্যামেরা বর্ণনা করে *সে যেখানে দাঁড়িয়ে সেখান থেকে*। ফলে একই ভঙ্গির
দুইটা সম্পূর্ণ আলাদা সংখ্যা আসে।

ব্যাপারটা এরকম — দুইজন মানুষ একটা চেয়ার কোথায় সেটা বলছে। একজন বলছে "আমার বাঁদিকে
তিন কদম"। আরেকজন বলছে "আমার সামনে পাঁচ কদম"। **দুইজনই সত্যি বলছে।** কিন্তু দুইটা
কথা মেলানো যাবে না, কারণ **প্রত্যেকে নিজের জায়গা থেকে মাপছে।**

## এটা কেন আসল সমস্যা

একজন ডাক্তার দেখতে চান রোগী আজ কীভাবে হাঁটছে আর ছয় মাস আগে কীভাবে হাঁটত। একজন কোচ
মেলাতে চান খেলোয়াড়ের আজকের থ্রো আর গত সপ্তাহেরটা। দুইজনেরই দরকার সংখ্যাগুলোর মানে
সব রেকর্ডিংয়ে **একই** হওয়া।

এখন সেটা হয় না। আর প্রচলিত সমাধানগুলোর জন্য এমন কিছু লাগে যা ছোট ক্লিনিকে থাকে না
— AI আবার ট্রেনিং করানোর টাকা, বা আগে থেকে মেপে রাখা যত্নের ক্যামেরা সেটআপ।

## আমি যা করেছি

ক্যামেরা থেকে না মেপে, **শরীর থেকেই মাপি।**

কম্পিউটার যে কঙ্কালটা বের করল, সেটা নিন। শরীরের ওপরেই দুইটা রেখা আঁকুন:

- একটা ধড় বরাবর উপরে (কোমর থেকে বুক)
- একটা কোমরের আড়াআড়ি (বাম কোমর থেকে ডান কোমর)

এই দুইটা রেখা **শরীরের নিজের**, তাই **শরীর ঘুরলে রেখা দুইটাও ঘোরে**। এগুলো দিয়ে
দিক ঠিক করুন, আর প্রতিটা জোড়াকে ক্যামেরার সাপেক্ষে না বলে **ওই দিকগুলোর সাপেক্ষে**
বলুন।

এখন দুইটা ক্যামেরাই একই উত্তর দেয়। প্রায় একই না — অঙ্কে ক্যামেরার নিজের কোণটা
**পুরোপুরি কেটে যায়**, এবং কোণটা কখনো মাপতেই হয় না। এজন্যই কোনো ক্যামেরা সেটআপ
লাগে না।

**সবচেয়ে ভালো তুলনা:** "কব্জিটা *আমার* বাঁদিকে তিন কদম" না বলে বলুন "কব্জিটা বুকের
সামনে আর মেরুদণ্ডের ডানে"। এই কথাটা **ঘরের যেকোনো জায়গা থেকে সত্যি**। সবাই একমত
হবে, কারণ এটা শরীর দিয়ে বলা, দর্শক দিয়ে না।

আর খরচ প্রায় নেই — আমি AI-টাকে আবার ট্রেনিং করাই না। ও যা দেয় সেটা নিয়ে পরে ঘুরিয়ে
দিই। AI-এর নিজের খরচের শতকরা এক ভাগেরও অনেক কম লাগে।

## সৎ কথাটা: এটা আমার আবিষ্কার না

দিকগুলো যেভাবে বানাই, সেটা **১৯৬৪ সালের একটা পদ্ধতি, যা দিয়ে মহাকাশযান কোন দিকে
মুখ করে আছে বের করা হয়** (TRIAD algorithm)। আর জোড়ার ভুল কীভাবে দিকের ভুলে ছড়ায়,
সেই অঙ্ক এসেছে **চিকিৎসা বিজ্ঞানের নড়াচড়া গবেষণা** থেকে। রিপোর্টে দুইটারই কৃতিত্ব
দেওয়া আছে।

তাই আমার আসল গবেষণা "আমি কিছু আবিষ্কার করেছি কি না" না। আসল প্রশ্ন:

> **এই কৌশলটা কখন কাজ করে, আর কখন কাজ করা বন্ধ করে দেয়?**

## যা পেয়েছি — তিনটা প্রশ্ন, একটা হ্যাঁ আর দুইটা না

একটা পুরনো নিয়ম আছে: **রেখা যত লম্বা, দিক তত নির্ভরযোগ্য।** দুইটা বিন্দু দিয়ে দিক
আঁকলে, আর দুইটা বিন্দুই যদি একটু ভুল থাকে, তাহলে বিন্দু দুইটা যত দূরে থাকবে ভুলটা
তত কম ক্ষতি করবে। (বন্দুকের নলের কথা ভাবুন — লম্বা নল বেশি নিখুঁতভাবে তাক করে।)

আমি এই নিয়মটা তিনটা স্তরে পরীক্ষা করেছি। প্রত্যেকটা পরীক্ষার **আগে** লিখে রেখেছি
কোনটাকে সফল বলব — যাতে পরে গোলপোস্ট সরাতে না পারি।

**১. লম্বা রেখা কি ভালো ফ্রেম দেয়? → হ্যাঁ।**
কাঁধ দুইটা কোমরের চেয়ে বেশি দূরে দূরে। কোমরের রেখার বদলে কাঁধের রেখা নিলে ভালো কাজ
করে, দুইটা আলাদা AI মডেলেই।

**২. কোন রেকর্ডিংটা বিশ্বাস করব, সেটা বলতে পারে? → না।**
এক পদ্ধতির ভেতরে সবার কোমরের চওড়া প্রায় একই। আলাদা করার মতো কিছু নেই।

**৩. কোন জোড়াটা সবচেয়ে খারাপ হবে, বলতে পারে? → না — বরং উল্টো।**
নিয়ম বলে কেন্দ্র থেকে যত দূরে, তত খারাপ। বাস্তবে ধড়ের সাথে আটকানো জোড়াগুলো *আরও
দূরে* থেকেও হাঁটু-কনুইয়ের চেয়ে **আড়াই গুণ ভালো**।

**কেন?** কারণ শরীর শক্ত কোনো জিনিস না। **শরীর ভাঁজ হয়।** যে জোড়া ভাঁজ হয় — কনুই,
হাঁটু — সেটা পেরোলেই AI-এর আরেকটা ভুল যোগ হয়: *জোড়াটা কতটা ভাঁজ হয়েছে* সেই ভুল।
আর ওই ভুলটা দূরত্বের ব্যাপারটার চেয়ে অনেক বড়।

**এই তৃতীয় ফলাফলটাই আমার আসল আবিষ্কার।** পুরনো নিয়মটা সত্যি, কিন্তু ওটা শুধু একটা
প্রশ্নের উত্তর দেয়: *ফ্রেমটা কী দিয়ে বানাব।* এর বেশি কিছু না।

## মূল সংখ্যা

একটা বড় প্রচলিত ডেটাসেটে — যেটা পদ্ধতি বানানোর সময় আমি একবারও ব্যবহার করিনি —
আমার পদ্ধতিতে দুইটা ক্যামেরার মিল **৭২% ভালো** হয়, আর **১৮০টার মধ্যে ১৭৯টা**
ক্যামেরা-জোড়ায় উন্নতি হয়।

## খারাপ খবরটা, যেটা আমি নিজের abstract-এ লিখেছি

আমি আরও সহজ একটা জিনিস পরীক্ষা করেছি: প্রতিটা ভঙ্গিকে শুধু **একটা নির্দিষ্ট নমুনা
কঙ্কালের** সাথে মিলিয়ে ঘুরিয়ে দেওয়া। এতে ট্রেনিং লাগে না, লেবেল লাগে না, ক্যামেরা
সেটআপ লাগে না — অর্থাৎ আমি আমার পদ্ধতির যেসব সুবিধা দাবি করি, ঠিক সেগুলোই।

**ওটা আমারটাকে হারিয়ে দিয়েছে। ১৮০টা জোড়ার প্রত্যেকটায়। দুইটা AI মডেলেই। পনেরোটা
কাজের সবগুলোতে।**

## তাহলে আমার কাজের মূল্য কী?

> সহজ পদ্ধতিটা **কাজে** জেতে। কিন্তু **সহজ পদ্ধতিটা আমার প্রশ্নটাই করতে পারে না।**
> ওর শরীরের ওপর কোনো রেখা আঁকা নেই, তাই লম্বা-খাটো করার মতো কিছুই নেই — আর যে
> পদ্ধতিতে রেখাই নেই, তার ভেতরে "লম্বা রেখা কি সাহায্য করে" এই প্রশ্নটা করা যায়
> না। আমার পদ্ধতি হলো সেই **মাপার যন্ত্র**, যেটা প্রশ্নটার উত্তর বের করা সম্ভব
> করেছে।

থার্মোমিটারের কথা ভাবুন। থার্মোমিটার ঘর গরম করে না। তবু ওটা রাখার মূল্য আছে, কারণ
কী ঘর গরম করে সেটা **ওটা দিয়েই** জানা যায়।

আর সৎ অর্ধেকটাও আমি স্পষ্ট বলি: **এই ডেটায় শুধু ক্যামেরাগুলোকে মেলাতে চাইলে সহজ
পদ্ধতিটাই ব্যবহার করুন।**

## যে জিনিসটা নিয়ে আমি আসলেই গর্বিত

- আমার নয়টা পরীক্ষার প্রত্যেকটার **আগে** ঠিক করে রেখেছি কোনটাকে সফল বলব, আর
  সময়-চিহ্ন সহ সংরক্ষণ করেছি — যাতে কেউ বলতে না পারে আমি পরে মত বদলেছি।
- **নয়টার মধ্যে পাঁচটা ব্যর্থ হয়েছে।** পাঁচটাই রিপোর্ট করেছি।
- ষষ্ঠটায় দেখা গেল প্রতিদ্বন্দ্বী পদ্ধতি ভালো। সেটা abstract-এ লিখেছি।
- একটা প্রোগ্রাম আমার রিপোর্টের **২৫২টা সংখ্যা** কাঁচা ডেটা ফাইলের সাথে মিলিয়ে
  দেখে, আর একটাও এদিক-ওদিক হলে fail করে।
- নিজের একটা ফলাফল ভুল জিনিস মাপছিল বুঝতে পেরে সেটা বলে দিয়েছি এবং ফিরিয়ে নিয়েছি।

**বেশিরভাগ প্রজেক্ট শুধু যা কাজ করেছে সেটা লেখে। আমারটা যা কাজ করেনি সেটাও লেখে —
এবং সময়-চিহ্ন দেখিয়ে দেয়।**
