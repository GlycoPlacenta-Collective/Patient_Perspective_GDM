"""
Prompts, structured-output schemas, and helpers for both LLM stages
====================================================================
Stage 1 - Self-use classification: does the author personally have/have had GDM?
Stage 2 - Narrative extraction: extract the author's GDM pathway experiences
          and assign each to a domain (SCREENING, DIAGNOSIS,
          ANTEPARTUM_MONITORING, BIRTH, POSTPARTUM).
 
Both stages are run through batch_pipeline.py 

"""

import json
from pathlib import Path
import pandas as pd


"""
Shared: User message
==========================================================
"""
def build_user_message(row):
    """Format a Reddit post into the user message for the classifier."""
    return f"posted in r/{row['subreddit']}\nposted on {row['date']}\n{row['message']}"

"""
Stage 1: Self-Use Classification
==========================================================
"""
SELF_USE_SYSTEM_PROMPT = """\
You are a research assistant helping classify Reddit posts for a study on \
gestational diabetes (GDM) experiences.

Your task is to determine whether the author of a post is personally describing \
their own experience with gestational diabetes during a current or past pregnancy.

Classify the post as self_use = true ONLY if the author:
- Has been diagnosed with gestational diabetes themselves, OR
- Is currently managing gestational diabetes in their own pregnancy

Classify as self_use = false if the post:
- Asks a general question without indicating personal diagnosis
- Discusses someone else's gestational diabetes (partner, friend, family member)
- Is about a previous pregnancy with no indication of current/past personal diagnosis
- Is from a healthcare provider discussing patients
- Is informational or hypothetical with no personal context

IMPORTANT — Undiagnosed/awaiting-test exclusion:
Simply having taken, being scheduled for, or awaiting results of a glucose \
screening or diagnostic test (1-hour GCT, 3-hour OGTT, 2-hour OGTT) is NOT \
by itself a GDM diagnosis. If the author has not yet received a positive \
result or explicit diagnosis, classify as self_use = false, even if they \
describe risk factors (obesity, PCOS, prediabetes, family history) or \
anxiety about an upcoming/pending test. Only classify as self_use = true \
once the post indicates the test came back positive or the author states \
they were diagnosed with or have GDM.

IMPORTANT — Pre-existing diabetes exclusion:
If the author indicates they have pre-existing type 1 diabetes (T1DM) or \
type 2 diabetes (T2DM), classify as self_use = false and set \
"pre_existing_diabetes" to the type indicated. This study focuses exclusively \
on gestational diabetes mellitus (GDM) — diabetes first detected in pregnancy \
that does not meet criteria for overt diabetes. Look for cues including but \
not limited to: "I'm type 1," "I've had type 2 for years," "T1D," "T2D," \
"I was diabetic before pregnancy," "pre-existing diabetes," "I've been on \
insulin since before I was pregnant." Note that some authors may have both \
pre-existing diabetes AND gestational diabetes mentioned in a post, or may \
be posting in a GDM subreddit despite having pre-existing diabetes — these \
should still be classified as self_use = false. If the author mentions being \
"pre-diabetic" or having a history of insulin resistance before pregnancy \
but was formally diagnosed with GDM during pregnancy, classify as \
self_use = true and note "pre_diabetic_history" = true.

Extract management_methods carefully:
- DO include any method the author personally uses/used to manage their \
OWN GDM: diet control, insulin, metformin, glyburide, CGM/Dexcom use, \
blood glucose finger-prick monitoring (and frequency if stated), and \
specific dietary guidance a provider gave them (e.g., food-pairing advice).
- DO NOT include medications or treatments unrelated to GDM management \
(e.g., blood thinners/anticoagulants, medications for unrelated \
conditions), advice given to someone else that the author is only \
relaying, or outcome/status descriptions that are not themselves a \
method (e.g., "sugars controlled" describes a result, not a method — \
only include the actual method, such as "diet controlled," if stated).

Additionally, extract the following contextual metadata when mentioned \
or clearly inferable from the post. For each field, return null if not \
mentioned or not inferable. Do not guess.

- "gravidity": Whether this is the author's first pregnancy or a \
subsequent pregnancy. Only set this when parity is EXPLICITLY stated \
or unambiguous — e.g., "first pregnancy," "FTM" (first-time mom), \
"second baby," "with my first I had GDM." Do NOT infer gravidity from \
a single passing mention of GDM, from the subreddit, or from indirect \
phrases like "my history" or "this time" without a clearer parity cue. \
When in doubt, return null rather than guessing. Classify as \
"first_pregnancy", "subsequent_pregnancy", or null.

- "gdm_recurrence": Whether the author has had a CONFIRMED GDM \
diagnosis in a prior pregnancy AND is either confirmed-diagnosed or \
managing GDM in another (current or subsequent) pregnancy. Look for \
explicit cues such as: "had GDM last time and now again," "this is my \
second GDM pregnancy," "didn't have it with my first but do now." \
Do NOT set this to true based on: GDM mentioned in only one of \
multiple pregnancies, ambiguous language like "my history" or "both \
times" without a clear statement that GDM occurred more than once, \
suspected-but-undiagnosed prior GDM, or a current pregnancy that has \
not yet been tested/diagnosed. When the post does not clearly support \
GDM occurring in more than one pregnancy, return null rather than \
guessing true. Classify as true, false, or null.

- "other_conditions": Conditions the AUTHOR THEMSELVES has, mentioned \
in the context of their own pregnancy or GDM care — including but not \
limited to PCOS, obesity, pre-eclampsia, hypertension, thyroid \
conditions, prior type 2 diabetes risk factors, history of \
macrosomia. Do NOT include conditions discussed only in general terms \
or about someone else, and do NOT include conception methods such as \
IVF or IUI (these are not medical conditions). Return as a list of \
strings or an empty list.

- "pregnancy_number": Specific pregnancy number if stated (e.g., \
"pregnant with #3"). Return as an integer or null.

- "pre_existing_diabetes": If the author indicates pre-existing T1DM \
or T2DM, return "T1DM" or "T2DM". If no pre-existing diabetes is \
indicated, return null.

- "pre_diabetic_history": If the author mentions being pre-diabetic \
or having insulin resistance before pregnancy but was diagnosed with \
GDM during pregnancy, return true. Otherwise return false.

Return a JSON object with the following schema:
{
  "self_use": true or false,
  "confidence": "high", "medium", or "low",
  "management_methods": ["<method1>", "<method2>"],
  "gravidity": "first_pregnancy", "subsequent_pregnancy", or null,
  "gdm_recurrence": true, false, or null,
  "other_conditions": ["<condition1>", "<condition2>"],
  "pregnancy_number": <integer or null>,
  "pre_existing_diabetes": "T1DM", "T2DM", or null,
  "pre_diabetic_history": true or false,
  "reasoning": "<one sentence explaining your classification>"
}"""

SELF_USE_SCHEMA = {
    "type": "object",
    "properties": {
        "self_use": {
            "type": "boolean",
            "description": "Whether the author personally has/had gestational diabetes."
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level in the classification."
        },
        "management_methods": {
            "type": "array",
            "items": {"type": "string"},
            "description": "GDM management methods mentioned (diet, insulin, metformin, etc.)"
        },
        "gravidity": {
            "type": ["string", "null"],
            "description": "Whether this is the author's first or a subsequent pregnancy. One of: 'first_pregnancy', 'subsequent_pregnancy', or null."
        },
        "pre_existing_diabetes": {
            "type": ["string", "null"],
            "description": "Pre-existing diabetes type if indicated. One of: 'T1DM', 'T2DM', or null."
        },
        "gdm_recurrence": {
            "type": ["boolean", "null"],
            "description": "Whether the author has had GDM in a prior pregnancy."
        },

        "other_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Co-occurring conditions mentioned in context of pregnancy or GDM care."
        },
        "pregnancy_number": {
            "type": ["integer", "null"],
            "description": "Specific pregnancy number if stated, or null."
        },
        "pre_diabetic_history": {
            "type": "boolean",
            "description": "True if author was pre-diabetic before pregnancy but diagnosed with GDM."
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining the classification."
        }
    },
    "required": [
        "self_use", "confidence", "management_methods",
        "gravidity", "gdm_recurrence", "other_conditions",
        "pregnancy_number", "pre_existing_diabetes", "pre_diabetic_history",
        "reasoning"
    ],
    "additionalProperties": False
}

"""
Stage 2: Narrative (pathway experience) Extraction
==========================================================
"""

GDM_EXPERIENCE_SYSTEM_PROMPT = """\
You are a research assistant for a study on patient experiences with the \
gestational diabetes mellitus (GDM) diagnostic and monitoring pathway. \
Your task is to extract self-reported experiences from a Reddit post and \
classify each into the appropriate pathway domain.

This study focuses exclusively on gestational diabetes mellitus (GDM). \
If the post indicates the author has pre-existing type 1 diabetes (T1DM) \
or type 2 diabetes (T2DM) rather than GDM, return an empty result. \
Posts should have been filtered upstream, but if a post about pre-existing \
diabetes reaches this stage, do not extract experiences from it.

Instructions:

* Only extract experiences the author attributes to their OWN GDM care. \
Do NOT extract experiences described about other people, hypothetical \
scenarios, or experiences unrelated to the GDM pathway.

* For each experience, identify:
  1. The specific experience described (in a short normalized phrase)
  2. The pathway domain it belongs to

* Pathway domains are listed below with illustrative examples. These \
examples are not exhaustive — extract ANY experience the author \
describes in connection with their GDM care, even if it does not match \
the examples listed.

  - SCREENING: Experiences related to glucose screening and diagnostic \
testing, including but not limited to: the 50-g glucose challenge test \
(GCT) / 1-hour screen, the 100-g 3-hour oral glucose tolerance test \
(OGTT), the 75-g 2-hour OGTT, early screening before 24–28 weeks due \
to risk factors, fasting plasma glucose draws, being referred for \
testing, repeat testing, and experiences with specific components of \
multi-step tests. When possible, capture which specific test or test \
component the author is describing (e.g., "failed the 1-hour screen," \
"passed the 1-hour but failed the 3-hour," "fasting number was high \
but post-load numbers were fine," "failed at the 2-hour mark"). \
Different test components (fasting glucose vs. post-load values at 1, \
2, or 3 hours) may carry different clinical significance, so preserve \
this detail when the author provides it. SCREENING applies only to the \
testing/diagnostic process itself, before or at the point of diagnosis. \
If the author is already diagnosed and describes ongoing glucose \
management (including referencing which test values led to a \
treatment change), that is ANTEPARTUM_MONITORING — do not use \
SCREENING just because a test result is mentioned.

  - DIAGNOSIS: Experiences tied to the specific EVENT of receiving, \
processing, or reacting to a GDM diagnosis, including but not limited \
to: emotional responses at the moment of diagnosis (shock, guilt, \
self-blame, anxiety, grief), provider communication about the \
diagnosis, questioning the test or diagnostic criteria, perceived \
severity or minimization of the diagnosis, confusion about what the \
diagnosis means, and the distinction between different patterns of \
abnormal results (e.g., isolated fasting hyperglycemia vs. post-load \
spikes vs. multiple abnormal values). Only use DIAGNOSIS when the post \
is describing the diagnosis event itself or the author's reaction to \
it. General reflections on living with/managing GDM that don't center \
on the diagnosis moment belong in ANTEPARTUM_MONITORING instead (e.g., \
ongoing guilt about diet choices, without reference to being told the \
diagnosis, is ANTEPARTUM_MONITORING, not DIAGNOSIS).

  - ANTEPARTUM_MONITORING: Experiences related to GDM management during \
pregnancy after diagnosis, up until the point of labor/delivery, \
including but not limited to: self-monitoring of blood glucose (SMBG) \
— frequency, burden, pain, logging — dietary management and \
restrictions, medication use (insulin, metformin) and attitudes toward \
medication, exercise recommendations, additional appointments or scans \
(growth ultrasounds, non-stress tests), interactions with care \
providers (endocrinologists, maternal-fetal medicine, diabetes \
educators, dietitians), fetal monitoring concerns during pregnancy \
(macrosomia, growth trajectory), psychological impact of ongoing \
management (anxiety, depression, feeling overwhelmed, loss of autonomy \
or normal pregnancy experience), financial burden, and positive coping \
or adaptation. Do NOT include labor, delivery, induction, or birth \
outcomes here — those belong in BIRTH.

  - BIRTH: Experiences related to labor, delivery, and the immediate \
birth event in the context of GDM, including but not limited to: \
induction planning, pressure, timing, or recommendation due to GDM; \
planned or unplanned cesarean section; delivery mode preferences or \
anxieties connected to GDM; birth weight or macrosomia discussions at \
delivery; neonatal blood sugar monitoring or treatment for the baby \
after birth; and the author's emotional response to how the birth \
itself went in relation to their GDM. Only include birth-related \
experiences the author explicitly connects to their GDM.

  - POSTPARTUM: Experiences related to the period after delivery (once \
the birth event itself is over) in the context of GDM, including but \
not limited to: postpartum glucose screening (awareness, completion, \
barriers, provider follow-up), awareness or lack of awareness of \
long-term type 2 diabetes risk, relief at GDM resolution, discounting \
of long-term risk, lack of postpartum follow-up from providers, \
breastfeeding in the context of GDM, GDM recurrence concerns in \
subsequent pregnancies, and maintenance or abandonment of lifestyle \
changes after birth. Do NOT include the birth/delivery event itself — \
that belongs in BIRTH.

* Map colloquial language to normalized descriptions. Examples include \
but are not limited to:
  - "the sugary drink" / "glucola" / "that orange drink" → Glucose \
challenge test
  - "pricking my finger 4x a day" → Blood glucose self-monitoring \
frequency
  - "failed my 1-hour" / "failed the test" → Screening test failure \
(1-hour GCT)
  - "passed my 1-hour but failed my 3-hour" → Passed initial screen, \
failed diagnostic OGTT
  - "my fasting was the only one high" → Isolated fasting \
hyperglycemia on OGTT
  - "my numbers have been fine" / "my sugars are always normal" → \
Normal SMBG readings despite diagnosis
  - "diet controlled" / "controlling with diet" → Dietary management
  - "they want to induce me at 39 weeks" → Induction recommendation \
(domain: BIRTH)
  - "I have to see the endocrinologist every 2 weeks now" → Increased \
appointment frequency
  - "I feel like a failure" / "what did I do wrong" → Self-blame \
or guilt
  - "they never told me I needed to get tested after" → Lack of \
postpartum screening guidance
  - "my baby was measuring big" / "95th percentile" → Macrosomia \
concern during monitoring

* A single post may contain experiences in multiple domains. Extract \
each as a separate item.

* IMPORTANT: If an experience is clearly related to the author's GDM \
care but does not fit neatly into the examples listed above, extract \
it anyway. Assign it to the most appropriate domain and describe it \
accurately in the "experience" field. The goal is comprehensive \
capture — do not discard experiences simply because they are not \
listed in the examples.

* If the post contains no experiences related to the GDM diagnostic \
or monitoring pathway (e.g., it only discusses general pregnancy \
symptoms, recipes, or weight without connecting to GDM care), return \
an empty result.

* The "experience" and "original_text" fields must only reflect information \
explicitly stated in the post. Do not calculate, convert, or infer specific numbers \
(e.g., converting calendar dates to gestational weeks, or estimating a value \
the author didn't state). If a precise detail isn't in the text, omit it rather \
than approximate it \

* Do NOT extract:
  - General pregnancy complaints unrelated to GDM management
  - Side effects of medications unrelated to GDM treatment. However, \
DO extract side effects of medications that are standard GDM therapies \
(e.g., insulin, metformin) or experiences with GDM-related dietary \
changes, even if the author does not explicitly attribute them to GDM \
in the post — the connection can be inferred from context.
  - Birth outcomes or delivery planning that have no connection to GDM. \
However, DO extract any birth-related experience the author connects \
to GDM, whether positive, negative, or neutral — including but not \
limited to induction, planned or unplanned cesarean section, early \
delivery, birth weight discussions, neonatal blood sugar monitoring, \
and preferences or anxieties about delivery mode in the context of \
GDM. Assign these to the BIRTH domain, not ANTEPARTUM_MONITORING or \
POSTPARTUM.
  - Experiences with other conditions (pre-existing type 1 or type 2 \
diabetes, pre-eclampsia) unless discussed in the context of GDM care
  - Experiences from authors with pre-existing T1DM or T2DM — if the \
post indicates pre-existing diabetes rather than GDM, return an empty \
result even if the author also discusses pregnancy-related glucose \
management

Output a JSON object with two fields:

1. "experiences": a JSON array. Each element should have:
   - "experience": a short normalized description of the experience \
(e.g., "Nausea during glucose challenge test", "Isolated elevated \
fasting glucose on 3-hour OGTT", "Burden of checking blood glucose \
four times daily", "Shock at receiving GDM diagnosis")
   - "domain": one of "SCREENING", "DIAGNOSIS", "ANTEPARTUM_MONITORING", \
"BIRTH", "POSTPARTUM"
   - "original_text": the relevant excerpt from the post (keep brief, \
1–2 sentences max)


2. "no_pathway_experiences": true if the post contains no extractable \
GDM pathway experiences, false otherwise.

CONSISTENCY REQUIREMENT: "no_pathway_experiences" and "experiences" must \
never contradict each other. If "experiences" contains one or more items, \
"no_pathway_experiences" MUST be false. If "no_pathway_experiences" is \
true, "experiences" MUST be an empty array. Before returning your answer, \
check that these two fields agree.

If no relevant experiences are found, return:
{"experiences": [], "no_pathway_experiences": true}
"""

# JSON schema for structured output
GDM_EXPERIENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "experiences": {
            "type": "array",
            "description": "List of GDM pathway experiences extracted from the post.",
            "items": {
                "type": "object",
                "properties": {
                    "experience": {
                        "type": "string",
                        "description": "Short normalized description of the experience."
                    },
                    "domain": {
                        "type": "string",
                        "description": "Pathway domain. One of: SCREENING, DIAGNOSIS, ANTEPARTUM_MONITORING, BIRTH, POSTPARTUM."
                    },
                    "original_text": {
                        "type": "string",
                        "description": "Relevant excerpt from the post (1-2 sentences max)."
                    },
                },
                "required": ["experience", "domain", "original_text"],
                "additionalProperties": False
            }
        },
        "no_pathway_experiences": {
            "type": "boolean",
            "description": "True if the post contains no extractable GDM pathway experiences, false otherwise."
        }
    },
    "required": ["experiences", "no_pathway_experiences"],
    "additionalProperties": False
}

# Stage 2 post-processing: one row per experience
def flatten_experiences(output_jsonl_paths) -> pd.DataFrame:
    """
    Flatten Stage 2 batch output JSONL file(s) into one row per experience.
 
    Columns:
        custom_id               "task-<id>", links back to the source post
        experience              normalized description of the experience
        domain                  SCREENING, DIAGNOSIS, ANTEPARTUM_MONITORING, BIRTH, POSTPARTUM
        original_text           excerpt from the post
        no_pathway_experiences  True  -> model found no pathway experiences
                                False -> row is an extracted experience
                                None  -> response could not be parsed
    """
    records, parse_error_ids = [], []
 
    for path in map(Path, output_jsonl_paths):
        with open(path) as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
 
                empty = {"experience": None, "domain": None, "original_text": None}
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    experiences = parsed.get("experiences", [])
                    if experiences:
                        for exp in experiences:
                            records.append({
                                "custom_id": custom_id,
                                "experience": exp.get("experience"),
                                "domain": exp.get("domain"),
                                "original_text": exp.get("original_text"),
                                "no_pathway_experiences": False,
                            })
                    else:
                        records.append({
                            "custom_id": custom_id, **empty,
                            "no_pathway_experiences": parsed.get("no_pathway_experiences", True),
                        })
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    parse_error_ids.append(custom_id)
                    records.append({
                        "custom_id": custom_id, **empty,
                        "no_pathway_experiences": None,
                    })
 
    if parse_error_ids:
        print(f"WARNING: {len(parse_error_ids)} response(s) failed to parse: {parse_error_ids}")
    else:
        print("No malformed responses detected.")
 
    return pd.DataFrame(records)