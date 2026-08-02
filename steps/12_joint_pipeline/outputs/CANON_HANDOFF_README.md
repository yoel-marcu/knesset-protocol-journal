# Canonicalization handoff — remaining segments

## Status
- Total gold segments needing canonicalization: 523
- Already done (Gemma, Colab, salvaged): 112 → `canonical_gold_salvaged.json`
- Already done (Claude, this session): 238 → `canonical_gold_claude_partial.json`
- **Remaining: 173 segments (255,803 words)** → `canon_remaining_for_team.json`

## What's needed
For each object in `canon_remaining_for_team.json` (keys: `seg_key`, `doc_id`, `seg_idx`, `date`, `raw`),
produce a canonicalized entry with this **exact schema**:

```json
{"seg_key": "...", "date": "...", "subject": "<3-6 word Hebrew title>",
 "canonical": "<neutral rewritten Hebrew text>",
 "decision": "אושר | נדחה | ללא הצבעה | נדחה להמשך",
 "amounts": ["<monetary amounts / request numbers as strings>"],
 "source": "<your name or model, e.g. \"tomer\" or \"gemma\">"}
```

## The exact prompt (Tomer's canonicalization prompt — keep verbatim for consistency)

**System:**
```
אתה עורך לשוני. אתה מקבל מקטע דיון מוועדת הכספים של הכנסת, ובו דוברים שונים
בסגנונות ואוצר מילים שונים. תפקידך לנסח מחדש את תוכן הדיון בלשון אחידה,
ניטרלית ועניינית - לשמר את כל המידע (נושאים, עמדות, נתונים, החלטות) אך
להסיר סגנון אישי, רטוריקה, ומאפייני דובר. אתה לא מסכם ולא מקצר - אתה מנסח מחדש.
```

**Instructions (per segment):**
```
נסח מחדש את מקטע הדיון הבא בלשון אחידה וניטרלית.

עקרונות:
- שמר את כל התוכן העובדתי: מה נדון, אילו עמדות הוצגו, נתונים מספריים, החלטות והצבעות.
- אחֵד את הרגיסטר: אותו אוצר מילים ענייני לכל הדוברים, ללא סלנג, רטוריקה, ברכות או ציטוט סגנוני.
- אל תסכם ואל תקצר באופן אגרסיבי - שמור על אורך דומה לתוכן המהותי של המקטע.
- כתוב כטקסט רציף בגוף שלישי ("הוצגה עמדה ש...", "סוכם כי...", "התקיימה הצבעה ש...").
- אל תוסיף פרשנות או מידע שאינו במקטע.

החזר JSON יחיד בלבד:
{"subject": "<כותרת נושא קצרה 3-6 מילים>", "canonical": "<הניסוח המחדש הניטרלי>",
 "decision": "<אושר/נדחה/ללא הצבעה/נדחה להמשך>", "amounts": ["<סכומים או מספרי פניות>"]}
```

## How to run it
Two options, whichever's easiest:
1. **Colab notebook** — reuse `steps/12_joint_pipeline/notebooks/canonicalize_gold.ipynb`
   (already built, resumable, Gemma-4-31B via Drive). Just point `IN_PATH` at
   `canon_remaining_for_team.json` instead of the original `canon_input.json`.
2. **Claude/any LLM directly** — batch the 177 segments (word-count-balance into
   several chunks so no single call is huge) and apply the prompt above.

## Merging back
Once you have output(s), merge them with the two "done" files
(`canonical_gold_salvaged.json` + `canonical_gold_claude_partial.json`) into one
523-entry `canonical_gold.json` (dedupe by `seg_key`), then run:
```
python code/build_segments.py
python code/run_journal.py --segments outputs/segments_canonical.json --tag canonical
```
