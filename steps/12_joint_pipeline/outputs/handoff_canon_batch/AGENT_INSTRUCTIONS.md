# Task: canonicalize 173 Hebrew Knesset committee text segments

You are an AI coding/text agent. Do this task directly — no need to ask the user
clarifying questions, everything required is in this folder.

## Input
`segments_to_canonicalize.json` in this same folder — a JSON list of 173 objects,
each with keys: `seg_key`, `doc_id`, `seg_idx`, `date`, `raw` (the Hebrew text to
rewrite).

## What to do
For **every single entry** in the input file, rewrite the `raw` Hebrew text into a
unified, neutral, third-person register, following these exact instructions:

**System role:**
```
אתה עורך לשוני. אתה מקבל מקטע דיון מוועדת הכספים של הכנסת, ובו דוברים שונים
בסגנונות ואוצר מילים שונים. תפקידך לנסח מחדש את תוכן הדיון בלשון אחידה,
ניטרלית ועניינית - לשמר את כל המידע (נושאים, עמדות, נתונים, החלטות) אך
להסיר סגנון אישי, רטוריקה, ומאפייני דובר. אתה לא מסכם ולא מקצר - אתה מנסח מחדש.
```

**Per-segment instructions:**
```
נסח מחדש את מקטע הדיון הבא בלשון אחידה וניטרלית.

עקרונות:
- שמר את כל התוכן העובדתי: מה נדון, אילו עמדות הוצגו, נתונים מספריים, החלטות והצבעות.
- אחֵד את הרגיסטר: אותו אוצר מילים ענייני לכל הדוברים, ללא סלנג, רטוריקה, ברכות או ציטוט סגנוני.
- אל תסכם ואל תקצר באופן אגרסיבי - שמור על אורך דומה לתוכן המהותי של המקטע.
- כתוב כטקסט רציף בגוף שלישי ("הוצגה עמדה ש...", "סוכם כי...", "התקיימה הצבעה ש...").
- אל תוסיף פרשנות או מידע שאינו במקטע.
```

## Output format
For each input entry, produce one JSON object with exactly these keys:
```json
{
  "seg_key": "<copy from input>",
  "date": "<copy from input>",
  "subject": "<short 3-6 word Hebrew topic title>",
  "canonical": "<the neutral rewritten Hebrew text>",
  "decision": "<one of: אושר / נדחה / ללא הצבעה / נדחה להמשך>",
  "amounts": ["<any monetary amounts or request numbers mentioned, as strings>"],
  "source": "<name of whoever/whatever produced this — e.g. your model name or teammate's name>"
}
```
Example of a finished entry:
```json
{
  "seg_key": "25_ptv_1355658__8",
  "date": "2022-12-13",
  "subject": "העברות תקציביות: אנרגיה, בטיחות בדרכים, מטרו",
  "canonical": "הוצגה בקשת העברה מס' 177 של משרד האנרגיה, לתקצוב 559 מיליון שקלים בהרשאה להתחייב...",
  "decision": "אושר",
  "amounts": ["559 מיליון ש\"ח", "330 אלף ש\"ח"],
  "source": "claude"
}
```

## Practical tips (learned the hard way)
- **Batch it.** Don't try to process all 173 in one giant call — split into small
  chunks (e.g. 4-8 segments at a time) and process sequentially or in a few
  parallel batches. Large single batches cause JSON-escaping errors and wasted
  retries.
- **Hebrew punctuation in JSON:** use proper gershayim (״) and geresh (׳) for
  Hebrew abbreviations (e.g. מע״מ, בג״ץ) instead of straight double/single quotes —
  straight quotes inside a JSON string will break parsing.
- **Validate as you go:** after writing each batch's output, parse it back with a
  JSON loader to confirm it's valid before moving on.
- Some segments may end mid-sentence in the source data (that's how the raw
  transcripts were cut) — that's expected, not an error; canonicalize what's there.
- If a segment has no formal vote recorded, use `decision: "ללא הצבעה"`.

## Final output
Merge **all 173** finished entries into a single JSON list and save it as:
`output_canonical.json` in this same folder.

Do not stop until all 173 entries are present in that single output file.
