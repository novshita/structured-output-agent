# Sample dataset

**These 24 postings are synthetic.** They were written to resemble real postings and to include specific hard cases on purpose. All companies, people, emails and URLs are fictional (URLs use the reserved `.example` domain). Treat the eval numbers as a stress test of the agent, not as a measure of accuracy on real-world traffic.

`labels/` holds hand-written ground truth for `01`–`12`. The eval scores `title`, `company`, `work_mode` and `salary` against these labels. `13`–`24` have no labels. They are there to exercise validity and retries, and several have no single correct answer.

## Labelling conventions

These match the rules in the extractor's system prompt:

- `title` is copied as written, without the company name or location. Titles are compared case- and punctuation-insensitively.
- `company` is the hiring company, not a recruiting agency.
- Salary amounts are integers in the stated currency (`120k` → `120000`, `18 LPA` → `1800000`).
- A single salary figure: "up to X" sets only `max`, "from X" sets only `min`, a plain "X" sets both.
- If several salary ranges are listed, the first one is used.
- Anything the posting doesn't state is `null` or `"unspecified"`.

## What each posting tests

| # | Case | Trap |
|---|---|---|
| 01 | Clean, well-structured posting | Control case |
| 02 | One-line social post | Almost no information; most fields must be `unspecified` or `null` |
| 03 | English title, German body | "65.000 €" uses a period as the thousands separator; "2 Tage Homeoffice" means hybrid |
| 04 | Three cities, two salary bands | Multiple locations; first-listed salary (GBP) |
| 05 | Internship, all-caps title | No salary; onsite inferred from "on site in Bergen every day" |
| 06 | Emoji-heavy contract role | Hourly salary; "1–2 years" → junior |
| 07 | Indian posting | "18-24 LPA" (lakhs per annum) must become 1,800,000–2,400,000 INR |
| 08 | Broken lines, all caps, emoji | URL without a scheme; noisy formatting |
| 09 | Raw HTML with cookie banner and footer | Boilerplate and HTML tags around the content |
| 10 | Polish/English mix | Monthly salary in "zł" (PLN); "stacjonarna" means onsite |
| 11 | Forwarded recruiter email | The recruiting agency is not the company |
| 12 | Part-time, "up to $28/hour" | Only `max` is set |
| 13 | Very long posting with benefits and EEO boilerplate | Salary buried near the end; "NY or Remote" work mode is ambiguous |
| 14 | Salary written as "$120,000 - $95,000" | Swapped range triggers the `cross_field` check |
| 15 | Confidential search | No company name; tempts the model to invent one |
| 16 | Fully Spanish | CLP amounts with a period as the thousands separator; monthly |
| 17 | Lowercase tweet | "loop" as the company; "140-170k" with no currency |
| 18 | Two roles in one posting | Which role to extract? |
| 19 | Japanese/English mix | "800万円" = 8,000,000 JPY |
| 20 | Key-value government listing | Salary with decimals ("$72,418.00") |
| 21 | Remote-first with required travel | Work mode is ambiguous; salary "competitive" (none) |
| 22 | Unpaid volunteer role | Employment type doesn't fit the schema's options |
| 23 | Job-board page with duplicated text | "Promoted", "Easy Apply" and "People also viewed" noise |
| 24 | Day-rate contract (£550/day) | "day" isn't an allowed `salary.period`, which tempts a `schema_enum` error |
