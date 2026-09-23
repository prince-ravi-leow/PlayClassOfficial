# Postprocessing

After tracking, raw tracks contain ID switches, mask overlaps, and
low-confidence periods. `build_dataset` detects these automatically and writes
template JSON files for you to review and fill in before the final dataset
build.

## Workflow

**Run 1 — generate templates:**

```sh
pixi run build_dataset
```

Reads `tracking_outputs.parquet` from `--tracking-dir` (default:
`data/results/tracking/sam3_best`) and writes postprocessing JSONs to
`--postprocessing-dir` (default: `data/postprocessing`).

For each new tracking directory, this writes:

- `tracking_issues.json` — detected problems (read-only reference)
- `tracking_postprocessing.json` — editable template with one entry per detected
  issue
- `bird_info.json` — bird descriptions from registration Excel, for reference

If any `tracking_postprocessing.json` has unfilled fields (`null` values), the
script stops and tells you which entries need attention.

**Fill in the JSON files** (see entry types below), then:

**Run 2 — build the dataset:**

```sh
pixi run build_dataset
```

---

## Entry types in `tracking_postprocessing.json`

### `trim` — delete bad frames

Drops track rows in a frame range. Auto-generated for `overlap` and `low_score`
issues.

```json
{
  "type": "trim",
  "cause": "overlap",
  "from": 1420,
  "to": 1510,
  "id": 2,
  "_time": "00:56-01:00"
}
```

- **`from` / `to`**: frame range to drop (inclusive).
- **`id`** _(optional)_: if present, only that tracking ID is trimmed; omit to
  trim all IDs in the range.
- **`cause`**: informational (`"overlap"`, `"low_score"`, `"merged_object"` —
  any string).
- **`_time`**: human-readable timestamp hint, ignored by the script.

---

### `id_switch` — merge a split track

Renames all rows of ID `from` _before_ `frame` to ID `to`, stitching two halves
into one continuous track.

```json
{
  "type": "id_switch",
  "frame": 1875,
  "from": 1,
  "to": 3,
  "_time": "01:15"
}
```

- **`frame`**: the switch frame (rows **before** this frame are renamed).
- **`from`**: the old (pre-switch) tracking ID — **fill this in**.
- **`to`**: the new (post-switch) tracking ID — auto-filled from the detected
  transition.

---

### `id_match` — assign bird identity

Maps a tracker-assigned ID to the protocol bird ID (from registration Excel).

```json
{
  "type": "id_match",
  "protocol_id": 1,
  "description": "Light plumage, blue paint",
  "tracking_id": 3,
  "frame": 500
}
```

- **`protocol_id`**: the bird's ID in the registration protocol.
- **`tracking_id`**: the post-merge tracking ID — **fill this in**.
- **`frame`**: a reference frame for verification (optional but recommended).

---

## Common edits

| Situation                                             | Action                                             |
| ----------------------------------------------------- | -------------------------------------------------- |
| Auto-selected trim ID is wrong                        | Change `"id"` to the correct tracking ID           |
| Trim range is too wide / narrow                       | Adjust `"from"` and `"to"`                         |
| No actual problem (false positive)                    | Delete the entry                                   |
| Two tracks need merging but no id_switch was detected | Add an `id_switch` entry manually                  |
| Extra trim needed (e.g. merged object)                | Add a `trim` entry with `"cause": "merged_object"` |
