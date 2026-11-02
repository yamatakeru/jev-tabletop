# Held-out evaluation (pre-registered in scripts/heldout.py)

| condition | expected | new: in tray | new | old: in tray | old |
|---|---|---|---|---|---|
| layout_seed7 | blue, green, red | blue, green, red | OK | red | FAIL |
| layout_seed8 | blue, green, red | blue, green, red | OK | blue, green, red | OK |
| layout_seed9 | blue, green, red | blue, green, red | OK | blue, green, red | OK |
| layout_seed10 | blue, green, red | blue, green, red | OK | blue, green, red | OK |
| layout_seed11 | blue, green, red | blue, green, red | OK | blue, green, red | OK |
| para_clear_floor | blue, green, red | blue, green, red | OK | blue, green, red | OK |
| para_ja_only_blue | blue | blue | OK | blue | OK |
| para_warm_colours | red, yellow | red | FAIL | red, yellow | OK |
| obj_ja_keep_purple | blue, green, orange_cylinder, red, yellow | blue, green, orange_cylinder, red, yellow | OK | blue, green, red, yellow | FAIL |
| obj_purple_and_orange | orange_cylinder, purple | orange_cylinder, purple | OK | green | FAIL |
| obj_round_one | orange_cylinder |  | FAIL |  | FAIL |

| metric | new | old |
|---|---|---|
| success | 9/11 | 7/11 |
| success (layout / phrasing / new object) | 5/5 / 2/3 / 2/3 | 4/5 / 3/3 / 0/3 |
| first-try grasp success | 27/27 | 23/25 |
| held check agrees with truth | 27/27 | 32/32 |
| held p when holding (min / max) | 0.64 / 0.90 (n=27) | 0.65 / 0.92 (n=24) |
| held p when not holding (min / max) | - (n=0) | 0.03 / 0.34 (n=8) |
| held misjudged | 0 | 0 |
| grasp error after zoom, median / max (mm) | 6.3 / 24.4 | 7.8 / 283.1 |
| zoom queries per pick, mean | 1.00 | 2.94 |
| Jev calls per pick | 6.1 | 5.2 |
| Jev latency per pick (s, sum of calls) | 2.22 | 1.65 |
| wall time per episode, median (s) | 46 | 45 |
