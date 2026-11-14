# Failure tests

Plan: test first. If it fails, record it and show we know why. If there is
time, add demos and test again.

## 1. Empty table

No brick. Should stay home.

Fix: check the gripper after grasping. If it closed all the way, there is
nothing in it.

## 2. Clutter

Two or more bricks touching or stacked. Should pick one.

Fix: record demos with cluttered bricks.

## 3. Wrong object

Not a Duplo (eraser, toy). Should say "wrong object", not good or defective.

Fix: a second Anomalib model trained on all bricks, good and defective. A high
score means it is not a brick.

## Results

| Test | Result before | Result after | Video |
|---|---|---|---|
| Empty table | | | |
| Clutter | | | |
| Wrong object | | | |
