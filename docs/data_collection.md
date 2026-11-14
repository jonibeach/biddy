# Data collection

One Studio dataset, `pick-inspect-sort` (project `biddy`, environment
`inspection-cell`, cameras top + inspection + wrist). Every episode is one
brick: grasp, inspect, regrasp, inspect, drop. The same videos feed the VLA
and Anomalib.

Studio keeps a recording session's episodes in its cache. They only reach the
dataset after **Finish recording**, so finish each session before training or
frame extraction.

## Station layout

The AprilTags are in their final positions (sheet and ID mapping:
[calibration/make_tags.py](../calibration/make_tags.py)). Their reference
pixel positions are in `calibration/reference_top.json` and
`reference_inspection.json`, taken from Studio's 640x480 stream.

| Camera | Tags seen | Object |
|---|---|---|
| top | 0, 1, 2, 3 | table corners |
| top | 4, 5, 7 | arm base |
| top | 8, 9 | inspection camera stand |
| top | 10 / 11 | good bowl / defective bowl |
| inspection | 12 to 15 | inspection background |

Tag 6 (arm base) is not detected in the top view. Clear it and save the
reference again if you want it tracked.

Before each recording session, check nothing moved. This reads Studio's
camera feed and works while Studio is recording:

    cd calibration
    ~/physical-ai-studio/application/backend/.venv/bin/python tag_check.py --studio --name top
    ~/physical-ai-studio/application/backend/.venv/bin/python tag_check.py --studio --name inspection

A tag that moved means the camera or that object was bumped. Put it back
before recording: the policy learns positions from the camera images. For the
same reason, never move or remove tags once recording has started; they are
part of every image the policy trains on.

The running cell makes the same check by itself before every brick
(`./biddy run`, [README](../README.md#apriltag-check)), so this manual check
is for recording sessions.

## Who sees which face

The wrist camera looks along the jaws, so it never sees the faces the jaws
close on. Faces are named by the first grasp:

| Face | Seen by | When |
|---|---|---|
| `top` (studs) | wrist | before the grasp and while carrying |
| `front` (toward the wrist camera) | wrist | before the grasp and while carrying |
| `back` | inspection | first grasp |
| `bottom` (tubes) | inspection | first grasp, brick tilted up |
| `side` (the two faces gripped first) | inspection | after the 90° regrasp |

In the wrist image the brick always sits in the same spot between the jaw
tips, so a fixed crop is enough for Anomalib.

## Bricks

- Good bricks: no marks at all. Keep them clean all weekend.
- Defective bricks: marker on exactly one face. Every face is inspected, so
  don't write IDs on the bricks. Keep defective bricks in their own cup and
  log each one in [brick_log.csv](brick_log.csv) (colour, size, defect).
- Defect size goes in rounds, easy to hard:
  1. `full`: the whole face coloured in
  2. `half`: half the face
  3. `stripe`: one line or scribble
  4. `dot`: a single dot
  Start a round only once the previous one is recorded.
- Spread defects over all five face types in the table above. Place each
  defective brick so its marked face ends up where the log says.
- Mix good and defective bricks within a session, about half each.

## One episode

Type the task in the Task box before pressing start, exactly one of:
- `put the brick in the good bowl`
- `put the brick in the defective bowl`

Never name the colour or the brick type in the task. The cell tells long
bricks from standard ones by shape at the side spot (`shape` column of
`./biddy check`), and a third phrase would be a task the cell never asks for.

Moves:

1. Start from the home pose, gripper open.
2. Move over the brick and hold about 0.5 s with the brick between the open
   jaws (wrist sees `top` and `front`).
3. Grasp and lift.
4. Bring the brick to the inspection spot: one fixed place, near the centre
   of the inspection image and close to the camera. Hold about 1 s showing
   `back`, then tilt up and hold about 1 s showing `bottom`.
5. Regrasp: put the brick down on the regrasp spot, turn the gripper 90°,
   grasp again.
6. Back to the same inspection spot, hold about 1 s per `side`.
7. Drop in the bowl named by the task, return home.

Keep every hold still and in the same place each episode: those frames become
the Anomalib images. Aim for 30 to 40 s per episode; move quickly between
holds rather than shortening them.

## Moving the leader arm

Set up once:
1. Tape the regrasp spot on the table. Pick a home pose for the leader arm
   (resting, gripper open) so every episode starts and ends the same.
2. Find the inspection pose: watch the inspection camera in Studio and move
   the brick until it sits near the centre and fills about a quarter of the
   image. Come back to that same pose every time.
3. Hold the leader by its wrist and handle, not the upper links. It moves
   more smoothly and stops dead for the holds.

Each episode:
1. Leader at home, gripper open. Start.
2. Swing the base toward the brick, then lower shoulder and elbow together in
   one smooth reach, stopping about 5 cm above the brick.
3. Roll the wrist so the jaws are square to the brick's sides, while still
   hovering. Hold 0.5 s.
4. Go straight down and close fully. A half grip slips in the regrasp.
5. Lift straight up about 5 cm before moving sideways.
6. Move in one path to the inspection pose. Hold 1 s (`back`). Tilt the wrist
   up, hold 1 s (`bottom`).
7. Lower the brick onto the regrasp spot until it touches, open, rise about
   3 cm, roll the wrist 90°, go down, close.
8. Back to the inspection pose. Hold 1 s, roll the wrist 180°, hold 1 s
   (both `side` faces).
9. Over the task's bowl, open, go home. Stop when at home and still.

What makes a demo good:
- Only the brick's start position changes between episodes. Route,
  inspection pose, regrasp spot and bowls stay the same.
- Steady speed with no pauses except the holds. The policy copies hesitation.
- No corrections on camera. Missed a grasp: discard and redo.
- Holds fully still, or the inspection frames blur.
- One or two joints at a time where possible: base, then reach, then wrist.
- Keep hands and the leader arm out of the follower's cameras.
- Do 2 or 3 practice runs before recording.

## Long bricks

A long brick can't be turned 90° in the jaws, so its route differs only in
the regrasp. Faces are named by the first grasp, as for standard bricks:

1. Hover, then grasp **one end** (jaws on the two long sides, close to the end).
2. Inspection spot: hold 1 s (`back`), tilt up, hold 1 s (`bottom`).
3. Regrasp: put it down on the regrasp spot, open, move along the brick to
   **the other end** (no wrist roll), close.
4. Inspection spot: hold 1 s, roll 180°, hold 1 s (the end that was in the
   jaws is now in view).
5. Drop in the task's bowl, home.

Same four holds and the same spots as a standard brick: if a long brick
sticks out of the inspection image, move the hold pose for **both** types
rather than giving long bricks their own pose. Record both long variants,
good and defective, about 10 to 15 full-route episodes in all.

## A perfect episode

Record only perfect episodes into `pick-inspect-sort`; extra still clips for
Anomalib go to `anomaly-extra`. Perfect means:

- starts and ends at the home pose, still
- one clean grasp and one clean regrasp, no retries
- all four holds at the usual spots, each still for at least 1 s
- the full route even for a defective brick (no stopping after the marked face)
- the bowl matches the task

After each session, **Finish recording**, then check only the new episodes:

    ./biddy check pick-inspect-sort --from <first new episode>

Delete in Studio every new episode with a flag after its task text, and
check that the `shape` column says `long` for long bricks once
`inspection.brick_types.long.min_elongation` is set in station.yaml.

## Positions

Vary where the brick starts and its rotation across the whole pick area.
Note the rough position in the brick log so we can test on positions the
policy never saw.

## Discard an episode if

- the grasp or regrasp slipped and needed another try
- a hold was skipped or the brick was outside the inspection image
- the wrong bowl was used for the task
