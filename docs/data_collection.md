# Data collection

One Studio dataset, `pick-inspect-sort` (project `biddy`, environment
`inspection-cell`, cameras top + inspection + wrist). Every episode is one
brick: grasp, inspect, regrasp, inspect, drop. The same videos feed the VLA
and Anomalib.

Studio keeps a recording session's episodes in its cache. They only reach the
dataset after **Finish recording**, so finish each session before training or
frame extraction.

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

Type the task in the Task box before pressing start:
- `put the brick in the good bowl`
- `put the brick in the defective bowl`

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

## Positions

Vary where the brick starts and its rotation across the whole pick area.
Note the rough position in the brick log so we can test on positions the
policy never saw.

## Discard an episode if

- the grasp or regrasp slipped and needed another try
- a hold was skipped or the brick was outside the inspection image
- the wrong bowl was used for the task
