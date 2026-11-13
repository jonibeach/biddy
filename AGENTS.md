# AGENTS.md

## Context

This repository is our team's entry to the Intel Physical AI Challenge at Junction 2026, a weekend hackathon. The full challenge text is in [docs/challenge.md](docs/challenge.md). Read it when you need details this file leaves out.

## Goal

Build an autonomous inspection cell with a real robot arm. The arm picks each Duplo brick, shows it to an inspection camera, and drops it into a "good" or "defective" bowl. No human controls it during the run.

## Hardware at our station

- SO-101 follower arm with a wrist camera
- SO-101 leader arm, used to teleoperate the follower when we record demonstrations
- Top-view camera and inspection camera. At least one is a RealSense D415 depth camera.
- Duplo bricks, two sorting bowls, and markers. We mark some bricks as defective ourselves.
- Intel Core Ultra Series 3 (Panther Lake) laptop with 64 GB RAM, CPU, GPU and NPU

## Pipeline

1. Teleoperate the arm with the leader arm to record demonstrations.
2. Fine-tune a vision-language-action (VLA) policy on the demonstrations in Physical AI Studio, using cloud GPUs.
3. In parallel, train an Anomalib model that tells good bricks from defective ones at the inspection point.
4. Optimize both models with OpenVINO and run them on the laptop's CPU, GPU and NPU.
5. Combine them into one loop: pick, inspect, sort.

## Tools

- Physical AI Studio: record demonstrations and fine-tune VLA policies
- Anomalib: anomaly detection
- OpenVINO and OpenVINO Physical AI: on-device inference and quantization
- Intel Robotics AI Suite: Intel's open-source robotics stack

## What the judges look for

- How few demonstrations are enough
- How well the policy handles bricks in new positions
- How fast the full loop runs on the laptop
- Reliability of the whole cell

Ideas they named:

- Use the anomaly score to trigger a re-grasp
- Share cameras between the policy and inspection
- Quantize models with OpenVINO

Stretch goals once the basic loop works: faster cycles, messier brick piles, new defect types, or our own twist on the task.

