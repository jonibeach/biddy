# Intel Physical AI Challenge: kickoff slides

Text of [slides.pdf](slides.pdf), the deck Intel shared in Discord on the first evening. Slides 4 to 7 and 9 are images in the PDF, so their text was copied out by hand.

The judging criteria on slide 7 differ from the rubric in [JUNCTION2026.md](https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md#judging-rubric). The organizers said in Discord that GitHub has the truth. Both are listed at the end of this file.

## 1. Title

Stop Simulating, Start Moving: See It. Detect It. Act on It. The Intel Physical AI Challenge.

## 2. Support team

Intel x Ignite Next Hackathon Support Team: Albert van Houten, Daniil Lyakhov, Jayabalaji Sathiyamoorthi, Olinca Beltran, Rajesh Gangireddy, Ada Betancourt, Marvin Mueller, Ricky Watts.

## 3. Challenge overview

Build an end-to-end Physical AI solution for robotic defect detection and intelligent manipulation using the Intel Physical AI Studio, Anomalib, and OpenVINO software stack.

Participants will design a robotics workflow that combines Vision-Language-Action (VLA) capabilities through Intel Physical AI Studio with Anomalib-based visual anomaly and defect detection.

The solution should enable the robot to perceive objects, identify defects, reason about the required action, and execute the appropriate manipulation task.

The completed solution will then be optimized and deployed on Intel Core Ultra Series 3 using OpenVINO, leveraging available CPU, GPU, and NPU acceleration to improve inference performance, latency, and deployment efficiency.

**Primary task:** distinguish acceptable ("perfect") from anomalous ("defective") objects, then physically sort or otherwise respond according to the rules revealed at kickoff.

**Closed-loop sequence:** PERCEIVE → DETECT → REASON → ACT → OPTIMIZE, and demonstrate the entire pipeline.

## 4. Required technology stack

1. **SO-101 Robotic Arm.** Physical manipulation platform used to execute the challenge task. The arm closes the loop from AI perception and reasoning to real-world action.
2. **Intel Physical AI Studio** (VLA workflow). Design and integrate the VLA-based robotics workflow, connecting perception, task reasoning, robot control, and application orchestration.
3. **Anomalib** (defect detection). Provide visual anomaly and defect detection, producing defect decisions and localization signals that can be consumed by the VLA / robot workflow.
4. **OpenVINO + Intel Core Ultra Series 3** (optimize + deploy). Optimize and deploy supported AI workloads with OpenVINO. Map inference across the integrated NPU, GPU, and CPU to improve latency and on-device efficiency.

## 5. End-to-end technical workflow

The solution is a closed-loop robotics pipeline: perception identifies defects, VLA reasoning determines the required response, the robot executes the action, and OpenVINO optimizes deployment for Intel on-device compute.

| Step | Stage | What happens |
|---|---|---|
| 1 | Perceive | Camera input and object observations |
| 2 | Detect | Anomalib anomaly score / localization |
| 3 | Reason | VLA task reasoning in Physical AI Studio |
| 4 | Act | SO-101 executes the manipulation response |
| 5 | Optimize | OpenVINO deployment on Core Ultra 3 |

## 6. Technical implementation expectations

1. **Perception & Defect Detection.** Capture object imagery from the robot workspace and apply Anomalib to identify anomalous or defective objects. The defect output should be available to the downstream robotics policy as a structured signal, label, score, or localized region.
2. **VLA Reasoning & Task Planning.** Use Intel Physical AI Studio to construct the VLA-driven application flow. The VLA layer interprets the task instruction and perception results, determines the appropriate response, and produces the action intent required by the robot controller.
3. **SO-101 Robot Execution.** Convert the selected action into executable robot behavior on the SO-101 arm. The implementation should connect AI decisions to robot motion, grasping, sorting, or other challenge-specific manipulation actions.
4. **OpenVINO Optimization.** Optimize supported models and inference stages with OpenVINO. Teams should consider model conversion, precision, graph optimization, preprocessing / postprocessing efficiency, and device selection to reduce end-to-end latency.
5. **Intel Core Ultra Series 3 Deployment.** Deploy the optimized solution on Intel Core Ultra Series 3. Use the integrated NPU, GPU, and CPU according to workload characteristics and operator support, while maintaining reliable robot control and application responsiveness.
6. **Closed-Loop Demonstration.** Demonstrate the full sequence from image acquisition and defect detection through VLA reasoning and physical robot action, followed by a concise explanation of the architecture and optimization decisions.

## 7. Judging criteria (slides version)

Solutions will be evaluated on technical integration, robotics execution, and efficient Intel deployment.

| Criteria | Description | Points |
|---|---|---|
| End-to-End Physical AI Solution | Demonstrates a complete workflow from visual perception and defect detection through decision-making and robotic action using the SO-101 arm. The solution should operate as one integrated closed-loop system rather than as independent component demonstrations. | 25 |
| Defect Detection with Anomalib | Effective use of Anomalib for visual anomaly and defect detection. Evaluation considers detection quality, model integration, handling of normal versus defective objects, and how defect outputs are exposed to and consumed by the downstream robotics workflow. | 20 |
| VLA & Physical AI Studio Integration | Demonstrates effective use of Intel Physical AI Studio and VLA-based reasoning to translate perception results and task instructions into appropriate robot actions. Integration across perception, reasoning, planning, and manipulation should be clearly demonstrated. | 20 |
| OpenVINO & Intel Core Ultra Series 3 Optimization | Optimizes and deploys supported AI workloads with OpenVINO on Intel Core Ultra Series 3. Evaluation considers inference latency, throughput, deployment efficiency, and appropriate use of the integrated NPU, GPU, and CPU for the implemented workload. | 20 |
| Robotic Execution & Reliability | Quality and robustness of SO-101 execution, including object localization, manipulation accuracy, repeatability, stable task completion, and recovery from incorrect or uncertain defect detections during the Challenge Day demonstration. | 10 |
| Innovation & Technical Demonstration | Originality of the approach, effective use of Intel technologies, overall technical completeness, and clarity of the live demonstration, system architecture, and optimization decisions. | 5 |
| **Total** | Overall Challenge Score | **100** |

## 8. HW/SW at a glance

**Hardware**

- Compute, one of:
  - HP OmniBook Ultra (C2NM8AV), Intel Core Ultra X9 Series 3 (388H) AI PC
  - Dell XPS 16 (DA16260), Intel Core Ultra X9 Series 3 (388H) AI PC
- Arm: LeRobot SO-ARM101, assembled 1 leader + 1 follower + 1 wrist camera
- Cameras: one RealSense Depth Camera 415 and one Pro QHD 2K/60fps webcam
- Other: camera desk mounts, USB hub, Legos/bowls, printed robot safety guidelines

**Software**

- Install as part of the challenge: Physical AI Studio, Anomalib v2.7.0
- Pre-installed: Ubuntu 26.04, NPU driver, iGPU driver
- Training: GPU cloud instance through Modal

## 9. End goal

The slide shows four photos:

1. A blue Duplo brick on the table next to the arm, a pink bowl and a blue bowl.
2. Arrow labelled **ACT**: the arm has placed the brick under the inspection camera (a RealSense on a small tripod). Labelled **Anomalib**.
3. Two arrows labelled **SmolVLA** lead to the last two photos. In one, the arm drops the brick in the pink bowl ("Defective objects"). In the other, the brick lies in the blue bowl ("Perfect objects").

So the example design uses ACT to bring the brick to inspection and SmolVLA, given the Anomalib result, to sort it.

**What teams must deliver**

- Working Physical AI prototype: closed-loop SO-101 solution from perception to action.
- VLA + Anomalib integration: VLA robotics workflow with visual anomaly / defect detection.
- OpenVINO-optimized deployment: optimized inference pipeline on Intel Core Ultra Series 3.
- Live demonstration: execute the Challenge-Day task and explain the architecture and optimization choices.

## 10. Happy hacking

https://developer.robotics.intel.com

## 11. End-to-end Physical AI workflows

A universal runtime that connects models, backends, and robots to standardize and scale deployment.

- **Physical AI Studio**: fine-tune imitation learning.
  - Imitation learning: teach robots to perform tasks from human demonstrations.
  - State-of-the-art policies: run ACT, SmolVLA, GR00T, LeRobot and Pi0.5, with support for Hugging Face.
  - Low-code visual interface: build physical AI workflows in an intuitive 3D experience.
- **Robotics AI Suite**: standardize workflows.
  - Plug-and-play integration: calibrate data from diverse inputs with integrations across diverse hardware setups.
  - ROS 2 ready: optimized workflows, reference applications and demos, with SLAM and control libraries.
  - Hardware integration: no need to build foundational plumbing from scratch, with support for DevKits and inputs.
- **OpenVINO Physical AI Framework**: deploy VLA policies to production.
  - Inference to production in a few lines of Python.
  - Heterogeneous compute: optimized for Intel Core and Core Ultra processors with CPU, built-in GPU, and NPU, processing what each engine does best automatically.
  - Sustain deterministic control loops with asynchronous and real-time chunking even as model complexity scales.

## 12. Physical AI Studio

Open-source, end-to-end framework to simplify development of Intel-optimized Physical AI models. Leverages emerging techniques, such as imitation learning, and streamlines the entire VLA lifecycle: from robot calibration and data collection, through model fine-tuning, to optimized model export for inference on Intel.

- **Key features**
  - Data collection based on imitation learning
  - Simple, modular API and CLI for training, inference, and benchmarking VLA policies
  - Export models to OpenVINO, ONNX, or Torch formats for accelerated inference
  - Benchmark policies on standardized simulation environments like LIBERO and PushT
  - Unified inference API across all export backends
- **Supported policies:** ACT, SmolVLA, Pi0, Pi0.5, GR00T N1, and more
- **Supported robots:** Trossen WidowX AI, SO-ARM101, AlohaMini, LeKiwi, and more

Learn more: https://github.com/open-edge-platform/physical-ai-studio/

## 13. OpenVINO Physical AI Framework

Build once, deploy everywhere. Brought to Physical AI inference and deployment.

- Training side (Physical AI Studio, LeRobot, other workflows): Robot Calibration → Dataset Collection → VLA Training → Optimization → Export
- Runtime side (OpenVINO Physical AI Framework): Robot State → Camera → Observation → Preprocess → Infer → Postprocess → Robot Action

Reducing deployment brittleness across industry:

- Unifying cameras, robots, and safety in one runtime.
- Safety is not an afterthought: action clamping, velocity limits, emergency stops and similar are built into the framework.
- Model predictive control loop, diffusion-based action generation and multi-modal pipelines as modular plugins to the runtime.
- Enabling deployments to use hybrid and asynchronous compute components.

## 14. Robotics AI Suite

Adapt and scale physical AI on Intel for fast time to value.

- Accelerate development for all kinds of robotics: industrial, service, mobile and humanoid. Locomotion, perception, and imitation learning. Run vision AI, media analytics, and Gen AI workloads with real-time controls.
- Innovate and scale fast with an open ecosystem: validated systems, SDK, and vision AI use cases. Near zero-touch deployment and remote device management. Intel's long-standing robotics expertise.
- Reduce complexity, improve TCO: latest generation processors, integrated CPUs, GPUs and NPUs, real-time controls and AI/perception from one platform.
- SDK content: vision libraries, real-time control frameworks, AI inference engines, orchestration-ready modules, hardware-aware tuning, ROS 2 and open standards.

## 15. Intel Core Ultra processors

All-in-one compute platform for real-time control and AI workloads. Up to 180 TOPS.

| Engine | Strength | Details |
|---|---|---|
| CPU | Fast response | Ideal for low-latency AI workloads through OpenVINO. Intel Time Coordinated Computing: resource partitioning (cache/memory), deterministic power management, cross-silicon time sync (Intel TCC + TSN). |
| NPU | Low power | Ideal for sustained, complex AI workloads with good AI performance per watt. 50 TOPS. Fast deployment through OpenVINO. |
| GPU | High throughput | Up to 12 Xe3 cores. AI deployment through OpenVINO. Compute acceleration through oneAPI. |

## Rubric comparison

The GitHub rubric is the one that counts.

| GitHub JUNCTION2026.md | Points | What judges look at |
|---|---|---|
| Task Success Rate | 25 | Accuracy, cycle time, and robustness to shifted objects and clutter |
| Intel Tech Integration | 25 | Offloading to the NPU/iGPU, and deployment with OpenVINO and the Robotics AI Suite |
| Dataset & Pipeline Quality | 20 | Clean teleoperation data and effective fine-tuning in Physical AI Studio |
| System Efficiency | 20 | Latency per inference step, thermal headroom, and power consumption on Core Ultra |
| Creativity | 10 | Original ideas beyond the baseline |
| **Total** | **100** | |
