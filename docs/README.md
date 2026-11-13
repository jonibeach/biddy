# Docs

Markdown copied from each project's repo, so it can be read offline. Images are not included.

| Folder | Source | Commit |
|---|---|---|
| `physical-ai-studio/` | [open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio) | `8f13241` |
| `physicalai-runtime/` | [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai) | `0ec06e9` |
| `robotics-ai-suite/` | [open-edge-platform/robotics-ai-suite](https://github.com/open-edge-platform/robotics-ai-suite) | `4544ddb` |
| `anomalib/` | [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib) | `437669f` |
| `lerobot/` | [huggingface/lerobot](https://github.com/huggingface/lerobot), SO-101, cameras and policy pages only | `fe1c545` |

## Start here

1. [Intel hackathon guide](robotics-ai-suite/docs/user-guide/resources/hackathon_resources.md): what is on the station laptop and how to launch Studio.
2. [Studio: getting started](physical-ai-studio/application/docs/03-getting-started.md), then [environment setup](physical-ai-studio/application/docs/04-environment-setup.md), [recording](physical-ai-studio/application/docs/05-recording-datasets.md), [training](physical-ai-studio/application/docs/06-training-policies.md), [remote training](physical-ai-studio/application/docs/07-remote-training.md) and [deploying](physical-ai-studio/application/docs/08-deploying-model-policies.md).
3. [Export a policy to OpenVINO](physical-ai-studio/library/docs/how-to/export/export_inference.md) and [run it](physicalai-runtime/docs/how-to/runtime/run-policy-on-robot.md) with the [SO-101 plugin](physicalai-runtime/packages/physicalai-so101-plugin/README.md) and [SO-101 example](physicalai-runtime/examples/so101/README.md).
4. Anomalib: [custom data](anomalib/docs/source/markdown/guides/how_to/data/custom_data.md), [folder datamodule](anomalib/docs/source/markdown/guides/reference/data/datamodules/image/folder.md), [Patchcore](anomalib/docs/source/markdown/guides/reference/models/image/patchcore.md), [deploy with OpenVINO](anomalib/docs/source/markdown/guides/reference/deploy/index.md), [training on Intel GPUs](anomalib/docs/source/markdown/guides/how_to/training_on_intel_gpus/index.md).
5. Robot side: [LeRobot SO-101](lerobot/docs/source/so101.mdx), [cameras](lerobot/docs/source/cameras.mdx), [RealSense on Intel](robotics-ai-suite/docs/user-guide/components/sensors/cameras/usb/realsense.md).
6. Faster VLA on Intel: [Pi0.5 optimization](robotics-ai-suite/docs/user-guide/ai_resources/openvino/pi05-optimization.md), [Pi0.5 with real-time chunking](robotics-ai-suite/docs/user-guide/software_references/humanoid/sample_pipelines/pi05_with_rtc.md), [ACT with OpenVINO](robotics-ai-suite/docs/user-guide/ai_resources/openvino/models/model_act.md).
