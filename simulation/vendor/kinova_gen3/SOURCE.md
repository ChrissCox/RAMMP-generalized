# Vendored model provenance

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/kinova_gen3

Copied from the existing local checkout at that revision on 2026-09-08. XML and
meshes are unchanged; the upstream BSD-3-Clause license is included. The model is
the simplified seven-joint Gen3 with position actuators and the original vision
bracelet. It has no gripper. Its camera is not a calibrated RealSense D405.

Use this model for joint-trajectory replay/physics checks only. It does not
validate the installed tool, contact sensing, grasps, feeding, safety limits or
hardware. Joint fixtures are test inputs, never substitute cuRobo plans.
