# RAMMP generalized ADL autonomy

Task-level autonomy code for one Kinova Gen3, with a working Python executor and simulation backends. The current build runs six reusable skills against synthetic ADL scene state and separately tests Gen3 joint dynamics in MuJoCo. It has no physical robot command transport and does not yet combine cuRobo, live perception and articulated-object physics into an ADL simulator.

Start with the [runtime runbook](docs/runtime.md), [project prompt](docs/project-prompt.md) and [design index](docs/design/00-README.md). The [design audit](docs/audit-2026-09-08.md) records the earlier foundation work.

The project is now also installed at `/home/abra/RAMMP-generalized` on the Jetson. See [Jetson setup and verification](docs/jetson.md) for remote editing, Linux commands and the completed Jetson checks. The Windows copy remains available; the two folders are not automatically synchronized.

OpenAI GPT-6 Astra (`gpt-6-astra`) proposes task DAGs through the Responses API. Deterministic validation, world-state commits, execution and rolling motion checks stay local. The [fixed catalog](skills/adl_skill_library.yaml) owns skill contracts; [explicit handlers](rammp_adl/handlers.py) implement `observe`, `move_to_pose`, `set_gripper`, `grasp`, `release` and `follow_constraint` for the fixture backend. Catalog hardware statuses remain planned; fixture capability registration cannot enable a robot.

Install and run from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[simulation,reasoning,cameras]"
.\.venv\Scripts\python.exe -m rammp_adl doctor
.\.venv\Scripts\python.exe -m rammp_adl simulate --scenario cabinet
.\.venv\Scripts\python.exe -m rammp_adl physics --rolling
.\.venv\Scripts\python.exe tools/check_design.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The runbook includes fault injection, trajectory traces, camera previews, Astra setup and ROS/container commands. The dependency-free design checker validates artifacts; the installed runtime tests exercise actual Python implementation. Synthetic task success, simulated joint tracking and robot validation are separate claims.

The Windows environment captured a local 640×480 RGB frame from camera index 0; Windows lists a Brio 105 camera. No RealSense device was enumerated, and access to the intended D405/Orbbec depth streams and calibration remains unresolved. ROS compilation, the Docker recipe and live Astra access have not been verified here. An Astra API key was unavailable. Camera previews are saved locally and are not uploaded to Astra by the runtime.

Deployment target: ROS 2 Humble on a provisionally assumed Jetson AGX Orin 64 GB, MAXN. Table mount now; wheelchair mount later. D405 wrist camera and temporary Orbbec scene camera remain the intended roles. GPU/cuRobo integration, driver streaming support, calibration and measured safety profiles are outstanding deployment work.

[AGENTS.md](AGENTS.md) gives coding agents the short implementation contract.
