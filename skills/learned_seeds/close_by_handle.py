def run(robot, part="handle"):
    """Close a hinged or sliding part by its handle: reach it where it is now, grasp it, move it back to closed, let go and back away."""
    movable = [found for found in robot.find(part) if found["movable_part"]]
    if not movable:
        raise RobotError("no movable " + part + " in view")
    handle = movable[0]
    robot.move_to(handle["id"], "pregrasp")
    robot.open_hand(0.08)
    robot.move_to(handle["id"], "grasp")
    robot.grasp(handle["id"])
    robot.move_part(handle["id"], handle["moves"]["from"], handle["moves"]["unit"])
    robot.release(handle["id"])
    robot.move_to(handle["id"], "retract")
    return {"part": handle["id"], "moved_to": handle["moves"]["from"], "unit": handle["moves"]["unit"]}
