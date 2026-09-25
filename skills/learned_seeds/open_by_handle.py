def run(robot, part="handle", amount=None):
    """Open a hinged or sliding part by its handle: reach it, grasp it, move it as far as asked (else as far as it goes), let go and back away."""
    movable = [found for found in robot.find(part) if found["movable_part"]]
    if not movable:
        raise RobotError("no movable " + part + " in view")
    handle = movable[0]
    moves = handle["moves"]
    target = moves["to"] if amount is None else amount
    robot.move_to(handle["id"], "pregrasp")
    robot.open_hand(0.08)
    robot.move_to(handle["id"], "grasp")
    robot.grasp(handle["id"])
    robot.move_part(handle["id"], target, moves["unit"])
    robot.release(handle["id"])
    robot.move_to(handle["id"], "retract")
    return {"part": handle["id"], "moved_to": target, "unit": moves["unit"]}
