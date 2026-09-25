def run(robot, part="handle", amount=None):
    """Open a part by its handle (as far as asked, else as far as it goes), then close it again: a door or drawer put back as it was."""
    opened = robot.use("open_by_handle", part=part, amount=amount)
    closed = robot.use("close_by_handle", part=part)
    return {"opened": opened, "closed": closed}
