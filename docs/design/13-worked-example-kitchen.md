# Bounded kitchen integration example

Integration scenario: open a cabinet, retrieve a supported utensil, then place it on a supported tray.

The sequence is composed by the same fixed vocabulary and executor:
observe cabinet handle -> empty preshape plus transit -> grasp handle -> follow_constraint -> release -> observe utensil -> transit -> grasp utensil -> move_to_pose at tray placement -> release after tray support verification.

Each arrow is a real plan dependency; preshape/transit is the only parallel branch where the aperture envelope and empty-gripper state permit it. Appliance heating, uncapping, carrying hot food, autonomous base motion, and manipulating unobserved objects are not implied by this example.

The two complete machine-readable trace fixtures are [cabinet](../../examples/cabinet.plan.json) and [bottle](../../examples/bottle.plan.json). This kitchen sequence is a scope illustration, not a third supposedly executable plan with invented poses or appliance interfaces.

If a grasp is not retained, commit the observed miss and replan while held; do not ask Astra to guide the arm through a cloud loop.
