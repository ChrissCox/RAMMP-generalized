"""The mechanism is read off the hand's measured path: a hinge's pivot and turn, a slide's direction and travel."""
import math
import unittest

import numpy as np

from rammp_adl.constraints import rotation_about
from rammp_adl.motion.articulation import ArticulationError, fit_articulation, fit_hinge, fit_slide, hinge_from_faces

AXIS = np.array([.015, -.006, 1.])
AXIS /= np.linalg.norm(AXIS)
PIVOT = np.array([.7836, .1791, .4326])                       # the bench door's hinge line, fitted 2026-09-24


def arc(turn, *, radius=.234, start_deg=-80., count=60, noise=0., seed=0):
    helper = np.cross(AXIS, [1., 0., 0.])
    u = helper/np.linalg.norm(helper)
    v = np.cross(AXIS, u)
    angles = np.radians(start_deg)-np.linspace(0., turn, count)
    points = PIVOT+radius*(np.outer(np.cos(angles), u)+np.outer(np.sin(angles), v))
    return points+np.random.default_rng(seed).normal(0., noise, points.shape)


class HingeTests(unittest.TestCase):
    def test_a_door_swung_thirty_degrees_predicts_where_its_handle_goes_next(self):
        prior = PIVOT+np.array([-.0147, -.0109, 0.])          # where discovery put it: 18 mm off
        path = arc(math.radians(30.), noise=.0004)
        fit = fit_hinge(path, axis=AXIS, prior_pivot=prior)
        self.assertAlmostEqual(abs(fit["turned"]), math.radians(30.), delta=math.radians(1.))
        self.assertLess(fit["rms_m"], .001)
        truth = arc(math.radians(90.))[-1]

        def predicted(pivot, radius):
            start = path[0]-pivot
            start -= (start @ AXIS)*AXIS
            start *= radius/np.linalg.norm(start)
            angle = -math.radians(90.)
            turned = (start*math.cos(angle)+np.cross(AXIS, start)*math.sin(angle))
            return pivot+turned+(path[0]-pivot) @ AXIS*AXIS
        fitted_miss = np.linalg.norm(predicted(np.asarray(fit["pivot"]), fit["radius"])-truth)
        prior_miss = np.linalg.norm(predicted(prior, np.linalg.norm((path[0]-prior)[:2]))-truth)
        # 30 degrees seen: the rest of the swing within 1.5 cm (discovery's prior misses by 2.6 cm); 45 seen: within 7 mm.
        self.assertLess(fitted_miss, .015)
        self.assertLess(fitted_miss, .6*prior_miss)
        wider = arc(math.radians(45.), noise=.0004)
        fit = fit_hinge(wider, axis=AXIS, prior_pivot=prior)
        path = wider
        self.assertLess(np.linalg.norm(predicted(np.asarray(fit["pivot"]), fit["radius"])-truth), .007)

    def test_a_short_arc_leans_on_the_prior_and_a_long_one_does_not(self):
        prior = PIVOT+np.array([.02, 0., 0.])
        short = fit_hinge(arc(math.radians(4.), noise=.0005), axis=AXIS, prior_pivot=prior)
        long = fit_hinge(arc(math.radians(40.), noise=.0005), axis=AXIS, prior_pivot=prior)
        self.assertLess(short["shift_m"], .015)                                       # stays near the prior
        self.assertLess(np.linalg.norm(np.subtract(long["pivot"], PIVOT)[:2]), .003)  # finds the hinge

    def test_the_turn_is_signed_about_the_axis(self):
        forward = fit_hinge(arc(.5), axis=AXIS, prior_pivot=PIVOT)
        back = fit_hinge(arc(.5)[::-1], axis=AXIS, prior_pivot=PIVOT)
        self.assertAlmostEqual(forward["turned"], -back["turned"], places=6)


class SlideAndChoiceTests(unittest.TestCase):
    def test_a_drawer_path_is_a_slide_with_its_direction_and_travel(self):
        points = np.array([.6, 0., .3])+np.outer(np.linspace(0., .12, 40), [-1., 0., 0.])
        fit = fit_slide(points+np.random.default_rng(1).normal(0., .0003, points.shape))
        np.testing.assert_allclose(fit["axis"], [-1., 0., 0.], atol=.01)
        self.assertAlmostEqual(fit["travelled"], .12, delta=.002)

    def test_a_hinge_that_moved_straight_is_called_a_slide_but_a_short_arc_is_not(self):
        straight = np.array([.6, 0., .3])+np.outer(np.linspace(0., .12, 40), [-1., 0., 0.])
        chosen = fit_articulation(straight, kind="revolute", axis=AXIS, prior_pivot=np.array([.6, .23, .3]))
        self.assertEqual((chosen["kind"], chosen["overruled"]), ("prismatic", "revolute"))
        short = fit_articulation(arc(math.radians(6.), noise=.0005), kind="revolute", axis=AXIS, prior_pivot=PIVOT)
        self.assertEqual(short["kind"], "revolute")
        self.assertEqual(fit_articulation(straight, kind="prismatic")["kind"], "prismatic")

    def test_too_few_points_say_nothing(self):
        with self.assertRaises(ArticulationError):
            fit_hinge(arc(.3, count=2), axis=AXIS, prior_pivot=PIVOT)
        with self.assertRaises(ArticulationError):
            fit_slide(np.zeros((2, 3)))



class FacesTests(unittest.TestCase):
    def test_two_views_of_a_turning_face_give_the_axis_one_view_cannot(self):
        face = np.array([-.9, -.44, 0.])
        face /= np.linalg.norm(face)
        placed = np.array([0., 0., 1.])                                       # vertical in the face: all one view gives
        leaned = rotation_about(face, math.radians(1.85)) @ placed            # the cabinet leans sideways
        turned = rotation_about(leaned, -.49) @ face
        fit, why = hinge_from_faces(face, turned, prior_axis=placed)
        self.assertEqual(why, "")
        np.testing.assert_allclose(fit["axis"], leaned, atol=1e-9)
        self.assertAlmostEqual(fit["turned"], .49, places=9)
        self.assertAlmostEqual(fit["axis_change_deg"], 1.85, places=6)
        fit, why = hinge_from_faces(face, rotation_about(leaned, -.05) @ face, prior_axis=placed)
        self.assertIsNone(fit)
        self.assertIn("too little", why)
        fit, why = hinge_from_faces(face, rotation_about(np.array([0., 1., 0.]), .5) @ face, prior_axis=placed)
        self.assertIsNone(fit)                                                # a floor seen, not the door turned
        self.assertIn("not the same panel", why)


if __name__ == "__main__":
    unittest.main()
