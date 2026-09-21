"""Per-domain DDS transport locality evidence for local-only camera imagery."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.ros_rgbd import RosRgbdSource
from rammp_adl.perception.ros_locality import (
    CYCLONEDDS_RMW, domain_transport_evidence, imagery_locality_report,
    local_imagery_config_path, render_cyclonedds_config)
from rammp_adl.perception.calibration_recording import read_recording_config


ROOT = Path(__file__).resolve().parents[1]
CYCLONE = {"RMW_IMPLEMENTATION": CYCLONEDDS_RMW}


def env(**overrides):
    """Explicit environment; never inherit the developer's own DDS settings."""
    base = {"ROS_LOCALHOST_ONLY": "0", "RMW_IMPLEMENTATION": CYCLONEDDS_RMW, "CYCLONEDDS_URI": ""}
    return base | overrides


def config_file(directory, body, name="cyclonedds.xml"):
    path = Path(directory)/name
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<CycloneDDS xmlns="https://cdds.io/config">\n'
                    + body + "\n</CycloneDDS>\n", encoding="utf-8")
    return path


LOOPBACK_87 = """  <Domain id="87">
    <General><Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces></General>
  </Domain>"""


class DomainEvidenceTests(unittest.TestCase):
    def test_process_wide_localhost_only_without_a_config_remains_accepted(self):
        evidence = domain_transport_evidence(87, environ=env(ROS_LOCALHOST_ONLY="1"))
        self.assertTrue(evidence["confined_to_loopback"])
        self.assertEqual(evidence["mechanism"], "ros_localhost_only")
        self.assertEqual(evidence["config_sources"], [])
        # The check is a configuration claim about DDS locators, never a capture.
        self.assertIn("not a packet capture", evidence["semantics"])

    def test_localhost_only_is_middleware_independent(self):
        evidence = domain_transport_evidence(87, environ=env(ROS_LOCALHOST_ONLY="1",
                                                             RMW_IMPLEMENTATION="rmw_fastrtps_cpp"))
        self.assertTrue(evidence["confined_to_loopback"])

    def test_unset_localhost_only_without_a_config_is_not_confined(self):
        for value in ("0", "", "true", "yes"):
            evidence = domain_transport_evidence(87, environ=env(ROS_LOCALHOST_ONLY=value))
            self.assertFalse(evidence["confined_to_loopback"])
            self.assertIsNone(evidence["mechanism"])
            self.assertIn("ROS_LOCALHOST_ONLY", evidence["reason"])

    def test_per_domain_config_confines_only_the_named_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            environ = env(CYCLONEDDS_URI="file://"+str(path))
            camera = domain_transport_evidence(87, environ=environ)
            driver = domain_transport_evidence(0, environ=environ)
        self.assertTrue(camera["confined_to_loopback"])
        self.assertEqual(camera["mechanism"], "cyclonedds_per_domain")
        self.assertEqual(camera["interfaces"], ["address=127.0.0.1"])
        self.assertEqual(len(camera["config_sources"]), 1)
        self.assertEqual(len(camera["config_sources"][0]["sha256"]), 64)
        self.assertFalse(driver["confined_to_loopback"])
        self.assertIn("no loopback interface", driver["reason"])

    def test_bare_path_and_repeated_sources_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            evidence = domain_transport_evidence(87, environ=env(CYCLONEDDS_URI=f"{path},file://{path}"))
        self.assertTrue(evidence["confined_to_loopback"])
        self.assertEqual(len(evidence["config_sources"]), 2)

    def test_interface_by_name_and_deprecated_scalar_are_understood(self):
        cases = {'<Domain id="87"><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General></Domain>': True,
                 '<Domain id="87"><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain>': True,
                 '<Domain id="87"><General><NetworkInterfaceAddress>::1</NetworkInterfaceAddress></General></Domain>': True,
                 '<Domain id="87"><General><NetworkInterfaceAddress>auto</NetworkInterfaceAddress></General></Domain>': False,
                 '<Domain id="87"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General></Domain>': False}
        for body, expected in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                path = config_file(tmp, body)
                evidence = domain_transport_evidence(87, environ=env(CYCLONEDDS_URI="file://"+str(path)))
            self.assertEqual(evidence["confined_to_loopback"], expected, body)

    def test_any_and_autodetermine_stanzas_defeat_confinement(self):
        bodies = ['  <Domain id="any"><General><Interfaces>'
                  '<NetworkInterface autodetermine="true"/></Interfaces></General></Domain>',
                  LOOPBACK_87 + '\n  <Domain id="any"><General><Interfaces>'
                  '<NetworkInterface autodetermine="true"/></Interfaces></General></Domain>',
                  '  <Domain><General><Interfaces>'
                  '<NetworkInterface address="192.168.1.11"/></Interfaces></General></Domain>']
        for body in bodies:
            with tempfile.TemporaryDirectory() as tmp:
                path = config_file(tmp, body)
                evidence = domain_transport_evidence(87, environ=env(CYCLONEDDS_URI="file://"+str(path)))
            self.assertFalse(evidence["confined_to_loopback"], body)

    def test_remote_peers_and_external_addresses_defeat_confinement(self):
        remote = LOOPBACK_87.replace("</General>", "</General><Discovery><Peers>"
                                     '<Peer address="192.168.1.20"/></Peers></Discovery>')
        external = LOOPBACK_87.replace("</General>", "</General>").replace(
            "</Interfaces>", "</Interfaces><ExternalNetworkAddress>192.168.1.11</ExternalNetworkAddress>")
        local_peer = LOOPBACK_87.replace("</General>", "</General><Discovery><Peers>"
                                         '<Peer address="localhost"/></Peers></Discovery>')
        for body, expected in ((remote, False), (external, False), (local_peer, True)):
            with tempfile.TemporaryDirectory() as tmp:
                path = config_file(tmp, body)
                evidence = domain_transport_evidence(87, environ=env(CYCLONEDDS_URI="file://"+str(path)))
            self.assertEqual(evidence["confined_to_loopback"], expected, body)

    def test_localhost_only_cannot_be_trusted_when_a_config_may_override_it(self):
        # rmw prepends its loopback stanza to CYCLONEDDS_URI; a later stanza can
        # reselect a routable interface. Require both mechanisms to agree.
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, '  <Domain id="any"><General><Interfaces>'
                                    '<NetworkInterface autodetermine="true"/></Interfaces></General></Domain>')
            evidence = domain_transport_evidence(87, environ=env(ROS_LOCALHOST_ONLY="1",
                                                                CYCLONEDDS_URI="file://"+str(path)))
            self.assertFalse(evidence["confined_to_loopback"])
            self.assertIn("CYCLONEDDS_URI", evidence["reason"])
            confining = config_file(tmp, LOOPBACK_87, name="confining.xml")
            agreed = domain_transport_evidence(87, environ=env(ROS_LOCALHOST_ONLY="1",
                                                              CYCLONEDDS_URI="file://"+str(confining)))
        self.assertTrue(agreed["confined_to_loopback"])
        self.assertEqual(agreed["mechanism"], "ros_localhost_only+cyclonedds_per_domain")

    def test_unreadable_unparsable_and_inline_sources_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp)/"broken.xml"
            broken.write_text("<CycloneDDS><Domain>", encoding="utf-8")
            oversize = Path(tmp)/"big.xml"
            oversize.write_text("<!--" + "x"*70000 + "-->", encoding="utf-8")
            cases = {"file://"+str(Path(tmp)/"absent.xml"): "could not be read",
                     "file://"+str(broken): "could not be parsed",
                     "file://"+str(oversize): "exceeds",
                     "file://"+tmp: "could not be read",
                     '<CycloneDDS><Domain id="87"/></CycloneDDS>': "inline",
                     ",".join("file://"+str(config_file(tmp, LOOPBACK_87, name=f"c{i}.xml"))
                              for i in range(9)): "at most"}
            for uri, reason in cases.items():
                evidence = domain_transport_evidence(87, environ=env(CYCLONEDDS_URI=uri))
                self.assertFalse(evidence["confined_to_loopback"], uri[:60])
                self.assertIn(reason, evidence["reason"])

    def test_non_cyclone_middleware_cannot_use_the_configuration_mechanism(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            evidence = domain_transport_evidence(87, environ=env(RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
                                                                CYCLONEDDS_URI="file://"+str(path)))
        self.assertFalse(evidence["confined_to_loopback"])
        self.assertIn(CYCLONEDDS_RMW, evidence["reason"])

    def test_domain_identifiers_are_validated(self):
        for domain in (True, -1, 233, "87", 1.0):
            with self.assertRaises(PerceptionError):
                domain_transport_evidence(domain, environ=env())


class LocalityReportTests(unittest.TestCase):
    def test_unconfined_non_imagery_domain_is_reported_but_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            report = imagery_locality_report(camera_domains=[87, 87], other_domains=[0],
                                             environ=env(CYCLONEDDS_URI="file://"+str(path)))
        self.assertTrue(report["ok"])
        self.assertEqual(report["camera_domains"], [87])
        self.assertEqual(report["unconfined_camera_domains"], [])
        self.assertFalse(report["domains"]["0"]["confined_to_loopback"])
        self.assertTrue(report["domains"]["87"]["confined_to_loopback"])
        self.assertEqual(report["reason"], "")

    def test_imagery_on_an_unconfined_domain_is_refused(self):
        report = imagery_locality_report(camera_domains=[87], other_domains=[0], environ=env())
        self.assertFalse(report["ok"])
        self.assertEqual(report["unconfined_camera_domains"], [87])
        self.assertIn("87", report["reason"])

    def test_a_camera_sharing_the_driver_domain_must_still_be_confined(self):
        report = imagery_locality_report(camera_domains=[0], other_domains=[0],
                                         environ=env(ROS_LOCALHOST_ONLY="1"))
        self.assertTrue(report["ok"])
        self.assertEqual(report["shared_domains"], [0])

    def test_at_least_one_camera_domain_is_required(self):
        with self.assertRaises(PerceptionError):
            imagery_locality_report(camera_domains=[], other_domains=[0], environ=env())


class GeneratedConfigTests(unittest.TestCase):
    def test_generated_document_confines_exactly_the_requested_domains(self):
        text = render_cyclonedds_config(loopback_domains=[87, 5])
        self.assertEqual(text, render_cyclonedds_config(loopback_domains=[5, 87, 5]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"generated.xml"
            path.write_text(text, encoding="utf-8")
            environ = env(CYCLONEDDS_URI="file://"+str(path))
            for domain in (5, 87):
                self.assertTrue(domain_transport_evidence(domain, environ=environ)["confined_to_loopback"])
            for domain in (0, 88):
                self.assertFalse(domain_transport_evidence(domain, environ=environ)["confined_to_loopback"])

    def test_generated_document_declares_no_wildcard_domain(self):
        self.assertNotIn('id="any"', render_cyclonedds_config(loopback_domains=[87]))
        with self.assertRaises(PerceptionError):
            render_cyclonedds_config(loopback_domains=[])

    def test_repository_document_matches_the_recording_camera_domains(self):
        config = read_recording_config(ROOT/"config/calibration-recording.json")
        domains = sorted({settings["domain_id"] for settings in config["cameras"].values()})
        path = local_imagery_config_path()
        self.assertEqual(path, ROOT/"config/cyclonedds-local-imagery.xml")
        self.assertEqual(path.read_text(encoding="utf-8"), render_cyclonedds_config(loopback_domains=domains))
        environ = env(CYCLONEDDS_URI="file://"+str(path))
        report = imagery_locality_report(camera_domains=domains,
                                         other_domains=[config["driver"]["domain_id"]], environ=environ)
        self.assertTrue(report["ok"], report["reason"])
        self.assertFalse(report["domains"][str(config["driver"]["domain_id"])]["confined_to_loopback"])


class SourceGuardTests(unittest.TestCase):
    def test_camera_source_refuses_an_unconfined_domain_before_importing_ros(self):
        with patch.dict(os.environ, env()):
            with self.assertRaisesRegex(PerceptionError, "ROS_LOCALHOST_ONLY"):
                RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di")

    def test_camera_source_accepts_a_per_domain_confined_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            environ = env(CYCLONEDDS_URI="file://"+str(path))
            with patch.dict(os.environ, environ), patch.dict("sys.modules", {"rclpy": None}):
                # Locality is decided before any ROS import; the absent module
                # proves the guard passed rather than silently skipping.
                with self.assertRaises((ImportError, TypeError, AttributeError)):
                    RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri",
                                  depth_info_topic="/di", domain_id=87)
            with patch.dict(os.environ, environ | {"ROS_DOMAIN_ID": "0"}):
                with self.assertRaisesRegex(PerceptionError, "ROS_LOCALHOST_ONLY"):
                    RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di")

    def test_default_domain_follows_the_ros_domain_id_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, LOOPBACK_87)
            environ = env(CYCLONEDDS_URI="file://"+str(path), ROS_DOMAIN_ID="87")
            with patch.dict(os.environ, environ), patch.dict("sys.modules", {"rclpy": None}):
                with self.assertRaises((ImportError, TypeError, AttributeError)):
                    RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di")
            for bad in ("abc", "-1", "233", ""):
                with patch.dict(os.environ, environ | {"ROS_DOMAIN_ID": bad}):
                    with self.assertRaises(PerceptionError):
                        RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di")


if __name__ == "__main__":
    unittest.main()
