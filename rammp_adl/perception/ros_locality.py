"""Per-ROS-domain DDS transport locality for local-only camera imagery.

Camera imagery must never leave this machine. A single process-wide
ROS_LOCALHOST_ONLY flag cannot express that while a read-only subscription in a
different domain must still reach a publisher that binds a routable interface,
because the flag pins every domain in the process to 127.0.0.1. This module
therefore resolves confinement separately for each ROS domain from the explicit
Cyclone DDS configuration named by CYCLONEDDS_URI.

Every result here is a configuration claim about the DDS locators a participant
will advertise. It is not a packet capture, not an egress audit, and it admits
no calibration, transform, capability or command. Anything this module cannot
positively prove is reported as unconfined.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree

from .geometry import PerceptionError


CYCLONEDDS_RMW = "rmw_cyclonedds_cpp"
MAX_CONFIG_SOURCES = 8
MAX_CONFIG_BYTES = 65536
LOOPBACK_INTERFACE_NAMES = ("lo",)
LOCALITY_SEMANTICS = ("configuration-derived confinement of advertised DDS locators; "
                      "not a packet capture and not an egress audit")
CONFIG_FILENAME = "cyclonedds-local-imagery.xml"


def local_imagery_config_path():
    """The repository document that confines the recorded camera domains."""
    return Path(__file__).resolve().parents[2]/"config"/CONFIG_FILENAME


def _domain(value):
    if type(value) is not int or not 0 <= value <= 232:
        raise PerceptionError("ROS domains must be integers in 0..232")
    return value


def _environment_domain(environ):
    raw = (environ.get("ROS_DOMAIN_ID") or "").strip()
    if not raw:
        return 0
    if not re.fullmatch(r"0|[1-9][0-9]*", raw):
        raise PerceptionError("ROS_DOMAIN_ID must be an integer in 0..232")
    return _domain(int(raw))


def _loopback_address(text):
    """Accept only addresses that are provably loopback, with an optional port."""
    text = (text or "").strip()
    if text.lower() == "localhost":
        return True
    candidates = [text]
    if text.startswith("[") and "]" in text:
        candidates.append(text[1:text.index("]")])
    elif text.count(":") == 1:
        candidates.append(text.rsplit(":", 1)[0])
    for candidate in candidates:
        try:
            return ipaddress.ip_address(candidate).is_loopback
        except ValueError:
            continue
    return False


def _tag(element):
    return element.tag.rpartition("}")[2]


def _children(element, name):
    return [child for child in element if _tag(child) == name]


def _descend(element, *names):
    found = [element]
    for name in names:
        found = [child for parent in found for child in _children(parent, name)]
    return found


def _interface(element):
    """Label one NetworkInterface selector and say whether it is loopback."""
    attributes = {key.lower(): (value or "").strip() for key, value in element.attrib.items()}
    if attributes.get("autodetermine", "false").lower() in ("true", "1", "yes"):
        return "autodetermine", False
    if "address" in attributes:
        return "address="+attributes["address"], _loopback_address(attributes["address"])
    if "name" in attributes:
        return "name="+attributes["name"], attributes["name"] in LOOPBACK_INTERFACE_NAMES
    return "unspecified", False


def _applies(element, domain_id):
    """Cyclone DDS selects a stanza by id; an unrecognised selector may apply."""
    raw = (element.get("id") or "").strip()
    if not raw or raw.lower() == "any":
        return True
    try:
        return int(raw) == domain_id
    except ValueError:
        return True


def _config_sources(uri):
    """Resolve the comma-separated file list; inline XML cannot be split safely."""
    parts = [part.strip() for part in uri.split(",")]
    parts = [part for part in parts if part]
    if len(parts) > MAX_CONFIG_SOURCES:
        raise PerceptionError(f"CYCLONEDDS_URI must name at most {MAX_CONFIG_SOURCES} configuration files")
    sources = []
    for part in parts:
        if part.startswith("<"):
            raise PerceptionError("inline CycloneDDS XML in CYCLONEDDS_URI is not accepted; name a file instead")
        path = Path(part[len("file://"):] if part.startswith("file://") else part)
        try:
            if not path.is_file():
                raise OSError("not a regular file")
            size = path.stat().st_size
        except OSError as exc:
            raise PerceptionError(f"CycloneDDS configuration {part} could not be read: {exc}") from exc
        if size > MAX_CONFIG_BYTES:
            raise PerceptionError(f"CycloneDDS configuration {part} exceeds {MAX_CONFIG_BYTES} bytes")
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PerceptionError(f"CycloneDDS configuration {part} could not be read: {exc}") from exc
        sources.append({"source": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data), "text": text})
    return sources


def domain_transport_evidence(domain_id, *, environ=None):
    """Report whether one ROS domain is confined to a loopback interface.

    Two mechanisms are accepted. ROS_LOCALHOST_ONLY=1 confines every domain in
    the process. A CYCLONEDDS_URI document confines the domains whose stanzas
    select only loopback interfaces. When both are present they must agree,
    because the middleware prepends its own stanza to that same list and a later
    stanza can reselect a routable interface.
    """
    domain_id = _domain(domain_id)
    environ = os.environ if environ is None else environ
    localhost_only = environ.get("ROS_LOCALHOST_ONLY", "") == "1"
    rmw = environ.get("RMW_IMPLEMENTATION", "").strip() or None
    uri = (environ.get("CYCLONEDDS_URI") or "").strip()
    evidence = {"domain_id": domain_id, "confined_to_loopback": False, "mechanism": None,
                "reason": "", "ros_localhost_only": localhost_only, "rmw_implementation": rmw,
                "cyclonedds_uri": uri or None, "config_sources": [], "interfaces": [],
                "semantics": LOCALITY_SEMANTICS}
    if not uri or rmw != CYCLONEDDS_RMW:
        if localhost_only:
            evidence.update(confined_to_loopback=True, mechanism="ros_localhost_only")
        elif uri:
            evidence["reason"] = f"CYCLONEDDS_URI is only meaningful with RMW_IMPLEMENTATION={CYCLONEDDS_RMW}"
        else:
            evidence["reason"] = ("neither ROS_LOCALHOST_ONLY=1 nor a CYCLONEDDS_URI document confining "
                                  f"ROS domain {domain_id} to a loopback interface is configured")
        return evidence
    try:
        sources = _config_sources(uri)
    except PerceptionError as exc:
        evidence["reason"] = str(exc)
        return evidence
    evidence["config_sources"] = [{key: source[key] for key in ("source", "sha256", "bytes")}
                                  for source in sources]
    interfaces, problems = [], []
    for source in sources:
        try:
            root = ElementTree.fromstring(source["text"])
        except ElementTree.ParseError as exc:
            evidence["reason"] = f"CycloneDDS configuration {source['source']} could not be parsed: {exc}"
            return evidence
        stanzas = _children(root, "Domain")
        # A document without Domain elements configures every domain at once.
        stanzas = [s for s in stanzas if _applies(s, domain_id)] if stanzas else [root]
        for stanza in stanzas:
            for element in _descend(stanza, "General", "Interfaces", "NetworkInterface"):
                label, loopback = _interface(element)
                interfaces.append(label)
                if not loopback:
                    problems.append("interface "+label)
            for element in _descend(stanza, "General", "NetworkInterfaceAddress"):
                text = (element.text or "").strip()
                interfaces.append("NetworkInterfaceAddress="+text)
                if text not in LOOPBACK_INTERFACE_NAMES and not _loopback_address(text):
                    problems.append("interface NetworkInterfaceAddress="+text)
            for element in _descend(stanza, "General", "ExternalNetworkAddress"):
                problems.append("ExternalNetworkAddress="+(element.text or "").strip())
            for element in _descend(stanza, "Discovery", "Peers", "Peer"):
                address = (element.get("address") or element.text or "").strip()
                if not _loopback_address(address):
                    problems.append("peer "+address)
    evidence["interfaces"] = interfaces
    if problems:
        evidence["reason"] = (f"CYCLONEDDS_URI configuration does not confine ROS domain {domain_id}: "
                              + "; ".join(sorted(set(problems))))
    elif not interfaces:
        evidence["reason"] = (f"CYCLONEDDS_URI configuration selects no loopback interface for ROS domain "
                              f"{domain_id}; unmatched domains keep the autodetermined routable interface")
    else:
        evidence.update(confined_to_loopback=True,
                        mechanism="ros_localhost_only+cyclonedds_per_domain" if localhost_only
                        else "cyclonedds_per_domain")
    return evidence


def imagery_locality_report(*, camera_domains, other_domains=(), environ=None):
    """Confinement evidence for every recorded domain; imagery domains must pass.

    A domain listed in other_domains carries no imagery, so it may stay routable.
    A domain that carries imagery must be confined even when it is shared.
    """
    cameras = sorted({_domain(domain) for domain in camera_domains})
    if not cameras:
        raise PerceptionError("At least one camera domain is required")
    others = sorted({_domain(domain) for domain in other_domains})
    domains = {str(domain): domain_transport_evidence(domain, environ=environ)
               for domain in sorted(set(cameras) | set(others))}
    unconfined = [domain for domain in cameras if not domains[str(domain)]["confined_to_loopback"]]
    reason = ""
    if unconfined:
        reason = "; ".join(f"ROS domain {domain} carries camera imagery but is not confined to loopback: "
                           + domains[str(domain)]["reason"] for domain in unconfined)
    return {"ok": not unconfined, "reason": reason, "camera_domains": cameras,
            "other_domains": others, "shared_domains": sorted(set(cameras) & set(others)),
            "unconfined_camera_domains": unconfined,
            "config_dependent": any(domains[str(domain)]["mechanism"] in (
                "cyclonedds_per_domain", "ros_localhost_only+cyclonedds_per_domain") for domain in cameras),
            "required_rmw_for_config": CYCLONEDDS_RMW, "domains": domains,
            "semantics": LOCALITY_SEMANTICS}


POLICY_SCHEMA_VERSION = 1


def deployment_policy(document):
    """Validate the deployment's declaration of the routable camera domains it publishes on.

    The deployment may publish imagery on the host network by its own design.
    This runtime never confines such streams; the policy names the domains it
    may subscribe to and why.
    """
    if not isinstance(document, dict) or document.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise PerceptionError("Imagery policy must be a version 1 document")
    domains = document.get("accepted_routable_domains")
    if (not isinstance(domains, list) or not domains
            or any(isinstance(d, bool) or type(d) is not int or not 0 <= d <= 232 for d in domains)):
        raise PerceptionError("Imagery policy must list accepted ROS domains in 0..232")
    reason = document.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PerceptionError("Imagery policy needs a stated reason")
    return {"accepted_domains": frozenset(domains), "reason": reason.strip(),
            "digest": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()}


def require_local_imagery(domain_id=None, *, environ=None, policy=None):
    """Refuse to create an imagery subscription on an unconfined ROS domain.

    The deployment policy is the one alternative to confinement: it names the
    routable domains the deployment already publishes on. The returned
    evidence records which mechanism admitted the subscription.
    """
    environ = os.environ if environ is None else environ
    domain_id = _environment_domain(environ) if domain_id is None else _domain(domain_id)
    evidence = domain_transport_evidence(domain_id, environ=environ)
    if evidence["confined_to_loopback"]:
        return evidence
    if policy is not None:
        accepted = deployment_policy(policy)
        if domain_id in accepted["accepted_domains"]:
            return {**evidence, "confined_to_loopback": False, "admitted": True,
                    "mechanism": "deployment_policy", "policy": accepted,
                    "semantics": "routable imagery accepted under the deployment policy; "
                                 "the stream is not confined by this runtime"}
        raise PerceptionError(f"Imagery on ROS domain {domain_id} is not confined and is not among the policy's "
                              "accepted domains; list it in config/imagery-locality.json to accept it")
    raise PerceptionError(
        "Local imagery requires ROS_LOCALHOST_ONLY=1, or a CYCLONEDDS_URI document confining ROS "
        f"domain {domain_id} to loopback, before creating the ROS context: {evidence['reason']}")


def render_cyclonedds_config(*, loopback_domains):
    """Render the deterministic document that pins imagery domains to loopback."""
    domains = sorted({_domain(domain) for domain in loopback_domains})
    if not domains:
        raise PerceptionError("At least one ROS domain must be confined to loopback")
    stanzas = "\n".join(f"""  <Domain id="{domain}">
    <General>
      <Interfaces>
        <NetworkInterface address="127.0.0.1"/>
      </Interfaces>
    </General>
  </Domain>""" for domain in domains)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Generated by rammp_adl.perception.ros_locality.render_cyclonedds_config.

  Each listed ROS domain carries local camera imagery and is pinned to the
  loopback interface, so those participants advertise no routable DDS locator.
  Domains that are not listed keep the Cyclone DDS defaults, which is how a
  read-only subscription reaches a driver that binds the real interface.

  This confines advertised locators by configuration. It is not a packet
  capture, and it does not constrain a publisher started by another process:
  launch camera drivers with the same document or with ROS_LOCALHOST_ONLY=1.

  Do not edit by hand. Regenerate it and rerun tests/test_ros_locality.py.
-->
<CycloneDDS xmlns="https://cdds.io/config">
{stanzas}
</CycloneDDS>
"""
