"""
tools/package_search_tool.py  —  Smart package matching with MAA YOJANA booking rules.

Searches BOTH maa_packages.json AND maa_robotic_surgeries.json using diagnosis keywords,
speciality matching, and TF-IDF-like scoring.  Then applies the 5 booking rules:

  1. Surgical + medical management packages cannot be booked together
  2. Stand-alone packages cannot be booked with any other package
  3. Add-on packages can only be booked alongside a regular package
  4. Implant packages auto-appear with their parent procedure
  5. Extended LOS packages can only be booked with a surgery package
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import PACKAGES_JSON, TOP_K_PACKAGES

logger = logging.getLogger("package_search")

# Robotic surgeries path (same directory as maa_packages.json)
ROBOTIC_JSON = PACKAGES_JSON.parent / "maa_robotic_surgeries.json"

# ── Package type constants ────────────────────────────────────────────
TYPE_REGULAR = "regular"
TYPE_ADDON = "addon"
TYPE_STANDALONE = "standalone"
TYPE_IMPLANT = "implant"
TYPE_EXTENDED_LOS = "extended_los"
TYPE_MEDICAL = "medical_management"
TYPE_DAY_CARE = "day_care"


# ── Loading ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_packages() -> list[dict[str, Any]]:
    """Load maa_packages.json once and cache in memory."""
    if not PACKAGES_JSON.exists():
        logger.warning(f"maa_packages.json not found at {PACKAGES_JSON}")
        return []
    with open(PACKAGES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} packages from {PACKAGES_JSON}")
    return data


@lru_cache(maxsize=1)
def _load_robotic_surgeries() -> list[dict[str, Any]]:
    """Load maa_robotic_surgeries.json once and cache in memory."""
    if not ROBOTIC_JSON.exists():
        logger.warning(f"maa_robotic_surgeries.json not found at {ROBOTIC_JSON}")
        return []
    with open(ROBOTIC_JSON, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} robotic surgery packages from {ROBOTIC_JSON}")
    return data


# ── Classification ────────────────────────────────────────────────────
def _classify_package(pkg: dict[str, Any], source: str) -> str:
    """Detect package type from metadata tags and fields."""
    if source == "robotic":
        pkg_type = (pkg.get("PACKAGE TYPE") or "").upper().strip()
        if pkg_type == "IMP":
            return TYPE_IMPLANT
        proc_type = (pkg.get("Procedure Type") or "").lower().strip()
        if "stand alone" in proc_type:
            return TYPE_STANDALONE
        if "add on" in proc_type:
            return TYPE_ADDON
        if "day care" in proc_type:
            return TYPE_DAY_CARE
        return TYPE_REGULAR

    # ── maa_packages.json ──
    name = (pkg.get("PACKAGE NAME") or "").upper()
    rate = pkg.get("RATE", 0)

    if "EXTENDED LOS" in name or "EXTENDED LENGTH OF STAY" in name:
        return TYPE_EXTENDED_LOS
    if "[STAND- ALONE]" in name or "[STAND-ALONE]" in name or "[STANDALONE]" in name:
        return TYPE_STANDALONE
    if "[ADD - ON PROCEDURE]" in name or "[ADD-ON" in name:
        return TYPE_ADDON
    if "[GOVT RESERVE / ADD ON]" in name:
        return TYPE_ADDON

    strat = (pkg.get("STRATIFICATION PACKAGE") or "").strip()
    if rate == 0 and strat != "NO STRATIFICATION":
        return TYPE_MEDICAL

    return TYPE_REGULAR


def _has_linked_implants(pkg: dict[str, Any], source: str) -> bool:
    """Check if a package has linked implant packages."""
    if source == "pmjay":
        implant = (pkg.get("IMPLANT PACKAGE") or "").strip()
        return implant != "" and implant.upper() != "NO IMPLANT"
    return False


def _find_linked_implants(
    pkg: dict[str, Any],
    all_robotic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find robotic IMP rows whose code starts with this package's code."""
    results = []
    pkg_code = pkg.get("PACKAGE CODE", "")
    if not pkg_code:
        return results
    for rpkg in all_robotic:
        r_code = rpkg.get("PACKAGE CODE", "")
        r_type = (rpkg.get("PACKAGE TYPE") or "").upper().strip()
        if r_type == "IMP" and r_code.startswith(pkg_code):
            results.append(rpkg)
    return results


# ── Scoring ───────────────────────────────────────────────────────────
def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenizer, strips short tokens."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _score_package(pkg: dict[str, Any], query_tokens: set[str]) -> float:
    """Score a package against query tokens (weighted fields)."""
    searchable = " ".join([
        pkg.get("PACKAGE NAME", pkg.get("Package Name", "")) * 3,
        pkg.get("SPECIALITY", pkg.get("Speciality", "")) * 2,
        pkg.get("PRE AUTH DOCUMENT", ""),
        pkg.get("CLAIM DOCUMENT", ""),
        pkg.get("PACKAGE CATEGORY", ""),
        pkg.get("Procedure Sub Category", ""),
    ])
    pkg_tokens = _tokenize(searchable)
    if not pkg_tokens:
        return 0.0

    overlap = query_tokens & pkg_tokens
    score = len(overlap) / max(len(query_tokens), 1)

    pkg_name = pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))
    direct_hits = len(query_tokens & _tokenize(pkg_name))
    score += direct_hits * 0.15

    return round(score, 4)


# ── Booking Rules Engine ─────────────────────────────────────────────
def apply_booking_rules(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the 5 MAA YOJANA booking rules.

    Returns grouped results: primary_packages, addon_packages,
    implant_packages, warnings, removed_packages.
    """
    primary, addons, implants = [], [], []
    standalones, extended_los, medical = [], [], []
    removed: list[dict] = []
    warnings: list[str] = []

    for pkg in packages:
        ptype = pkg.get("package_type", TYPE_REGULAR)
        if ptype == TYPE_STANDALONE:
            standalones.append(pkg)
        elif ptype == TYPE_ADDON:
            addons.append(pkg)
        elif ptype == TYPE_IMPLANT:
            implants.append(pkg)
        elif ptype == TYPE_EXTENDED_LOS:
            extended_los.append(pkg)
        elif ptype == TYPE_MEDICAL:
            medical.append(pkg)
        else:
            primary.append(pkg)

    # ── Rule 2: Stand-alone isolation ─────────────────────────────
    if standalones:
        best = max(standalones, key=lambda p: p.get("alignment_score", 0))
        sa_imps = [
            i for i in implants
            if i.get("parent_code") == best.get("package_code", "???")
        ]
        rest = primary + addons + extended_los + medical
        rest += [s for s in standalones if s is not best]
        rest += [i for i in implants if i not in sa_imps]
        if rest:
            warnings.append(
                f"🔒 '{best.get('package_name', 'Standalone')}' is STAND-ALONE "
                f"— {len(rest)} other package(s) excluded."
            )
            for r in rest:
                r["removal_reason"] = "Standalone rule"
        return {
            "primary_packages": [best],
            "addon_packages": [],
            "implant_packages": sa_imps,
            "warnings": warnings,
            "removed_packages": rest,
        }

    # ── Rule 1: Surgical + medical exclusion ─────────────────────
    has_surgical = any(p.get("rate", 0) > 0 for p in primary)
    if has_surgical and medical:
        warnings.append(
            f"⚕️ Surgical + medical management cannot be booked together. "
            f"{len(medical)} medical package(s) removed."
        )
        for m in medical:
            m["removal_reason"] = "Surgical + medical exclusion"
        removed.extend(medical)
        medical = []

    # ── Rule 3: Add-ons require a regular package ────────────────
    if addons and not primary:
        warnings.append(
            f"➕ {len(addons)} add-on package(s) need a regular package — none matched."
        )
        for a in addons:
            a["removal_reason"] = "No regular package"
        removed.extend(addons)
        addons = []

    # ── Rule 5: Extended LOS requires a surgery package ──────────
    if extended_los and not has_surgical:
        warnings.append(
            "🏥 Extended LOS packages require a surgery package — none matched."
        )
        for e in extended_los:
            e["removal_reason"] = "No surgery for extended LOS"
        removed.extend(extended_los)
        extended_los = []

    # Valid extended LOS packages behave like add-ons
    addons.extend(extended_los)
    # Non-excluded medical packages go into primary
    primary.extend(medical)

    return {
        "primary_packages": primary,
        "addon_packages": addons,
        "implant_packages": implants,
        "warnings": warnings,
        "removed_packages": removed,
    }


# ── Main Search ───────────────────────────────────────────────────────
def search_packages(
    diagnosis: str,
    speciality: str = "",
    extra_keywords: str = "",
    surgery_name: str = "",
    procedure_name: str = "",
    top_k: int = TOP_K_PACKAGES,
    use_smart_agent: bool = True,
) -> dict[str, Any]:
    """
    Returns structured package results with booking rule enforcement.

    Args:
        diagnosis: Primary diagnosis or condition
        speciality: Medical specialty/department
        extra_keywords: Additional search terms
        surgery_name: Name of surgery if applicable
        procedure_name: Name of procedure if applicable
        top_k: Number of top packages to return
        use_smart_agent: If True, uses the intelligent LLM-powered agent

    Result keys: primary_packages, addon_packages, implant_packages,
                 warnings, removed_packages, total_matched
    """
    # Try smart agent first for better results
    if use_smart_agent:
        try:
            from tools.smart_package_agent import search_packages_smart
            result = search_packages_smart(
                diagnosis=diagnosis,
                speciality=speciality,
                extra_keywords=extra_keywords,
                surgery_name=surgery_name,
                procedure_name=procedure_name,
                top_k=top_k,
            )
            logger.info("Using Smart Package Agent for intelligent matching")
            return result
        except Exception as e:
            logger.warning(f"Smart agent failed, falling back to basic search: {e}")

    # Fallback to basic token-based search
    packages = _load_packages()
    robotic = _load_robotic_surgeries()
    empty: dict[str, Any] = {
        "primary_packages": [], "addon_packages": [],
        "implant_packages": [], "warnings": [],
        "removed_packages": [], "total_matched": 0,
    }

    if not packages and not robotic:
        return empty

    query = f"{diagnosis} {speciality} {extra_keywords} {surgery_name} {procedure_name}"
    query_tokens = _tokenize(query)
    if not query_tokens:
        return empty

    # ── Score all packages ────────────────────────────────────────
    scored: list[tuple[float, dict, str]] = []
    for pkg in packages:
        s = _score_package(pkg, query_tokens)
        if s > 0:
            scored.append((s, pkg, "pmjay"))
    for pkg in robotic:
        s = _score_package(pkg, query_tokens)
        if s > 0:
            scored.append((s, pkg, "robotic"))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Build classified result list ──────────────────────────────
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for score, pkg, source in scored[:top_k * 2]:
        pkg_type = _classify_package(pkg, source)
        code = pkg.get("PACKAGE CODE", str(pkg.get("SR. NO.", "")))
        if code in seen:
            continue
        seen.add(code)

        if source == "pmjay":
            entry: dict[str, Any] = {
                "source": "MAA YOJANA Package",
                "package_code": pkg.get("PACKAGE CODE", ""),
                "package_name": pkg.get("PACKAGE NAME", "").replace("\\n", " ").strip(),
                "speciality": pkg.get("SPECIALITY", ""),
                "category": pkg.get("PACKAGE CATEGORY", ""),
                "rate": pkg.get("RATE", 0),
                "govt_reserve": pkg.get("GOVT RESERVE", "NO"),
                "pre_auth_documents": pkg.get("PRE AUTH DOCUMENT", ""),
                "claim_documents": pkg.get("CLAIM DOCUMENT", ""),
                "alignment_score": int(score * 100),
                "package_type": pkg_type,
            }

            # ── Rule 4: Auto-include linked implants ──────────
            if _has_linked_implants(pkg, source):
                entry["has_implant"] = True
                entry["implant_details"] = pkg.get("IMPLANT PACKAGE", "")
                for imp in _find_linked_implants(pkg, robotic):
                    ic = imp.get("PACKAGE CODE", "")
                    if ic not in seen:
                        seen.add(ic)
                        all_results.append({
                            "source": "Implant Package",
                            "package_code": ic,
                            "package_name": imp.get("Package Name", "").replace("\\n", " ").strip(),
                            "speciality": imp.get("Speciality", ""),
                            "category": "Implant",
                            "rate": imp.get("Rate", 0),
                            "alignment_score": int(score * 80),
                            "package_type": TYPE_IMPLANT,
                            "parent_code": pkg.get("PACKAGE CODE", ""),
                        })

            all_results.append(entry)
        else:
            all_results.append({
                "source": "Robotic Surgery",
                "package_code": pkg.get("PACKAGE CODE", str(pkg.get("SR. NO.", ""))),
                "package_name": pkg.get("Package Name", "").replace("\\n", " ").strip(),
                "speciality": pkg.get("Speciality", ""),
                "category": pkg.get("PACKAGE TYPE", ""),
                "rate": pkg.get("Rate", 0),
                "procedure_type": pkg.get("Procedure Type", ""),
                "mandatory_documents": pkg.get("Mandatory Documents", ""),
                "claim_documents": pkg.get("Mandatory Documents - Claim Processing", ""),
                "alignment_score": int(score * 100),
                "package_type": pkg_type,
            })

    # ── Apply booking rules ───────────────────────────────────────
    ruled = apply_booking_rules(all_results)
    ruled["primary_packages"] = ruled["primary_packages"][:top_k]
    ruled["total_matched"] = (
        len(ruled["primary_packages"])
        + len(ruled["addon_packages"])
        + len(ruled["implant_packages"])
    )

    logger.info(
        f"Package search: '{diagnosis}' → "
        f"{len(ruled['primary_packages'])} primary, "
        f"{len(ruled['addon_packages'])} addon, "
        f"{len(ruled['implant_packages'])} implant"
    )
    return ruled
