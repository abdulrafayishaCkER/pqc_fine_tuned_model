#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate PQC/Web Crypto Q&A generator (merged from dataset1.py + d1.py).

Built for phi-2 style fine-tuning: JSONL lines shaped like:
{
  "instruction": "<string>",
  "context": { "kb_refs": ["<KB_TAG>", "..."] },
  "response": {
    "mode": "QA",
    "justification": ["<string>", "<string>", "<string>"],
    "answer": "<string or bullet list>",
    "safety": { "private_keys_access": "forbidden", "policy_notes": "" }
  }
}

Highlights
- Rich CLI (TOTAL/OUT/VARIANTS_PER_PROMPT/SEED/STYLE/LANG/MIX/SHARD_SIZE/DUMP_SOURCES)
- Union of generators; when overlapping, keeps the broader “context window”
- Strong de-dup: normalized-instruction, [instruction,answer] SHA-1, and 3-gram near-duplicate filter
- Top-up loop with adaptive paraphrasing & optional kb_refs enrichment
- Sharded writer & optional KB sources dump (JSON & CSV)

Defaults
- TOTAL: 30000
- VARIANTS_PER_PROMPT: 6
- OUT: qa_pqc_dataset.jsonl
- STYLE: mixed  (plain|bulleted|mixed)
- LANG: en
"""
from __future__ import annotations
import random, itertools, re, math, csv, json, hashlib
from collections import defaultdict
import json, random, re, sys, hashlib, itertools, math, csv
from pathlib import Path
# ---------------- Prompt-variant helpers: synonyms, aliases, connector styles ----------------
import re, random

# interchangeable “quantum-safe” wordings
_QSAFE_CANON = "quantum-safe"
_QSAFE_VARIANTS = [
    "quantum-safe", "quantum resistant", "quantum-resistant",
    "post-quantum secure", "post-quantum compliant",
    "PQC-ready", "PQC safe", "PQC-ready"
]
_QSAFE_RX = re.compile(r"\b(post[\s-]?quantum(?:\s+(?:secure|compliant))?|quantum[\s-]?resistan[t]?|quantum[\s-]?safe|PQC(?:[\s-]?ready|[\s-]?safe)?)\b", re.I)

# algorithm aliases shown in QUESTIONS (answers remain unchanged)
_ALGO_ALIASES = {
    # KEMs
    r"\bML[- ]KEM[- ](512|768|1024)\b": [r"ML-KEM-\1", r"Kyber-\1", r"Kyber \1", r"ML KEM \1"],
    r"\bKyber[- ](512|768|1024)\b":     [r"Kyber-\1", r"ML-KEM-\1", r"ML KEM \1"],

    # Signatures
    r"\bML[- ]DSA(?:\s*\(Dilithium\))?\b": ["ML-DSA (Dilithium)", "Dilithium", "ML-DSA"],
    r"\bSLH[- ]DSA(?:\s*\(SPHINCS\+\))?\b": ["SLH-DSA (SPHINCS+)", "SPHINCS+", "SLH-DSA"],
    r"\bFalcon[- ]?512\b": ["Falcon-512", "Falcon"],

    # Common cert wording
    r"\bcertificate(s?)\b": ["certificate\\1", "cert\\1"],
}

# connector styles for hybrids in QUESTIONS (cosmetic)
_HYBRID_CONNECTORS = [" + ", " / ", " & ", " and ", " with ", " using "]
# Normalize all connector variants for dedupe (see §2)
_HYBRID_RX = re.compile(r"\s*(\+|/|&|and|with|using)\s*", re.I)

def _rand_sub_once(pattern: str, s: str, choices: list[str]) -> str:
    """Replace the first match of `pattern` in s with a random choice (keeps backrefs)."""
    rx = re.compile(pattern, re.I)
    m = rx.search(s)
    if not m:
        return s
    repl = random.choice(choices)
    # expand backrefs like \1 if present in the chosen alias
    return s[:m.start()] + rx.sub(repl, s[m.start():], count=1)

def apply_prompt_variants(q: str) -> str:
    """
    Light-touch variants for QUESTIONS only:
      - swap ‘quantum-safe’ phrasing
      - swap common alg aliases (Kyber/ML-KEM, Dilithium/ML-DSA, SPHINCS+/SLH-DSA, Falcon)
      - vary hybrid connectors (X25519 + Kyber ↔ X25519 with Kyber, etc.)
    Deterministically safe: never alters semantics.
    """
    s = q

    # 1) quantum-safe synonyms
    if _QSAFE_RX.search(s) and random.random() < 0.75:
        s = _QSAFE_RX.sub(lambda _: random.choice(_QSAFE_VARIANTS), s, count=1)

    # 2) algorithm aliases (one or two swaps per question max)
    swaps = 0
    for patt, aliases in _ALGO_ALIASES.items():
        if swaps >= 2:
            break
        before = s
        s = _rand_sub_once(patt, s, aliases)
        if s != before:
            swaps += 1

    # 3) hybrid connector styling (purely cosmetic in QUESTION text)
    if random.random() < 0.7:
        def _hyb_style(m):
            return random.choice(_HYBRID_CONNECTORS)
        s = _HYBRID_RX.sub(_hyb_style, s)

    return s


# ---------- CLI parsing ----------
def parse_cli(argv):
    cfg = {
        "TOTAL": 30000,
        "OUT": Path("qa_pqc_dataset.jsonl"),
        "VARIANTS_PER_PROMPT": 6,
        "SEED": 20240919,
        "STYLE": "mixed",           # plain|bulleted|mixed
        "LANG": "en",
        "MIX": None,                # JSON string (category->weight); merged then normalized
        "SHARD_SIZE": 0,            # single file by default; if >0, shards
        "DUMP_SOURCES": ""          # if set, writes <prefix>.json and <prefix>.csv
    }
    pos = []
    for a in argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k in ("TOTAL", "N", "COUNT"):
                cfg["TOTAL"] = int(v)
            elif k in ("OUT", "OUTPUT", "PATH"):
                cfg["OUT"] = Path(v)
            elif k in ("VARIANTS_PER_PROMPT", "VARIANTS", "V"):
                cfg["VARIANTS_PER_PROMPT"] = max(1, int(v))
            elif k == "SEED":
                cfg["SEED"] = None if v.lower() in ("none","rand","random") else int(v)
            elif k == "STYLE":
                cfg["STYLE"] = v.lower()
            elif k == "LANG":
                cfg["LANG"] = v.lower()
            elif k == "MIX":
                try:
                    cfg["MIX"] = json.loads(v)
                except Exception:
                    print("WARN: MIX not valid JSON; ignoring", file=sys.stderr)
            elif k == "SHARD_SIZE":
                cfg["SHARD_SIZE"] = int(v)
            elif k == "DUMP_SOURCES":
                cfg["DUMP_SOURCES"] = v
        else:
            pos.append(a)
    if pos:
        cfg["TOTAL"] = int(pos[0])
    if len(pos) > 1:
        cfg["OUT"] = Path(pos[1])
    if len(pos) > 2:
        cfg["VARIANTS_PER_PROMPT"] = max(1, int(pos[2]))
    return cfg

args = parse_cli(sys.argv)
TOTAL = args["TOTAL"]
OUT = args["OUT"]
VARIANTS_PER_PROMPT = args["VARIANTS_PER_PROMPT"]
STYLE = args["STYLE"]
LANG = args["LANG"]
SHARD_SIZE = args["SHARD_SIZE"]
DUMP_SOURCES = args["DUMP_SOURCES"]
if args["SEED"] is not None:
    random.seed(args["SEED"])

# ---------- Knowledge base tags & URLs ----------
# Start from the richer d1.py set, then add aliases/extra tags used by dataset1.py.
KB = {
    # Core PQC standards
    "NIST_PQC_HUB": "NIST PQC project hub",
    "FIPS203": "FIPS-203 (ML-KEM / Kyber)",
    "FIPS204": "FIPS-204 (ML-DSA / Dilithium)",
    "FIPS205": "FIPS-205 (SLH-DSA / SPHINCS+)",
    "NIST_PQC_FAQ": "NIST PQC FAQs (Grover/AES)",
    "NISTIR8105": "NISTIR 8105 (PQC survey)",
    "SP800_208": "NIST SP 800-208 (XMSS/LMS)",
    "NIST_AUG2024_NEWS": "NIST PQC standards (Aug 2024)",

    # Policy / timelines
    "NSM10": "NSM-10 White House memo",
    "OMB_M2302": "OMB M-23-02 PQC migration memo",
    "CNSA2": "NSA CNSA 2.0 algorithms/timing",
    "CISA_QR_FACTS": "CISA Quantum-Readiness factsheet",
    "WH_PQC_2024_BG": "White House PQC backgrounder (2024)",

    # IETF / LAMPS / IKEv2 / TLS drafts
    "RFC8391": "RFC 8391 (XMSS)",
    "RFC8554": "RFC 8554 (LMS)",
    "RFC8784": "RFC 8784 (PQC PSKs for IKEv2)",
    "RFC9370": "RFC 9370 (multi-KEX in IKEv2)",
    "TLS_AUTHKEM_DRAFT": "TLS AuthKEM draft",
    "LAMPS_COMP_SIGS": "LAMPS composite PQ signatures draft",
    "LAMPS_COMP_KEM": "LAMPS composite PQ KEM draft",
    "PQUIP_TRACKER": "IETF PQUIP tracker",
    "CNSA1_TLS": "CNSA 1.0 TLS profile (RFC 9151)",
    "CNSA1_IPSEC": "CNSA 1.0 IPsec profile (RFC 9206)",
    "CNSA1_SSH": "CNSA 1.0 SSH profile (RFC 9212)",

    # Regional / international
    "ETSI_QSC": "ETSI Quantum-Safe cryptography hub",
    "ENISA_PQC": "ENISA PQC mitigation",
    "BSI_TR_02102_1": "BSI TR-02102-1 crypto recs",
    "NCSC_UK_PQC": "UK NCSC PQC prep",
    "ANSSI_PQC": "ANSSI PQC hub",
    "CCCS_PQC": "Canada CCCS quantum-safe guidance",
    "SG_CSA_PQC": "Singapore CSA PQC adoption",
    "AU_ACSC_PQC": "Australia ACSC PQC prep",
    "CRYPTREC_PQC": "Japan CRYPTREC PQC portal",
    "ISO_14888_4": "ISO/IEC 14888-4:2024 (XMSS/XMSS-MT)",

    # Migration playbooks
    "NCCOE_MPQC": "NCCoE migration to PQC project",
    "NCCOE_FACT": "NCCoE MPQC fact sheet",
    "SP1800_38A": "NCCoE SP 1800-38A (prelim)",
    "SP1800_38B": "NCCoE SP 1800-38B (prelim)",
    "SP1800_38C": "NCCoE SP 1800-38C (prelim)",
    "NCCOE_WHITEPAPER": "NIST PQC migration white paper",

    # Libraries / toolchains
    "OQS_SITE": "Open Quantum Safe site",
    "LIBOQS": "liboqs",
    "OQS_PROVIDER": "OpenSSL 3 OQS provider",
    "PQCLEAN": "PQClean",
    "BOUNCY_PQC": "Bouncy Castle PQC",
    "XMSS_REF": "XMSS reference code",
    "OPENSSH_9": "OpenSSH 9.0 hybrid KEX notes",

    # Adoption (browsers/cloud/OS/CDN)
    "CHROMIUM_HYBRID": "Chromium X25519Kyber768 in TLS",
    "CF_PQC_GA": "Cloudflare PQC GA inbound/outbound",
    "CF_PQC_ORIG": "Cloudflare PQC to origins",
    "CF_PQ_2024": "Cloudflare state of PQ Internet",
    "MS_PQC_2024": "Microsoft PQC (Schannel/SymCrypt)",
    "MS_PQC_2025": "Microsoft PQC update (Aug 2025)",
    "APPLE_TLS26": "Apple iOS/macOS 26 PQC TLS prep",
    "AKAMAI_PQC_2025": "Akamai PQC TLS considerations",

    # PKI & S/MIME
    "CABF_SMC013": "CA/B Forum S/MIME Ballot SMC-013",

    # Academic / specs / refs
    "KYBER_SPEC": "CRYSTALS-Kyber spec (R3)",
    "DILITHIUM_SPEC": "CRYSTALS-Dilithium spec (R3)",
    "PQC_BOOK": "Post-Quantum Cryptography (Springer)",

    # Testing / certification
    "ACVP_HOME": "ACVP homepage",
    "ACVP_SERVER": "ACVP server releases",
    "CAVP_SAMPLE": "CAVP ML-KEM validation example",
    "WOLFSSL_CAVP": "wolfSSL CAVP blog (ML-KEM/ML-DSA)",
    "ATSEC_FIRST": "atsec: first PQC certs",

    # Community
    "NIST_PQC_FORUM": "NIST PQC Forum",
    "LF_PQCA": "Linux Foundation PQC Alliance",

    # RFC/TLS/etc tags reused
    "TLS13": "RFC 8446 (TLS 1.3)",
    "TLS_DEPREC": "RFC 8996 (Deprecate TLS 1.0/1.1)",
    "OQS": "OQS provider hybrid groups",
    "SHOR": "Shor's algorithm",
    "SIKE": "SIKE (broken 2022)",
    "RAINBOW": "Rainbow (broken 2022)",
    "MCELIECE": "Classic McEliece (KEM)",
    "NTRU": "NTRU/NTRU Prime (KEM)",
    "CISA": "CISA quantum-risk (HNDL)",
    "NCSC": "UK NCSC PQC guidance",
    "OPENSSH": "OpenSSH hybrid KEX",
    "HPKE": "RFC 9180 (HPKE)",
    "KEMTLS": "KEMTLS (KEM auth)",
    "MLS": "RFC 9420 (MLS)",
    "LAMPS": "IETF LAMPS PQ cert profiles",
    "PQXDH": "Signal PQXDH",
    "QUIC": "RFC 9000 (QUIC)",

    # Aliases/extra tags used by dataset1.py generators
    "RFC791": "RFC 791 (IPv4: 4 octets)",
    "RFC1918": "RFC 1918 (private IPv4 ranges)",
    "RFC1123": "RFC 1123 (hostname LDH rules)",
    "RFC5890": "RFC 5890 (IDNA2008 A-/U-labels)",
    "IANA_PORTS": "IANA port registry (0–65535)",
    "ETSI": "ETSI quantum-safe migration guidance"
}

KB_URLS = {
    "NIST_PQC_HUB": "https://csrc.nist.gov/projects/post-quantum-cryptography",
    "FIPS203": "https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf",
    "FIPS204": "https://csrc.nist.gov/pubs/fips/204/final",
    "FIPS205": "https://csrc.nist.gov/pubs/fips/205/final",
    "NIST_PQC_FAQ": "https://csrc.nist.gov/projects/post-quantum-cryptography/faqs",
    "NISTIR8105": "https://nvlpubs.nist.gov/nistpubs/ir/2016/nist.ir.8105.pdf",
    "SP800_208": "https://csrc.nist.gov/news/2020/stateful-hash-based-signature-schemes-sp-800-208",
    "NIST_AUG2024_NEWS": "https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards",

    "NSM10": "https://irp.fas.org/offdocs/nsm/nsm-10.pdf",
    "OMB_M2302": "https://www.whitehouse.gov/wp-content/uploads/2022/11/M-23-02-M-Memo-on-Migrating-to-Post-Quantum-Cryptography.pdf",
    "CNSA2": "https://media.defense.gov/2025/May/30/2003728741/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS.PDF",
    "CISA_QR_FACTS": "https://www.cisa.gov/resources-tools/resources/quantum-readiness-factsheet",
    "WH_PQC_2024_BG": "https://bidenwhitehouse.archives.gov/wp-content/uploads/2024/07/REF_PQC-Report_FINAL_Send.pdf",

    "RFC8391": "https://datatracker.ietf.org/doc/html/rfc8391",
    "RFC8554": "https://www.rfc-editor.org/rfc/rfc8554.html",
    "RFC8784": "https://www.rfc-editor.org/rfc/rfc8784.pdf",
    "RFC9370": "https://datatracker.ietf.org/doc/html/rfc9370",
    "TLS_AUTHKEM_DRAFT": "https://datatracker.ietf.org/doc/draft-celi-wiggers-tls-authkem/",
    "LAMPS_COMP_SIGS": "https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-sigs/",
    "LAMPS_COMP_KEM": "https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-kem/",
    "PQUIP_TRACKER": "https://github.com/ietf-wg-pquip/state-of-protocols-and-pqc",
    "CNSA1_TLS": "https://datatracker.ietf.org/doc/rfc9151/",
    "CNSA1_IPSEC": "https://datatracker.ietf.org/doc/rfc9206/",
    "CNSA1_SSH": "https://datatracker.ietf.org/doc/rfc9212/",

    "ETSI_QSC": "https://www.etsi.org/technologies/quantum-safe-cryptography",
    "ENISA_PQC": "https://www.enisa.europa.eu/publications/post-quantum-cryptography/",
    "BSI_TR_02102_1": "https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/TechGuidelines/TG02102/BSI-TR-02102-1.html",
    "NCSC_UK_PQC": "https://www.ncsc.gov.uk/whitepaper/next-steps-preparing-for-post-quantum-cryptography",
    "ANSSI_PQC": "https://cyber.gouv.fr/post-quantum-cryptography",
    "CCCS_PQC": "https://www.cyber.gc.ca/en/guidance/quantum-safe-guidance-gs-quantum-securise",
    "SG_CSA_PQC": "https://www.csa.gov.sg/insights-news/insights/quantum-safe-cryptography-standards-adoption",
    "AU_ACSC_PQC": "https://www.cyber.gov.au/resources-business-and-government/hardening-your-organisation/cyber-security-guidance/preparing-post-quantum-cryptography",
    "CRYPTREC_PQC": "https://www.cryptrec.go.jp/en/pqcrypto.html",
    "ISO_14888_4": "https://www.iso.org/standard/80492.html",

    "NCCOE_MPQC": "https://www.nccoe.nist.gov/crypto-agility-considerations-migrating-post-quantum-cryptographic-algorithms",
    "NCCOE_FACT": "https://www.nccoe.nist.gov/sites/default/files/2023-08/mpqc-fact-sheet.pdf",
    "SP1800_38A": "https://www.nccoe.nist.gov/sites/default/files/2023-04/pqc-migration-nist-sp-1800-38a-preliminary-draft.pdf",
    "SP1800_38B": "https://www.nccoe.nist.gov/sites/default/files/2023-12/pqc-migration-nist-sp-1800-38b-preliminary-draft.pdf",
    "SP1800_38C": "https://www.nccoe.nist.gov/sites/default/files/2023-12/pqc-migration-nist-sp-1800-38c-preliminary-draft.pdf",
    "NCCOE_WHITEPAPER": "https://csrc.nist.rip/publications/detail/white-paper/2021/08/04/migration-to-post-quantum-cryptography/final",

    "OQS_SITE": "https://openquantumsafe.org/",
    "LIBOQS": "https://github.com/open-quantum-safe/liboqs",
    "OQS_PROVIDER": "https://github.com/open-quantum-safe/oqs-provider",
    "PQCLEAN": "https://github.com/PQClean/PQClean",
    "BOUNCY_PQC": "https://www.bouncycastle.org/specifications.html#pqc",
    "XMSS_REF": "https://github.com/XMSS/xmss-reference",
    "OPENSSH_9": "https://www.openssh.com/txt/release-9.0",

    "CHROMIUM_HYBRID": "https://blog.chromium.org/2023/08/protecting-chrome-traffic-with-hybrid.html",
    "CF_PQC_GA": "https://blog.cloudflare.com/post-quantum-cryptography-ga/",
    "CF_PQC_ORIG": "https://blog.cloudflare.com/post-quantum-to-origins/",
    "CF_PQ_2024": "https://blog.cloudflare.com/pq-2024/",
    "MS_PQC_2024": "https://techcommunity.microsoft.com/blog/microsoft-security-blog/microsofts-quantum-resistant-cryptography-is-here/4238780",
    "MS_PQC_2025": "https://www.microsoft.com/en-us/security/blog/2025/08/20/quantum-safe-security-progress-towards-next-generation-cryptography/",
    "APPLE_TLS26": "https://support.apple.com/en-us/122756",
    "AKAMAI_PQC_2025": "https://www.akamai.com/blog/security/post-quantum-cryptography-implementation-considerations-tls",

    "CABF_SMC013": "https://cabforum.org/2025/07/02/ballot-smc-013/",
    "KYBER_SPEC": "https://pq-crystals.org/kyber/data/kyber-specification-round3-20210131.pdf",
    "DILITHIUM_SPEC": "https://pq-crystals.org/dilithium/data/dilithium-specification-round3-20210208.pdf",
    "PQC_BOOK": "https://link.springer.com/book/10.1007/978-3-540-88702-7",

    "ACVP_HOME": "https://pages.nist.gov/ACVP/",
    "ACVP_SERVER": "https://github.com/usnistgov/ACVP-Server/releases",
    "CAVP_SAMPLE": "https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/details?product=19778",
    "WOLFSSL_CAVP": "https://www.wolfssl.com/ml-kem-and-ml-dsa-at-the-cavp/",
    "ATSEC_FIRST": "https://www.atsec.com/first-post-quantum-cryptographic-algorithm-certificates-published/",

    # Extra URLs for dataset1 aliases
    "RFC791": "https://www.rfc-editor.org/rfc/rfc791.html",
    "RFC1918": "https://www.rfc-editor.org/rfc/rfc1918.html",
    "RFC1123": "https://www.rfc-editor.org/rfc/rfc1123.html",
    "RFC5890": "https://www.rfc-editor.org/rfc/rfc5890.html",
    "IANA_PORTS": "https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml",
    "TLS13": "https://www.rfc-editor.org/rfc/rfc8446.html",
    "TLS_DEPREC": "https://www.rfc-editor.org/rfc/rfc8996.html",
    "ETSI": "https://www.etsi.org/technologies/quantum-safe-cryptography"
}

# ---------- TLS / crypto catalogs ----------
CLASSICAL_SIGN = [
    ("RSA", [1024, 2048, 3072, 4096, 6144, 8192]),
    ("ECDSA", ["P-256", "P-384", "P-521", "brainpoolP256r1", "brainpoolP384r1"]),
    ("DSA", [2048, 3072]),
    ("Ed25519", None),
    ("Ed448", None)
]
CLASSICAL_KEX = [
    ("ECDH", ["P-256", "P-384", "P-521", "brainpoolP256r1", "brainpoolP384r1"]),
    ("X25519", None),
    ("X448", None),
    ("DH", ["Group 14", "Group 15", "Group 16", "Group 17", "Group 18", "Group 19", "Group 20", "Group 21", "Group 24"])
]
PQC_KEM = [("ML-KEM", ["512", "768", "1024"], ["Kyber-512", "Kyber-768", "Kyber-1024"])]
PQC_SIG = [
    ("ML-DSA", ["2","3","5"], ["Dilithium2","Dilithium3","Dilithium5"]),
    ("SLH-DSA", ["128s","128f","192s","256s"], ["SPHINCS+ 128s","SPHINCS+ 128f","SPHINCS+ 192s","SPHINCS+ 256s"]),
    ("Falcon", ["512","1024"], ["Falcon-512","Falcon-1024"])
]
TLS13_SUITES = [
    "TLS_AES_128_GCM_SHA256",
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_CCM_SHA256",
    "TLS_AES_128_CCM_8_SHA256"
]
TLS12_SUITES = [
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
]
HYB_UNDERSCORE = [
    "x25519_kyber768","p256_kyber768","x448_kyber1024","p384_kyber1024",
    "x25519_mlkem768","p256_mlkem768","x448_mlkem1024","p384_mlkem1024"
]
HYB_CONCAT = [
    "X25519MLKEM768","SecP256r1MLKEM768","X448MLKEM1024","SecP384r1MLKEM1024"
]
HYB_PLUS = [
    "X25519+ML-KEM-768","P-256+ML-KEM-768","X448+ML-KEM-1024","P-384+ML-KEM-1024"
]
HYBRID_ALL = HYB_UNDERSCORE + HYB_CONCAT + HYB_PLUS
SYMMETRIC_AEAD = ["AES-128-GCM", "AES-256-GCM", "ChaCha20-Poly1305", "AES-128-CCM", "AES-128-CCM-8"]
SYMMETRIC_OLD = ["3DES-CBC", "DES-CBC", "RC4"]
HASHES = ["SHA-256", "SHA-384", "SHA-512", "SHA-1 (deprecated)", "MD5 (broken)"]

# ---------- Validators ----------
_octet = r"(25[0-5]|2[0-4]\d|1?\d?\d)"
IPv4_RE = re.compile(rf"^{_octet}\.{_octet}\.{_octet}\.{_octet}$")
def is_valid_ipv4(ip: str) -> bool:
    return IPv4_RE.match(ip) is not None

# --- Helper: RFC1918 private IPv4 check (drop-in) ---
def is_private_ipv4(ip: str) -> bool:
    """
    Return True iff ip is syntactically valid IPv4 and within RFC 1918 ranges:
    10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16.
    (Does not include loopback/link-local/CGNAT.)
    """
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        octets = list(map(int, parts))
        if not all(0 <= x <= 255 for x in octets):
            return False
        a, b, c, d = octets
    except Exception:
        return False

    if a == 10:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    return False


HOST_LABEL_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
def is_valid_hostname(host: str) -> bool:
    labels = host.rstrip(".").split(".")
    if not labels: return False
    if sum(len(l) for l in labels) + (len(labels)-1) > 253: return False
    return all(HOST_LABEL_RE.match(l) for l in labels)

def is_valid_port(p) -> bool:
    try:
        x = int(p); return 0 <= x <= 65535
    except Exception:
        return False

# ---------- Blacklist to avoid scanner copy-paste ----------
SCANNER_BLACKLIST = [
    # union of both scripts
    re.compile(r"Protocol\s*:\s*TLS\s*1\.3.*Key\s*exchange\s*:\s*(?:X25519MLKEM768|X25519Kyber768).*Server\s*signature\s*:\s*ECDSA\s+with\s+SHA-256.*Cipher\s*:\s*AES_128_GCM", re.I|re.S),
    re.compile(r"\bProtocol\s+TLS\s*1\.3.*Key\s*exchange\s+(?:X25519MLKEM768|X25519Kyber768).*Server\s*signature\s+ECDSA\s+with\s+SHA-256.*Cipher\s+AES_128_GCM", re.I|re.S),
    re.compile(r"Protocol\s*:\s*TLS\s*1\.3.*Key\s*exchange\s*:\s*X25519MLKEM768.*Server\s*signature\s*:\s*ECDSA\s+with\s+SHA-256.*Cipher\s*:\s*AES_128_GCM", re.I|re.S),
    re.compile(r"\bProtocol\s+TLS\s*1\.3.*Key\s*exchange\s+X25519MLKEM768.*Server\s*signature\s+ECDSA\s+with\s+SHA-256.*Cipher\s+AES_128_GCM", re.I|re.S),
]

# ---------- Paraphrasing ----------
CODE_TOKEN_RE = re.compile(
    r"(\bTLS_[A-Z0-9_]+\b|"
    r"\bX[0-9]{3,4}MLKEM[0-9]{3,4}\b|"
    r"\bSecP[0-9]+r1MLKEM[0-9]{3,4}\b|"
    r"\b[A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b)"
)

WRAPPER_PREFIX_RE = re.compile(
    r"^\s*(?:please\s*advise\s*:|pls\s*advise\s*:|given\s*this\s*:|in\s*short\s*:|question\s*:)\s*",
    re.IGNORECASE,
)

INSTR_SYNS = [
    (r"\bPQC\b", "post-quantum"),
    (r"\bpost-quantum\b", "quantum-resistant"),
    (r"\bquantum-resistant\b", "post-quantum"),
    (r"\bquantum-safe\b", "quantum-secure"),
    (r"\bquantum-secure\b", "quantum-safe"),
    (r"\bconfiguration\b", "setup"),
    (r"\bsetup\b", "configuration"),
    (r"\buses\b", "employs"),
    (r"\butilizes\b", "uses"),
    (r"\bis using\b", "uses"),
    (r"\bcompliant\b", "meeting requirements"),
    (r"\bresistant\b", "resilient"),
    (r"\bsecurity\b", "safety"),
    (r"\bWhat changes are needed\b", "What should we change"),
    (r"\bquantum attacks\b", "quantum threats"),
    (r"\battacks by quantum computers\b", "quantum threats"),
]

def _structural_variants(text: str) -> list[str]:
    """
    Disabled: avoid wrapper phrases like 'Please advise:', 'Given this:', 'In short:', or 'Question:'.
    Just return the text as-is.
    """
    return [text.strip()]

def _strip_wrappers(s: str) -> str:
    """
    Remove any leading wrapper phrase(s) repeatedly, e.g.:
    'Given this: Given this: ...' or mixed-case/spacing variants.
    """
    t = s.lstrip()
    while True:
        m = WRAPPER_PREFIX_RE.match(t)
        if not m:
            break
        t = t[m.end():].lstrip()
    return t


def safe_paraphrase(text: str, heavy_prob: float = 0.35) -> str:
    # Structural rewrites are disabled to avoid wrapper prefixes.
    text = _strip_wrappers(text)
    # Light paraphrasing: replace 1–2 synonyms, skipping code tokens
    parts = CODE_TOKEN_RE.split(text)
    out_segments = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:
            out_segments.append(seg)
        else:
            new_seg = seg
            k = random.randint(1, min(2, len(INSTR_SYNS)))
            for pattern, replacement in random.sample(INSTR_SYNS, k=k):
                new_seg = re.sub(pattern, replacement, new_seg)
            out_segments.append(new_seg)
    t = "".join(out_segments)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r"\s+\?", "?", t)
    if not t.endswith("?"):
        t = t.rstrip(".") + "?"
    return _strip_wrappers(t)


def fix_instruction_grammar(q: str) -> str:
    t = _strip_wrappers(q.strip())
    t = re.sub(r"^Does this(?!\s+meet\b)", "Is this", t, flags=re.I)
    t = re.sub(r"\bIs this\s+PQC[- ]?safe\b", "Is this PQC safe", t, flags=re.I)
    t = re.sub(r"\bIs this\s+post-quantum\s+compliant\b", "Is this post-quantum compliant", t, flags=re.I)
    t = re.sub(r"\?+$", "?", t)
    if not t.endswith("?"):
        t += "?"
    return t


def qa_item(q: str, a: str, just: list[str], refs: list[str], style: str = "mixed") -> dict:
    """Builds a QA item; supports bulleted answers to increase uniqueness."""
    style = STYLE if style == "mixed" else style
    s = style
    if s == "mixed":
        s = random.choice(["plain", "bulleted"])
    if s == "bulleted":
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", a) if p.strip()]
        bullets = ["• " + p.rstrip(".") for p in parts[:3]]
        a_fmt = "\n".join(bullets) if len(bullets) >= 2 else a
    else:
        a_fmt = a
    return {
        "instruction": q,
        "context": {"kb_refs": refs},
        "response": {
            "mode": "QA",
            "justification": just[:3],
            "answer": a_fmt,
            "safety": {"private_keys_access": "forbidden", "policy_notes": ""}
        }
    }

# ---------- Dedupe helpers ----------
WS_COLLAPSE_RE = re.compile(r"\s+")
PUNCT_FOLD_RE = re.compile(r"[“”\"'`‘’]+|[—–]+")
ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b", re.I)
COMMA_NUMBER_RE = re.compile(r"(\d),(?=\d{3}\b)")

def norm_instruction(s: str) -> str:
    t = s.strip().lower()
    t = ARTICLES_RE.sub(" ", t)
    t = COMMA_NUMBER_RE.sub(r"\1", t)     # 4,096 -> 4096
    t = PUNCT_FOLD_RE.sub("", t)
    t = WS_COLLAPSE_RE.sub(" ", t).strip()
    return t

# Lightweight near-dup filter (token 3-grams with keyed buckets)
TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-/.]+")
def _token_trigrams(text: str) -> set[tuple]:
    toks = TOKEN_RE.findall(text.lower())
    if len(toks) < 3:
        return set([(tuple(toks),)])
    return set(tuple(toks[i:i+3]) for i in range(len(toks)-2))

def _near_key(text: str) -> tuple:
    toks = TOKEN_RE.findall(text.lower())
    first = toks[0] if toks else ""
    second = toks[1] if len(toks) > 1 else ""
    last = toks[-1] if toks else ""
    return (first, second, last)

def _similar_enough(grams_a: set, grams_b: set) -> bool:
    if not grams_a or not grams_b:
        return False
    inter = len(grams_a & grams_b)
    if inter == 0:
        return False
    contain = inter / float(min(len(grams_a), len(grams_b)))
    jacc = inter / float(len(grams_a | grams_b))
    return contain >= 0.90 or jacc >= 0.85

# ---------- Convenience text ----------
AES128_NOTE = "AES-128 is acceptable today (~2^64 work under Grover); prefer AES-256-GCM for long-term safety."
AES256_NOTE = "AES-256-GCM offers a high margin (≈2^128 under Grover)."
HYBRID_HINT = "Use TLS 1.3 with a hybrid key_share (e.g., X25519+Kyber-768)."

# ---------- Seed Generators (kept/broadened) ----------


def g_generic_pqc():
    # --- Strong 5-variant PQC definition seeds (richer kb_refs, no wrappers) ---
    _defs_refs = ["SHOR","FIPS203","FIPS204","FIPS205","NIST_PQC_FAQ","NIST_PQC_HUB","NISTIR8105","NIST_AUG2024_NEWS"]
    _defs_just = ["PQC remains secure vs Shor", "Kyber for key establishment", "Dilithium/SPHINCS+ for signatures"]
    _def_pairs = [
        (
            "What is post-quantum cryptography (PQC)?",
            "Post-quantum cryptography (PQC) is a family of public-key algorithms designed to remain secure even against large, fault-tolerant quantum computers. "
            "In practice, use ML-KEM (Kyber) for key establishment and ML-DSA (Dilithium) or SLH-DSA (SPHINCS+) for digital signatures."
        ),
        (
            "Explain PQC for someone rolling out TLS, VPN, and code signing.",
            "Think: KEM for key agreement + signature for authentication. For web/TLS and VPNs, use Kyber (ML-KEM), ideally in a hybrid with a classical curve during transition. "
            "For certificates, code signing, and S/MIME, use Dilithium or SPHINCS+. This protects forward secrecy and authentication once large quantum computers arrive."
        ),
        (
            "Is 'quantum-resistant cryptography' the same as PQC—what does it cover?",
            "Yes—PQC means public-key schemes designed to resist quantum attacks like Shor. Use Kyber for key establishment and Dilithium/SPHINCS+ for signatures; "
            "deploy hybrids first to stay compatible while you migrate."
        ),
        (
            "Where should we use PQC first?",
            "Start at the transport/PKI boundary: enable hybrid key exchange in TLS 1.3 and IKEv2 so sessions gain quantum-resilient forward secrecy, then roll out PQC signatures for X.509, S/MIME, and code signing. "
            "Keep modern AEADs; prefer AES-256-GCM for long-term margin."
        ),
        (
            "How is PQC different from quantum key distribution (QKD)?",
            "PQC is software-based cryptography (algorithms like Kyber/Dilithium) that runs over today’s networks. QKD is a hardware approach over quantum channels. "
            "They solve different problems; standards and policy focus on PQC for broad interoperability."
        ),
    ]
    for q, a in _def_pairs:
        yield qa_item(q, a, _defs_just, _defs_refs)

    # --- Your existing seeds (kept) ---
    defs = [
        "What is post-quantum cryptography (PQC)?",
        "What is PQC and why do we need it?",
        "What is PQC?",
        "What is Post Quantum Cryptography?",
        "What does 'quantum-resistant cryptography' mean in practice?",
        "Explain PQC like I’m a developer integrating TLS and code signing.",
        "Is PQC part of modern cryptography or just a research topic?"
    ]
    why_now = [
        "Why do organizations need PQC now if large quantum computers don’t exist yet?",
        "What is 'harvest now, decrypt later' and how does it change our timelines?",
        "What risks do quantum computers pose to RSA/ECC and when should we act?"
    ]
    where_used = [
        "Where is PQC used in practice (TLS, VPNs, SSH, email, code signing)?",
        "Which parts of a typical stack adopt PQC first?",
        "Do we need PQC for both transport (TLS/VPN) and PKI (certificates/signatures)?"
    ]
    algos = [
        "Which PQC algorithms should we plan for (KEMs and signatures)?",
        "What are Kyber, Dilithium, and SPHINCS+ used for?",
        "Are there alternatives like Falcon or Classic McEliece we should know about?"
    ]
    migration = [
        "What’s a sensible migration plan from RSA/ECC to PQC?",
        "How do hybrids help us move from classical to PQC without breaking clients?",
        "What guidance should we follow for a phased PQC rollout?"
    ]
    tradeoffs = [
        "What are the performance and size trade-offs when adopting PQC?",
        "Will TLS handshakes or certificates get much larger with PQC?",
        "Do symmetric ciphers like AES need to change for PQC?"
    ]
    qkd = [
        "Is quantum key distribution (QKD) the same as PQC?",
        "Do we need QKD hardware if we adopt PQC?",
    ]
    iot_kms_pki = [
        "How should constrained IoT devices approach PQC for signatures?",
        "What is the PQC alternative to RSA key wrapping in KMS/HSM systems?",
        "How do we make our PKI ready for PQC certificates?"
    ]

    a_defs = ("Post-quantum cryptography (PQC) is a family of public-key algorithms designed to remain secure even against "
              "large, fault-tolerant quantum computers. A practical baseline is ML-KEM (Kyber) for key establishment and "
              "ML-DSA (Dilithium) or SLH-DSA (SPHINCS+) for digital signatures.")
    a_why_now = ("Timelines are uncertain and migrations take years. The 'harvest now, decrypt later' threat means adversaries "
                 "can record traffic today and decrypt it once quantum capabilities arrive. Start deploying hybrid key exchanges "
                 "and plan for PQC signatures to protect long-lived data.")
    a_where_used = ("Start at the transport boundary and in PKI: enable TLS 1.3 hybrid key_share groups (e.g., X25519+ML-KEM-768) "
                    "for forward secrecy, then introduce PQC signatures (Dilithium/SPHINCS+) for certificates, S/MIME, and code signing. "
                    "VPNs (IKEv2), SSH, KMS/HSM wrapping, and WebAuthn will follow as tooling matures.")
    a_algos = ("Use ML-KEM (Kyber) for KEM/key establishment and ML-DSA (Dilithium) or SLH-DSA (SPHINCS+) for signatures. "
               "Falcon-512 is attractive where tiny signatures and fast verification matter. Alternatives like Classic McEliece offer "
               "strong security with very large public keys; NTRU/NTRU Prime are lattice options seen in some hybrids.")
    a_migration = ("Adopt TLS 1.3; enable hybrid key_share groups so sessions are quantum-resilient when both endpoints support them. "
                   "Introduce PQC signatures in certificates alongside classical (dual-sign/dual-path) during transition. "
                   "Follow national and standards guidance (CNSA 2.0, NCSC/CISA, ETSI).")
    a_tradeoffs = ("Expect larger public keys/ciphertexts and higher CPU for KEM ops; certificates grow when you switch signature "
                   "algorithms to PQC. Symmetric ciphers largely remain the same: AES-128 is still acceptable (~2^64 under Grover), "
                   "but prefer AES-256-GCM for long-term margin.")
    a_qkd = ("QKD is a hardware-based key exchange using quantum channels; PQC is software-based encryption/signatures that run "
             "over today’s networks. They address different problems—PQC is the practical path to protect most systems.")
    a_iot = ("Constrained devices should consider Falcon-512 for compact signatures and fast verification, or Dilithium2 when policy "
             "prefers standardized simplicity at the cost of larger signatures. For key wrapping, replace RSA with ML-KEM (Kyber) "
             "encapsulation in KMS/HSM workflows; prepare PKI to issue and verify PQC certificates.")

    for q in defs:
        yield qa_item(q, a_defs,
                      ["PQC remains secure vs Shor", "Kyber for key establishment", "Dilithium/SPHINCS+ for signatures"],
                      ["SHOR","FIPS203","FIPS204","FIPS205","NIST_PQC_FAQ"])
    for q in why_now:
        yield qa_item(q, a_why_now,
                      ["Harvest-now/decrypt-later risk", "Migrations take years", "Use hybrids first"],
                      ["CISA","NCSC","CNSA2","NIST_PQC_FAQ"])
    for q in where_used:
        yield qa_item(q, a_where_used,
                      ["TLS 1.3 hybrid key_share first", "Add PQC signatures in PKI", "Extend to VPN/SSH/KMS/WebAuthn"],
                      ["TLS13","OQS","FIPS203","FIPS204","FIPS205"])
    for q in algos:
        yield qa_item(q, a_algos,
                      ["Kyber (ML-KEM) for KEX", "Dilithium/SPHINCS+ for signatures", "Falcon/NTRU/McEliece context"],
                      ["FIPS203","FIPS204","FIPS205","NTRU","MCELIECE"])
    for q in migration:
        yield qa_item(q, a_migration,
                      ["TLS 1.3 + hybrids", "Dual-sign/dual-path transition", "Follow CNSA/ETSI/NCSC guidance"],
                      ["TLS13","CNSA2","ETSI","NCSC"])
    for q in tradeoffs:
        yield qa_item(q, a_tradeoffs,
                      ["PQC increases sizes/CPU", "Certs grow with PQC signatures", "Prefer AES-256-GCM long-term"],
                      ["NIST_PQC_FAQ","TLS13","FIPS204","FIPS205"])
    for q in qkd:
        yield qa_item(q, a_qkd,
                      ["QKD ≠ PQC", "PQC is software-deployable", "Use PQC for mainstream systems"],
                      ["FIPS203","FIPS204","FIPS205"])
    for q in iot_kms_pki:
        yield qa_item(q, a_iot,
                      ["Falcon-512 for constrained devices", "Kyber for DEK encapsulation", "Prepare PKI for PQC"],
                      ["FIPS203","FIPS204","FIPS205"])


def g_tls_configs():
    # 5 prompt variants per config; plus diversified verdict phrasing to reduce template feel.
    ask_variants = [
        "Is this post-quantum compliant?",
        "Is this PQC-safe?",
        "Is this configuration quantum-resistant?",
        "Does this meet post-quantum requirements?",
        "Would this resist quantum attacks?"
    ]
    verdict_phrasings = [
        "Not PQC-compliant: {kx} is classical. Migrate to TLS 1.3 with a hybrid key_share (e.g., {hybrid}). {tip}",
        "Quantum-vulnerable: {kx} is a classical KEX broken by Shor. Use TLS 1.3 with a hybrid key_share (e.g., {hybrid}). {tip}",
        "Not quantum-safe: {kx} is classical; deploy TLS 1.3 hybrids (e.g., {hybrid}) and keep modern AEADs. {tip}",
        "{kx} is classical and does not provide quantum resistance. Enable TLS 1.3 hybrid groups (e.g., {hybrid}). {tip}",
        "This setup is not PQC-ready because {kx} is classical. Move to TLS 1.3 hybrids (e.g., {hybrid}). {tip}",
    ]
    aes_tips = [
        "AES-128 remains acceptable (~2^64 under Grover), but prefer AES-256-GCM for future-proofing.",
        "Prefer AES-256-GCM for long-term security; AES-128-GCM is still acceptable today.",
        "Symmetric encryption is less affected by quantum attacks; nevertheless, AES-256-GCM is advisable."
    ]

    for fam, groups in CLASSICAL_KEX:
        if groups:
            for g in groups:
                for suite, ask in itertools.product(TLS13_SUITES + TLS12_SUITES, ask_variants):
                    kx = f"{fam} {g}"
                    is_tls13 = suite in TLS13_SUITES
                    FFDHE_MAP = {
                        "Group 14": "ffdhe2048",
                        "Group 15": "ffdhe3072",
                        "Group 16": "ffdhe4096",
                        "Group 17": "ffdhe6144",
                        "Group 18": "ffdhe8192"
                    }
                    # Normalize DH names in TLS 1.3 context
                    if fam == "DH" and is_tls13:
                        if g in FFDHE_MAP:
                            kx = f"DHE {FFDHE_MAP[g]}"
                        else:
                            continue

                    hybrid_suggestion = random.choice(HYBRID_ALL)
                    aes_tip = random.choice(aes_tips)
                    q = f"Our TLS uses {kx} with {suite}. {ask}"
                    verdict = random.choice(verdict_phrasings)
                    a = verdict.format(kx=kx, hybrid=hybrid_suggestion, tip=aes_tip)

                    just = ["Classical key exchange is quantum-vulnerable", "Use TLS 1.3 hybrid key_share", "Prefer AES-256-GCM"]
                    refs = ["TLS13", "TLS_DEPREC", "OQS", "NIST_PQC_FAQ"]
                    yield qa_item(q, a, just, refs)
        else:
            # KEX families without explicit group lists (e.g., X25519)
            for suite, ask in itertools.product(TLS13_SUITES + TLS12_SUITES, ask_variants):
                kx = fam
                hybrid_suggestion = random.choice(HYBRID_ALL)
                aes_tip = random.choice(aes_tips)
                q = f"Our TLS uses {kx} with {suite}. {ask}"
                verdict = random.choice(verdict_phrasings)
                a = verdict.format(kx=kx, hybrid=hybrid_suggestion, tip=aes_tip)

                just = ["Classical key exchange (not quantum-safe)", "Switch to TLS 1.3 hybrid key_share", "Use AES-256-GCM"]
                refs = ["TLS13", "TLS_DEPREC", "OQS", "NIST_PQC_FAQ"]
                yield qa_item(q, a, just, refs)


def g_tls_canonical():
    # Suite-specific notes for TLS 1.3 (for richer, non-duplicative answers)
    def note_tls13(suite: str) -> str:
        if suite == "TLS_AES_256_GCM_SHA384":
            return "Strong default with larger symmetric margin; widely accelerated on modern CPUs."
        if suite == "TLS_AES_128_GCM_SHA256":
            return "Widely deployed and fast; acceptable today, but prefer 256-bit keys for long-term safety."
        if suite == "TLS_CHACHA20_POLY1305_SHA256":
            return "Excellent on devices without AES acceleration; constant-time software performance is strong."
        if suite == "TLS_AES_128_CCM_SHA256":
            return "Standards-compliant AEAD; typically slower than GCM in common stacks."
        if suite == "TLS_AES_128_CCM_8_SHA256":
            return "Shortened tag variant for constrained contexts; avoid as a general default."
        return "AEAD suite defined by TLS 1.3."

    # 5 prompt templates for TLS 1.3
    tmpl13 = [
        "Our TLS 1.3 uses {kx} with {suite}. Is this post-quantum compliant?",
        "Is {suite} a good choice for TLS 1.3 if our key exchange is {kx}?",
        "TLS 1.3 stack: {kx} + {suite}. Does this meet PQC requirements?",
        "With TLS 1.3 ({suite}) and {kx} key exchange, are we quantum-safe?",
        "For a PQC migration, is {suite} acceptable if the KEX is {kx}?"
    ]
    # 5 prompt templates for TLS 1.2
    tmpl12 = [
        "We’re on TLS 1.2 using {kx} and {suite}. Is this quantum-safe during transition?",
        "TLS 1.2 config: {kx} + {suite}. Can we rely on this until we roll out PQC?",
        "Is {suite} in TLS 1.2 acceptable for PQC if our key exchange is {kx}?",
        "Does TLS 1.2 with {kx} and {suite} satisfy post-quantum guidance?",
        "While migrating, is TLS 1.2 ({suite}) with {kx} adequate against quantum risks?"
    ]

    # Answer variant banks (selected randomly to add diversity while staying faithful)
    a13_variants = [
        lambda kx, suite: (
            f"Not fully PQC-compliant: {kx} is a classical Diffie-Hellman exchange vulnerable to Shor’s algorithm. "
            f"Keep the AEAD suite ({note_tls13(suite)}). To gain quantum resilience, enable a TLS 1.3 hybrid key_share "
            f"(e.g., X25519+ML-KEM-768 or p256_mlkem768) so secrets derive from both classical and Kyber. "
            "Next, adopt PQC certificate signatures (Dilithium or SPHINCS+) to remove the classical authentication weak link."
        ),
        lambda kx, suite: (
            f"{kx} is classical (not quantum-safe). Your cipher choice {suite} is fine as an AEAD, but PQC readiness depends on key exchange "
            "and certificate signatures. Enable hybrid groups (e.g., X25519+ML-KEM-768 / p256_mlkem768) and plan PQC certs."
        ),
        lambda kx, suite: (
            f"No—{kx} alone doesn’t provide post-quantum security. Keep {suite} (AEAD); add a hybrid key_share so the KDF mixes secrets from "
            "both the classical curve and Kyber. Phase in PQC certificate signatures to close the remaining gap."
        ),
        lambda kx, suite: (
            f"Partially modern, but not PQC-safe: {kx} remains classical. Deploy TLS 1.3 hybrids (e.g., X25519+ML-KEM-768) now; "
            "prefer AES-256-GCM long-term and move certificates to Dilithium/SPHINCS+ when supported."
        ),
    ]
    a12_variants = [
        lambda kx, suite: (
            f"{suite} is a solid AEAD, but TLS 1.2 has no standardized hybrid/PQC key exchange. Treat {kx}+{suite} as **interim only**. "
            "Migrate to TLS 1.3 and enable hybrid key_share (e.g., X25519+ML-KEM-768); plan PQC certificate signatures."
        ),
        lambda kx, suite: (
            f"Not PQC-ready: {kx} is classical and TLS 1.2 lacks hybrid groups. Keep {suite} during transition, "
            "but prioritize moving to TLS 1.3 hybrids and PQC certificate paths."
        ),
        lambda kx, suite: (
            f"Short answer: no. {kx} is classical and TLS 1.2 can’t negotiate Kyber hybrids. Upgrade to TLS 1.3 with hybrid key_share; "
            "retain AEADs (AES-256-GCM or ChaCha20-Poly1305) and add PQC signatures."
        ),
        lambda kx, suite: (
            f"Only acceptable as a stop-gap. PQC requires hybrid KEX and PQC signatures, neither standardized in TLS 1.2. "
            "Move to TLS 1.3 hybrids (e.g., X25519+ML-KEM-768) and modern AEADs."
        ),
    ]

    # Canonical classical KEX labels to call out explicitly
    kx13 = ["X25519", "ECDH P-256", "ECDH P-384"]
    kx12 = ["ECDHE with P-256", "ECDHE with P-384", "X25519"]

    # Emit TLS 1.3 variants (5 prompt variants × multiple answer phrasings)
    for kx in kx13:
        for suite in TLS13_SUITES:
            for t in tmpl13:
                q = t.format(kx=kx, suite=suite)
                a = random.choice(a13_variants)(kx, suite)
                yield qa_item(q, a,
                              ["Classical KEX breaks under Shor", "Use TLS 1.3 hybrid key_share", "Keep AEAD; prefer AES-256-GCM long-term"],
                              ["SHOR","TLS13","OQS","FIPS203","FIPS204","FIPS205","NIST_PQC_FAQ"])

    # Emit TLS 1.2 variants (clear 'interim only' guidance; 5 prompts × multiple answers)
    for kx in kx12:
        for suite in TLS12_SUITES:
            for t in tmpl12:
                q = t.format(kx=kx, suite=suite)
                a = random.choice(a12_variants)(kx, suite)
                yield qa_item(q, a,
                              ["TLS 1.2 lacks hybrid KEX", "Move to TLS 1.3 hybrids", "Use strong AEAD suites"],
                              ["TLS13","TLS_DEPREC","OQS","NIST_PQC_FAQ","FIPS203"])

    # General seeds (5 questions; short, high-signal; answers are stable + precise)
    generals = [
        ("Is AES-128-GCM acceptable in a PQC-ready TLS 1.3 stack?",
         "Yes, today—Grover’s algorithm yields ~2^64 effective work for 128-bit keys. For long-term margin, prefer AES-256-GCM. "
         "PQC readiness is determined by negotiating a hybrid key_share and, later, using PQC certificate signatures."),
        ("Does switching to ChaCha20-Poly1305 make TLS 1.3 post-quantum secure?",
         "Cipher choice doesn’t make a handshake PQC-safe by itself. Post-quantum readiness comes from the key exchange "
         "(hybrid groups like X25519+ML-KEM-768) and the certificate signature algorithm."),
        ("If we already run TLS 1.3, do we still need hybrids?",
         "Yes. TLS 1.3 fixes legacy issues and mandates AEADs, but its classical KEX (X25519/ECDH) is not quantum-safe. "
         "Enable hybrid key_share groups now; add PQC signatures as ecosystem support matures."),
        ("Does TLS_AES_256_GCM_SHA384 provide better post-quantum safety than TLS_AES_128_GCM_SHA256?",
         "Both are secure today; AES-256-GCM offers a larger symmetric margin against Grover. Pick based on hardware performance and data lifetime."),
        ("Will enabling hybrids change our TLS record encryption?",
         "No. Hybrids affect key exchange only. Record encryption stays with your AEAD (AES-GCM/ChaCha20-Poly1305); the change is in the secrets derived by HKDF.")
    ]
    for q, a in generals:
        yield qa_item(q, a,
                      ["Grover affects symmetric margins", "PQC hinges on key_share & signatures", "TLS 1.3 ≠ PQC by default"],
                      ["NIST_PQC_FAQ","TLS13","OQS","FIPS203","FIPS204"])



def g_hybrid_details():
    # --- Overview (5 prompt variants, 3 answer variants) ---
    overview_prompts = [
        "Which hybrid TLS key exchange groups are common?",
        "List common TLS 1.3 hybrid key_share groups we can enable.",
        "What hybrid groups (classical + Kyber) exist for TLS 1.3?",
        "What are typical TLS hybrid KEX groups supported in practice?",
        "Examples of TLS 1.3 hybrid key_share names?"
    ]
    excerpt = ", ".join(HYBRID_ALL[:8]) + "…"
    overview_answers = [
        lambda: (f"Common TLS 1.3 hybrid key_share groups include {excerpt}. Each combines a classical curve "
                 "(X25519/P-256/X448/P-384) with Kyber (ML-KEM) so the KDF mixes both secrets."),
        lambda: (f"Typical hybrids: {excerpt}. They pair ECDH (X25519/P-256/X448/P-384) with ML-KEM (Kyber); the handshake "
                 "derives traffic secrets from both components."),
        lambda: (f"Expect names like {excerpt}. These negotiate a classical ECDH + Kyber KEM, giving forward secrecy even if one side is later weakened.")
    ]
    for p in overview_prompts:
        yield qa_item(
            p,
            random.choice(overview_answers)(),
            ["Hybrid = classical + PQC", "Examples of hybrid groups", "Used in TLS 1.3"],
            ["TLS13","OQS","FIPS203"]
        )

    # --- Per-group explainers (4 prompt variants, 3 answer variants per group) ---
    group_prompts = [
        'What does the TLS hybrid group "{name}" mean?',
        'Explain the TLS 1.3 hybrid key_share "{name}".',
        'In "{name}", what are the classical and PQC parts?',
        'How does the hybrid group "{name}" compose security?'
    ]
    for name in HYBRID_ALL:
        if name.lower().startswith("x25519"):
            classical_part = "X25519 (ECDH)"
        elif name.lower().startswith("x448"):
            classical_part = "X448 (ECDH)"
        elif name.lower().startswith("p256") or name.startswith("SecP256r1"):
            classical_part = "secp256r1 (P-256)"
        elif name.lower().startswith("p384") or name.startswith("SecP384r1"):
            classical_part = "secp384r1 (P-384)"
        else:
            classical_part = "a classical component"
        answers = [
            lambda: (f'"{name}" is a hybrid key_share: {classical_part} + Kyber (ML-KEM). TLS 1.3 derives secrets from both via HKDF, '
                     "so the handshake keeps forward secrecy if one part is broken."),
            lambda: (f'"{name}" combines {classical_part} with ML-KEM (Kyber). Both contributions feed the KDF; security holds unless both components fail.'),
            lambda: (f'"{name}" pairs classical ECDH ({classical_part}) with a Kyber KEM. It’s negotiated when both peers support it; otherwise TLS falls back to classical.')
        ]
        q = random.choice(group_prompts).format(name=name)
        a = random.choice(answers)()
        yield qa_item(
            q, a,
            ["Hybrid composition", f"Classical part: {classical_part}", "Kyber adds quantum resistance"],
            ["TLS13","OQS","FIPS203"]
        )

    # --- Negotiation / fallback (4 prompts, 2 answers) ---
    neg_prompts = [
        "How does negotiation work if one peer doesn't support hybrid groups?",
        "What happens if a client lacks hybrid key_share support?",
        "Hybrid groups and TLS negotiation—how is fallback handled?",
        "If hybrids fail to negotiate, what does TLS pick?"
    ]
    neg_answers = [
        "Servers negotiate a hybrid only when both sides offer it; otherwise a classical group is chosen. Keep hybrids enabled so compatible clients upgrade automatically.",
        "Hybrid requires mutual support. If absent on either side, TLS selects a classical ECDH group; leave hybrids on by default to benefit capable peers."
    ]
    for p in neg_prompts:
        yield qa_item(
            p, random.choice(neg_answers),
            ["Hybrid requires mutual support", "Graceful fallback to classical", "Enable hybrids by default"],
            ["TLS13","OQS"]
        )

    # --- Certificate sizes (4 prompts, 2 answers) ---
    cert_prompts = [
        "Do hybrid TLS groups increase certificate sizes?",
        "Will enabling hybrids make our certificates larger?",
        "Does choosing a hybrid key_share affect X.509 size?",
        "Are cert chains impacted by hybrid KEX?"
    ]
    cert_answers = [
        "No. Hybrid affects key exchange (key_share) only. Certificates grow only when you adopt PQC signatures like Dilithium or SPHINCS+.",
        "Certificate size is independent of key_share. Growth happens when you switch the certificate signature algorithm to a PQC scheme."
    ]
    for p in cert_prompts:
        yield qa_item(
            p, random.choice(cert_answers),
            ["Hybrid ≠ bigger certs", "Cert size depends on signature alg", "PQC sigs are larger"],
            ["TLS13","FIPS204","FIPS205"]
        )

    # --- MTU / handshake size (4 prompts, 2 answers) ---
    mtu_prompts = [
        "Are there MTU or handshake size pitfalls with hybrids?",
        "Will hybrids cause ClientHello/ServerHello to exceed path MTU?",
        "Hybrid key_share and packetization—what should we watch?",
        "Operational issues from larger key_share payloads?"
    ]
    mtu_answers = [
        "ClientHello/ServerHello get slightly larger. Usually fine, but watch middleboxes/MTU on constrained links; enable fragmentation and monitor handshake failures.",
        "Expect a larger key_share in the hello messages. Validate PMTUD and track failure rates; tune MSS/MTU if you observe fragmentation."
    ]
    for p in mtu_prompts:
        yield qa_item(
            p, random.choice(mtu_answers),
            ["Hybrid increases key_share bytes", "Middleboxes/MTU can bite", "Observe and tune deployment"],
            ["TLS13","OQS"]
        )

    # --- Security composition (5 prompts, 2 answers) ---
    comp_prompts = [
        "Do hybrid groups make security strictly 'the stronger of the two'?",
        "How do hybrids combine classical and Kyber secrets in practice?",
        "Does the KDF really mix both classical and PQC contributions?",
        "If one component breaks, does the session remain secure?",
        "What’s the failure model for hybrid KEX?"
    ]
    comp_answers = [
        "Security derives from combining both secrets in the KDF. Properly composed, the session resists attacks unless both components fail.",
        "Yes—the HKDF mixes both parts. The model is defense-in-depth: compromise needs both classical and PQC components to fail."
    ]
    for p in comp_prompts:
        yield qa_item(
            p, random.choice(comp_answers),
            ["KDF mixes both secrets", "Resists single-component failures", "Defense-in-depth design"],
            ["TLS13","OQS","FIPS203"]
        )

    # --- TLS 1.2 limitation (4 prompts, 2 answers) ---
    tls12_prompts = [
        "Can we use hybrid KEX with TLS 1.2?",
        "Is there a standardized hybrid/PQC KEX for TLS 1.2?",
        "Does TLS 1.2 support Kyber hybrids?",
        "Hybrid key_share in TLS 1.2—possible?"
    ]
    tls12_answers = [
        "There’s no standardized hybrid/PQC KEX for TLS 1.2. Move to TLS 1.3 and enable hybrid key_share groups.",
        "No. Hybrids are a TLS 1.3 path. Upgrade to TLS 1.3 to negotiate Kyber + ECDH."
    ]
    for p in tls12_prompts:
        yield qa_item(
            p, random.choice(tls12_answers),
            ["TLS 1.2 lacks hybrid KEX", "TLS 1.3 required", "Enable hybrid groups"],
            ["TLS13","OQS"]
        )

def g_ipsec_ikev2():
    # --- Core matrix: 5 prompt variants per (auth,kex) pair, one consistent answer ---
    core_prompts = [
        "Our IPsec VPN uses {auth} for authentication and {kex} for key exchange. Is this PQC-safe?",
        "IPsec/IKEv2 config: auth={auth}, kex={kex}. Quantum-resistant or not?",
        "Given IKEv2 with {auth} (auth) and {kex} (KEX), are we post-quantum secure?",
        "Is an IPsec setup using {auth} + {kex} compliant with PQC guidance (e.g., CNSA 2.0)?",
        "Does {auth} authentication with {kex} key exchange in IKEv2 meet post-quantum requirements?"
    ]
    auths = ["RSA-2048","RSA-3072","RSA-4096","ECDSA P-256","Ed25519"]
    kexes = ["DH Group 14","DH Group 15","DH Group 16","ECDH P-256","X25519"]

    base_answer = (
        "No. Both components are classical and fall to a large quantum computer. "
        "Adopt ML-KEM (Kyber) for key establishment—ideally as part of a hybrid—and move authentication to PQC signatures "
        "(Dilithium or SPHINCS+). For transition, use RFC 8784 (post-quantum PSKs) and RFC 9370 (multiple KEX) to compose interim hybrids."
    )
    base_just = [
        "Classical auth/KEX are quantum-vulnerable",
        "Use Kyber for key establishment",
        "Leverage RFC 8784/9370 for transition"
    ]
    base_refs = ["FIPS203","FIPS204","FIPS205","RFC8784","RFC9370","CNSA2"]

    for auth, kex in itertools.product(auths, kexes):
        for t in core_prompts:
            q = t.format(auth=auth, kex=kex)
            yield qa_item(q, base_answer, base_just, base_refs)

    # --- Focused guidance seeds (each with 4–5 prompt variants, one answer) ---

    # Bigger classical groups aren't a quantum fix
    dh_prompts = [
        "Is moving from DH Group 14 to Group 18 enough for quantum safety in IKEv2?",
        "Would upgrading DH from group 14 → 18 make our IKEv2 quantum-resistant?",
        "Do larger classical DH groups (e.g., 18) mitigate quantum risk in IPsec?",
        "If we switch IKEv2 to DH group 18, are we PQC-safe?",
        "Does raising DH group size address Shor’s algorithm in IKEv2?"
    ]
    dh_answer = (
        "No. Bigger classical groups don’t help against Shor. Use Kyber for key establishment and plan PQC signatures for authentication."
    )
    for p in dh_prompts:
        yield qa_item(p, dh_answer,
                      ["Shor breaks all classical DL", "Adopt Kyber KEM", "Plan PQC signatures"],
                      ["SHOR","FIPS203","FIPS204"])

    # RFC 8784 (PQC PSKs) interim
    psk_prompts = [
        "Can RFC 8784 post-quantum PSKs make IKEv2 quantum-resistant today?",
        "Does adding RFC 8784 PQ PSKs provide adequate quantum protection for VPNs?",
        "Are RFC 8784 PQ PSKs a practical near-term defense for IKEv2?",
        "How effective are RFC 8784 pre-shared keys against quantum threats?",
        "Should we deploy RFC 8784 right now while waiting for native KEMs?"
    ]
    psk_answer = (
        "RFC 8784 lets you inject high-entropy PSKs to add a quantum-resistant shared secret. It’s a useful interim, but long-term you should deploy standardized PQ KEMs and signatures."
    )
    for p in psk_prompts:
        yield qa_item(p, psk_answer,
                      ["PQ PSKs are interim", "Add entropy against quantum", "Aim for KEM + PQC sigs"],
                      ["RFC8784","CNSA2","FIPS203"])

    # RFC 9370 multi-KEX hybrids
    mke_prompts = [
        "How does RFC 9370 help us run hybrid IKEv2?",
        "What does RFC 9370 change for combining classical + PQ KEX in IKEv2?",
        "Can IKEv2 derive keys from both DH and a PQC KEM via RFC 9370?",
        "How do we compose a hybrid IKE_SA using RFC 9370?",
        "What’s the role of RFC 9370 in PQC migration for IPsec?"
    ]
    mke_answer = (
        "RFC 9370 allows multiple concurrent key exchanges, so you can combine a classical DH with a PQ KEM and derive keys from both."
    )
    for p in mke_prompts:
        yield qa_item(p, mke_answer,
                      ["Multi-KEX enables hybrids", "Combine classical + PQC", "Keys derive from both"],
                      ["RFC9370","CNSA2","FIPS203"])

    # Data-plane ciphers
    dp_prompts = [
        "Do we need to change data-plane ciphers in IPsec for PQC?",
        "With PQC adoption, should we replace AES-GCM on the IPsec data plane?",
        "Are data-plane AEAD choices affected by quantum threats in IPsec?",
        "For long-lived tunnels, which AEAD should we prefer under PQ considerations?",
        "Does PQC mainly affect control-plane (IKE) or also the ESP data-plane?"
    ]
    dp_answer = (
        "Usually no. Keep AEADs like AES-GCM; for long-term margin prefer AES-256-GCM. The quantum-sensitive parts are key exchange and authentication."
    )
    for p in dp_prompts:
        yield qa_item(p, dp_answer,
                      ["AEADs remain fine", "Prefer AES-256-GCM", "Focus on KEX & auth"],
                      ["NIST_PQC_FAQ","CNSA2"])

    # Pragmatic plan
    plan_prompts = [
        "What’s a pragmatic IKEv2 PQC migration plan?",
        "Give us a stepwise plan to migrate IKEv2/IPsec to PQC.",
        "How do we phase IKEv2 toward PQC without breaking peers?",
        "Recommend a rollout plan for IKEv2 PQC with minimal disruption.",
        "Our VPN fleet is large—what staged approach should we take for PQC?"
    ]
    plan_answer = (
        "1) Inventory and enable RFC 8784 PSKs. 2) Trial hybrids via RFC 9370 in pilot sites. "
        "3) Roll out Kyber for KEX and add PQC signatures as vendor support lands."
    )
    for p in plan_prompts:
        yield qa_item(p, plan_answer,
                      ["Inventory then pilot", "Use 8784/9370 for ramp", "Adopt Kyber + PQC sigs"],
                      ["RFC8784","RFC9370","FIPS203","FIPS204","CNSA2"])

    # Concise explainer (combined 8784 + 9370)
    combo_prompts = [
        "What do RFC 8784 and RFC 9370 enable for IKEv2 during PQC transition?",
        "Summarize how RFC 8784 and 9370 help VPNs before full PQC.",
        "How do RFC 8784 PSKs and RFC 9370 multi-KEX work together?",
        "Why are RFC 8784 and 9370 important stepping stones to PQC for IPsec?",
        "Explain 8784 + 9370 as an interim path to hybrid IKEv2."
    ]
    combo_answer = (
        "RFC 8784 lets you inject PQ pre-shared keys into IKEv2. RFC 9370 allows multiple key exchanges (enabling hybrids). "
        "Together they support interim quantum resistance for VPNs."
    )
    for p in combo_prompts:
        yield qa_item(p, combo_answer,
                      ["RFC 8784: PQ PSKs", "RFC 9370: multi-KEX", "Useful for hybrid migration"],
                      ["RFC8784","RFC9370","CNSA2"])


def g_wireguard_openvpn():
    topics = [
        # WireGuard X25519 (classical)
        {
            "prompts": [
                "We use WireGuard (NoiseIK with X25519). Is that post-quantum safe?",
                "WireGuard with X25519 KEX — PQC status?",
                "Is a standard WireGuard deployment (X25519) quantum-resistant?",
                "Does WireGuard’s default X25519 handshake protect against quantum attacks?",
                "Are we PQC-ready if our VPN is WireGuard (NoiseIK/X25519)?"
            ],
            "answer": (
                "Not yet. WireGuard’s KEX is classical (X25519). To mitigate now, layer a PQC-ready tunnel beneath or above it "
                "(e.g., TLS 1.3 with a hybrid key_share) or track upstream for hybrid KEM support."
            ),
            "just": ["Current KEX is classical", "Prefer TLS 1.3 hybrids", "PQC signatures follow"],
            "refs": ["TLS13","OQS","CNSA2","NIST_PQC_HUB"]
        },

        # OpenVPN over TLS 1.2
        {
            "prompts": [
                "OpenVPN over TLS 1.2 with ECDHE and RSA certs — PQC status?",
                "Is OpenVPN on TLS 1.2 (ECDHE + RSA) acceptable against quantum threats?",
                "Does TLS 1.2 in OpenVPN provide any PQC protection?",
                "For OpenVPN, is TLS 1.2 with ECDHE/RSA enough during the PQC transition?",
                "Should we upgrade OpenVPN from TLS 1.2 to reach PQC readiness?"
            ],
            "answer": (
                "TLS 1.2 lacks standardized hybrid KEX. Move to TLS 1.3 and enable hybrid key_share groups (e.g., X25519+ML-KEM-768). "
                "Adopt PQC certificate signatures as client/server support matures."
            ),
            "just": ["Current KEX is classical", "Prefer TLS 1.3 hybrids", "PQC signatures follow"],
            "refs": ["TLS13","OQS","CNSA2","NIST_PQC_HUB"]
        },

        # WireGuard optional PSK
        {
            "prompts": [
                "Would adding WireGuard’s optional preshared key (PSK) make it quantum-resistant?",
                "Does WireGuard’s PSK mode solve PQC for the handshake?",
                "Is a strong out-of-band PSK sufficient to protect WireGuard from quantum attacks?",
                "Does enabling the PSK option in WireGuard provide post-quantum security?",
                "Is PSK-on-WireGuard a valid long-term PQC solution?"
            ],
            "answer": (
                "A strong out-of-band PSK can add defense in depth, but it’s not a substitute for standardized PQ KEMs. "
                "Treat it as an interim hardening step, and plan for hybrid/PQC key exchange."
            ),
            "just": ["Current KEX is classical", "Prefer TLS 1.3 hybrids", "PQC signatures follow"],
            "refs": ["TLS13","OQS","CNSA2","NIST_PQC_HUB"]
        },

        # WG over TLS hybrid
        {
            "prompts": [
                "Can we run WireGuard inside a TLS 1.3 hybrid tunnel (WG-over-TLS) to gain PQC now?",
                "Is layering WireGuard over a TLS 1.3 hybrid session a good mitigation?",
                "Does WG-over-TLS (hybrid key_share) deliver PQC benefits today?",
                "Can we encapsulate WireGuard within a hybrid TLS tunnel to get quantum resilience?",
                "Operationally, is WG-over-hybrid-TLS a reasonable approach before native WG hybrids?"
            ],
            "answer": (
                "Yes. Terminate TLS 1.3 with a hybrid key_share, then carry WireGuard inside that channel. "
                "Trade-off: extra overhead and operational complexity."
            ),
            "just": ["Current KEX is classical", "Prefer TLS 1.3 hybrids", "PQC signatures follow"],
            "refs": ["TLS13","OQS","CNSA2","NIST_PQC_HUB"]
        },

        # OpenVPN TLS 1.3 + ChaCha question
        {
            "prompts": [
                "If we upgrade OpenVPN to TLS 1.3 and ChaCha20-Poly1305, are we PQC-ready?",
                "Does choosing ChaCha20-Poly1305 in TLS 1.3 make OpenVPN post-quantum secure?",
                "Is cipher choice (e.g., ChaCha20-Poly1305) enough for PQC in OpenVPN?",
                "OpenVPN on TLS 1.3 with ChaCha — does that solve PQC?",
                "Will switching to ChaCha20-Poly1305 achieve quantum resistance for OpenVPN?"
            ],
            "answer": (
                "Cipher choice alone isn’t enough. PQC readiness comes from key exchange (hybrid groups like X25519+ML-KEM-768) "
                "and later PQC certificate signatures."
            ),
            "just": ["Current KEX is classical", "Prefer TLS 1.3 hybrids", "PQC signatures follow"],
            "refs": ["TLS13","OQS","CNSA2","NIST_PQC_HUB"]
        },
    ]

    for t in topics:
        for p in t["prompts"]:
            yield qa_item(p, t["answer"], t["just"], t["refs"])

    # QUIC note (4–5 prompts, one answer)
    quic_prompts = [
        "Does using QUIC instead of TCP help with PQC for VPN-over-TLS?",
        "If we move to QUIC, do we gain PQC benefits automatically?",
        "QUIC vs TCP for VPN tunnels—any difference for PQC readiness?",
        "Does QUIC change the PQC story for our TLS-based VPN?",
        "Is QUIC a shortcut to post-quantum security?"
    ]
    quic_answer = (
        "QUIC uses TLS 1.3 for key establishment, so the PQC story is the same: enable hybrid key_share and plan for PQC certificate signatures."
    )
    for p in quic_prompts:
        yield qa_item(
            p, quic_answer,
            ["QUIC relies on TLS 1.3", "Enable hybrid key_share", "Add PQC cert signatures"],
            ["QUIC","TLS13","OQS"]
        )


def g_email_pki_smime_pgp():
    # --- Baseline per-algorithm: 5 prompt variants → one consistent answer ---
    algs = ["RSA-2048","RSA-3072","RSA-4096","ECDSA P-256","Ed25519","Ed448"]
    base_prompts = [
        "Our enterprise email uses {rs} + SHA-256 for digital signatures. PQC path?",
        "We sign S/MIME with {rs} (SHA-256). What’s the post-quantum migration?",
        "{rs} + SHA-256 for email signatures—quantum-safe or not, and what should we do?",
        "Email signing algorithm is {rs}. How do we transition to PQC?",
        "Is {rs} acceptable for S/MIME in a PQC plan? What’s the alternative?"
    ]
    base_answer = (
        "{rs} is not quantum-safe. Keep SHA-256, migrate signatures to Dilithium or SPHINCS+. "
        "During transition, use dual-signing (classical + PQC) and track CA/B Forum S/MIME guidance for policy enablement."
    )
    base_just = ["Classical signature not quantum-safe", "Dilithium/SPHINCS+ recommended", "Dual-sign in transition"]
    base_refs = ["FIPS204","FIPS205","CABF_SMC013","LAMPS"]
    for rs in algs:
        for t in base_prompts:
            q = t.format(rs=rs)
            a = base_answer.format(rs=rs)
            yield qa_item(q, a, base_just, base_refs)

    # --- Policy (SMC-013): 5 prompt variants → one answer ---
    smc_prompts = [
        "Does CA/Browser Forum S/MIME Ballot SMC-013 allow PQC algorithms?",
        "Under SMC-013, can we use Dilithium/ML-KEM for S/MIME?",
        "Is PQC (ML-DSA, ML-KEM) permitted by current CA/B S/MIME policy?",
        "What does SMC-013 say about post-quantum algorithms in S/MIME?",
        "Are PQC certs/messages in scope of SMC-013?"
    ]
    smc_answer = (
        "Yes. It enables ML-DSA (Dilithium) and ML-KEM for S/MIME certificates and messages within policy profiles."
    )
    for p in smc_prompts:
        yield qa_item(p, smc_answer,
                      ["S/MIME PQC enablement", "ML-DSA/ML-KEM allowed", "Policy-side support emerging"],
                      ["CABF_SMC013","FIPS204","FIPS203","LAMPS"])

    # --- Need PQC for both encryption & signatures: 5 prompts → one answer ---
    both_prompts = [
        "For S/MIME, do we need PQC for both encryption and signatures?",
        "Is PQC required for S/MIME confidentiality and authentication?",
        "Do S/MIME deployments need Kyber and Dilithium/SPHINCS+ together?",
        "Should we adopt PQC for both CEK encapsulation and message signing?",
        "What’s PQC for S/MIME: KEM only, signatures only, or both?"
    ]
    both_answer = (
        "Yes. Use ML-KEM (Kyber) to encapsulate the content-encryption key for confidentiality and Dilithium/SPHINCS+ for message signatures."
    )
    for p in both_prompts:
        yield qa_item(p, both_answer,
                      ["Confidentiality needs ML-KEM", "Authentication needs PQC sigs", "Cover both risks"],
                      ["FIPS203","FIPS204","FIPS205"])

    # --- Size/MTU impact: 5 prompts → one answer ---
    size_prompts = [
        "Will PQC signatures bloat email size or break MTU?",
        "Do Dilithium/SPHINCS+ signatures significantly increase S/MIME message size?",
        "Are there gateway/archiving limits impacted by larger PQC signatures?",
        "Will PQC signature sizes affect MIME or transport behavior?",
        "Operational impact: do PQC signatures cause fragmentation issues?"
    ]
    size_answer = (
        "PQC signatures are larger than RSA/ECDSA. Expect bigger MIME parts and plan for gateway/archiving limits; "
        "functionally it works with adjusted limits."
    )
    for p in size_prompts:
        yield qa_item(p, size_answer,
                      ["PQC signatures are larger", "Plan for system limits", "Functionality remains intact"],
                      ["FIPS204","FIPS205","LAMPS"])

    # --- Dual-signing / downgrade safety: 5 prompts → one answer ---
    dual_prompts = [
        "Is dual-signing (RSA + Dilithium) safe against downgrade attacks?",
        "How do we prevent classical-only acceptance when dual-signing S/MIME?",
        "Does dual-signing expose us to downgrade if verifiers ignore PQC?",
        "What policy setting avoids downgrade with dual-signed mail?",
        "Best practice for enforcing PQC validation during dual-signing?"
    ]
    dual_answer = (
        "Yes if verifiers are configured to require acceptance of the PQC signature. Enforce policy to reject messages that only validate classically."
    )
    for p in dual_prompts:
        yield qa_item(p, dual_answer,
                      ["Require PQC acceptance", "Avoid classical-only fallback", "Policy enforcement matters"],
                      ["LAMPS","FIPS204","FIPS205"])

    # --- HNDL: signatures alone aren’t enough: 5 prompts → one answer ---
    hndl_prompts = [
        "Does using only a PQC signature protect against 'harvest now, decrypt later' for email?",
        "If we sign with PQC but keep classical key encapsulation, are we safe from HNDL?",
        "Is PQC signing sufficient for recorded email traffic risk?",
        "HNDL risk: do we need Kyber for S/MIME or are signatures sufficient?",
        "How do we mitigate HNDL for email beyond signatures?"
    ]
    hndl_answer = (
        "No. HNDL targets confidentiality. You must adopt ML-KEM for key encapsulation in addition to PQC signatures."
    )
    for p in hndl_prompts:
        yield qa_item(p, hndl_answer,
                      ["HNDL is about decryption", "Add ML-KEM for KEK/CEK", "Signatures alone aren’t enough"],
                      ["CISA_QR_FACTS","FIPS203","NIST_PQC_FAQ"])

    # --- Rollout plan: 5 prompts → one answer ---
    plan_prompts = [
        "What’s a pragmatic S/MIME PQC rollout plan?",
        "Give us a practical migration path to PQC for enterprise email.",
        "How should we phase S/MIME toward Dilithium/Kyber?",
        "Stepwise plan to enable PQC for S/MIME without breaking clients?",
        "What sequence should we follow for PQC S/MIME deployment?"
    ]
    plan_answer = (
        "1) Enable dual-signed certs (RSA/ECDSA + Dilithium). 2) Add Kyber for key encapsulation. "
        "3) Phase to PQC-only as ecosystem support matures."
    )
    for p in plan_prompts:
        yield qa_item(p, plan_answer,
                      ["Dual-sign first", "Add ML-KEM for KEK/CEK", "Phase to PQC-only"],
                      ["FIPS203","FIPS204","CABF_SMC013"])


def g_jose_cose_webauthn():
    # --- WebAuthn / Passkeys PQC: 5 prompts → one answer ---
    webauthn_prompts = [
        "How will WebAuthn/Passkeys handle PQC?",
        "Are passkeys moving to PQC signatures, and what’s required?",
        "What changes are needed in WebAuthn to support PQC?",
        "Passkeys and post-quantum: how do we get there?",
        "Do authenticators and clients need updates for PQC in WebAuthn?"
    ]
    webauthn_answer = (
        "Authenticators must support PQC signatures (e.g., Dilithium/Falcon) and clients must accept the corresponding COSE algorithm IDs. "
        "During migration, accept both classical and PQC attestations."
    )
    for p in webauthn_prompts:
        yield qa_item(p, webauthn_answer,
                      ["PQC alg IDs needed", "TLS 1.3 hybrid first", "Token sigs follow later"],
                      ["WEBAUTHN","LAMPS","TLS13","OQS","FIPS204"])

    # --- JOSE/COSE readiness: 5 prompts → one answer ---
    jose_prompts = [
        "Are COSE/JWS/JWE ready for PQC signatures/encryption?",
        "Is JOSE/COSE standardization sufficient to carry PQC algs today?",
        "Do COSE registries include PQC algorithms for signatures/KEM?",
        "Can we ship PQC tokens using JOSE/COSE now?",
        "What’s the current status of PQC alg IDs in COSE/JWS/JWE?"
    ]
    jose_answer = (
        "JOSE/COSE registries can carry PQC alg IDs; adoption depends on platform/tooling. Until then, ensure TLS 1.3 hybrid for transport."
    )
    for p in jose_prompts:
        yield qa_item(p, jose_answer,
                      ["PQC alg IDs needed", "TLS 1.3 hybrid first", "Token sigs follow later"],
                      ["WEBAUTHN","LAMPS","TLS13","OQS","FIPS204"])

    # --- Do we need PQC for tokens or just TLS? 5 prompts → one answer ---
    tokens_prompts = [
        "Do we need PQC for OAuth/OIDC tokens or just for TLS?",
        "Is PQC necessary for ID/Access tokens if TLS is hybrid?",
        "Priority: PQC for TLS vs token signatures—what comes first?",
        "Short-lived tokens and PQC—how urgent is this compared to TLS?",
        "Should we focus on PQC token signatures now or later?"
    ]
    tokens_answer = (
        "Prioritize TLS 1.3 hybrids first (most exposure). Add PQC token signatures as libraries standardize—many tokens are short-lived and less exposed to HNDL."
    )
    for p in tokens_prompts:
        yield qa_item(p, tokens_answer,
                      ["PQC alg IDs needed", "TLS 1.3 hybrid first", "Token sigs follow later"],
                      ["WEBAUTHN","LAMPS","TLS13","OQS","FIPS204"])

    # --- Extra focused seeds, each expanded to 5 prompts → one answer ---

    # Passkeys: attestations & assertions
    passkey_prompts = [
        "For passkeys, do we need both PQC attestations and assertions?",
        "Should WebAuthn upgrade both attestation and assertion to PQC?",
        "Is PQC needed for device attestation as well as auth assertions?",
        "Do RPs need to accept PQC on both attest and assert paths?",
        "During migration, how do we handle dual algorithms for passkeys?"
    ]
    passkey_answer = (
        "Attestation (device→platform) and assertion (auth→RP) both benefit from PQC signatures. Accept dual algorithms during migration."
    )
    for p in passkey_prompts:
        yield qa_item(p, passkey_answer,
                      ["Attestation + assertion matter", "Use PQC signatures", "Dual-accept during migration"],
                      ["WEBAUTHN","FIPS204"])

    # Hybrid TLS but classical tokens
    risk_prompts = [
        "If TLS is hybrid but ID tokens remain classical, what is the risk?",
        "We upgraded TLS to hybrid—how risky are classical token signatures?",
        "Does hybrid TLS reduce the urgency for PQC token signing?",
        "Risk tradeoff: PQC TLS now, classical tokens for a while?",
        "Is keeping classical token signatures acceptable temporarily?"
    ]
    risk_answer = (
        "Main risk is limited because tokens are short-lived. Prioritize PQC for TLS first; plan PQC token signatures as libraries mature."
    )
    for p in risk_prompts:
        yield qa_item(p, risk_answer,
                      ["Short-lived token risk lower", "TLS boundary first", "Plan PQC token sigs"],
                      ["CISA_QR_FACTS","TLS13","FIPS204"])

    # Native COSE alg IDs vs libraries now
    adopt_prompts = [
        "Should we wait for native COSE PQC alg IDs or use libraries now?",
        "Adopt PQC libraries today, or hold for standardized COSE IDs?",
        "What’s the recommended path: hybrid TLS now, COSE PQC later?",
        "Is it OK to rely on library-specific PQC algs before COSE finalization?",
        "Transition plan for JOSE/COSE PQC: what should we do today?"
    ]
    adopt_answer = (
        "Do both: keep TLS 1.3 hybrid today and evaluate PQC-capable JOSE/COSE libraries. Move to native IDs when platforms standardize."
    )
    for p in adopt_prompts:
        yield qa_item(p, adopt_answer,
                      ["Deploy TLS hybrids now", "Evaluate PQC libs", "Adopt native IDs later"],
                      ["TLS13","OQS","LAMPS"])



def g_dnssec_doh_dot():
    # --- DNSSEC needs PQC signatures: 5 prompt variants → one answer ---
    dnssec_prompts = [
        "Does DNSSEC need PQC signatures?",
        "Are post-quantum signatures necessary for DNSSEC zones?",
        "Should DNSSEC migrate from RSA/ECDSA to PQC signatures?",
        "Is DNSSEC expected to adopt Dilithium/SPHINCS+ for zone signing?",
        "Quantum-safe DNSSEC: do we need PQC signatures for RRSIG/DNSKEY?"
    ]
    dnssec_answer = (
        "Eventually yes. DNSSEC signs zones, so move to Dilithium/SPHINCS+ in the future. "
        "Expect larger DNSKEY/RRSIG records and plan EDNS(0) and TCP fallback."
    )
    for p in dnssec_prompts:
        yield qa_item(p, dnssec_answer,
                      ["DNSSEC uses signatures", "PQC increases sizes", "DoT/DoH inherit TLS PQC story"],
                      ["TLS13","FIPS204","FIPS205","LAMPS","PQUIP_TRACKER"])

    # --- DoH/DoT PQC posture: 5 prompt variants → one answer ---
    doh_dot_prompts = [
        "DoT/DoH and PQC — what changes?",
        "For DNS over TLS/HTTPS, what’s required for PQC?",
        "DoH/DoT PQC plan: hybrid now, PQC certs later?",
        "How do we make DoT/DoH quantum-resilient?",
        "Is the PQC migration for DoH/DoT identical to any TLS service?"
    ]
    doh_dot_answer = (
        "DoT/DoH ride on TLS 1.3, so enable hybrid key_share now and adopt PQC certificate signatures later—identical to any TLS service."
    )
    for p in doh_dot_prompts:
        yield qa_item(p, doh_dot_answer,
                      ["DNSSEC uses signatures", "PQC increases sizes", "DoT/DoH inherit TLS PQC story"],
                      ["TLS13","FIPS204","FIPS205","LAMPS","PQUIP_TRACKER"])

    # --- Operational breakage first: packet size/MTU: 5 variants → one answer ---
    ops_prompts = [
        "Operationally, what breaks first when DNSSEC adopts PQC?",
        "What’s the first operational issue when DNSSEC moves to PQC signatures?",
        "Biggest DNSSEC pain switching to Dilithium/SPHINCS+: what is it?",
        "Where do PQC DNSSEC deployments hit limits first?",
        "What should we tune when DNSSEC keys/signatures grow under PQC?"
    ]
    ops_answer = (
        "Packet size. Larger DNSKEY/RRSIG may exceed typical UDP limits; ensure EDNS(0) with larger buffers and allow TCP fallback in resolvers and firewalls."
    )
    for p in ops_prompts:
        yield qa_item(p, ops_answer,
                      ["Bigger packets with PQC", "Enable EDNS(0)+TCP fallback", "Adjust resolver/firewall limits"],
                      ["PQUIP_TRACKER","FIPS204","FIPS205"])

    # --- DoH/DoT early data & performance: 5 variants → one answer ---
    perf_prompts = [
        "Does PQC affect DoH/DoT early data or performance?",
        "Will PQC change DoH/DoT 0-RTT or early data semantics?",
        "Performance impact of PQC on DoH/DoT handshakes?",
        "Under PQC, what changes for DoH/DoT latency/bytes?",
        "How does PQC influence DoH/DoT handshake size and timing?"
    ]
    perf_answer = (
        "PQC mainly affects the TLS handshake (key_share and cert). Early data semantics remain the same; monitor handshake size and latency."
    )
    for p in perf_prompts:
        yield qa_item(p, perf_answer,
                      ["Handshake grows with PQC", "Early data unchanged", "Monitor latency"],
                      ["TLS13","PQUIP_TRACKER"])


def g_kms_hsm_envelope():
    # --- KMS: RSA→Kyber for DEK wrap — 5 prompts → one answer ---
    wrap_prompts = [
        "We wrap DEKs with RSA-2048 in our KMS. PQC alternative?",
        "Our KMS uses RSA-OAEP for DEK wrapping—what’s the PQC path?",
        "How do we replace RSA wrapping of DEKs with a quantum-safe method?",
        "DEK wrapping today is RSA; what PQC algorithm should we use?",
        "For envelope encryption, what replaces RSA key wrap under PQC?"
    ]
    wrap_answer = (
        "Replace RSA wrapping with ML-KEM (Kyber) to encapsulate DEKs. Envelope encryption remains the right pattern."
    )
    for p in wrap_prompts:
        yield qa_item(p, wrap_answer,
                      ["Switch RSA wrap → Kyber", "Envelope encryption still good", "Use PQC signatures for auth"],
                      ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"])

    # --- HSM readiness — 5 prompts → one answer ---
    hsm_prompts = [
        "Are HSMs ready for Kyber/Dilithium for key management?",
        "Do current HSMs support PQC (Kyber/Dilithium) yet?",
        "What’s HSM vendor status for PQC KEM/signatures?",
        "If HSMs lack PQC, how do we proceed now?",
        "Path to PQC with existing HSMs—what are the options?"
    ]
    hsm_answer = (
        "Vendors are adding PQC; until then, use hybrid patterns or external services that expose Kyber."
    )
    for p in hsm_prompts:
        yield qa_item(p, hsm_answer,
                      ["Switch RSA wrap → Kyber", "Envelope encryption still good", "Use PQC signatures for auth"],
                      ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"])

    # --- Envelope model validity — 5 prompts → one answer ---
    env_prompts = [
        "Is envelope encryption still valid in a PQC world?",
        "Do we keep envelope encryption when migrating to PQC?",
        "Does PQC change the envelope encryption pattern?",
        "Is the KEM-DEM approach still recommended under PQC?",
        "Under PQC, how should we construct envelope encryption?"
    ]
    env_answer = (
        "Yes—use Kyber for key encapsulation and PQC signatures for metadata, keeping the envelope model."
    )
    for p in env_prompts:
        yield qa_item(p, env_answer,
                      ["Switch RSA wrap → Kyber", "Envelope encryption still good", "Use PQC signatures for auth"],
                      ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"])

    # --- Archived backups rewrap — 5 prompts → one answer ---
    rewrap_prompts = [
        "How do we migrate archived backups that used RSA key wrapping?",
        "Best practice to handle old backups wrapped with RSA-OAEP?",
        "Rewrapping archives for PQC: what’s the sequence?",
        "Strategy for long-lived backups with classical DEK wrap?",
        "What should we do with historical backups wrapped by RSA?"
    ]
    rewrap_answer = (
        "Dual-wrap new DEKs with both RSA and Kyber, then rotate to Kyber-only as restore paths get upgraded."
    )
    for p in rewrap_prompts:
        yield qa_item(p, rewrap_answer,
                      ["Dual-wrap during transition", "Rotate to Kyber-only", "Keep restores compatible"],
                      ["FIPS203","CISA_QR_FACTS","NCCOE_MPQC"])

    # --- Hybrid wrapping (RSA + Kyber) — 5 prompts → one answer ---
    hybrid_prompts = [
        "Is 'hybrid wrapping' for DEKs useful (RSA + Kyber)?",
        "Should we store both RSA and Kyber encapsulations during transition?",
        "Is it reasonable to dual-encapsulate the same DEK with RSA and Kyber?",
        "Does keeping both RSA and Kyber headers help migration?",
        "Hybrid KEM/WRAP for DEKs—good interim approach?"
    ]
    hybrid_answer = (
        "Yes as an interim step. Encapsulate the same DEK with both and store both headers; later drop RSA once all services support Kyber."
    )
    for p in hybrid_prompts:
        yield qa_item(p, hybrid_answer,
                      ["Interim hybrid is viable", "Store both headers", "Drop RSA when ready"],
                      ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"])

    # --- KEK/DEK hierarchy under PQC — 5 prompts → one answer ---
    hierarchy_prompts = [
        "What changes in our KEK/DEK hierarchy with PQC?",
        "Does PQC alter the KEK/DEK layering we use today?",
        "Under PQC, how should KEKs and DEKs be organized?",
        "Do we need a new hierarchy for keys in PQC migrations?",
        "What’s the PQC-ready approach to KEK/DEK management?"
    ]
    hierarchy_answer = (
        "The hierarchy stays the same: DEKs protect data, KEKs protect DEKs. Swap RSA-OAEP for Kyber encapsulation and use PQC signatures for key metadata."
    )
    for p in hierarchy_prompts:
        yield qa_item(p, hierarchy_answer,
                      ["Hierarchy unchanged", "Replace RSA-OAEP with Kyber", "Sign metadata with PQC"],
                      ["FIPS203","FIPS204","NIST_PQC_FAQ"])

    # --- Rotation frequency / HNDL — 5 prompts → one answer ---
    rotation_prompts = [
        "How often should we rotate keys to mitigate 'harvest now, decrypt later'?",
        "Key rotation guidance for HNDL in a PQC transition?",
        "Does PQC change DEK rotation frequency for long-lived data?",
        "Rotation strategy to reduce HNDL exposure with Kyber wrap?",
        "How do we tune key lifetimes under HNDL and PQC plans?"
    ]
    rotation_answer = (
        "Increase rotation frequency for long-lived data and move to Kyber DEK encapsulation. Shorter DEK lifetimes reduce exposure even if captures occur."
    )
    for p in rotation_prompts:
        yield qa_item(p, rotation_answer,
                      ["Shorten DEK lifetimes", "Move to Kyber encapsulation", "Reduce HNDL exposure"],
                      ["CISA_QR_FACTS","FIPS203","NIST_PQC_FAQ"])



def g_cloud_cdn_platforms():
    # CDN/front-end deployment posture — 5 prompts → one answer
    deploy_prompts = [
        "How do CDNs/frontends deploy PQC for TLS?",
        "What is the right PQC plan for CDN edges and front doors?",
        "How should a CDN enable post-quantum TLS on the edge?",
        "Best way to roll out PQC on CDN and front-end tiers?",
        "Practical steps for PQC on edge TLS termination?"
    ]
    deploy_answer = (
        "Enable TLS 1.3 hybrid groups at the edge, then extend to origins (front-end to origin). "
        "Watch client compatibility and MTU."
    )
    for p in deploy_prompts:
        yield qa_item(p, deploy_answer,
                      ["Edge first, then origins", "Hybrid negotiates when both support it", "Vendor rollouts accelerate adoption"],
                      ["CHROMIUM_HYBRID","CF_PQC_GA","CF_PQC_ORIG","MS_PQC_2024","MS_PQC_2025","APPLE_TLS26","AKAMAI_PQC_2025","CF_PQ_2024"])

    # Chrome/Cloudflare support implications — 5 prompts → one answer
    support_prompts = [
        "Chrome/Cloudflare support for X25519+Kyber — what does that mean for us?",
        "If major browsers/CDNs support hybrid X25519+Kyber, what should we do?",
        "Client-side hybrid readiness (Chrome etc.): how should servers react?",
        "With Cloudflare and Chrome supporting hybrids, is it time to enable them?",
        "What does wide client hybrid support imply for our edge?"
    ]
    support_answer = (
        "It means many clients can already negotiate hybrid. Turn it on server-side to gain PQC forward secrecy."
    )
    for p in support_prompts:
        yield qa_item(p, support_answer,
                      ["Edge first, then origins", "Hybrid negotiates when both support it", "Vendor rollouts accelerate adoption"],
                      ["CHROMIUM_HYBRID","CF_PQC_GA","CF_PQ_2024"])

    # Vendor announcements impact — 5 prompts → one answer
    vendors_prompts = [
        "Microsoft/Apple/Akamai PQC announcements — practical impact?",
        "OS/browser/CDN PQC roadmaps: what does it change for us?",
        "How do recent vendor PQC releases affect our rollout plan?",
        "Platform support (Apple/Microsoft/Akamai) and PQC — what to plan?",
        "Do vendor PQC timelines justify enabling hybrids now?"
    ]
    vendors_answer = (
        "Vendors are rolling out PQC in OS stacks, browsers, and CDNs—plan phased enablement and compatibility testing."
    )
    for p in vendors_prompts:
        yield qa_item(p, vendors_answer,
                      ["Edge first, then origins", "Hybrid negotiates when both support it", "Vendor rollouts accelerate adoption"],
                      ["MS_PQC_2024","MS_PQC_2025","APPLE_TLS26","AKAMAI_PQC_2025","CHROMIUM_HYBRID"])

    # Safe rollout plan (canary/telemetry) — 5 prompts → one answer
    rollout_prompts = [
        "What’s a safe rollout plan for hybrid TLS on a CDN edge?",
        "How should we canary PQC hybrids at the edge?",
        "Rollout strategy for enabling hybrid groups on CDN?",
        "What KPIs/steps should govern CDN hybrid enablement?",
        "How do we stage edge hybrid TLS without regressions?"
    ]
    rollout_answer = (
        "1) Enable X25519+Kyber on a small POP cohort. 2) Telemetry: measure negotiated-groups rate and handshake failures. "
        "3) Ramp by region; keep classical fallback for non-supporting clients."
    )
    for p in rollout_prompts:
        yield qa_item(p, rollout_answer,
                      ["Start small and meter", "Measure negotiated groups", "Gradually ramp with fallback"],
                      ["CHROMIUM_HYBRID","CF_PQ_2024","CF_PQC_GA"])

    # MTU/PMTUD considerations — 5 prompts → one answer
    mtu_prompts = [
        "Will hybrid TLS increase packet sizes or cause MTU/PMTUD issues?",
        "Do hybrids risk fragmentation on CDN edges?",
        "How to handle MTU and PMTUD when enabling hybrid TLS?",
        "What about ClientHello/ServerHello size with hybrids?",
        "Any sizing issues for QUIC initial flights with hybrids?"
    ]
    mtu_answer = (
        "ClientHello/ServerHello grow due to larger key_share. Usually fine, but validate PMTUD and initial QUIC flights; "
        "adjust edge MSS/MTU if you see fragmentation."
    )
    for p in mtu_prompts:
        yield qa_item(p, mtu_answer,
                      ["Larger key_share payload", "Validate PMTUD/initial flights", "Tune MSS/MTU if needed"],
                      ["TLS13","CF_PQC_GA","QUIC"])

    # Caching/WAF/DDoS posture — 5 prompts → one answer
    posture_prompts = [
        "Does hybrid TLS change CDN caching, WAF, or DDoS posture?",
        "Operational impact of hybrids on WAF/caching/DDoS?",
        "Do hybrids require CDN policy changes for WAF/DDoS?",
        "How do hybrids affect edge capacity planning?",
        "Any layer-7 changes from enabling hybrid TLS?"
    ]
    posture_answer = (
        "Not materially. Cost is concentrated in handshake CPU and a few extra bytes on the wire. "
        "Budget capacity and TLS terminator cores accordingly."
    )
    for p in posture_prompts:
        yield qa_item(p, posture_answer,
                      ["Handshake CPU overhead", "Wire size slightly larger", "Capacity plan terminators"],
                      ["CF_PQ_2024","TLS13"])

    # Edge-only benefits when origin not ready — 5 prompts → one answer
    edge_prompts = [
        "We can’t update origins yet—can we still get PQC benefits at the edge?",
        "Is edge-only hybrid useful before origins are upgraded?",
        "Do we gain value if only client↔edge is hybrid?",
        "Can we defer origin upgrades but enable edge hybrids now?",
        "What if origins stay classical for a while?"
    ]
    edge_answer = (
        "Yes. Do client↔edge hybrid now, keep edge↔origin classical until your origin stack supports PQC; "
        "some providers already support hybrid to origins."
    )
    for p in edge_prompts:
        yield qa_item(p, edge_answer,
                      ["Edge-first is valid", "Back-end can follow later", "Providers offer origin hybrids"],
                      ["CF_PQC_ORIG","CF_PQC_GA","TLS13"])

    # QUIC/h3 and 0-RTT — 5 prompts → one answer
    quic_prompts = [
        "Any impact on QUIC/h3 or 0-RTT with hybrid?",
        "Does hybrid TLS change HTTP/3 behavior or 0-RTT?",
        "QUIC under hybrids: what changes operationally?",
        "How do hybrids affect QUIC initial congestion window?",
        "Is 0-RTT affected by enabling hybrid KEX?"
    ]
    quic_answer = (
        "QUIC uses TLS 1.3 under the hood. Hybrid affects KEX only; 0-RTT semantics are unchanged. "
        "Monitor handshake bytes and initial congestion window."
    )
    for p in quic_prompts:
        yield qa_item(p, quic_answer,
                      ["QUIC rides TLS 1.3", "0-RTT semantics unchanged", "Watch handshake size/cwnd"],
                      ["QUIC","TLS13","CHROMIUM_HYBRID"])

    # PQC certificates necessary? — 5 prompts → one answer
    certs_prompts = [
        "Do we need PQC certificates to deploy hybrid TLS at the edge?",
        "Are PQC-signed certs required to enable hybrid KEX?",
        "Can we ship hybrids without PQC cert chains first?",
        "Edge hybrids and certs: do PQC signatures need to be ready?",
        "Is hybrid deployment blocked on PQC certificates?"
    ]
    certs_answer = (
        "No. Hybrid KEX is independent of certificate signatures. PQC certs can come later as CA policy and client support mature."
    )
    for p in certs_prompts:
        yield qa_item(p, certs_answer,
                      ["Hybrid ≠ PQC certs", "Add PQC certs later", "Track CA/standards work"],
                      ["LAMPS","TLS13","CABF_SMC013"])

    # Success metrics — 5 prompts → one answer
    kpi_prompts = [
        "How do we measure success after enabling hybrid on the CDN?",
        "What KPIs show hybrid TLS rollout is healthy on edge?",
        "Which metrics indicate we can ramp PQC hybrids globally?",
        "Telemetry to watch after turning on hybrid groups?",
        "How to gate progressive enablement of hybrids at edge?"
    ]
    kpi_answer = (
        "Track negotiated hybrid group percentage, handshake failure rate, median handshake latency, and fallback rates. "
        "Roll forward if KPIs stay green."
    )
    for p in kpi_prompts:
        yield qa_item(p, kpi_answer,
                      ["Instrument negotiated groups", "Watch failures/latency", "Gate rollout on KPIs"],
                      ["CF_PQ_2024","CHROMIUM_HYBRID","CF_PQC_GA"])


def g_hpke_kemtls_mls():
    # HPKE basics — 5 prompts → one answer
    hpke_basic_prompts = [
        "What is HPKE and how does PQC fit?",
        "Explain HPKE and where post-quantum KEMs slot in.",
        "How do we make HPKE quantum-resistant?",
        "HPKE composition: where to replace with Kyber?",
        "HPKE overview for PQC migration?"
    ]
    hpke_basic_answer = (
        "HPKE composes KEM+KDF+AEAD. Swap DHKEM for a PQC KEM (Kyber) or a composite KEM to get quantum resistance while keeping HKDF and AEAD unchanged."
    )
    for p in hpke_basic_prompts:
        yield qa_item(p, hpke_basic_answer,
                      ["HPKE = KEM+KDF+AEAD", "Replace KEM with Kyber/composite", "Keep KDF/AEAD layers"],
                      ["HPKE","FIPS203","TLS13"])

    # HPKE modes — 5 prompts → one answer
    hpke_modes_prompts = [
        "Which HPKE modes exist and does PQC change them?",
        "HPKE base/psk/auth/auth-psk: any PQC differences?",
        "Do HPKE modes need updates for PQC?",
        "How do HPKE modes interact with a PQC KEM?",
        "Are HPKE modes stable if we switch to Kyber?"
    ]
    hpke_modes_answer = (
        "HPKE has base, psk, auth, and auth-psk modes. The modes don’t change—security hinges on the KEM choice. Using Kyber or a hybrid KEM gives PQC strength in any mode."
    )
    for p in hpke_modes_prompts:
        yield qa_item(p, hpke_modes_answer,
                      ["Modes: base/psk/auth/auth-psk", "Security depends on KEM", "Kyber/hybrid gives PQC"],
                      ["HPKE","FIPS203"])

    # KDF/AEAD pairs — 5 prompts → one answer
    hpke_kdf_aead_prompts = [
        "Which KDF/AEAD pairs are good in a PQC HPKE profile?",
        "Recommended HPKE KDF and AEAD under PQC?",
        "Do we need to change KDF/AEAD when using Kyber in HPKE?",
        "Best-practice KDF/AEAD for PQC-ready HPKE?",
        "HPKE cryptographic primitives to keep under PQC?"
    ]
    hpke_kdf_aead_answer = (
        "HKDF-SHA256/384 with AES-GCM or ChaCha20-Poly1305 are common and remain fine in PQ settings. Prefer 256-bit margins where feasible."
    )
    for p in hpke_kdf_aead_prompts:
        yield qa_item(p, hpke_kdf_aead_answer,
                      ["HKDF-SHA256/384 typical", "AES-GCM/ChaCha20 are fine", "Favor 256-bit margins"],
                      ["HPKE","TLS13"])

    # Composite/hybrid HPKE — 5 prompts → one answer
    hpke_hybrid_prompts = [
        "Can HPKE be made hybrid (classical + PQC) without new APIs?",
        "How to combine X25519 and Kyber in HPKE?",
        "Composite KEM in HPKE: does it work today?",
        "Hybrid HPKE secrets: how are they combined?",
        "Is a composite KEM path viable for PQC HPKE?"
    ]
    hpke_hybrid_answer = (
        "Yes—use a composite KEM that encapsulates with X25519 (DHKEM) and Kyber, then combine secrets via the HPKE KDF. Libraries prototype this today."
    )
    for p in hpke_hybrid_prompts:
        yield qa_item(p, hpke_hybrid_answer,
                      ["Composite KEM approach", "Combine secrets in KDF", "Prototype support exists"],
                      ["HPKE","FIPS203","OQS_SITE","LIBOQS"])

    # KEMTLS: what is it — 5 prompts → one answer
    kemtls_prompts = [
        "What is KEMTLS?",
        "Explain KEMTLS in the context of PQC.",
        "How does KEMTLS differ from classic TLS?",
        "What problem does KEMTLS solve for PQC auth?",
        "Why replace CertificateVerify with KEMs in KEMTLS?"
    ]
    kemtls_answer = (
        "KEMTLS replaces signature-based CertificateVerify with KEM-based possession proofs, enabling PQC authentication with KEMs and reducing signature overhead."
    )
    for p in kemtls_prompts:
        yield qa_item(p, kemtls_answer,
                      ["KEM auth not signatures", "PQC-ready handshake", "Lower signature overhead"],
                      ["KEMTLS","FIPS203","TLS13"])

    # KEMTLS client authentication & migration — 5 prompts → one answer
    kemtls_client_prompts = [
        "Is client authentication possible in KEMTLS and how to migrate?",
        "How do clients authenticate in KEMTLS?",
        "KEMTLS migration path alongside classic TLS 1.3?",
        "Can we pilot KEMTLS without breaking compatibility?",
        "Client auth in KEMTLS: what’s the rollout approach?"
    ]
    kemtls_client_answer = (
        "Yes—clients prove possession via KEM encapsulation. Pilot KEMTLS in controlled environments while maintaining classic TLS 1.3 for broad compatibility."
    )
    for p in kemtls_client_prompts:
        yield qa_item(p, kemtls_client_answer,
                      ["Client KEM auth works", "Pilot alongside TLS 1.3", "Controlled migration path"],
                      ["KEMTLS","TLS13"])

    # MLS & PQC relationship — 5 prompts → one answer
    mls_prompts = [
        "How does MLS relate to PQC?",
        "Is MLS ready for PQC KEMs like Kyber?",
        "Where does MLS use HPKE and how to make it PQC-ready?",
        "Can TreeKEM in MLS adopt Kyber or a hybrid?",
        "PQC in MLS group updates: what changes?"
    ]
    mls_answer = (
        "MLS uses HPKE (TreeKEM) for group key updates. Replacing DHKEM with Kyber (or a hybrid) makes group updates quantum-resistant with moderate size increases."
    )
    for p in mls_prompts:
        yield qa_item(p, mls_answer,
                      ["MLS builds on HPKE", "Swap in Kyber/hybrid", "Moderate overhead"],
                      ["MLS","HPKE","FIPS203"])

    # PQC signatures in MLS — 5 prompts → one answer
    mls_sig_prompts = [
        "Do we also need PQC signatures in MLS?",
        "If MLS uses Kyber for HPKE, do we still need PQC signatures?",
        "What should authenticate MLS if KEMs are PQC?",
        "Are Dilithium/SPHINCS+ required for MLS authentication?",
        "Does MLS rely on signatures even with PQC KEMs?"
    ]
    mls_sig_answer = (
        "Yes—MLS still authenticates with signatures. Consider Dilithium or SPHINCS+ for auth while using Kyber in HPKE for confidentiality."
    )
    for p in mls_sig_prompts:
        yield qa_item(p, mls_sig_answer,
                      ["MLS uses signatures", "Use Dilithium/SPHINCS+ for auth", "Kyber for confidentiality"],
                      ["MLS","FIPS204","FIPS205","HPKE"])

    # Overhead expectations for Kyber in MLS — 5 prompts → one answer
    mls_overhead_prompts = [
        "What overhead should we expect when using Kyber in MLS?",
        "Does Kyber increase MLS path update sizes?",
        "Operational cost of swapping DHKEM→Kyber in MLS?",
        "How does PQC affect MLS latency and packetization?",
        "What to measure when enabling Kyber in MLS?"
    ]
    mls_overhead_answer = (
        "Public keys and ciphertexts are larger, so path updates grow. In many group sizes the overhead is acceptable; "
        "measure update latency and packetization."
    )
    for p in mls_overhead_prompts:
        yield qa_item(p, mls_overhead_answer,
                      ["Keys/ciphertexts grow", "Path updates larger", "Measure latency/packing"],
                      ["MLS","FIPS203"])


def g_broken_alts():
    # ---- Broken schemes (multiple prompt variants → one answer each) ----
    sike_prompts = [
        "Is SIKE still secure after the 2022 attack?",
        "What happened to SIKE in 2022—is it still safe?",
        "Did SIKE survive the 2022 cryptanalysis?",
        "Is the SIKE KEM considered secure today?",
        "Status update: is SIKE viable post-2022?"
    ]
    sike_answer = "No. Efficient classical attacks broke SIKE in 2022; it was dropped from standardization."
    for p in sike_prompts:
        yield qa_item(p, sike_answer,
                      ["Broken by classical attacks", "Removed from tracks", "Use NIST-selected algorithms"],
                      ["SIKE","RAINBOW","NIST_PQC_HUB"])

    rainbow_prompts = [
        "Is the Rainbow signature scheme secure?",
        "Did Rainbow remain secure after the 2022 breaks?",
        "Is Rainbow still recommended today?",
        "What is Rainbow’s security status?",
        "Was Rainbow standardized by NIST?"
    ]
    rainbow_answer = "No. Rainbow was broken in 2022 (private-key recovery). It wasn’t standardized."
    for p in rainbow_prompts:
        yield qa_item(p, rainbow_answer,
                      ["Broken by classical attacks", "Removed from tracks", "Use NIST-selected algorithms"],
                      ["SIKE","RAINBOW","NIST_PQC_HUB"])

    # ---- Alternates (viable but not in first FIPS wave) ----
    mce_prompts = [
        "What is Classic McEliece, and is it secure against quantum attacks?",
        "Is Classic McEliece a quantum-safe KEM?",
        "How do we view Classic McEliece in 2025?",
        "Classic McEliece: trade-offs and security?",
        "Should we consider Classic McEliece in designs?"
    ]
    mce_answer = ("Classic McEliece is a code-based KEM believed quantum-resistant; the trade-off is very large public keys. "
                  "It remains an alternate with long study history.")
    for p in mce_prompts:
        yield qa_item(p, mce_answer,
                      ["Code-based and long-studied", "No known breaks", "Very large public keys"],
                      ["MCELIECE"])

    ntru_prompts = [
        "What is NTRU (or NTRU Prime), and is it quantum-safe?",
        "Is NTRU/NTRU Prime considered a PQC KEM?",
        "Should we use NTRU or NTRU Prime?",
        "How mature is NTRU/NTRU Prime for deployment?",
        "Is NTRU Prime used in real systems?"
    ]
    ntru_answer = ("NTRU/NTRU Prime are lattice-based KEMs considered quantum-safe with no efficient attacks; "
                   "an NTRU Prime variant is used in OpenSSH hybrids.")
    for p in ntru_prompts:
        yield qa_item(p, ntru_answer,
                      ["Lattice-based KEM", "No efficient attacks", "Used in hybrids"],
                      ["NTRU","OPENSSH"])

    hqc_prompts = [
        "What is HQC (Hamming Quasi-Cyclic), and is it secure?",
        "Is HQC a viable PQC KEM?",
        "HQC status and trade-offs?",
        "How does HQC compare to lattice KEMs?",
        "Should we track HQC for future use?"
    ]
    hqc_answer = ("HQC is a code-based KEM from the NIST process. It’s believed quantum-resistant but has larger sizes than lattice KEMs; "
                  "still a promising alternate.")
    for p in hqc_prompts:
        yield qa_item(p, hqc_answer,
                      ["Code-based KEM", "Larger sizes", "Promising alternate"],
                      ["HQC"])

    bike_prompts = [
        "What is BIKE (Bit Flipping Key Encapsulation), and is it secure?",
        "Is BIKE still in play as a PQC option?",
        "BIKE security and parameter history?",
        "How should we treat BIKE today?",
        "Is BIKE standardized yet?"
    ]
    bike_answer = ("BIKE is a QC-MDPC code-based KEM. Earlier weaknesses led to parameter adjustments; no complete break. "
                   "Considered quantum-resistant but not standardized.")
    for p in bike_prompts:
        yield qa_item(p, bike_answer,
                      ["QC-MDPC KEM", "Parameters adjusted", "Not standardized"],
                      ["BIKE"])

    std_prompts = [
        "Are McEliece or NTRU standardized like Kyber?",
        "Did McEliece or NTRU make the first NIST FIPS set?",
        "Standardization status: McEliece/NTRU vs Kyber?",
        "Are NTRU/McEliece in FIPS alongside Kyber?",
        "Where do McEliece and NTRU stand in standards?"
    ]
    std_answer = ("Not in the first NIST FIPS set. They are alternates; track IETF PQUIP and ecosystem adoption for profiles and interop.")
    for p in std_prompts:
        yield qa_item(p, std_answer,
                      ["Not in first FIPS set", "Alternates to track", "Watch IETF PQUIP/interops"],
                      ["NIST_PQC_HUB","PQUIP_TRACKER"])


def g_comparisons():
    # ---- Classical vs PQC signatures (variant phrasings per pair) ----
    classical_sigs = ["RSA-2048","RSA-3072","ECDSA P-256","Ed25519","brainpoolP384r1"]
    pqc_sigs = ["ML-DSA (Dilithium)","SLH-DSA (SPHINCS+)","Falcon-512"]
    sig_q_variants = [
        "Which is post-quantum safe: {cls} or {pq}?",
        "Between {cls} and {pq}, which withstands quantum attacks?",
        "Does {cls} or {pq} survive Shor’s algorithm?",
        "For long-term security, choose {cls} or {pq}?",
        "Is {cls} or {pq} better for a PQC migration?"
    ]
    for cls in classical_sigs:
        for pq in pqc_sigs:
            ans = (f"{pq} is designed to resist quantum attacks; {cls} is classical and would be broken by Shor.")
            for t in sig_q_variants:
                q = t.format(cls=cls, pq=pq)
                yield qa_item(q, ans,
                              ["PQC vs classical", f"{cls}: quantum-vulnerable", f"{pq}: quantum-resistant"],
                              ["SHOR","FIPS204","FIPS205"])

    # ---- Classical KEX vs hybrid KEX for TLS ----
    classical_kex = ["ECDH P-256","X25519","DH Group 14","brainpoolP384r1"]
    sample_hybrids = random.sample(HYBRID_ALL, k=min(4, len(HYBRID_ALL)))
    kex_q_variants = [
        "For TLS key exchange, which is PQC-ready: {cls} or {hy}?",
        "Which TLS KEX resists quantum: {cls} or {hy}?",
        "Pick a PQC-ready TLS KEX: {cls} vs {hy}?",
        "Is {cls} or {hy} better for post-quantum TLS?",
        "Which TLS key_share should we prefer: {cls} or {hy}?"
    ]
    for cls in classical_kex:
        for hy in sample_hybrids:
            ans = f"{hy} is PQC-ready (includes Kyber); {cls} is classical."
            for t in kex_q_variants:
                q = t.format(cls=cls, hy=hy)
                yield qa_item(q, ans,
                              ["Hybrid includes Kyber", f"{cls} is classical", "Use TLS 1.3 hybrid"],
                              ["TLS13","OQS","FIPS203"])

    # ---- Symmetric cipher suitability (each algorithm with variants) ----
    sym_q_variants = [
        "Is {enc} suitable for post-quantum use?",
        "Is {enc} OK in the post-quantum era?",
        "Post-quantum readiness of {enc}?",
        "Should we keep using {enc} after PQC migration?",
        "Is {enc} recommended under PQ considerations?"
    ]
    for enc in SYMMETRIC_AEAD + SYMMETRIC_OLD:
        if enc in SYMMETRIC_AEAD:
            ans = "Yes. It’s a modern AEAD. Grover reduces 128-bit keys to ~2^64; prefer 256-bit variants for long-term safety."
            refs = ["NIST_PQC_FAQ","TLS13"]
            just = ["AEADs remain strong", "Grover halves 128-bit exponent", "Prefer 256-bit keys"]
        else:
            ans = "No—deprecated or insecure. Use AEADs such as AES-GCM or ChaCha20-Poly1305 as required by TLS 1.3."
            refs = ["TLS13"]
            just = ["Legacy cipher is weak", "TLS 1.3 requires AEADs", "Pick AES-GCM/ChaCha20-Poly1305"]
        for t in sym_q_variants:
            q = t.format(enc=enc)
            yield qa_item(q, ans, just, refs)

    # ---- Targeted comparisons (each with 4–5 prompt variants) ----
    # AES-128-GCM vs AES-256-GCM for long-lived data
    aes_prompts = [
        "Which should we prefer for long-term TLS: TLS_AES_128_GCM_SHA256 or TLS_AES_256_GCM_SHA384?",
        "For very long lifetimes, pick AES-128-GCM or AES-256-GCM in TLS 1.3?",
        "What’s better vs Grover for TLS: 128-GCM or 256-GCM?",
        "Long-term margin choice: TLS_AES_128_GCM_SHA256 vs TLS_AES_256_GCM_SHA384?",
        "Which AEAD has a larger symmetric margin for PQC timelines?"
    ]
    aes_answer = ("Both are strong; for very long lifetimes pick AES-256-GCM to retain a larger margin against Grover while monitoring performance.")
    for p in aes_prompts:
        yield qa_item(p, aes_answer,
                      ["Both secure today", "AES-256-GCM has larger margin", "Balance with performance"],
                      ["TLS13","NIST_PQC_FAQ"])

    # X25519 vs X25519+Kyber
    hyb_prompts = [
        "Which is more PQC-ready for TLS key exchange: X25519 or X25519+ML-KEM-768?",
        "Between X25519 and X25519+Kyber, which should we deploy?",
        "Is X25519 enough, or do we need X25519+ML-KEM-768?",
        "For post-quantum TLS, pick X25519 or the hybrid with Kyber?",
        "Which key_share is PQC-ready: classical X25519 or hybrid X25519+Kyber?"
    ]
    hyb_answer = "X25519+ML-KEM-768 is hybrid and PQC-ready; X25519 alone is classical and not quantum-safe."
    for p in hyb_prompts:
        yield qa_item(p, hyb_answer,
                      ["Hybrid = classical+PQC", "X25519 alone is classical", "Prefer hybrid groups"],
                      ["TLS13","OQS","FIPS203"])

    # SHA-1 vs SHA-256/384
    hash_prompts = [
        "Which hash should we use going forward: SHA-1 or SHA-256/SHA-384?",
        "Is SHA-1 acceptable vs SHA-256/SHA-384 now?",
        "For new deployments, choose SHA-1 or SHA-256/384?",
        "Post-quantum era: SHA-1 vs SHA-256/384?",
        "Modern guidance on SHA-1 compared to SHA-256/384?"
    ]
    hash_answer = "Use SHA-256/384 (or SHA-3). SHA-1 is deprecated and collision-prone and should not be used."
    for p in hash_prompts:
        yield qa_item(p, hash_answer,
                      ["SHA-1 deprecated", "Use SHA-2/3", "Follow modern guidance"],
                      ["TLS13"])

    # Ed25519 vs Dilithium2 for code signing
    cs_prompts = [
        "For code signing on new platforms, pick Ed25519 or Dilithium2?",
        "Should new code-signing use Dilithium2 instead of Ed25519?",
        "Code-signing choice: Ed25519 vs Dilithium2 for PQC readiness?",
        "Which signature for long-term code signing: Ed25519 or Dilithium2?",
        "Is Dilithium2 preferred over Ed25519 for PQC timelines?"
    ]
    cs_answer = ("Dilithium2 is PQC and suitable for long-term resilience; Ed25519 is classical. "
                 "If you need immediate compatibility, dual-sign and phase out Ed25519.")
    for p in cs_prompts:
        yield qa_item(p, cs_answer,
                      ["Dilithium2 is PQC", "Ed25519 is classical", "Dual-sign during migration"],
                      ["FIPS204","NIST_PQC_HUB"])

    # RSA-4096 vs Dilithium3 for certificates
    cert_prompts = [
        "Which is more appropriate for certificates now: RSA-4096 or Dilithium3?",
        "Cert choice today: RSA-4096 vs Dilithium3?",
        "Should new certs move to Dilithium3 over RSA-4096?",
        "Long-term PKI: RSA-4096 or Dilithium3?",
        "Which certificate signature is PQC-ready: RSA-4096 or Dilithium3?"
    ]
    cert_answer = ("Dilithium3 offers PQC security; RSA-4096 is classical and falls to Shor. "
                   "Use Dilithium as policy and ecosystem allow.")
    for p in cert_prompts:
        yield qa_item(p, cert_answer,
                      ["Dilithium3 = PQC", "RSA-4096 is classical", "Adopt PQC as policy allows"],
                      ["FIPS204","SHOR"])

    # Kyber-512 vs Kyber-768
    kyber_prompts = [
        "Kyber-512 vs Kyber-768 for web handshakes—what to choose?",
        "Which Kyber level should we prefer for web: 512 or 768?",
        "TLS hybrids: pick ML-KEM-512 or ML-KEM-768?",
        "For browser-facing services, Kyber-512 or Kyber-768?",
        "Is Kyber-768 the right default for web?"
    ]
    kyber_answer = ("Kyber-768 is the typical web/hybrid choice for stronger margins. "
                    "Kyber-512 may be acceptable in constrained contexts per policy.")
    for p in kyber_prompts:
        yield qa_item(p, kyber_answer,
                      ["Kyber-768 typical on web", "Kyber-512 for constrained cases", "Follow policy guidance"],
                      ["FIPS203","OQS"])

    # ---- Symmetric vs asymmetric under quantum ----
    rsa_aes_prompts = [
        "Which is post-quantum secure: RSA-2048 or AES-256?",
        "Between RSA-2048 and AES-256, which resists quantum attacks?",
        "Is AES-256 still strong against quantum vs RSA-2048?",
        "Quantum impact: RSA-2048 vs AES-256?",
        "For PQC timelines, should we trust RSA-2048 or AES-256 more?"
    ]
    rsa_aes_answer = ("AES-256 remains robust (≈2^128 under Grover). RSA-2048 is broken by Shor on a large fault-tolerant quantum computer.")
    for p in rsa_aes_prompts:
        yield qa_item(p, rsa_aes_answer,
                      ["Grover vs AES-256", "Shor vs RSA", "Symmetric vs asymmetric impact"],
                      ["SHOR","NIST_PQC_FAQ"])


def g_definitions():
    # --- Classical signature definitions (multiple prompt variants per algo) ---
    for algo, sizes in CLASSICAL_SIGN:
        size_info = f" (keys: {', '.join(map(str, sizes))})" if sizes else ""
        prompts = [
            f"What is {algo}{size_info}, and is it quantum-safe?",
            f"Explain {algo}{size_info}. Does it survive quantum attacks?",
            f"{algo}{size_info}: definition and PQ safety?",
            f"How does {algo}{size_info} work, and is it post-quantum secure?",
            f"Is {algo}{size_info} suitable in a PQC migration?"
        ]
        if algo == "RSA":
            ans = ("RSA is based on integer factorization (commonly 2048–4096-bit keys). It is not quantum-safe—Shor would break it. "
                   "Plan to move to Dilithium or SPHINCS+.")
            refs = ["SHOR", "FIPS204", "FIPS205"]
        elif algo == "ECDSA":
            ans = ("ECDSA is an elliptic-curve signature (P-256, P-384, etc.). Not quantum-safe because ECDLP falls to Shor. "
                   "Migrate to Dilithium (ML-DSA) or SPHINCS+ (SLH-DSA).")
            refs = ["SHOR", "FIPS204", "FIPS205"]
        elif algo == "DSA":
            ans = ("DSA is based on discrete logs (≈2048–3072-bit primes). Not quantum-safe. Replace with PQC signatures.")
            refs = ["SHOR", "FIPS204", "FIPS205"]
        else:
            ans = (f"{algo} is an ECC signature (Edwards curves). Secure classically, not quantum-safe. "
                   "Migrate to Dilithium or SPHINCS+.")
            refs = ["SHOR", "FIPS204", "FIPS205"]
        just = [f"{algo} definition", "Shor breaks it (not quantum-safe)", "Use PQC signatures instead"]
        for p in prompts:
            yield qa_item(p, ans, just, refs)

    # --- Extra definition seeds (multi-variant) ---
    rsa_bigger_prompts = [
        "Does increasing RSA key size (e.g., 8192) make it quantum-safe?",
        "Is RSA-8192 quantum-safe compared to RSA-2048?",
        "Will bigger RSA keys protect us from quantum attacks?",
        "Do very large RSA moduli defeat Shor’s algorithm?",
        "Is moving to RSA-8192 a PQC mitigation?"
    ]
    rsa_bigger_answer = ("No. Larger RSA keys help against classical attacks but not against Shor’s algorithm. "
                         "Switch to PQC signatures such as Dilithium or SPHINCS+.")
    for p in rsa_bigger_prompts:
        yield qa_item(p, rsa_bigger_answer,
                      ["Key size doesn’t stop Shor", "Switch to PQC signatures", "Plan phased migration"],
                      ["SHOR","FIPS204","FIPS205"])

    brainpool_prompts = [
        "Are brainpool curves (e.g., brainpoolP384r1) quantum-safe?",
        "Is ECC brainpoolP384r1 safe against quantum computers?",
        "Brainpool ECC vs quantum: safe or not?",
        "Do brainpool curves survive Shor’s algorithm?",
        "Should we rely on brainpool curves during PQC transition?"
    ]
    brainpool_answer = ("No. All classical ECC, including brainpool curves, falls to Shor. "
                        "Use PQC signatures and/or hybrid approaches during transition.")
    for p in brainpool_prompts:
        yield qa_item(p, brainpool_answer,
                      ["ECC (any curve) is classical", "Shor breaks ECC", "Adopt PQC/Hybrid"],
                      ["SHOR","FIPS204","FIPS205"])

    # --- Classical key exchange definitions (multi-variant per family/group) ---
    kex_variants = [
        "What is {fam} {g}, and is it quantum-safe for key exchange?",
        "Explain {fam} {g}. PQC status?",
        "Is {fam} {g} safe against quantum attacks?",
        "{fam} {g}: definition and PQ suitability?",
        "Should we keep {fam} {g} in a PQC migration?"
    ]
    for fam, groups in CLASSICAL_KEX:
        if groups:
            for g in groups:
                ans = (f"{fam} {g} is a classical key exchange and not quantum-safe (Shor breaks it). "
                       f"Use ML-KEM (Kyber), possibly hybridized with {fam} for compatibility.")
                just = [f"{fam} {g} explained", "Not quantum-safe (Shor)", "Use Kyber or hybrid"]
                refs = ["SHOR", "FIPS203", "TLS13", "OQS"]
                for t in kex_variants:
                    q = t.format(fam=fam, g=g)
                    yield qa_item(q, ans, just, refs)
        else:
            base_prompts = [
                f"What is {fam}, and is it quantum-safe for key exchange?",
                f"Explain {fam}. PQC status?",
                f"Is {fam} quantum-safe for KEX?",
                f"{fam}: definition and PQ suitability?",
                f"Should we keep using {fam} during PQC transition?"
            ]
            ans = (f"{fam} is a classical KEX; not quantum-safe. Use Kyber KEM or a TLS 1.3 hybrid.")
            just = [f"{fam} = classical KEX", "Not quantum-safe", "Use Kyber or hybrid TLS 1.3"]
            refs = ["SHOR", "FIPS203", "TLS13", "OQS"]
            for q in base_prompts:
                yield qa_item(q, ans, just, refs)

    # --- Core PQC algorithm definitions (multi-variant) ---
    kyber_prompts = [
        "What is Kyber?",
        "What is the ML-KEM (Kyber) algorithm used for?",
        "Kyber (ML-KEM): definition and usage?",
        "Explain ML-KEM (Kyber) for practitioners.",
        "Is Kyber used for key exchange or signatures?"
    ]
    kyber_answer = ("ML-KEM (Kyber) is a post-quantum KEM standardized by NIST. "
                    "It establishes shared secrets—e.g., via hybrid TLS 1.3 key_share groups.")
    for p in kyber_prompts:
        yield qa_item(p, kyber_answer,
                      ["Kyber ML-KEM definition", "Post-quantum key exchange", "Used in hybrid TLS"],
                      ["FIPS203","TLS13","OQS"])

    dilithium_prompts = [
        "What is ML-DSA (Dilithium)?",
        "Explain the Dilithium signature scheme.",
        "Dilithium (ML-DSA): definition and use cases?",
        "Is Dilithium intended to replace RSA/ECDSA?",
        "Where do we use Dilithium in practice?"
    ]
    dilithium_answer = ("ML-DSA (Dilithium) is a lattice-based signature standardized in FIPS-204, "
                        "intended to replace RSA/ECDSA in certs and code signing.")
    for p in dilithium_prompts:
        yield qa_item(p, dilithium_answer,
                      ["Dilithium = lattice-based PQC signature", "FIPS-204 standardized", "Quantum-safe certificates"],
                      ["FIPS204"])

    sphincs_prompts = [
        "What is SLH-DSA (SPHINCS+)?",
        "Is SPHINCS+ hash-based?",
        "SPHINCS+: definition and trade-offs?",
        "When should we use SPHINCS+ vs Dilithium?",
        "Does SPHINCS+ have larger signatures?"
    ]
    sphincs_answer = ("SLH-DSA (SPHINCS+) is a stateless hash-based signature (FIPS-205). "
                       "Quantum-resistant with larger signatures and slower signing.")
    for p in sphincs_prompts:
        yield qa_item(p, sphincs_answer,
                      ["Hash-based PQC signature", "FIPS-205 standardized", "Larger signatures/slower signing"],
                      ["FIPS205"])

    falcon_prompts = [
        "What is Falcon-512 and when is it used?",
        "Falcon signatures: definition and pros/cons?",
        "Is Falcon-512 suitable for constrained/verifier-heavy cases?",
        "Falcon vs Dilithium: when to pick Falcon?",
        "Do Falcon signatures tend to be small?"
    ]
    falcon_answer = ("Falcon-512 is a lattice-based signature with compact signatures and fast verification—attractive for constrained or verifier-heavy settings. "
                     "Track policy/standardization; many environments default to Dilithium first.")
    for p in falcon_prompts:
        yield qa_item(p, falcon_answer,
                      ["Compact signatures", "Fast verification", "Policy may prefer Dilithium"],
                      ["OQS","NIST_PQC_HUB"])

    # (Optionally include brief alternates for completeness)
    mce_prompts = [
        "Classic McEliece: what is it and is it quantum-safe?",
        "Explain Classic McEliece and its trade-offs.",
        "Is Classic McEliece viable despite large public keys?",
        "Classic McEliece status in 2025?",
        "Should we track Classic McEliece for interop?"
    ]
    mce_answer = ("Classic McEliece is a code-based KEM believed quantum-resistant; the trade-off is very large public keys. "
                  "It remains an alternate with long study history.")
    for p in mce_prompts:
        yield qa_item(p, mce_answer,
                      ["Code-based and long-studied", "No known breaks", "Very large public keys"],
                      ["MCELIECE"])

    ntru_prompts = [
        "What is NTRU / NTRU Prime?",
        "Is NTRU considered quantum-safe?",
        "Where is NTRU/NTRU Prime used today?",
        "NTRU vs Kyber: positioning?",
        "Should we consider NTRU in designs?"
    ]
    ntru_answer = ("NTRU/NTRU Prime are lattice-based KEMs considered quantum-safe with no efficient attacks; "
                   "an NTRU Prime variant is used in OpenSSH hybrids.")
    for p in ntru_prompts:
        yield qa_item(p, ntru_answer,
                      ["Lattice-based KEM", "No efficient attacks", "Used in hybrids"],
                      ["NTRU","OPENSSH"])

    # --- Levels/parameters explainer (multi-variant) ---
    level_prompts = [
        "What do the levels in ML-DSA (e.g., Dilithium2/3/5) and ML-KEM (Kyber-512/768/1024) mean?",
        "Explain security levels: Dilithium2/3/5 and Kyber-512/768/1024.",
        "How should we choose between Dilithium2/3/5 and Kyber-512/768/1024?",
        "Do higher Dilithium/Kyber levels just cost more CPU/bytes?",
        "Which levels are typical for web vs high-assurance use?"
    ]
    level_answer = ("They reflect increasing security/performance trade-offs. 2/512 target lower cost; "
                    "3/768 is a common web choice; 5/1024 provides larger margins at higher cost.")
    for p in level_prompts:
        yield qa_item(p, level_answer,
                      ["Levels = security/cost trade-off", "Kyber-768 common on web", "Higher levels cost more"],
                      ["FIPS203","FIPS204"])



def g_tls_suite_reasoning():
    # --- TLS 1.3 AEAD suites with tailored notes (variants per suite) ---
    suite_variants = [
        "Why is {suite} considered suitable in TLS 1.3?",
        "Is {suite} a good AEAD choice for TLS 1.3 in a PQC plan?",
        "Does {suite} fit a PQC-ready TLS 1.3 stack?",
        "Is {suite} acceptable for long-term TLS 1.3 deployments?",
        "Should we prefer {suite} when enabling hybrid key_share?"
    ]
    for suite in TLS13_SUITES:
        if suite == "TLS_AES_128_GCM_SHA256":
            note = "Strong today; for very long-lived secrecy consider AES-256-GCM for a larger post-quantum margin."
        elif suite == "TLS_AES_256_GCM_SHA384":
            note = "High-margin choice with SHA-384; pairs well with hybrid KEX for long-term resilience."
        elif suite == "TLS_CHACHA20_POLY1305_SHA256":
            note = "Great on non-AES hardware (mobile/ARM). Security-wise comparable to AES-GCM; margin governed by key length (128-bit)."
        elif suite == "TLS_AES_128_CCM_SHA256":
            note = "Standards-compliant AEAD; typically slower than GCM in common stacks."
        else:  # TLS_AES_128_CCM_8_SHA256
            note = "Shortened tag variant for constrained contexts; avoid as a general default."
        ans = (f"{suite} is an AEAD cipher suite defined for TLS 1.3. TLS 1.3 permits only AEADs, which pair well with hybrid key_share deployments. {note}")
        for t in suite_variants:
            yield qa_item(t.format(suite=suite),
                          ans,
                          ["AEAD suite in TLS 1.3", "Drops weak ciphers", "Works with hybrid key_share"],
                          ["TLS13"])

    # --- TLS 1.2 suites during transition (variants) ---
    t12_variants = [
        "Can we use {suite} during the PQC transition?",
        "Is {suite} acceptable while we migrate to PQC?",
        "Does {suite} in TLS 1.2 meet PQC needs?",
        "Is {suite} fine as an interim before TLS 1.3 hybrids?",
        "Should we keep {suite} until we roll out hybrids?"
    ]
    for suite in TLS12_SUITES:
        ans = (f"{suite} is TLS 1.2. Use only as an interim; TLS 1.2 has no standard hybrid/PQC KEX. "
               "Prefer migrating services to TLS 1.3 to enable hybrid key_share and prepare for PQC certificates.")
        for t in t12_variants:
            yield qa_item(t.format(suite=suite),
                          ans,
                          ["TLS 1.2 interim only", "No hybrid KEX in TLS 1.2", "Move to TLS 1.3"],
                          ["TLS13","TLS_DEPREC"])

    # --- Extra reasoning seeds (variants) ---
    why13_prompts = [
        "Why does TLS 1.3 help with post-quantum migration more than TLS 1.2?",
        "TLS 1.3 vs 1.2: why better for PQC?",
        "What makes TLS 1.3 a better PQC baseline?",
        "Why is TLS 1.3 preferred for hybrid key_share?",
        "PQC migration: benefits of TLS 1.3 over 1.2?"
    ]
    why13_answer = ("TLS 1.3 mandates AEAD ciphers, streamlines the handshake, and supports hybrid key_share groups—enabling PQC readiness without legacy baggage.")
    for p in why13_prompts:
        yield qa_item(p, why13_answer,
                      ["AEAD-only requirement", "Simpler handshake", "Hybrid groups enabled"],
                      ["TLS13"])

    aead_margin_prompts = [
        "Between TLS_AES_128_GCM_SHA256 and TLS_AES_256_GCM_SHA384, which should we pick for very long-lived data?",
        "Long-term TLS choice: 128-GCM or 256-GCM?",
        "Which AEAD offers a larger symmetric margin against Grover?",
        "For long-lived secrecy, prefer AES-128-GCM or AES-256-GCM?",
        "PQC timelines: 128-bit vs 256-bit AEAD in TLS 1.3?"
    ]
    aead_margin_answer = ("Both are strong; choose AES-256-GCM if you want a larger symmetric margin against Grover, balancing any performance impact.")
    for p in aead_margin_prompts:
        yield qa_item(p, aead_margin_answer,
                      ["Both secure today", "AES-256-GCM has larger margin", "Balance performance vs margin"],
                      ["TLS13","NIST_PQC_FAQ"])

    chacha_prompts = [
        "Is TLS_CHACHA20_POLY1305_SHA256 acceptable in a PQC-ready stack?",
        "ChaCha20-Poly1305 in TLS 1.3: OK for PQC plans?",
        "Should we keep ChaCha20-Poly1305 when moving to hybrids?",
        "Does ChaCha20-Poly1305 remain strong under PQ considerations?",
        "Is ChaCha20-Poly1305 fine on mobile/ARM in PQC migrations?"
    ]
    chacha_answer = ("Yes. It’s a modern AEAD and widely implemented. Under PQ considerations, symmetric ciphers remain robust; "
                     "prefer 256-bit keys when available for very long-term data.")
    for p in chacha_prompts:
        yield qa_item(p, chacha_answer,
                      ["ChaCha20-Poly1305 is AEAD", "PQ impact limited on symmetric", "Prefer larger key sizes for longevity"],
                      ["TLS13","NIST_PQC_FAQ"])

    deprec_prompts = [
        "Why are TLS 1.0 and 1.1 discouraged for use now?",
        "Is it OK to keep TLS 1.0/1.1 enabled during PQC transition?",
        "TLS 1.0/1.1 status in modern deployments?",
        "Should we disable TLS 1.0/1.1 when planning PQC?",
        "Are TLS 1.0/1.1 deprecated?"
    ]
    deprec_answer = ("Deprecated per RFC 8996 due to security issues. TLS 1.3 is recommended (AEAD-only, supports hybrid key_share).")
    for p in deprec_prompts:
        yield qa_item(p, deprec_answer,
                      ["RFC 8996 deprecates TLS1.0/1.1", "Old versions have issues", "TLS 1.3 recommended"],
                      ["TLS_DEPREC","TLS13"])


def g_ipsec_ssh():
    # ----- IPsec (classical -> PQC) -----
    ipsec_q_variants = [
        "Our IPsec VPN uses {auth} for authentication and {kex} for key exchange. Is this PQC-safe?",
        "IPsec setup: auth={auth}, KEX={kex}. Quantum-resistant or not?",
        "With IPsec using {auth} (auth) and {kex} (KEX), are we post-quantum secure?",
        "Is an IPsec configuration with {auth} + {kex} acceptable in a PQC migration?",
        "Evaluate IPsec ({auth} auth, {kex} key exchange): PQC status?"
    ]
    for auth, kex in itertools.product(
        ["RSA-2048", "RSA-3072", "RSA-4096", "ECDSA P-256", "ECDSA P-384", "ECDSA P-521", "Ed25519"],
        ["DH Group 14", "DH Group 15", "DH Group 16", "ECDH P-256", "ECDH P-384", "ECDH P-521", "X25519", "X448"]
    ):
        ans = (f"No. Both {auth} and {kex} are classical and fall to a large quantum computer. "
               "Adopt ML-KEM (Kyber) for key establishment—ideally as part of a hybrid—and move authentication to PQC signatures (Dilithium/SPHINCS+).")
        for t in ipsec_q_variants:
            q = t.format(auth=auth, kex=kex)
            yield qa_item(q, ans,
                          ["Classical auth & KEX are quantum-vulnerable", "Use Kyber (PQC) for key exchange", "Use Dilithium/SPHINCS+ for auth"],
                          ["FIPS203","FIPS204","FIPS205","CNSA2"])

    # ----- IKEv2 migration specifics -----
    ike_prompts = [
        "How can IKEv2 move toward PQC without breaking existing peers?",
        "IKEv2 PQC migration: how to phase in without disrupting interoperability?",
        "What’s the safe path to add PQC to IKEv2 while keeping classical peers working?",
        "Practical steps for IKEv2 to adopt PQC with backward compatibility?",
        "How do we introduce hybrids in IKEv2 without outages?"
    ]
    ike_answer = ("Use RFC 8784 to inject PQ pre-shared keys, and RFC 9370 to negotiate multiple key exchanges (classical+PQC) "
                  "in one IKE_SA. Phase in PQC while retaining classical fallback.")
    for p in ike_prompts:
        yield qa_item(p, ike_answer,
                      ["RFC 8784: PQ PSKs", "RFC 9370: multi-KEX", "Phased migration with fallback"],
                      ["RFC8784","RFC9370","CNSA2"])

    # ----- SSH specifics -----
    ssh_sig_variants = [
        "Is {sig} sufficient for post-quantum SSH authentication?",
        "{sig} for SSH host/user keys: PQC-safe?",
        "SSH using {sig} signatures — quantum-resistant?",
        "Should we keep {sig} for SSH in a PQC plan?",
        "Evaluate {sig} for SSH auth under PQC."
    ]
    ssh_sig_ans = ("No. {sig} is classical. For quantum-resistant SSH, use PQC signatures once standardized (e.g., Dilithium or SPHINCS+). "
                   "Bigger classical keys don’t fix quantum attacks.")
    for sig in ["Ed25519", "Ed448", "ECDSA P-256", "ECDSA P-384", "RSA-2048", "RSA-3072", "RSA-4096"]:
        for t in ssh_sig_variants:
            q = t.format(sig=sig)
            a = ssh_sig_ans.format(sig=sig)
            yield qa_item(q, a,
                          ["Classical algorithm (not quantum-safe)", "Need PQC signatures for SSH", "Larger classical keys don’t help"],
                          ["SHOR","FIPS204","FIPS205"])

    # sntrup761 hybrid explainer (variants)
    sntrup_prompts = [
        "What does sntrup761x25519-sha512@openssh.com provide in SSH?",
        "Explain sntrup761x25519-sha512@openssh.com: is it hybrid/PQC?",
        "SSH KEX sntrup761x25519-sha512@openssh.com — what security does it give?",
        "Is sntrup761x25519-sha512@openssh.com quantum-resistant in SSH?",
        "How does the sntrup761x25519-sha512@openssh.com method work?"
    ]
    sntrup_answer = ("A hybrid KEX: X25519 (ECDH) + sntrup761 (NTRU Prime), hashed with SHA-512. Session secrets derive from both for quantum resistance "
                     "and compatibility; host-key signatures remain classical unless upgraded.")
    for p in sntrup_prompts:
        yield qa_item(p, sntrup_answer,
                      ["Hybrid SSH KEX", "Combines X25519 + sntrup761", "Auth still classical unless updated"],
                      ["OPENSSH","OQS"])

    # RSA size myth (variants)
    rsa_size_prompts = [
        "Is increasing RSA key size in IPsec/SSH a viable quantum mitigation?",
        "Will RSA-4096 or RSA-8192 make IPsec/SSH quantum-safe?",
        "Do bigger RSA keys protect IPsec/SSH against quantum attacks?",
        "Is upsizing RSA a short-term PQC fix for VPN/SSH?",
        "Does increasing RSA key size help against Shor in practice?"
    ]
    rsa_size_answer = ("No. It improves classical security but does not resist Shor’s algorithm. Use hybrid KEX and transition to PQC signatures.")
    for p in rsa_size_prompts:
        yield qa_item(p, rsa_size_answer,
                      ["Key size ≠ quantum safety", "Use hybrid/PQC instead", "Plan migration"],
                      ["SHOR","FIPS203","FIPS204","FIPS205"])



def g_validation_rich():
    # ----- Helpers (unchanged core logic) -----
    def _gen_valid_ipv4():
        return f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    def _gen_invalid_ipv4():
        options = [
            f"{random.randint(256,999)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            f"{random.randint(0,255)}..{random.randint(0,255)}.{random.randint(0,255)}",
            f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}. 12",
            f"01.02.03.004a",
            f"-1.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            f"{random.randint(0,255)}.{random.randint(0,255)}.xyz.{random.randint(0,255)}"
        ]
        return random.choice(options)
    def _gen_hostname(valid=True):
        letters = "abcdefghijklmnopqrstuvwxyz"
        digits = "0123456789"
        def label():
            L = random.randint(1, min(20, 63))
            chars = letters + letters.upper() + digits + "-"
            s = "".join(random.choice(chars) for _ in range(L))
            if s[0] == "-": s = "a" + s[1:]
            if s[-1] == "-": s = s[:-1] + "b"
            return s
        if valid:
            labels = [label() for _ in range(random.randint(2,4))]
            host = ".".join(labels)
            if len(host) > 253:
                host = host[:253]
            return host
        else:
            bads = [
                "-dashstart.com", "space in name.com", ".leadingdot.com", "a..b.com",
                "toolonglabel-" + "a"*60 + ".com", "label-ends-.bad", "bad_host"
            ]
            return random.choice(bads)
    def _gen_port(valid=True):
        if valid:
            return random.choice([22, 53, 80, 123, 443, 587, 993, 3306, 5432, 8080, 65535])
        else:
            return random.choice([-1,0,65536, "eighty", "", 1.5, None, "443/tcp"])

    # ----- Prompt variant templates -----
    ipv4_valid_templates = [
        "Is '{val}' a valid IPv4 address?",
        "Validate IPv4: '{val}' — is it well-formed?",
        "Does '{val}' conform to IPv4 format (x.x.x.x, 0–255)?",
        "IPv4 check for '{val}': valid or not?",
        "Is the address '{val}' syntactically valid IPv4?"
    ]
    ipv4_invalid_templates = [
        "Is '{val}' a valid IPv4 address?",
        "IPv4 syntax check: '{val}' — valid?",
        "Does '{val}' qualify as a proper IPv4 address?",
        "Validate IPv4 string '{val}': correct format?",
        "Is '{val}' correctly formed IPv4?"
    ]
    rfc1918_templates = [
        "Is '{val}' an RFC 1918 private IPv4 address?",
        "Private vs public: is '{val}' in RFC 1918 space?",
        "Does '{val}' fall within RFC 1918 ranges?",
        "Classify '{val}': RFC 1918 private or public?",
        "Is '{val}' considered a private IPv4 (RFC 1918)?"
    ]
    host_valid_templates = [
        "Is '{val}' a valid ASCII hostname (LDH rule)?",
        "Hostname check for '{val}': valid under LDH?",
        "Does '{val}' satisfy LDH (letters/digits/hyphens, 1–63 per label)?",
        "Validate hostname '{val}': LDH compliant?",
        "Is '{val}' syntactically valid as a hostname?"
    ]
    host_invalid_templates = [
        "Is '{val}' a valid ASCII hostname (LDH rule)?",
        "Does '{val}' violate LDH hostname rules?",
        "Hostname syntax check: '{val}' — valid?",
        "Is '{val}' acceptable as an LDH hostname?",
        "Validate '{val}' against LDH constraints."
    ]
    port_valid_templates = [
        "Is '{val}' a valid TCP port number?",
        "Port check: is '{val}' within 0–65535?",
        "Validate TCP port '{val}': allowed range?",
        "Does '{val}' qualify as a valid port?",
        "Is '{val}' acceptable as a TCP port?"
    ]
    port_invalid_templates = [
        "Is '{val}' a valid TCP port number?",
        "Check '{val}' as a TCP port: valid?",
        "Is '{val}' within the 0–65535 port range?",
        "Validate '{val}' as a TCP port value.",
        "Does '{val}' represent a valid TCP port?"
    ]

    # ----- IPv4 validity (valid) -----
    for _ in range(120):
        ip = _gen_valid_ipv4()
        ans = "Yes. It’s a correctly formatted IPv4 address (four octets in the range 0–255)."
        for t in ipv4_valid_templates:
            q = t.format(val=ip)
            yield qa_item(q, ans,
                          ["IPv4 = four octets 0–255", "Dot-separated", "Range checking required"],
                          ["RFC791"])

    # ----- IPv4 validity (invalid) -----
    for _ in range(120):
        ip = _gen_invalid_ipv4()
        ans = "No. A valid IPv4 address must have exactly four numbers (0–255) separated by dots."
        for t in ipv4_invalid_templates:
            q = t.format(val=ip)
            yield qa_item(q, ans,
                          ["IPv4 formatting rules", "Octets in 0–255", "Exactly four octets"],
                          ["RFC791"])

    # ----- RFC 1918 private ranges -----
    for _ in range(120):
        ip = _gen_valid_ipv4()
        is_priv = is_private_ipv4(ip)
        ans = ("Yes, that address is within the private IPv4 ranges (RFC 1918)."
               if is_priv else
               "No, that IP is not in the private ranges (public IP).")
        for t in rfc1918_templates:
            q = t.format(val=ip)
            yield qa_item(q, ans,
                          ["RFC 1918 private ranges", "10/8, 172.16/12, 192.168/16", "Public vs private"],
                          ["RFC1918"])

    # ----- Hostname validity (valid) -----
    for _ in range(120):
        h = _gen_hostname(valid=True)
        ans = ("Yes. Meets LDH rules (labels 1–63 chars, letters/digits/hyphens, no leading/trailing hyphen; total ≤253). "
               "IDNs should be punycode.")
        for t in host_valid_templates:
            q = t.format(val=h)
            yield qa_item(q, ans,
                          ["LDH: letters/digits/hyphens", "No edge hyphens", "Total length ≤253"],
                          ["RFC1123","RFC5890"])

    # ----- Hostname validity (invalid) -----
    for _ in range(120):
        h = _gen_hostname(valid=False)
        ans = ("No. Hostnames must use letters/digits/hyphens per label (1–63 chars), not start/end with hyphen, total length ≤253.")
        for t in host_invalid_templates:
            q = t.format(val=h)
            yield qa_item(q, ans,
                          ["Violates LDH constraints", "Label length or characters invalid", "Must not start/end with '-'"],
                          ["RFC1123","RFC5890"])

    # ----- TCP port validity (valid) -----
    for _ in range(120):
        p = _gen_port(valid=True)
        ans = "Yes. Port numbers are integers 0–65535 inclusive."
        for t in port_valid_templates:
            q = t.format(val=p)
            yield qa_item(q, ans,
                          ["Ports must be integer 0–65535", "0 and 65535 are extremes", "IANA port registry applies"],
                          ["IANA_PORTS"])

    # ----- TCP port validity (invalid) -----
    for _ in range(120):
        p = _gen_port(valid=False)
        ans = "No. Valid TCP ports are integers between 0 and 65535."
        for t in port_invalid_templates:
            q = t.format(val=p)
            yield qa_item(q, ans,
                          ["Non-integer or out of range", "Valid range 0–65535", "Check parsing errors"],
                          ["IANA_PORTS"])



def g_iot_expanded():
    # ---- Parametric device matrix with prompt variants ----
    rams = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    mhzs = [16, 32, 40, 48, 80, 96, 120, 150, 200, 240, 300]
    sigs = ["ECDSA", "Ed25519", "Ed448"]

    iot_prompt_variants = [
        "IoT node ({ram} KB RAM, {mhz} MHz) uses {sig} for signatures. PQC option?",
        "Constrained device {ram}KB/{mhz}MHz currently on {sig}. What PQC signature should we pick?",
        "For an IoT MCU ({ram}KB RAM @ {mhz}MHz) with {sig} today, what’s the PQC replacement?",
        "Device profile: {ram}KB RAM, {mhz}MHz CPU, signatures via {sig}. Post-quantum plan?",
        "We ship {sig} on {ram}KB/{mhz}MHz nodes. Which PQC algorithm is practical?"
    ]
    iot_answer = (
        "For constrained devices, Falcon-512 (small signatures, fast verification) is attractive. "
        "Dilithium2 is standardized and simpler to implement but has larger signatures; "
        "SPHINCS+ is hash-based, larger still, but with minimal assumptions."
    )
    for ram, mhz, sig in itertools.product(rams, mhzs, sigs):
        for t in iot_prompt_variants:
            q = t.format(ram=ram, mhz=mhz, sig=sig)
            yield qa_item(
                q, iot_answer,
                [f"{sig} is not quantum-safe", "Falcon-512: compact & fast verify", "Dilithium2: standardized; SPHINCS+ minimal assumptions"],
                ["FIPS204","FIPS205"]
            )

    # ---- Curated IoT scenarios with variants ----
    scenario_blocks = [
        (
            [
                "Battery IoT sensor verifies firmware at boot; which PQC signature is practical?",
                "Secure-boot on a battery sensor: which PQC signature fits best?",
                "For boot-time firmware validation on a tiny sensor, which PQC signature should we choose?",
                "Firmware authenticity at boot on constrained nodes — Falcon, Dilithium, or SPHINCS+?",
                "What PQC signature minimizes boot-time overhead for sensors?"
            ],
            "Prioritize verification speed and signature size: Falcon-512 keeps updates small and verifies quickly. "
            "Dilithium2 is simpler and standardized; if flash/RAM allow, it’s a safe default. "
            "SPHINCS+ works where simplicity of assumptions matters most."
        ),
        (
            [
                "Low-power node signs telemetry; verifier is a powerful cloud service. PQC choice?",
                "Telemetry signing on device, verification in cloud — which PQC signature?",
                "For sensor-side signing and cloud-side verification, what PQC signature is sensible?",
                "MCU signs, backend verifies: best PQC signature option?",
                "What PQC signature fits low-power signers with strong verifiers?"
            ],
            "Signing cost on-device matters: Dilithium2 has relatively fast, simple signing; "
            "Falcon-512 signing is feasible but more complex; "
            "SPHINCS+ signing is slower/larger—use only if you strongly prefer hash-based designs."
        ),
        (
            [
                "DTLS 1.3/CoAP for IoT: what to do for key establishment?",
                "For CoAP over DTLS 1.3, which KEM choice makes it PQC-ready?",
                "IoT DTLS stack: how do we enable post-quantum key establishment?",
                "DTLS on constrained devices — classical vs Kyber?",
                "How should DTLS/CoAP integrate PQC KEMs?"
            ],
            "Use Kyber (ML-KEM) for the KEM part (or hybrid classical+Kyber during transition). "
            "Keep AEAD as specified; symmetric ciphers remain strong under PQ considerations."
        ),
        (
            [
                "Will larger PQC signatures break our bootloader update channel (MTU limits)?",
                "Do PQC signature sizes cause fragmentation for firmware updates?",
                "Update transport MTU vs PQC signatures — what should we expect?",
                "Are Dilithium/Falcon signatures going to exceed our FOTA packet limits?",
                "How to handle firmware signature size increases on constrained links?"
            ],
            "Possibly. Dilithium2 signatures are ~2–3 KB vs ~0.7 KB for Falcon-512. "
            "Plan chunking/fragmentation and measure flash/EEPROM overhead before rollout."
        ),
        (
            [
                "We use MQTT/TLS with client certs on constrained devices. Migration path?",
                "MQTT over TLS on small MCUs — how to phase in PQC?",
                "Client-cert IoT (MQTT/TLS): what’s the PQC migration plan?",
                "How do we bring PQC to MQTT client certs without breaking devices?",
                "TLS client certs on IoT: sequence for PQC rollout?"
            ],
            "Enable TLS 1.3 hybrids (e.g., X25519+Kyber) first for forward secrecy, then migrate client/server certs to Dilithium2 or Falcon-512 as toolchains mature."
        ),
        (
            [
                "Do we need PQC attestation for IoT device identity, not just TLS?",
                "IoT identity/attestation: should we adopt PQC signatures too?",
                "Beyond transport, do device identities need PQC?",
                "Is PQC required for device certificates and attestation tokens?",
                "What about PQC for onboarding/attestation credentials?"
            ],
            "Yes. Transport secrecy comes from Kyber in TLS/DTLS, but device identity and attestation also need PQC signatures. "
            "Plan issuance with Dilithium or Falcon (policy permitting) and test verifier acceptance."
        ),
    ]
    for prompts, ans in scenario_blocks:
        for p in prompts:
            yield qa_item(
                p, ans,
                ["Constrained resources drive choice", "Hybrid KEX first, then PQC certs", "Test MTU/flash impact"],
                ["FIPS203","FIPS204","FIPS205","TLS13","NCCOE_MPQC"]
            )

    # ---- Quick chooser (with variants) ----
    chooser_blocks = [
        (
            [
                "Which PQC signature should small MCUs start with?",
                "Best first PQC signature for tiny MCUs?",
                "On constrained microcontrollers, what’s the default PQC signature?",
                "For small embedded targets, which PQC signature do we begin with?",
                "Starter choice for MCU-class PQC signatures?"
            ],
            "Default to Dilithium2 for simplicity/availability; pick Falcon-512 when bandwidth/verify speed are critical; "
            "reserve SPHINCS+ for environments favoring hash-based schemes despite size."
        ),
        (
            [
                "Is Ed25519 with bigger keys a quantum fix for IoT?",
                "Can larger Ed25519/ECDSA keys make IoT quantum-safe?",
                "Would upsizing classical ECC keys protect IoT devices from quantum attacks?",
                "Does increasing ECC key size help against Shor for IoT?",
                "Is bigger-key Ed25519 an acceptable PQC mitigation?"
            ],
            "No. Bigger classical keys do not resist Shor. Move to PQC signatures and Kyber for KEM."
        ),
    ]
    for prompts, ans in chooser_blocks:
        for p in prompts:
            yield qa_item(
                p, ans,
                ["Dilithium2 as safe default", "Falcon-512 when size/verify dominate", "Shor breaks classical ECC"],
                ["SHOR","FIPS203","FIPS204","FIPS205"]
            )


def g_smartcards():
    # ---- Core scenarios with variants ----
    blocks = [
        (
            [
                "A payment processor signs transactions on smartcards using ECDSA P-256. What is the PQC transition plan?",
                "Smartcard estate on ECDSA P-256 — how do we move to PQC?",
                "Payments smartcards use ECDSA P-256 today. Migration path to PQC?",
                "We issue P-256 smartcards; what’s the PQC plan?",
                "Card-based ECDSA P-256 signatures — post-quantum roadmap?"
            ],
            "Adopt Falcon-512 on new cards (tiny signatures, fast verification). If policy requires, use Dilithium2 with increased APDU fragmentation. "
            "Dual-sign (ECC+PQC) during transition and update terminals/HSMs to verify PQC."
        ),
        (
            [
                "Smartcard applets use Ed25519 for signatures. Which PQC signature fits and why?",
                "Ed25519 on cards today — Falcon or Dilithium for PQC?",
                "For card applets currently on Ed25519, which PQC signature is practical?",
                "Choosing a PQC signature for Ed25519-based smartcards?",
                "Card signature migration from Ed25519 — recommended PQC?"
            ],
            "Use Falcon-512 for compact signatures and quick verification; Dilithium2 if standardized-only policy is required. "
            "Update readers/middleware to accept PQC OIDs and larger certs."
        ),
        (
            [
                "Can we keep RSA-2048 on smartcards and be quantum-safe if we use a larger key?",
                "Is upsizing RSA on smartcards enough for quantum resistance?",
                "Would RSA-4096 on cards be PQC-secure?",
                "Does a larger RSA modulus make card signatures safe vs quantum?",
                "Smartcard RSA key length vs quantum safety — does it help?"
            ],
            "No. Shor’s algorithm breaks RSA at any size. Implement PQC signatures (Dilithium or Falcon) and phase in dual-signing until relying parties are upgraded."
        ),
        (
            [
                "Will larger PQC signatures fit in EMV/PIV smartcard flows?",
                "Do PQC signature sizes break EMV/PIV data paths?",
                "APDU/chain impact of PQC signatures in EMV/PIV?",
                "Will cert chains/APDUs grow too much with PQC on cards?",
                "Can EMV/PIV ecosystems carry PQC signature sizes?"
            ],
            "They can, but expect bigger APDUs and certificate chains. Falcon-512 reduces on-wire size; Dilithium2 increases it—test terminal interop and issuer scripts."
        ),
        (
            [
                "Do we need new card profiles and OIDs for PQC?",
                "Are profile/OID updates required for PQC on cards?",
                "Card profiles for PQC: what changes?",
                "Will middleware/readers need PQC OIDs for card support?",
                "X.509/OID considerations for PQC smartcards?"
            ],
            "Yes. Card profiles, middleware, and CA policies must recognize PQC algorithms and OIDs; plan issuance/reader updates in lockstep."
        ),
        (
            [
                "Is dual-signing on-card advisable?",
                "Should cards emit both ECC and PQC signatures during transition?",
                "On-card dual signatures: good practice for migration?",
                "Card dual-sign (classic+PQC) — recommended?",
                "Do we support dual-signed artifacts on smartcards?"
            ],
            "Yes, during transition. Produce both ECC and PQC signatures so legacy verifiers continue working while PQC-capable verifiers validate the new algorithm."
        ),
        (
            [
                "On-card key generation vs off-card import for PQC — which is practical first?",
                "Should we generate PQC keys on the card or import them initially?",
                "PQC key management on cards: on-card gen or off-card provisioning?",
                "Early PQC rollout: off-card keygen/import for smartcards?",
                "Best practice for PQC keys on cards in phase 1?"
            ],
            "Early deployments often start with off-card key generation and import, then move to on-card keygen as vendor firmware matures. "
            "Ensure issuer/HSM tooling understands PQC key formats and certificate OIDs."
        ),
        (
            [
                "What’s a pragmatic validation plan for PQC smartcards?",
                "How do we test PQC smartcards end-to-end before scale-out?",
                "Acceptance criteria to roll out PQC on cards?",
                "Pilot validation steps for PQC card issuance?",
                "How to measure readiness for PQC smartcard expansion?"
            ],
            "Pilot small BIN/issuer ranges, validate APDU timing and failure rates, ensure terminals/HSMs verify PQC chains, and confirm dual-sign fallbacks. "
            "Proceed when interop KPIs (success rates/latency) stay green."
        ),
    ]
    for prompts, ans in blocks:
        for p in prompts:
            yield qa_item(
                p, ans,
                ["Shor breaks RSA/ECC", "Falcon vs Dilithium trade-offs", "Dual-sign during migration"],
                ["SHOR","FIPS204","FIPS205","NCCOE_MPQC"]
            )



def g_storage_wrap():
    # ---- Matrix with prompt variants (envelope: data cipher + DEK wrap) ----
    matrix_variants = [
        "We encrypt backups with {enc} and wrap DEKs with {wrap}. Is this PQC-secure?",
        "Current setup: data uses {enc}, DEKs wrapped by {wrap}. Quantum-safe?",
        "Backups: {enc} for payload, {wrap} for key wrap — post-quantum status?",
        "Is our envelope encryption PQC-ready? Data: {enc}; DEK wrapping: {wrap}.",
        "Does {wrap} wrapping make our {enc}-encrypted backups quantum-resistant?"
    ]
    for wrap, enc in itertools.product(
        ["RSA-2048","RSA-3072","RSA-4096","RSA-8192"],
        ["AES-128","AES-256","ChaCha20-Poly1305"]
    ):
        if enc == "AES-128":
            aes_note = "AES-128 is acceptable today; under Grover the effective work is ~2^64,"
        else:
            aes_note = f"{enc} is robust vs quantum (symmetric ciphers hold up well),"
        a = (f"{aes_note} but {wrap} is classical and not quantum-safe. Replace RSA wrapping with ML-KEM (Kyber) "
             "to encapsulate the DEK.")
        for t in matrix_variants:
            q = t.format(enc=enc, wrap=wrap)
            yield qa_item(
                q, a,
                ["Symmetric cipher margins under Grover", f"{wrap} not quantum-safe", "Use Kyber for key wrapping"],
                ["FIPS203","NIST_PQC_FAQ"]
            )

    # ---- Curated scenarios & best practices (each with prompt variants) ----
    scenario_blocks = [
        (
            [
                "We use RSA-OAEP to wrap DEKs in our KMS. What’s the PQC alternative?",
                "KMS uses RSA-OAEP for DEK wrapping — how to make this PQC-ready?",
                "What replaces RSA-OAEP for DEK wrap in a post-quantum design?",
                "PQC plan for key wrapping in KMS that currently uses RSA-OAEP?",
                "How do we convert RSA-OAEP DEK wrapping to a quantum-safe approach?"
            ],
            "Use a KEM-DEM approach with ML-KEM (Kyber) for key encapsulation and an AEAD (e.g., AES-256-GCM) for payload "
            "encryption. This preserves envelope encryption and crypto agility.",
            ["KEM-DEM with Kyber", "Keep AEAD for payload", "Maintain crypto agility"],
            ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"]
        ),
        (
            [
                "Do we need to rewrap old backup DEKs immediately?",
                "Should we rewrap historical backup keys right away for PQC?",
                "Priority for rewrapping archived DEKs under PQC?",
                "How urgent is rewrapping legacy DEKs for backups?",
                "When should we rewrap old DEKs vs rotating on schedule?"
            ],
            "Prioritize rewrapping for archives with long confidentiality lifetimes. New material should use Kyber; schedule "
            "a phased rewrap of historic DEKs as part of rotation.",
            ["Prioritize long-lived archives", "New DEKs via Kyber", "Phase historic rewraps"],
            ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"]
        ),
        (
            [
                "Can HPKE help for object storage/key wrapping?",
                "Is HPKE a good way to modernize key wrapping for objects?",
                "HPKE for envelope encryption: does swapping KEM to Kyber work?",
                "How does HPKE enable PQC key wrapping for blobs?",
                "Could we adopt HPKE with Kyber for object encryption headers?"
            ],
            "Yes. HPKE composes KEM+KDF+AEAD. Swapping DHKEM for Kyber yields a PQC-ready envelope while keeping HKDF/AEAD "
            "unchanged.",
            ["HPKE = KEM+KDF+AEAD", "Swap DHKEM→Kyber", "Keep HKDF/AEAD layers"],
            ["HPKE","FIPS203","NCCOE_MPQC"]
        ),
        (
            [
                "Tape archives: AES-128 data, RSA-4096 wrapped DEKs. Risk?",
                "We used AES-128 and RSA-4096 wraps for tapes — what’s the quantum risk?",
                "Archive tapes with RSA-wrapped DEKs: what should we change?",
                "Is RSA-4096 DEK wrapping the weak link for long-lived tapes?",
                "How to mitigate HNDL on tapes using RSA-wrapped DEKs?"
            ],
            "Data cipher is acceptable short-term; the RSA wrapping is the weak link under HNDL. Move to Kyber-wrapped DEKs and "
            "consider re-encrypting highly sensitive sets.",
            ["HNDL targets key wrapping", "Kyber-wrap DEKs", "Re-encrypt sensitive sets as needed"],
            ["CISA_QR_FACTS","FIPS203","NCCOE_MPQC"]
        ),
    ]
    for prompts, ans, just, refs in scenario_blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)

    # ---- Operational guidance (variants) ----
    ops_blocks = [
        (
            [
                "What operational steps make storage encryption quantum-ready?",
                "Ops checklist to make envelope encryption PQC-ready?",
                "How do we operationalize PQC for backups and object storage?",
                "Practical steps to harden storage encryption for PQC?",
                "What should we change in ops to be PQC-ready for key wrapping?"
            ],
            "Adopt Kyber for DEK wrap, prefer AES-256-GCM for payloads, enable crypto-agile metadata (algorithm IDs/versions), "
            "and automate periodic rewrap/rotation.",
            ["Kyber for wrap; AES-256 preferred", "Record alg/version for agility", "Automate rewrap/rotation"],
            ["FIPS203","TLS13","NIST_PQC_FAQ"]
        ),
        (
            [
                "Is ChaCha20-Poly1305 fine for backups in a PQC plan?",
                "Can we keep ChaCha20-Poly1305 for data at rest post-quantum?",
                "ChaCha20-Poly1305 vs AES-GCM for PQC storage — acceptable?",
                "Do symmetric choices change for PQC backups?",
                "Are we okay with ChaCha20-Poly1305 while moving to Kyber wraps?"
            ],
            "Yes. It’s a modern AEAD; symmetric primitives remain strong. The urgent change is replacing RSA wrapping with Kyber.",
            ["Symmetric AEADs remain strong", "Focus on replacing RSA wrap", "Kyber is the priority change"],
            ["NIST_PQC_FAQ","TLS13","FIPS203"]
        ),
    ]
    for prompts, ans, just, refs in ops_blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)


def g_quantum_risk():
    # ---- Risk & mitigation, with prompt variants and tight answers ----
    blocks = [
        (
            [
                "What is the 'harvest now, decrypt later' risk and how do we mitigate it?",
                "Explain HNDL and how to defend against it.",
                "What does harvest-now-decrypt-later mean for my org?",
                "If attackers record traffic today, how do we stay safe post-quantum?",
                "HNDL risk: what immediate steps reduce exposure?"
            ],
            "Adversaries record ciphertexts today and wait for quantum attacks to break RSA/ECC later. Mitigate by enabling "
            "TLS 1.3 hybrid key_share now (e.g., X25519+Kyber), migrating certificates/signatures to PQC over time, and "
            "rewrapping stored keys (DEKs/KEKs) with Kyber in your KMS.",
            ["HNDL explains urgency", "Hybrid TLS today", "Rewrap keys with Kyber"],
            ["CISA","NCSC","TLS13","NIST_PQC_FAQ"]
        ),
        (
            [
                "Does AES-128 fall immediately to quantum computers?",
                "Is AES-128 broken by quantum today?",
                "Do we need to stop using AES-128 right now because of quantum?",
                "Quantum effect on AES-128 vs AES-256 — what’s the deal?",
                "Is AES-128 acceptable during PQC transition?"
            ],
            "No. Grover’s algorithm gives a square-root speedup, making AES-128 roughly 2^64 effective work. It’s acceptable "
            "today, but prefer AES-256 for long-lived confidentiality.",
            ["Grover halves 128-bit", "AES-128 OK now", "Prefer AES-256 long-term"],
            ["NIST_PQC_FAQ","TLS13"]
        ),
        (
            [
                "Which data should we prioritize against 'harvest now, decrypt later'?",
                "What information is most at risk from HNDL?",
                "How do we triage data classes for quantum risk?",
                "Which datasets should migrate first for PQC protection?",
                "Prioritizing for HNDL: where to start?"
            ],
            "Prioritize data with long confidentiality lifetimes: health/financial records, legal/diplomatic material, trade "
            "secrets, backups/archives, and telemetry that retains strategic value.",
            ["Prioritize long-lived data", "Risk from stored ciphertexts", "Act before quantum arrival"],
            ["CISA","NCSC","NIST_PQC_FAQ"]
        ),
        (
            [
                "Should we re-encrypt data-at-rest or just rewrap keys?",
                "Re-encrypt vs rewrap for PQC — which first?",
                "Fastest quantum-risk reduction for storage: rewrap or re-encrypt?",
                "Do we need full re-encryption or only DEK/KEK rewrap?",
                "What’s the right order: Kyber rewrap then bulk re-encrypt?"
            ],
            "Start by rewrapping DEKs/KEKs with Kyber (fastest risk reduction). Re-encrypt bulk data when feasible or during "
            "natural rotation; focus first on sensitive long-term sets.",
            ["Rewrap first for speed", "Bulk re-encrypt later", "Target sensitive archives"],
            ["FIPS203","NIST_PQC_FAQ","NCCOE_MPQC"]
        ),
        (
            [
                "Is quantum key distribution (QKD) required to be quantum-safe?",
                "Do we need QKD hardware if we adopt PQC?",
                "Is QKD necessary for compliance on post-quantum security?",
                "PQC vs QKD — which should we implement?",
                "Can we be quantum-safe without QKD?"
            ],
            "No. Policies and standards emphasize software-based PQC (Kyber/Dilithium/SPHINCS+) over QKD. QKD isn’t required "
            "for compliance or broad interoperability today.",
            ["PQC is software-based", "Kyber/Dilithium/SPHINCS+ focus", "Interop/compliance first"],
            ["CISA","NCSC","NIST_PQC_FAQ"]
        ),
        (
            [
                "How do we validate a TLS 1.3 hybrid rollout practically?",
                "What KPIs confirm our hybrid TLS deployment is healthy?",
                "Operational checks for enabling X25519+Kyber in production?",
                "How to canary and measure a hybrid TLS rollout?",
                "What telemetry proves hybrid key_share is safe to scale?"
            ],
            "Track handshake success rates, ClientHello/ServerHello sizes, MTU/fragmentation, and CPU impact. Roll out by "
            "cohort with canaries and fallbacks to classical when peers lack hybrid.",
            ["Measure handshake health", "Watch MTU/CPU overhead", "Use staged rollout"],
            ["TLS13","NCCOE_MPQC","OQS"]
        ),
        (
            [
                "Does QUIC 0-RTT change the PQC story?",
                "Is 0-RTT with QUIC affected by PQC adoption?",
                "Any special PQC concerns for QUIC/h3 0-RTT?",
                "QUIC early data and PQC — what should we know?",
                "Does hybrid KEX alter QUIC 0-RTT risks?"
            ],
            "0-RTT replay semantics are unchanged. PQC helps the forward secrecy of new sessions via hybrid KEX, but 0-RTT "
            "remains replay-prone; treat as before.",
            ["0-RTT replay unchanged", "Hybrid boosts PFS", "Treat 0-RTT carefully"],
            ["QUIC","TLS13","NIST_PQC_FAQ"]
        ),
        # Myth-busters with variants
        (
            [
                "Will larger RSA/ECC keys protect us from quantum attacks?",
                "Is switching to RSA-8192 or P-521 enough for quantum safety?",
                "Do bigger classical keys stop Shor’s algorithm?",
                "Can we delay PQC by using much larger RSA/ECC keys?",
                "Does key size alone mitigate quantum risk for public-key crypto?"
            ],
            "No. Shor’s algorithm breaks RSA and ECC regardless of key size. Move to Kyber for KEM and Dilithium/SPHINCS+ for signatures.",
            ["Shor breaks RSA/ECC", "Adopt Kyber + PQC sigs", "Key size doesn’t save classical"],
            ["SHOR","FIPS203","FIPS204","FIPS205"]
        ),
        (
            [
                "Do we need to replace symmetric ciphers to be quantum-safe?",
                "Are AES-GCM/ChaCha20-Poly1305 obsolete in a PQC world?",
                "Should we swap out symmetric algorithms when adopting PQC?",
                "Symmetric crypto under quantum: keep or change?",
                "Do AEAD choices change when moving to PQC?"
            ],
            "Generally no. Modern AEADs remain strong; the main change is adopting PQC for key establishment and signatures "
            "while favoring 256-bit keys for margin.",
            ["Symmetric largely fine", "Prefer 256-bit keys", "Focus KEM/signature swap"],
            ["NIST_PQC_FAQ","TLS13","FIPS203"]
        ),
    ]

    for prompts, ans, just, refs in blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)


def g_regulatory():
    blocks = [
        (
            [
                "Under NSA CNSA 2.0, is TLS with X25519 and RSA-2048 compliant?",
                "Does a stack using X25519 key exchange and RSA-2048 certs meet CNSA 2.0?",
                "Are X25519 KEX and RSA-2048 signatures acceptable under CNSA 2.0?",
                "CNSA 2.0 check: TLS with classical X25519 + RSA—compliant or not?",
                "For CNSA 2.0, can we keep X25519 and RSA-2048 in production TLS?"
            ],
            "No. CNSA 2.0 calls for quantum-resistant solutions. Use TLS 1.3 with a hybrid key_share (e.g., X25519+Kyber) and adopt PQC signatures (Dilithium/SPHINCS+) as the ecosystem matures.",
            ["Classical not compliant", "Hybrid TLS 1.3 needed", "Adopt PQC signatures"],
            ["CNSA2","TLS13","FIPS203","FIPS204","FIPS205"]
        ),
        (
            [
                "What does OMB M-23-02 require federal agencies to do about PQC?",
                "OMB M-23-02: what are the concrete PQC tasks for agencies?",
                "Per OMB M-23-02, what steps must agencies take for PQC migration?",
                "Which PQC actions are mandated by OMB M-23-02?",
                "How does OMB M-23-02 guide our PQC roadmap?"
            ],
            "Inventory cryptography, identify vulnerable systems, prioritize migrations, publish roadmaps, and report progress annually through the 2030s.",
            ["Inventory & prioritize", "Roadmap & reporting", "Multi-year transition"],
            ["OMB_M2302"]
        ),
        (
            [
                "What does NSM-10 say about timelines for PQC migration?",
                "NSM-10 timeline: when should systems be quantum-resistant?",
                "How does NSM-10 frame the PQC transition schedule?",
                "NSM-10 expectations for achieving quantum resilience?",
                "By when does NSM-10 aim to mitigate quantum risk?"
            ],
            "It directs a timely transition to quantum-resistant cryptography, aiming to mitigate as much risk as feasible by 2035 across national systems.",
            ["Policy sets end-state", "Aim ≈2035 mitigation", "Plan early, iterate"],
            ["NSM10"]
        ),
        (
            [
                "Are organizations outside government expected to follow these policies?",
                "Do private companies need to align with CNSA 2.0/NIST PQC guidance?",
                "Should non-federal orgs mirror federal PQC policies?",
                "Why should private orgs track CNSA 2.0 and NIST PQC?",
                "Are federal PQC directives relevant to commercial teams?"
            ],
            "They’re not mandatory for private orgs, but they’re strong signals. Aligning with CNSA 2.0/NIST guidance reduces vendor risk and eases audits.",
            ["De facto market baseline", "Easier vendor alignment", "Audit-friendly approach"],
            ["CISA_QR_FACTS","NIST_PQC_HUB"]
        ),
        (
            [
                "Does policy require TLS 1.3 and deprecate older TLS?",
                "Is TLS 1.3 the baseline for PQC migration by policy?",
                "Are TLS 1.0/1.1 deprecated and TLS 1.2 insufficient for PQC?",
                "Do we need TLS 1.3 for hybrid KEX under current policies?",
                "Policy guidance on TLS versions during PQC transition?"
            ],
            "Yes, move to TLS 1.3. TLS 1.0/1.1 are deprecated, and TLS 1.2 lacks standardized hybrid KEX; treat TLS 1.3 as the PQC baseline.",
            ["TLS 1.3 is baseline", "Old TLS deprecated", "TLS 1.2 lacks hybrid"],
            ["TLS13","TLS_DEPREC"]
        ),
        (
            [
                "Is QKD required by policy to achieve quantum safety?",
                "Do current policies require QKD hardware for compliance?",
                "Is QKD necessary if we adopt standardized PQC?",
                "Policy stance: PQC vs QKD—what’s required?",
                "Can we be compliant without deploying QKD?"
            ],
            "No. Current federal guidance emphasizes standardized PQC algorithms (Kyber, Dilithium, SPHINCS+), not QKD.",
            ["PQC standards first", "No QKD requirement", "Focus on Kyber/Dilithium"],
            ["CISA_QR_FACTS","NIST_PQC_HUB"]
        ),
        (
            [
                "What quick wins demonstrate policy traction in 90 days?",
                "90-day PQC wins to show progress?",
                "Short-term actions to align with PQC policy?",
                "What near-term milestones prove PQC momentum?",
                "First-quarter PQC deliverables for leadership?"
            ],
            "Publish a crypto inventory/BOM, enable TLS 1.3 hybrids on a pilot tier, start Kyber rewrap in KMS for new DEKs, and define issuance policy for PQC/dual-signed certs.",
            ["Inventory + hybrid pilot", "Kyber rewrap for DEKs", "Define PQC issuance"],
            ["OMB_M2302","CNSA2","NCCOE_MPQC"]
        ),
    ]
    for prompts, ans, just, refs in blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)


def g_regional_guidance():
    blocks = [
        (
            [
                "How do ETSI and ENISA view PQC migration?",
                "ETSI/ENISA guidance: what are the PQC next steps?",
                "What’s the European stance (ETSI/ENISA) on PQC rollout?",
                "According to ETSI/ENISA, how should we transition to PQC?",
                "ETSI/ENISA: recommended path for PQC adoption?"
            ],
            "Adopt TLS 1.3, use hybrid KEX as an interim step, and move to standardized PQC algorithms (Kyber for KEM; Dilithium/SPHINCS+ for signatures). Emphasize interoperability planning and staged rollouts.",
            ["TLS 1.3 + hybrids", "Standardized PQC", "Interop planning"],
            ["ETSI_QSC","ENISA_PQC","TLS13","FIPS203","FIPS204","FIPS205"]
        ),
        (
            [
                "UK NCSC next steps for PQC?",
                "What does the UK NCSC recommend for PQC adoption?",
                "NCSC (UK) guidance: how to start with PQC?",
                "Practical PQC actions per UK NCSC?",
                "How should UK orgs sequence PQC tasks?"
            ],
            "Inventory systems, enable TLS 1.3 everywhere, trial hybrids in controlled segments, and coordinate vendor/CA readiness for PQC certificates.",
            ["Inventory and assess", "Hybrid trials first", "Vendor/CA coordination"],
            ["NCSC_UK_PQC","TLS13"]
        ),
        (
            [
                "Germany BSI TR-02102-1 — how does it impact PQC plans?",
                "BSI TR-02102-1: implications for PQC migration?",
                "What does Germany’s BSI recommend for PQC?",
                "How to align PQC with BSI TR-02102-1?",
                "BSI guidance on symmetric margins and PQC?"
            ],
            "Follow BSI crypto recommendations; plan to move from classical algorithms to standardized PQC as toolchains mature, preferring AES-256 for symmetric margins.",
            ["BSI guidance baseline", "Plan PQC adoption", "Prefer AES-256"],
            ["BSI_TR_02102_1","TLS13","FIPS203"]
        ),
        (
            [
                "France ANSSI — PQC posture?",
                "What is ANSSI’s recommendation for PQC transition?",
                "ANSSI guidance: TLS and PQC—what to do?",
                "How does ANSSI suggest phasing in PQC?",
                "ANSSI view on hybrids and NIST algorithms?"
            ],
            "ANSSI recommends preparing for PQC with TLS 1.3, hybrid transition, and adoption of NIST-selected algorithms once profiles are established.",
            ["TLS 1.3 + hybrids", "Adopt NIST PQC", "Phased deployment"],
            ["ANSSI_PQC","TLS13","FIPS203","FIPS204","FIPS205"]
        ),
        (
            [
                "Canada CCCS — practical guidance?",
                "How does CCCS advise prioritizing PQC work?",
                "CCCS: which datasets should move first to PQC?",
                "Canadian guidance on TLS hybrids and PQC certs?",
                "CCCS approach to quantum risk triage?"
            ],
            "Assess quantum risk by data lifetime, enable TLS 1.3 hybrids, and plan for PQC certificates and key management, prioritizing long-lived sensitive data.",
            ["Lifetime-based triage", "Hybrid TLS rollout", "PQC certs + KMS"],
            ["CCCS_PQC","TLS13","FIPS203"]
        ),
        (
            [
                "Singapore CSA — adoption stance?",
                "CSA Singapore: what’s the PQC path?",
                "Singapore CSA guidance for starting PQC?",
                "How does Singapore’s CSA view PQC pilots?",
                "CSA advice on tracking PQC ecosystem readiness?"
            ],
            "Encourages early PQC planning, pilots with hybrid KEX, and tracking ecosystem support for PQC signatures and certificates.",
            ["Early planning", "Hybrid pilots", "Ecosystem tracking"],
            ["SG_CSA_PQC","TLS13"]
        ),
        (
            [
                "Australia ACSC — where to start?",
                "ACSC guidance for PQC migration steps?",
                "What does Australia’s ACSC recommend for PQC?",
                "ACSC: TLS, crypto-agility, and pilots—how to proceed?",
                "Australian posture on enabling PQC quickly?"
            ],
            "Harden with TLS 1.3, plan crypto agility, and test hybrids on key external interfaces before widening the rollout.",
            ["TLS 1.3 everywhere", "Crypto agility", "Pilot then scale"],
            ["AU_ACSC_PQC","TLS13"]
        ),
        (
            [
                "Japan CRYPTREC — considerations?",
                "How should we align with CRYPTREC on PQC?",
                "CRYPTREC perspective on adopting NIST PQC?",
                "Japan guidance for hybrids and PQC certificates?",
                "CRYPTREC and global interop for PQC—what to know?"
            ],
            "Monitor CRYPTREC recommendations while aligning to NIST PQC for global interoperability; stage adoption via hybrids and PQC certificates.",
            ["Align to NIST PQC", "Hybrid transition", "Global interop"],
            ["CRYPTREC_PQC","TLS13","FIPS203","FIPS204","FIPS205"]
        ),
    ]
    for prompts, ans, just, refs in blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)

def g_migration_playbooks():
    blocks = [
        (
            [
                "Propose a 3-step plan to move a bank from ECC-based cryptography to lattice-based PQC without breaking legacy clients.",
                "Bank migration: how do we shift from ECC to PQC in three steps while keeping legacy clients working?",
                "What’s a phased (3-step) bank plan to migrate from ECC to PQC?",
                "Give a 3-phase rollout for banks to adopt PQC without client breakage.",
                "How should a bank sequence ECC→PQC in three concrete steps?"
            ],
            "1) Enable TLS 1.3 everywhere and inventory crypto (protocols, certs, KMS wraps). "
            "2) Deploy hybrid KEX (e.g., X25519+Kyber768) and introduce dual-signed certs where possible; monitor MTU/CPU/handshake telemetry. "
            "3) Transition to PQC-only signatures and KEM as client/server support matures; shorten classical cert validity and retire ECC in phases.",
            ["TLS1.3 + inventory", "Hybrid & dual-sign", "PQC-only in phases"],
            ["NCCOE_MPQC","SP1800_38A","SP1800_38B","SP1800_38C","TLS13","FIPS203","FIPS204"]
        ),
        (
            [
                "How should an enterprise transition from using RSA-4096 for code signing to PQC algorithms?",
                "Enterprise code signing: plan to move from RSA-4096 to PQC?",
                "What is a practical path from RSA-4096 code signing to Dilithium/SPHINCS+?",
                "Steps to migrate code-signing workflows from RSA-4096 to PQC?",
                "How do we phase code-signing from classical RSA to PQC?"
            ],
            "Phase 1: Dual-sign binaries/artifacts (RSA + Dilithium) with deterministic build provenance. "
            "Phase 2: Ship PQC-only signatures once toolchains and verifiers are upgraded. "
            "Phase 3: Revoke/retire RSA paths; update policy/OIDs and attestation docs.",
            ["Dual-sign then cutover", "Toolchain readiness", "Policy/OID updates"],
            ["NCCOE_MPQC","FIPS204","FIPS205"]
        ),
        (
            [
                "What is a practical PQC rollout plan for IPsec/IKEv2 VPNs?",
                "IKEv2 VPNs: how do we adopt PQC without breaking peers?",
                "Plan a stepwise IKEv2 migration to PQC/hybrid.",
                "How to phase in PQC for IKEv2 at scale?",
                "IKEv2: concrete steps to add PQC KEX and signatures?"
            ],
            "Start with RFC 8784 (PQC PSKs) for interim protection, then use RFC 9370 multi-KEX to add hybrid DH+Kyber. "
            "Migrate authentication to PQC signatures as profiles/clients become available.",
            ["Interim PQ PSKs", "Hybrid multi-KEX", "Move auth to PQC"],
            ["RFC8784","RFC9370","CNSA2"]
        ),
        (
            [
                "How do we migrate enterprise S/MIME to PQC?",
                "S/MIME: steps to adopt PQC for encryption and signatures?",
                "What’s the S/MIME migration path to Kyber + Dilithium?",
                "Enterprise email: plan for PQC S/MIME without disruption.",
                "Rollout plan to enable PQC in S/MIME at scale?"
            ],
            "Issue dual-capable S/MIME profiles (classical + PQC) as CAs/tooling allow (see CA/B guidance). "
            "Prioritize executive/long-lived mailboxes, update gateways/DLP, and enforce PQC-only issuance on new enrollments once ecosystem readiness is proven.",
            ["Dual-capable profiles", "Prioritize long-lived mailboxes", "PQC-only when ready"],
            ["CABF_SMC013","LAMPS","FIPS204","FIPS205"]
        ),
        (
            [
                "What’s a zero-downtime PQC plan for a multi-region CDN + origins?",
                "CDN + origin: how to deploy PQC without outages?",
                "Plan hybrid TLS on CDN edge/origins with no downtime.",
                "How should a CDN roll out PQC across regions safely?",
                "Multi-region edge: steps to enable hybrids and PQC certs?"
            ],
            "Enable TLS 1.3 hybrid groups at the edge first, then extend to origin links. Canary by region, track handshake and error budgets, and gradually add PQC certificate signatures after hybrid stability.",
            ["Edge-first hybrid", "Canary + telemetry", "Then PQC certs"],
            ["TLS13","OQS","FIPS203"]
        ),
        (
            [
                "Service mesh / Kubernetes: how to phase in PQC?",
                "Kubernetes service mesh: steps to adopt PQC mTLS?",
                "What’s the plan to add hybrid KEX and PQC certs in a mesh?",
                "Mesh migration to PQC with minimal disruption?",
                "How do we roll out PQC in mTLS between services?"
            ],
            "Upgrade mesh to TLS 1.3, introduce hybrid KEX in sidecars or mTLS providers, rotate service certs to PQC, and ensure MTU/overhead are acceptable under real traffic.",
            ["Mesh TLS 1.3", "Hybrid mTLS", "Monitor MTU/overhead"],
            ["TLS13","NCCOE_MPQC","FIPS203","FIPS204"]
        ),
    ]

    # Acceptance / tracking criteria blocks (expanded)
    kpi_blocks = [
        (
            [
                "What acceptance criteria signal we’re ready to expand PQC beyond pilots?",
                "Which KPIs show we can scale PQC rollouts safely?",
                "Readiness gates to move from PQC pilot to broad deployment?",
                "What SLOs confirm PQC is stable enough to ramp?",
                "Define go/no-go metrics for PQC expansion."
            ],
            "≥99.9% hybrid TLS handshake success over 7 days; <1% MTU-related fragmentation increase; <5% peak CPU overhead on terminators; no cert pathing regressions; rollback tested and documented.",
            ["SLO-based rollout", "Watch MTU/CPU/regressions", "Rollback readiness"],
            ["NCCOE_MPQC","TLS13"]
        ),
        (
            [
                "How should we measure progress across teams?",
                "What does a PQC scorecard per system look like?",
                "How to track PQC posture by product/service?",
                "Which fields belong in a crypto/PQC BOM?",
                "What reporting structure helps leadership see PQC progress?"
            ],
            "Maintain a crypto BOM and PQC scorecard per system: TLS version, KEX mode, signature alg, KMS wrap alg, data-lifetime class, negotiated-hybrid rate, and planned cutover date.",
            ["Crypto BOM tracking", "Team-level accountability", "Cutover planning"],
            ["OMB_M2302","NCCOE_MPQC"]
        ),
        (
            [
                "What’s a minimal rollback plan for PQC pilots?",
                "Rollback criteria for hybrid TLS/KEM deployments?",
                "How do we design safe rollback for PQC changes?",
                "Rollback SLOs when hybrids cause regressions?",
                "What proves rollback is safe before ramping PQC?"
            ],
            "Pre-stage configs/feature flags; validate classical-only fallback; keep dual certificate chains live; automated canary aborts on KPI breach; practice rollback in staging with traffic replay.",
            ["Pre-stage fallbacks", "Automated aborts", "Rehearsed rollback"],
            ["NCCOE_MPQC","TLS13"]
        ),
    ]

    for prompts, ans, just, refs in blocks + kpi_blocks:
        for p in prompts:
            yield qa_item(p, ans, just, refs)



def g_positive_cases():
    """
    Expanded with 4–5 prompt variants per assessment type.
    """
    verdict_prompts = [
        "TLS 1.3 with {hy} key exchange, {sig} certificate, and {cipher}: is this PQC-safe?",
        "Config check: {hy} + {sig} + {cipher} — post-quantum ready?",
        "Is this TLS stack quantum-resistant: {hy} / {sig} / {cipher}?",
        "Does {hy} with {sig} cert and {cipher} meet PQC requirements?",
        "Would {hy}+{sig}+{cipher} resist a large quantum adversary?"
    ]
    caveat_prompts = [
        "We run TLS 1.3 with {hy}, {sig} certs, and {cipher}. Any PQC caveats?",
        "With {hy} + {sig} + {cipher}, what should we watch operationally?",
        "Are there deployment gotchas for {hy}/{sig}/{cipher}?",
        "What PQC risks remain in {hy} + {sig} + {cipher}?",
        "Before scaling {hy}/{sig}/{cipher}, what checks should we do?"
    ]
    improve_prompts = [
        "What (if anything) should we improve in this stack: {hy} + {sig} + {cipher}?",
        "How can we optimize {hy}/{sig}/{cipher} for PQC readiness?",
        "Tuning advice for {hy} + {sig} + {cipher} in production?",
        "What upgrades would strengthen {hy}/{sig}/{cipher} further?",
        "Any optimizations for {hy}/{sig}/{cipher} on mixed hardware?"
    ]
    full_vs_trans_prompts = [
        "Is {hy}/{sig}/{cipher} fully post-quantum or just transitional?",
        "Does {hy} + {sig} + {cipher} count as PQC-complete, or is it a transition profile?",
        "Classify {hy}/{sig}/{cipher}: fully PQC or hybrid-transition?",
        "Would you consider {hy}/{sig}/{cipher} end-state PQC?",
        "Is {hy}/{sig}/{cipher} PQC-complete for TLS today?"
    ]
    falcon_prompts = [
        "Any policy caveats when using Falcon-512 with {hy} and {cipher}?",
        "Compliance issues to consider for Falcon-512 in {hy}/{cipher} stacks?",
        "When should we prefer Dilithium over Falcon-512 for {hy} + {cipher}?",
        "Falcon-512 with {hy} and {cipher}: where could policy block us?",
        "Is Falcon-512 appropriate for regulated environments using {hy} + {cipher}?"
    ]

    for hy, sig, cipher in itertools.product(
        HYBRID_ALL,
        ["ML-DSA (Dilithium)", "SLH-DSA (SPHINCS+)", "Falcon-512"],
        ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]
    ):
        cipher_note = (
            "ChaCha20-Poly1305 is very strong on non-AES hardware."
            if cipher.endswith("CHACHA20_POLY1305_SHA256")
            else "AES-256-GCM offers a high security margin."
        )

        # Verdict
        a1 = (
            f"Yes. Hybrid {hy} includes Kyber (quantum-resistant), the certificate uses a PQC signature ({sig}), and {cipher_note} "
            "Overall this stack is post-quantum ready, assuming peers negotiate the hybrid and validate the PQC certificate chain."
        )
        for p in verdict_prompts:
            yield qa_item(
                p.format(hy=hy, sig=sig, cipher=cipher), a1,
                ["Hybrid KEX with Kyber", "PQC certificate signature", "Modern AEAD cipher"],
                ["TLS13","OQS","FIPS203","FIPS204","FIPS205","NIST_PQC_FAQ"]
            )

        # Caveats / checklist
        a2 = (
            "Looks good. Validate client compatibility and watch handshake/MTU overhead. "
            "Ensure your CA profile, OIDs, and verification paths accept the PQC algorithm; prefer Dilithium levels aligned to policy (e.g., ML-DSA level 2/3). "
            "Retain classical fallbacks only for legacy peers and shorten their validity."
        )
        for p in caveat_prompts:
            yield qa_item(
                p.format(hy=hy, sig=sig, cipher=cipher), a2,
                ["Monitor client compatibility", "Check PQC cert paths/OIDs", "Manage MTU/handshake overhead"],
                ["TLS13","OQS","FIPS203","FIPS204","FIPS205"]
            )

        # Improvements
        a3 = (
            f"You’re in good shape. Keep {cipher} as-is; prefer AES-256-GCM on AES-accelerated servers and ChaCha20-Poly1305 on CPU-bound edges. "
            "Enable telemetry (handshake success, CPU, fragmentation), and plan staged PQC-only issuance once ecosystem coverage is sufficient."
        )
        for p in improve_prompts:
            yield qa_item(
                p.format(hy=hy, sig=sig, cipher=cipher), a3,
                ["Cipher choice by hardware", "Telemetry-driven rollout", "PQC-only issuance later"],
                ["TLS13","OQS","FIPS203","FIPS204","FIPS205"]
            )

        # Fully vs transitional
        a4 = (
            "It’s effectively PQC-complete for TLS: session secrecy derives from Kyber via the hybrid, and authentication uses a PQC signature. "
            "It remains ‘transitional’ only insofar as classical components are still present for interop, which is fine."
        )
        for p in full_vs_trans_prompts:
            yield qa_item(
                p.format(hy=hy, sig=sig, cipher=cipher), a4,
                ["Secrecy via Kyber hybrid", "Auth via PQC signature", "Interop keeps classical present"],
                ["TLS13","OQS","FIPS203","FIPS204","FIPS205"]
            )

        # Falcon-specific policy caveats
        if sig == "Falcon-512":
            a5 = (
                "Falcon-512 is attractive for constrained/verifier-heavy scenarios. If your policy requires only finalized FIPS algorithms, "
                "prefer ML-DSA (Dilithium). Otherwise, Falcon can reduce signature sizes and verification cost."
            )
            for p in falcon_prompts:
                yield qa_item(
                    p.format(hy=hy, cipher=cipher), a5,
                    ["Falcon is compact/fast verify", "Policy may require Dilithium", "Choose per compliance needs"],
                    ["FIPS204","FIPS205","OQS"]
                )




# ---------------- Category registry & mix ----------------
GENERATORS = {
    # Broad / kept from d1.py
    "GENERIC_PQC": g_generic_pqc,
    "TLS_CANONICAL": g_tls_canonical,
    "TLS_CONFIGS": g_tls_configs,
    "HYBRID_DETAILS": g_hybrid_details,
    "IPSEC_IKEV2": g_ipsec_ikev2,
    "WIREGUARD_OPENVPN": g_wireguard_openvpn,
    "EMAIL_PKI_SMIME_PGP": g_email_pki_smime_pgp,
    "JOSE_COSE_WEBAUTHN": g_jose_cose_webauthn,
    "DNSSEC_DOH_DOT": g_dnssec_doh_dot,
    "KMS_HSM_ENVELOPE": g_kms_hsm_envelope,
    "CLOUD_CDN_PLATFORMS": g_cloud_cdn_platforms,
    "HPKE_KEMTLS_MLS": g_hpke_kemtls_mls,
    "BROKEN_ALTS": g_broken_alts,
    "COMPARISONS": g_comparisons,
    "VALIDATION_RICH": g_validation_rich,
    "IOT_EXPANDED": g_iot_expanded,
    "SMARTCARDS": g_smartcards,
    "STORAGE_WRAP": g_storage_wrap,
    "QUANTUM_RISK": g_quantum_risk,
    "REGULATORY": g_regulatory,
    "REGIONAL_GUIDE": g_regional_guidance,
    "MIGRATION_PLAYBOOKS": g_migration_playbooks,
    "POSITIVE_CASES": g_positive_cases,

    # Unique keepers from dataset1.py (no overlapping category kept)
    "DEFINITIONS": g_definitions,
    "TLS_SUITE_RSNG": g_tls_suite_reasoning,
    "IPSEC_SSH": g_ipsec_ssh,
}

DEFAULT_MIX = {
    # PQC core & migration focus (upweighted)
    "GENERIC_PQC": 0.045,
    "DEFINITIONS": 0.095,
    "TLS_CONFIGS": 0.120,
    "TLS_CANONICAL": 0.037,
    "HYBRID_DETAILS": 0.055,
    "HPKE_KEMTLS_MLS": 0.045,
    "MIGRATION_PLAYBOOKS": 0.040,
    "QUANTUM_RISK": 0.038,
    "POSITIVE_CASES": 0.028,
    "COMPARISONS": 0.035,

    # Transport/VPN/SSH (kept strong, more practical Qs)
    "IPSEC_IKEV2": 0.050,
    "IPSEC_SSH": 0.033,
    "WIREGUARD_OPENVPN": 0.022,
    "TLS_SUITE_RSNG": 0.020,

    # PKI/email/KMS/storage (where you added lots of depth)
    "EMAIL_PKI_SMIME_PGP": 0.041,
    "KMS_HSM_ENVELOPE": 0.038,
    "STORAGE_WRAP": 0.038,
    "SMARTCARDS": 0.030,

    # Web/platform rollout (edge/origin/browser)
    "CLOUD_CDN_PLATFORMS": 0.028,

    # Ecosystem & policy
    "REGULATORY": 0.030,
    "REGIONAL_GUIDE": 0.018,

    # Other protocol areas
    "DNSSEC_DOH_DOT": 0.016,
    "JOSE_COSE_WEBAUTHN": 0.025,
    "IOT_EXPANDED": 0.033,

    # Reference/alt algos (kept, but trimmed)
    "BROKEN_ALTS": 0.020,

    # Generic validators (trimmed hard; they’re not PQC)
    "VALIDATION_RICH": 0.020,
}

# Allow MIX override from CLI; keep only known keys, then normalize
if isinstance(args.get("MIX"), dict):
    for k in list(args["MIX"].keys()):
        if k not in GENERATORS:
            del args["MIX"][k]
    MIX = {**DEFAULT_MIX, **args["MIX"]}  # CLI > default
else:
    MIX = dict(DEFAULT_MIX)

total_prop = sum(MIX.values())
if total_prop <= 0:
    MIX = dict(DEFAULT_MIX)
    total_prop = sum(MIX.values())
MIX = {k: v/total_prop for k, v in MIX.items()}

def _count_targets(total_seeds: int) -> dict:
    counts = {cat: int(total_seeds * prop) for cat, prop in MIX.items()}
    delta = total_seeds - sum(counts.values())
    if delta != 0:
        largest_cat = max(counts, key=lambda c: counts[c])
        counts[largest_cat] += delta
    return counts

# ---------------- Output helpers ----------------
def _dump_sources(prefix: str):
    if not prefix:
        return
    jpath = Path(prefix + ".json")
    cpath = Path(prefix + ".csv")
    with jpath.open("w", encoding="utf-8") as jf:
        json.dump({"kb": KB, "urls": KB_URLS}, jf, ensure_ascii=False, indent=2)
    with cpath.open("w", encoding="utf-8", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["tag", "label", "url"])
        for tag, label in KB.items():
            w.writerow([tag, label, KB_URLS.get(tag, "")])

def _write_jsonl_sharded(items, out_path: Path, shard_size: int):
    if shard_size and shard_size > 0:
        n = len(items)
        shards = math.ceil(n / shard_size)
        for i in range(shards):
            part = items[i*shard_size:(i+1)*shard_size]
            spath = out_path.with_name(out_path.stem + f".part-{i+1:04d}.jsonl")
            with spath.open("w", encoding="utf-8") as f:
                for qa in part:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        return shards
    else:
        with out_path.open("w", encoding="utf-8") as f:
            for qa in items:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        return 1



# ---------------- Normalizers used by near-duplicate index ----------------
_WORD_RX = re.compile(r"[A-Za-z0-9\+\-/&_]+")


# The near-duplicate detection can be very slow for large N (O(N^2) comparisons).
# We keep it but you may choose to adjust thresholds or disable it for speed.
_ND_DOC_GRAMS = []
_ND_INV = defaultdict(set)
_ND_BKTS = defaultdict(set)

def _normalize_for_index(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"[\s\r\n]+", " ", t)
    # ... [same normalization rules as before] ...
    return t

def _token_trigrams(s: str) -> set[tuple[str, str, str]]:
    t = _normalize_for_index(s)
    toks = re.findall(r"[A-Za-z0-9\+\-/&_]+", t)
    if len(toks) < 3:
        toks += [""] * (3 - len(toks))
    return set(zip(toks, toks[1:], toks[2:]))

def _near_key(s: str) -> tuple[str, str, str]:
    t = _normalize_for_index(s)
    toks = re.findall(r"[A-Za-z0-9\+\-/&_]+", t) or [""]
    first = toks[0]
    second = toks[1] if len(toks) > 1 else ""
    last = toks[-1]
    return (first, second, last)

def _near_dupe(ikey: str,
               jacc_thresh: float = 0.85,
               contain_thresh: float = 0.90) -> bool:
    grams = _token_trigrams(ikey)
    bkey = _near_key(ikey)
    alt_bkey = (bkey[0], "", bkey[2])
    cand_ids = set()
    for g in grams:
        cand_ids.update(_ND_INV.get(g, ()))
        if len(cand_ids) > 1200:
            break
    if cand_ids:
        cand_ids &= (_ND_BKTS.get(bkey, set()) | _ND_BKTS.get(alt_bkey, set()))
    for idx in cand_ids:
        g2 = _ND_DOC_GRAMS[idx]
        inter = len(grams & g2)
        if inter == 0:
            continue
        if inter / float(min(len(grams), len(g2))) >= contain_thresh or \
           inter / float(len(grams | g2)) >= jacc_thresh:
            return True
    # Unique: insert into index
    doc_id = len(_ND_DOC_GRAMS)
    _ND_DOC_GRAMS.append(grams)
    _ND_BKTS[bkey].add(doc_id)
    _ND_BKTS[alt_bkey].add(doc_id)
    for g in grams:
        _ND_INV[g].add(doc_id)
    return False

if __name__ == "__main__":
    # Ensure VARIANTS_PER_PROMPT is at least 1 to avoid division by zero:
    VARIANTS_PER_PROMPT = max(1, VARIANTS_PER_PROMPT)
    SEEDS_TOTAL = max(1, math.ceil(TOTAL / VARIANTS_PER_PROMPT))
    counts = _count_targets(SEEDS_TOTAL)

    seed_pool = []
    for category, need in counts.items():
        gen = GENERATORS.get(category)
        if gen is None:
            continue
        items = list(gen())
        if not items:
            continue
        if len(items) >= need:
            picks = random.sample(items, need)
        else:
            shortfall = need - len(items)
            # random.choices will raise IndexError if items is empty:contentReference[oaicite:2]{index=2}, but we checked already.
            picks = items + random.choices(items, k=shortfall)
        seed_pool.extend(picks)

    expanded = []
    seen_fp = set()
    seen_instr = set()

    def add_item(item) -> bool:
        qtext = item.get("instruction", "")
        qtext = fix_instruction_grammar(_strip_wrappers(qtext))
        item["instruction"] = qtext
        if any(p.search(qtext) for p in SCANNER_BLACKLIST):
            return False
        ikey = norm_instruction(qtext)
        if ikey in seen_instr or _near_dupe(ikey):
            return False
        fp = hashlib.sha1(json.dumps(
            [qtext, item["response"]["answer"]],
            ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        if fp in seen_fp:
            return False
        seen_instr.add(ikey)
        seen_fp.add(fp)
        expanded.append(item)
        return True

    # Base seeds and paraphrase variants
    for seed in seed_pool:
        base = json.loads(json.dumps(seed, ensure_ascii=False))
        base["instruction"] = fix_instruction_grammar(apply_prompt_variants(base["instruction"]))
        add_item(base)
        for _ in range(max(0, VARIANTS_PER_PROMPT - 1)):
            v = json.loads(json.dumps(seed, ensure_ascii=False))
            v["instruction"] = fix_instruction_grammar(apply_prompt_variants(safe_paraphrase(v["instruction"])))
            if random.random() < 0.45:
                refs = v["context"]["kb_refs"]
                extra_refs = random.sample(list(KB.keys()), k=min(2, len(KB)))
                v["context"]["kb_refs"] = sorted(set(refs + extra_refs))
            add_item(v)

    # Top-up loop: continue until TOTAL items (or max_tries)
    suffixes = [
        "Bottom line: enable TLS 1.3 hybrids and plan for PQC certificates.",
        "Prefer AES-256-GCM for long-term symmetric margin.",
        "Note: hybrids negotiate only when both endpoints support them."
    ]
    tries = 0
    # Allow more attempts if needed to reach large TOTAL:
    max_tries = max(TOTAL * 5, 200_000)
    # Remove stale-limit so we don't stop early
    while len(expanded) < TOTAL and tries < max_tries:
        tries += 1
        v = json.loads(json.dumps(random.choice(expanded), ensure_ascii=False))
        v["instruction"] = fix_instruction_grammar(apply_prompt_variants(safe_paraphrase(v["instruction"])))
        if random.random() < 0.45:
            ans = v["response"]["answer"]
            if STYLE in ("mixed", "bulleted") and "\n• " not in ans and random.random() < 0.6:
                parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", ans) if p.strip()]
                bullets = ["• " + p.rstrip(".") for p in parts[:3]] or ["• " + ans]
                v["response"]["answer"] = "\n".join(bullets)
            else:
                v["response"]["answer"] = (
                    ans
                    + (" " if not ans.endswith((" ", "\n")) else "")
                    + random.choice(suffixes)
                ).strip()
        if random.random() < 0.4:
            refs = v["context"]["kb_refs"]
            extra_refs = random.sample(list(KB.keys()), k=min(2, len(KB)))
            v["context"]["kb_refs"] = sorted(set(refs + extra_refs))
        add_item(v)

    if len(expanded) > TOTAL:
        expanded = expanded[:TOTAL]

    shards = _write_jsonl_sharded(expanded, OUT, SHARD_SIZE)
    if DUMP_SOURCES:
        _dump_sources(DUMP_SOURCES)

    print(
        f"Wrote {len(expanded)} UNIQUE QA items to "
        f"{OUT if shards == 1 else OUT.with_name(OUT.stem + '.part-*.jsonl')} "
        f"(variants per seed: {VARIANTS_PER_PROMPT}; style={STYLE}; shards={shards})"
    )
       