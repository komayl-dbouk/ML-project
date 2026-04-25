from __future__ import annotations

import pandas as pd


BENIGN_FAMILY = "Benign"


FAMILY_BY_LABEL: dict[str, str] = {
    "BenignTraffic": BENIGN_FAMILY,
    "Backdoor_Malware": "Malware",
    "BrowserHijacking": "WebAttack",
    "CommandInjection": "WebAttack",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DNS_Spoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce",
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    "MITM-ArpSpoofing": "Spoofing",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "SqlInjection": "WebAttack",
    "Uploading_Attack": "WebAttack",
    "VulnerabilityScan": "Recon",
    "XSS": "WebAttack",
}


def label_to_family(label: str) -> str:
    if label not in FAMILY_BY_LABEL:
        raise KeyError(f"No family mapping defined for label '{label}'")
    return FAMILY_BY_LABEL[label]


def add_family_column(df: pd.DataFrame, label_col: str = "label", family_col: str = "family") -> pd.DataFrame:
    out = df.copy()
    out[family_col] = out[label_col].map(label_to_family)
    missing = out[out[family_col].isna()][label_col].unique().tolist()
    if missing:
        raise ValueError(f"Missing family mapping for labels: {missing}")
    return out


def families_in_order() -> list[str]:
    seen: list[str] = []
    for label in FAMILY_BY_LABEL:
        family = FAMILY_BY_LABEL[label]
        if family not in seen:
            seen.append(family)
    return seen


def labels_for_family(family: str) -> list[str]:
    return [label for label, mapped_family in FAMILY_BY_LABEL.items() if mapped_family == family]

