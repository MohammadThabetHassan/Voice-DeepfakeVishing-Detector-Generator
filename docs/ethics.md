# Ethics Policy — Voice Deepfake Vishing Detector & Generator

## Purpose

This project is an academic graduation research project submitted at a UAE university. Its purpose is to:
1. Demonstrate the feasibility of detecting AI-synthesised (deepfake) voices.
2. Educate researchers and security professionals about voice-cloning technology.
3. Provide reproducible baselines for future research in voice anti-spoofing.

---

## Strict Use Limitations

This software is provided for **research and educational purposes only**. The following uses are **prohibited**:

- Using the voice generator to impersonate any real person without their written consent.
- Using the system to conduct vishing (voice phishing) attacks.
- Applying the tool to circumvent identity verification or biometric authentication systems.
- Commercial deployment without proper legal review.
- Any use that violates local or international laws (including UAE Federal Law on Combatting Cybercrimes).

---

## Consent Requirements

Before cloning any voice, the following must hold:

1. **Explicit written consent** from the speaker whose voice is being cloned.
2. The consent must specify: (a) the intended use, (b) the generated content, (c) the storage/retention policy.
3. Consent must be revocable at any time.
4. The generated voice must be clearly labelled as synthetic in any downstream use.

The UI enforces a mandatory consent checkbox before any generation is allowed.

---

## Data Privacy

- **No audio is stored** by the backend. Uploaded files are processed in memory and any temporary files are deleted immediately after the HTTP response is sent.
- No personally identifiable information (PII) is logged beyond minimal request metadata (timestamp, file size).
- Users should not upload recordings of third parties without consent.

---

## Transport Security

- HTTPS/TLS protects audio data **in transit**.
- TLS does **not** authenticate voice authenticity. End-to-end encryption does not prevent a deepfake audio from being transmitted securely. This distinction must be understood.

---

## Research Responsibility

Researchers using this codebase are responsible for:

- Complying with the terms of any dataset used (VoxCeleb, OSR, etc.).
- Not releasing trained models without disclosing their limitations.
- Citing this work appropriately in any publications.
- Reporting discovered vulnerabilities responsibly (not publicly exploiting them).

---

## References

- IEEE Code of Ethics: https://www.ieee.org/about/corporate/governance/p7-8.html
- ASVspoof Challenge Ethics: https://www.asvspoof.org/
- UAE Federal Decree-Law No. 34 of 2021 on Combatting Rumours and Cybercrime
