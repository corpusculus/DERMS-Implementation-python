# Key Project Assumptions and Definitions

This document records all study assumptions made for the DERMS MVP project.
Reviewers and interviewers should read this to understand the scope and limitations.

---

## 1. Feeder Choices

| Role | Feeder |
|---|---|
| Development / debug | IEEE 13-bus or IEEE 34-bus |
| Final demonstration | IEEE 123-bus |

**Rationale:** Smaller feeder enables fast debugging. Larger feeder provides portfolio-quality final results.

---

## 2. PV Penetration Definition

$$
\text{PV penetration} = \frac{\text{total installed PV nameplate kW}}{\text{feeder peak load kW}}
$$

This is the definition used consistently throughout the study. It may differ from utility-specific definitions.

---

## 3. Hosting Capacity Definition

> "The maximum total PV nameplate capacity that can be connected under the chosen study assumptions while maintaining project voltage criteria over the selected simulation horizon."

This is a project-specific definition, not a universal utility standard.

---

## 4. Voltage Criteria

| Threshold | Value (pu) | Purpose |
|---|---|---|
| Lower planning limit | 0.95 | Hard limit — violation |
| Upper planning limit | 1.05 | Hard limit — violation |
| Soft lower | 0.97 | Warning band (informational) |
| Soft upper | 1.03 | Triggers reactive power absorption |
| Curtailment activation | 1.05 | Triggers active power curtailment |

---

## 5. Time Resolution

| Parameter | Value |
|---|---|
| Simulation type | Quasi-static time series (QSTS) |
| Horizon | 24 hours |
| Step size | 5 minutes |
| Timesteps per run | 288 |

---

## 6. DER Model Assumptions

- PV systems are modelled as aggregated resources at selected buses ($10$–$20$ DERs for the MVP).
- Each PV has a rated apparent power $S_i$ (kVA).
- Real power availability depends on the irradiance profile.
- Reactive power capability at any instant is:

$$
Q_{\max,i}(t) = \sqrt{S_i^2 - P_i(t)^2}
$$

- Positive $Q$ means reactive power injection (leading); negative $Q$ means absorption (lagging).

---

## 7. Standards Alignment Wording

Use **exactly** this wording when referencing standards:
- ✅ "IEEE 1547-2018 aligned inverter functionality"
- ✅ "functional mapping to standard concepts"
- ✅ "not a certification or formal compliance demonstration"
- ❌ Do **not** say "IEEE 1547 compliant DERMS"

---

## 8. Voltage Equipment Behavior

**Development feeder:** Regulator tap positions and capacitor states are **frozen** to isolate DERMS impact.

**Final feeder:** Primarily frozen controls; optionally one case with active legacy controls for comparison.

---

## 9. Communication Model

Real protocols (IEEE 2030.5, DNP3) are **not** implemented.

Instead:
- Measurements are read from OpenDSS via Python.
- Setpoints are written back through Python.
- Commands are logged to CSV/JSON as if issued by a physical DERMS.

---

## 10. Limitations

- QSTS only — no sub-second dynamics, no transient stability.
- Centralized perfect telemetry assumed — no communication delays or noise.
- Simplified inverter model — actual hardware dynamics are ignored.
- No formal IEEE certification or compliance testing.
- Results are valid only under the stated scenarios and feeder models.
