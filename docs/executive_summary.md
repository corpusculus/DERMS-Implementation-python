# DERMS MVP Executive Summary

## Overview

DERMS MVP is a reproducible Python and OpenDSS study showing how centralized control of distributed PV inverters can reduce feeder overvoltage during high-solar periods. The project models a 24-hour quasi-static time-series workflow, compares uncontrolled and controlled operating modes, and exports artifacts that make the results inspectable.

This repository is intended as a technical demonstration and portfolio project. It is not a production DERMS, not a compliance claim, and not a substitute for utility-grade planning studies.

## Problem

High PV penetration can raise feeder voltages above the 0.95 to 1.05 pu planning band during low-load, high-generation periods. In a baseline case with no DER coordination, the feeder can experience sustained midday overvoltage.

## Approach

The project uses:

- OpenDSS for feeder power-flow simulation
- Python for orchestration, controls, and analysis
- IEEE 13-bus for development runs
- IEEE 123-bus for larger final studies
- CSV input profiles for load, PV production, and DER placement

Control modes included in the repository:

- Baseline operation with no DERMS intervention
- Heuristic Volt-VAR control with curtailment only when voltage remains too high
- Optimization-based dispatch that prioritizes voltage correction with limited control effort
- Optional battery-supported mode for extended analysis

## What The Repository Produces

The runnable workflow exports:

- snapshot voltage files for feeder validation
- 24-hour QSTS result tables
- voltage and violation plots
- DER command logs
- comparison reports across control modes

## Qualitative Findings

Across representative runs, the controlled modes reduce overvoltage duration relative to the uncontrolled baseline. The heuristic controller provides an interpretable first layer of control, while the optimization-based controller is designed to achieve similar or better voltage performance with more selective use of reactive power and curtailment.

The repository is strongest as a reproducible engineering study:

- the inputs are explicit
- the commands are inspectable
- the outputs are written to predictable result directories
- the control logic is separated cleanly from feeder simulation and analysis

## Limitations

- QSTS only; no transient or sub-second dynamics
- centralized perfect telemetry assumption
- no communication latency or protocol stack
- simplified inverter behavior relative to field devices
- no claim of IEEE 1547 certification or formal compliance

## Recommended Public Positioning

Publish this project as a reproducible DERMS simulation case study. Keep the public messaging centered on:

- feeder voltage control under high PV penetration
- config-driven Python and OpenDSS workflow
- comparison between baseline, heuristic, and optimization control
- transparent study assumptions and exported results

Avoid publishing unfinished KPI tables or claiming production readiness. When fixed metrics are needed, regenerate them from the documented commands and publish the resulting artifacts alongside the code.
