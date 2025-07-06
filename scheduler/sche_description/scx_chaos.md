# scx_chaos

## Overview

scx_chaos is a variant of the scx_p2dq scheduler that adds chaos testing capabilities. It is based on the P2DQ (Preemptive Distributed Queuing) scheduler implementation.

## Description

This scheduler is designed for testing and debugging purposes, introducing controlled chaos into scheduling decisions to help identify potential issues in the scheduling framework or workloads.

## Features

- Based on scx_p2dq implementation
- Includes chaos testing functionality
- Useful for stress testing and debugging

## Use Case

This scheduler is primarily intended for development and testing environments where you want to stress test scheduling behavior and identify potential race conditions or edge cases.

## Production Ready?

No, this scheduler is designed for testing purposes only and should not be used in production environments.