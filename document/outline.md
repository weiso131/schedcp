# Towards Agentic OS: An LLM Agent Framework for Linux Schedulers

## I. Introduction

Operating system schedulers face a fundamental challenge: kernel policies cannot understand what applications need. This semantic gap leads to suboptimal performance across modern computing infrastructure. In cloud platforms, system administrators who manage schedulers are not the developers who understand application behavior. On personal devices, users lack the kernel expertise to optimize their systems for gaming or creative workloads. Meanwhile, workloads themselves are increasingly dynamic—a machine learning training job may shift from compute-intensive to I/O-bound phases, requiring different scheduling strategies that no human can adjust in real-time.

Recent advances in reinforcement learning have attempted to address scheduler optimization through automated parameter tuning. Systems like Decima and Firm have shown promise in specific domains. However, these approaches remain fundamentally limited: they cannot understand application-level requirements such as whether a workload is latency-critical or throughput-oriented. They miss optimization opportunities that require semantic understanding—for instance, prioritizing compilation tasks based on code dependencies could significantly accelerate software builds, but no existing system can automatically discover and implement such strategies.

The rapid advancement of Large Language Model (LLM) agents in 2024-2025 presents an unprecedented opportunity. Modern LLM agents can understand complex system behaviors, generate code, and reason about optimization strategies. However, our motivation experiments reveal significant challenges: automatically generating a basic FIFO scheduler from scratch required 33 minutes, 221 API calls with Claude Code through many trial-and-error iterations, costing ~$6, and sometimes the generated code degraded system performance. This highlights the need for carefully designed interfaces that balance AI capabilities with practical constraints. We propose treating these AI agents as expert system administrators—but with appropriate safety constraints and interfaces designed for their unique capabilities and limitations.

This paper presents the first framework for using fully automatic LLM agents to dynamically optimize Linux schedulers. Our framework can be leveraged by any AI agent, from open-source Gemini CLI to proprietary Agents like Claude Code. A key insight is that LLM agents operate on the control plane, not the data plane—while scheduling decisions occur at microsecond timescales in the kernel, workload patterns change at minute-to-hour timescales, well-suited for LLM optimization. Our system enables AI agents to select, configure, modify, or generate entirely new scheduling algorithms tailored to specific workloads. Built on the production-ready sched_ext infrastructure, our framework maintains a self-evolving library of schedulers that grows and improves through experience.

To achieve this vision while addressing the challenges revealed in our experiments, our design is guided by four key principles derived from treating AI agents as context engineering systems and similar to human experts: (1) **Decoupling** the "what to optimize" (AI's domain) from "how to observe and act" (system's domain); (2) **Context Balance** to provide sufficient information for decisions while controlling costs, and give feedback earlier; (3) **Composable Tools** that leverage LLM Agent's dynamic planning, code generation and tool usage capabilities; and (4) **Safety-First Design** that treats AI as potentially non-cautious actors requiring defensive interfaces. These principles ensure our system remains effective as AI models evolve while preventing catastrophic failures. We think these principles also are generalizable to other domains and systems.

Following these principles, we implement a framework with five core components: (1) static analysis and testing for safe code deployment, (2) scheduler libraries with reusable optimization patterns, (3) reinforcement learning for continuous improvement, (4) profiling tools for workload characterization, and (5) a unified interface enabling any LLM to optimize schedulers. This modular design evolves with AI capabilities without system redesign. Once generated, schedulers execute as native eBPF code with no LLM overhead in the critical path, achieving up to 80% speedup for Linux kernel builds and 50% latency reduction for scheduler benchmarks.

We make the following contributions:
• Design and implementation of the first comprehensive LLM agent framework for OS scheduler optimization
• A modular extension system that can be leveraged by any AI agent, from open-source to proprietary models
• A set of design principles for AI-system interfaces that balance capability, cost, and safety
• A self-evolving scheduler library that grows through experience and improves with model capabilities
• Evaluation demonstrating 30-80% performance improvements across diverse workloads
• Discussion on the challenges and future directions of AI-driven system optimizations.

The remainder of this paper is organized as follows. Section II provides background on scheduler evolution and LLM capabilities. Section III details our motivation through concrete experiments. Section IV presents our system design. Section V details implementation. Section VI evaluates performance across multiple workloads. Section VII discusses related work, Section VIII presents future work and impact, and Section IX concludes.

## II. Background

### A. Linux Scheduling and sched_ext

1. **Traditional Linux Schedulers**
   - CFS (Completely Fair Scheduler): Default scheduler focusing on fairness
   - Real-time schedulers: SCHED_FIFO, SCHED_RR for time-critical tasks
   - Limitations: One-size-fits-all approach, lack of application awareness

2. **sched_ext (Scheduler Extensions)**
   - BPF-based framework for custom schedulers
   - Allows safe kernel scheduler development without kernel modifications
   - Production-ready with minimal overhead
   - Enables rapid prototyping and deployment

3. **eBPF Technology**
   - Extended Berkeley Packet Filter for programmable kernel functionality
   - Safety guarantees through verification
   - JIT compilation for near-native performance
   - Wide adoption in networking, tracing, and now scheduling


4. **Infrastructure Challenges**
   - Separation of deployment and optimization roles
   - Heterogeneous workloads on shared infrastructure
   - Dynamic workload characteristics
   - Need for workload-specific optimizations

5. **Emerging Workload Types**
   - ML/AI training and inference
   - Serverless and Function-as-a-Service
   - Microservices with complex dependencies
   - Batch processing with varying job sizes

6. **Performance Requirements**
   - Latency-sensitive: Web services, databases
   - Throughput-oriented: Batch processing, analytics
   - Resource-efficient: Edge computing, IoT
   - Application-specific: Build systems, testing suites

### B. LLMs and Autonomous Agents

1. **Reinforcement Learning for OS**
   - Recent work at top conferences (OSDI, SOSP, EuroSys)
   - Focus on parameter tuning and policy learning
   - Examples: Decima (SIGCOMM'19), Firm (OSDI'20)
   - Limitations: Cannot understand application semantics

2. **LLMs for Code Generation**
   - Recent advances in code synthesis (Codex, Claude, GPT-4)
   - Ability to understand natural language specifications
   - Growing capability in systems programming
   - Challenge: Generating correct, efficient kernel code

3. **AI Agents and Autonomous Systems**
   - Evolution from simple automation to intelligent agents
   - Claude Code, GitHub Copilot Workspace, Devin
   - Capability to understand, plan, and execute complex tasks
   - 2024 as inflection point for agent capabilities

## III. Motivation

### A. The Semantic Gap Problem

1. **Domain Knowledge Gap in Modern Infrastructure**
   - In modern infrastructure, especially cloud platforms and serverless environments, system managers who optimize the system are not the ones who deploy the applications
   - System managers don't know the requirements and behavior of their workloads
   - **Edge and Personal Device Users**: Even more challenging for non-CS experts
     - Personal computer users are not programmers and don't know how to optimize applications
     - Edge device operators lack systems programming expertise
     - Gaming enthusiasts want better performance but lack kernel knowledge
     - Creative professionals (video editors, 3D artists) need optimized systems but aren't CS experts
   - Understanding workloads requires deep domain knowledge
   - Example: Traditional DevOps personnel cannot easily optimize ML workloads
   - LLM Agents can help understand workload patterns and requirements, providing specifications and suggestions for optimizations
   - **Democratization**: AI agents make expert-level optimization accessible to all users, not just system administrators

2. **Technical Complexity of Scheduler Development**
   - Designing and implementing new schedulers for Linux kernel requires deep domain knowledge
   - Steep learning curve for kernel programming and BPF development
   - LLM domain knowledge can help bridge this expertise gap

3. **Dynamic Workload Adaptation**
   - Workloads of target systems change over time
   - Impossible for humans to redesign scheduler algorithms every hour
   - AI can dynamically adapt to changing workload patterns

### B. Motivation Experiments

#### Experiment Setup
- Used most advanced fully automatic coding agent (Claude Code)
- Prompt: "write a scheduler in eBPF"

#### Results
- **Success**: AI successfully synthesized a basic FIFO scheduler without human help
- **Performance Issues**:
  - Generation time: 33 minutes
  - LLM API calls: 221
  - Multiple trial-and-error iterations
  - Web browsing and Linux source code searching required
  - Cost: ~$6
- **Comparison**: Experienced human takes only 5 minutes for same task

#### Key Observations

1. AI-generated code doesn't always improve system performance
   - Sometimes worse due to higher scheduler overhead
2. Cost and time are prohibitive for production use
3. Process requires wide privileges and extensive iterations

- Create comparison Human Expert vs Naive AI vs Our System
- Show cost breakdown and time analysis
- Include flowchart of failed attempts to highlight challenges

### C. Challenges （In one paragraph）

1. **Safety and Reliability**
   - How to ensure generated code won't break the system?
   - How to prevent soft-lockups, stalls, starvation?
   - How to avoid negative impact on target workloads?

2. **Efficiency and Cost**
   - How to ensure reasonable generation time?
   - How to minimize cost for production deployment?

## IV. System Design

### A. Design Principles

Keep in mind that we are building a system interface that can be used by AI. As the AI Agent becomes more powerful and general, all software we currently design will be used by and maintained by AI in next few years.

**Core Insight**: AI agents are fundamentally context engineering systems - they need sufficient information to make decisions but not so much that costs become prohibitive. It's the same as human experts and engineers when doing profiling and optimization: we need to use the right tool to collect the profiling data and write programable policy or improve alogorithm with the right framework. choosing the wrong toolset and framework will cost a lot of extra time, when time is money. As a system researcher, our goal is not to design better AI agents, but to design better system and interfaces that can be used by AI agents.

#### 1. **Decoupling and Role Separation for better AI model**

   - **AI Agents as Performance Engineers**: Treat AI agents like human experts
   - **Clear Interface Boundaries**: System provides tools and observations, AI gathers context information and provides decisions and actions
   - **Evolution-Ready Design**: Future models can leverage same interfaces without system changes
   - **Principle**: Separate "what to do" (AI problem) from "how to observe and act" (system problem)

#### 2. **Context and Feedback Balance for cost-efficiency**

   - **Context Engineering Challenge**: AI agents need sufficient information to make good decisions
   - **Information Filtering**: Not too much (cost/token limits), not too little (poor decisions)
   - **Adaptive Context Window**:
     - Start with minimal context for simple decisions
     - Progressively add detail for complex scenarios
     - Learn which context matters for which workload types
     - Adding feedback loop or previous results to the context window can help the agent make better decisions
   - **Cost-Efficiency Trade-off**: Expose APIs to control information granularity

#### 3. **Composable Tool Architecture for scalable**

   - **Programmable, Not Prescriptive**: Leverage LLM's code generation abilities as the orginal unix philosophy
   - **Tool Decomposition**:
     - Atomic tools for basic operations (read stats, modify parameters)
     - Compositional tools for complex workflows
     - No fixed workflows - AI decides tool sequences
   - **Dynamic Tool Chains**: AI can create new tool combinations for novel scenarios
   - **Examples**:
     - Basic: `get_cpu_stats()`, `set_scheduler_param()`
     - Composed: `profile_workload()` = multiple stat reads + analysis
     - Generated: AI writes custom analysis scripts on demand

#### 4. **Safety-First Interface Design for AI Agents**
**Core Principle**: Treat AI as a potentially non-cautious human that can make wrong decisions - design interfaces that inherently prevent catastrophic failures.

   - **Defensive API Design**: 
     - No single API call can crash the system
     - All operations have built-in bounds and limits and need to be validated before execution
     - Example: `set_cpu_limit()` has hard max of 95% to prevent starvation
   
   - **Staged Execution with Validation**:
     - Preview mode: Show what would happen before execution
     - Dry-run capabilities for all destructive operations
     - Mandatory validation checks between generation and deployment
   
   - **Constrained Action Space**:
     - Whitelist allowed operations, not blacklist dangerous ones
     - Graduated permissions: earn trust through successful operations
     - Example: New AI agents can only tune parameters within 20% of defaults
   
   - **Automatic Safety Rails**:
     - Resource limits enforced at system level, not trust AI to respect them
     - Automatic circuit breakers when performance degrades
     - Watchdog timers for all AI-initiated operations
   
   - **Human-in-the-Loop Fallback only when necessary**:
     - fully automatic system is the goal so we should only use human-in-the-loop when necessary
     - Critical operations require confirmation for first N times
     - Anomaly detection triggers human review
     - Clear audit trail for all AI decisions and actions

### B. System Architecture

Our framework provides a modular extension system that can be used by any AI agent—from open-source models like Llama to proprietary systems like GPT-4 or Claude. As AI capabilities grow, the framework evolves to leverage these improvements without requiring system redesign.

#### 1. Multi-Layer RL LLM Agent

**Key Innovation**: Multi-layer reinforcement learning LLM agent for intelligent scheduler management

**Agent Layers**:

- **Decision Layer**: High-level strategy selection (configure, modify, or create)
- **Implementation Layer**: Code generation and modification
- **Learning Layer**: Reinforcement learning from performance feedback

**Capabilities**:
- **Selection**: Choose existing scheduler based on workload description and historical performance
- **Configuration**: Automatically tune parameters for new scenarios  
- **Modification**: Adapt existing schedulers for slightly different workloads
- **Creation**: Write new scheduler code when no suitable scheduler exists

#### 2. AI-Managed Scheduler Library

**Library Entry Components**:
- **Description**: Textual summary of purpose, characteristics, use-cases
- **Configuration Parameters**: Clearly defined adjustable parameters
- **Source Code**: Executable eBPF C code and Rust userspace components
- **Historical Performance Data**: Metrics from previous uses (makespan, throughput, latency, fairness)
- **Test Results**: Static analysis and runtime test outcomes

#### 3. Static Analysis & Testing Framework

**Purpose**: Ensure AI-generated schedulers are safe and correct before deployment

**Key Features**:
- **BPF Safety Verification**: Pre-validate code against kernel verifier rules
- **Performance Estimation**: Predict scheduler overhead before deployment
- **Automated Testing Pipeline**: Unit tests, integration tests, and benchmarks
- **Sandbox Environment**: Safe testing without affecting production
- **Early Feedback Loop**: Rapid iteration based on test results

#### 4. Scheduler Optimization Libraries

**Purpose**: Pre-built scheduler templates and patterns for AI retrieval and adaptation

**Key Components**:
- **Production Scheduler Collection**: Battle-tested schedulers (scx_rusty, scx_layered, etc.)
- **Algorithm Primitives**: FIFO, SJF, CFS-style, work-stealing patterns
- **Optimization Patterns**: Workload-specific optimizations (batch, interactive, ML)
- **Smart Retrieval**: Semantic search with performance-based ranking
- **Self-Evolution**: Automatically extract patterns from successful AI-generated schedulers

#### 5. Profiling & Monitoring Toolkit

**Purpose**: Comprehensive workload analysis and system observability

**Core Tools**:
- **Workload Profiling**: CPU patterns, memory access, I/O behavior analysis
- **Real-time Monitoring**: Scheduler metrics, latencies, throughput tracking
- **Historical Analysis**: Pattern mining and predictive analytics
- **Low-overhead Integration**: eBPF probes, PMU counters, tracing frameworks
- **Anomaly Detection**: Automatic alerts for performance degradation

#### 6. Unified Agent Extension Framework

**Purpose**: Enable any AI agent to optimize schedulers through standardized interfaces

**Key Features**:
- **Model-Agnostic Interface**: Works with GPT, Claude, Llama, and future models
- **MCP Server**: Standardized API following Model Context Protocol
- **Tool Library**: Atomic operations and composite workflows
- **Safety Governance**: Permission system, audit logging, resource quotas
- **Plugin Architecture**: Extensible without core system changes

### C. Reinforcement Learning Integration

**Purpose**: Enable continuous improvement through reinforcement learning

**Key Algorithms**:
- **In-Context Learning**: Memory-based adaptation using performance feedback
- **Bayesian Optimization**: Efficient parameter tuning with uncertainty modeling
- **Multi-Armed Bandits**: Balance exploration and exploitation for scheduler selection
- **Workload Pattern Learning**: Detect and adapt to changing workload phases
- **Meta-Learning**: Transfer knowledge across similar workloads for rapid adaptation

### D. Continuous Learning and Evolution

1. **Continuous Learning**: Each workload execution enriches the library
2. **Feedback Integration**: Performance data guides future decisions
3. **Adaptive Strategies**: Agent learns which schedulers work for which workloads
4. **Library Growth**: New schedulers added as needed for novel workloads

### E. Architecture Overview

**Additional Diagrams to Include:** State diagram for scheduler lifecycle (development → testing → deployment → monitoring)
- Add sequence diagram showing agent workflow:
  1. Workload analysis
  2. Library search
  3. Decision (configure/modify/create)  
  4. Validation
  5. Deployment
  6. Feedback collection


```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Production System                               │
│                                                                         │
│  ┌─────────────────┐                    ┌─────────────────────────┐   │
│  │   Application   │                    │  Performance Monitor    │   │
│  │   Workloads     │◄──────────────────►│  Daemon                │   │
│  └────────┬────────┘                    └───────────┬─────────────┘   │
│           │                                          │                  │
│           ▼                                          │ Metrics         │
│  ┌─────────────────┐                                │                  │
│  │  Linux Kernel   │                                │                  │
│  │  ┌────────────┐ │                                ▼                  │
│  │  │ sched_ext  │ │                    ┌─────────────────────────┐   │
│  │  │   (BPF)    │◄├────────────────────┤   MCP Server          │   │
│  │  └────────────┘ │     Deploy         │  (Tool Interface)      │   │
│  └─────────────────┘                    └───────────┬─────────────┘   │
│                                                      │                  │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                                              Workload Analysis
                                              Performance Data
                                                       │
┌──────────────────────────────────────────────────────▼──────────────────┐
│                        AI Agent System                                   │
│                                                                         │
│  ┌─────────────────────┐        ┌──────────────────────────────────┐  │
│  │  Multi-Layer RL     │        │   Scheduler Library              │  │
│  │  LLM Agent          │        │                                  │  │
│  │                     │        │  ┌──────────────┐ ┌────────────┐│  │
│  │ ┌─────────────────┐ │◄──────►│  │ Description  │ │HistoricalP││  │
│  │ │ Decision Layer  │ │Retrieve│  │ Config Params│ │erformance ││  │
│  │ ├─────────────────┤ │        │  │ Source Code  │ │   Data    ││  │
│  │ │Implementation   │ │        │  │ Test Results │ └────────────┘│  │
│  │ │     Layer       │ │        │  └──────────────┘                │  │
│  │ ├─────────────────┤ │        └──────────────────────────────────┘  │
│  │ │ Learning Layer  │ │                                               │
│  │ │ (In-Context RL) │ │        ┌──────────────────────────────────┐  │
│  │ └─────────────────┘ │        │  Safety & Validation Pipeline    │  │
│  └──────────┬──────────┘        │                                  │  │
│             │                   │  ┌──────────┐ ┌───────────────┐  │  │
│             ▼                   │  │ Static   │ │   Sandbox     │  │  │
│  ┌──────────────────┐           │  │ Analysis │ │   Testing     │  │  │
│  │ Code Generation  │───────────►  │          │ │               │  │  │
│  │ - eBPF C Code   │  Generate  │  └──────────┘ └───────────────┘  │  │
│  │ - Rust Userspace│           └──────────────────────────────────┘  │
│  └──────────────────┘                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## V. Implementation

### A. MCP Server Implementation

**Architecture**:
- RESTful API endpoints for agent communication
- WebSocket support for real-time monitoring
- Authentication and rate limiting for safety

### B. Scheduler Library Management

**Storage Format**:
- JSON metadata for descriptions and parameters
- Git-based version control for source code
- SQLite database for performance history
- Semantic embeddings for similarity search

**Library Operations**:
- Automatic indexing of new schedulers
- Performance regression detection
- Pattern extraction from successful implementations

### C. Code Generation Pipeline

**Generation Process**:
1. Template selection based on workload type
2. Parameter injection and customization
3. Static validation before compilation
4. Incremental testing in sandbox

**Optimization Techniques**:
- Caching of common code patterns
- Reuse of verified code snippets
- Profile-guided refinement

### D. Safety Implementation

**Validation Layers**:
- Syntax checking with clang-format
- BPF verifier pre-check
- Resource usage estimation
- Deadlock and starvation detection

**Runtime Protection**:
- Automatic fallback to default scheduler
- Performance anomaly detection
- Resource quota enforcement
- Audit logging of all operations


## VI. Evaluation

### A. Research Questions

We investigate five key research questions to validate the effectiveness and efficiency of our AI-driven scheduler optimization framework:

- **RQ1**: Can LLM agents effectively configure existing schedulers?
- **RQ2**: Can LLM agents generate new schedulers for specific workloads?
- **RQ3**: What is the cost and efficiency of AI-driven scheduler generation?
- **RQ4**: How much can RL improve performance after initial generation?
- **RQ5**: How effectively can LLMs understand workloads?

### B. Experimental Setup

**Hardware Platform**: 32-core AMD EPYC 7543 with 256GB DDR4-3200, NVMe SSDs, 10Gbps network, Linux 6.12 with sched_ext

**AI Agents Tested**: Claude Code (Opus 4), GPT-4, Gemini Pro, Llama-3 70B (local)

**Workload Suite**:
- **Latency-sensitive**: schbench (web service simulation), database queries
- **Throughput-oriented**: Linux kernel compilation, video transcoding
- **Batch processing**: Data analytics, log processing, unit testing
- **Mixed workloads**: Git operations, Chromium test suite

### C. Performance Impact of AI-Driven Optimization (Figure 1)

**Figure 1**: Performance comparison of scheduler configurations with RL improvement
- Grouped bar chart showing three stages: Baseline CFS, LLM-configured, RL-improved
- X-axis: Different workloads (schbench, kernel build, TPC-H, video)
- Y-axis: Performance improvement (%)
- Shows initial LLM gains (30-80%) and additional RL improvements (10-12%)

**Key Results**:
- schbench: 50% lower p99 latency, 30% higher throughput (selected scx_layered)
- Linux kernel build: 80% speedup from 312s to 173s (selected scx_rusty)
- LLM correctly identifies workload characteristics in 85% of cases
- RL optimization adds 10-12% additional gain beyond LLM configuration
- Total improvement: 25-27% over baseline CFS
- Convergence within 50-60 episodes of RL training

### D. Novel Scheduler Synthesis for Batch Workloads (Figure 2)

**Figure 2**: AI-generated scheduler performance on batch workloads
- Grouped bar chart comparing Default CFS vs AI-Generated schedulers
- Shows strategy selection (SJF, LJF, Hybrid) for different workloads
- Demonstrates 30-50% improvements through optimal algorithm selection

**Key Results**:
- Unit tests (avg wait): 45% reduction using SJF
- Compilation (makespan): 32% improvement using LJF
- Data analytics: 29% speedup with hybrid approach
- Claude Opus identifies theoretically optimal strategies

### E. Cost Reduction and Optimization Efficiency (Figure 3)

**Figure 3**: Cost reduction through framework optimizations
- Multi-line graph showing metrics over optimization stages
- Lines for: Generation time, API calls, Cost, Success rate
- X-axis: Naive → With Library → With RL
- Shows 85-88% cost reduction

**Key Results**:
- Generation time: 33 min → 5 min (85% reduction)
- API calls: 221 → 28 (87% reduction)
- Cost: $6.00 → $0.75 (88% reduction)
- Success rate: 65% → 95% (+30pp improvement)

### F. Workload Classification and Understanding (Figure 4)

**Figure 4**: Confusion matrix of workload classification
- Heatmap showing classification accuracy across workload types
- Categories: CPU-intensive, I/O-bound, Memory-intensive, Latency-critical, Batch
- Overall accuracy: 89.6%

**Key Results**:
- Latency-critical: 96% accuracy (best)
- CPU-intensive: 94% accuracy
- I/O-bound: 86% accuracy
- Memory-intensive: 82% accuracy
- Classification cost: $0.15 per analysis

## VII. Related Work

**Positioning Strategy:**
- Highlight unique aspects: code generation, self-evolution, production readiness
- Acknowledge limitations of current approach while showing clear advantages

### A. RL-based Scheduler Optimization
- Previous work at top conferences on RL for schedulers
- Limited to parameter tuning, not code generation
- Cannot understand application-level requirements

### B. AI for Systems
- Recent survey: https://arxiv.org/html/2507.12472v1
- Growing interest in AI-powered system optimization
- Our work among first for fully automatic execution agents

### C. sched_ext Framework
- Enables safe BPF-based scheduler development
- Production-ready infrastructure
- Perfect platform for AI-generated schedulers

## VIII. Future Work (one paragraph)

### A. Extended Scope
- Beyond schedulers: cache policies, DVFS, network, sysctl
- Unified framework for OS optimization
- Cross-component optimization

### B. Broader Impact
- Democratize OS optimization
- Enable application-specific kernel policies
- Bridge the gap between application needs and kernel capabilities


## IX. Conclusion (one paragraph)

We present the first framework for using fully automatic LLM agents to dynamically optimize Linux schedulers. By maintaining a self-evolving library of schedulers and leveraging reinforcement learning for continuous improvement, our system achieves significant performance gains while reducing the expertise required for OS optimization. This work opens new possibilities for adaptive, application-aware operating systems that can automatically optimize themselves for changing workloads. This work has implications beyond schedulers. As AI agents become more powerful, the interfaces we design today will shape how AI interacts with systems software tomorrow. Our framework demonstrates that with proper design, AI can democratize system optimization—making expert-level performance accessible to users from cloud operators to gamers, while paving the way for truly self-optimizing operating systems.