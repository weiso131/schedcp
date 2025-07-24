# AI-Powered Dynamic Operating System Optimization with LLM Agents: A Framework for Self-Evolving Linux Schedulers

## I. Introduction

### A. Problem Statement
- The OS kernel policy cannot understand what the application needs
- System managers who optimize systems are not the ones who deploy them, lacking knowledge of workload requirements and behavior
- Understanding workloads requires deep domain knowledge (e.g., traditional DevOps cannot easily optimize ML workloads)
- Workloads change over time - impossible for humans to redesign scheduler algorithms hourly, but AI can

### B. Current State and Limitations
- Many RL algorithms for Linux schedulers proposed at top conferences and applied in production
- However, RL algorithms cannot understand application-level requirements:
  - Is it latency-critical or throughput-critical?
  - What about pure application-level optimization goals?
  - Example: In software building process, if scheduler can prioritize based on code dependencies, we can achieve huge wins compared to baseline
  - No one designs kernel schedulers for such specific cases, but AI agents can

### C. Research Significance
- Among the first frameworks using fully automatic execution LLM AI agents for optimizing OS and computer systems
- Only feasible now (2025) due to recent dramatic improvements in AI agent performance

## II. Background

### A. Evolution of Linux Schedulers

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

### B. AI/ML in Systems Optimization

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

### A. Key Motivations
1. **Domain Knowledge Gap in Modern Infrastructure**
   - In modern infrastructure, especially cloud platforms and serverless environments, system managers who optimize the system are not the ones who deploy the applications
   - System managers don't know the requirements and behavior of their workloads
   - Understanding workloads requires deep domain knowledge
   - Example: Traditional DevOps personnel cannot easily optimize ML workloads
   - LLM Agents can help understand workload patterns and requirements, providing specifications and suggestions for optimizations

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

### C. Research Challenges
1. **Safety and Reliability**
   - How to ensure generated code won't break the system?
   - How to prevent soft-lockups, stalls, starvation?
   - How to avoid negative impact on target workloads?

2. **Efficiency and Cost**
   - How to ensure reasonable generation time?
   - How to minimize cost for production deployment?

### D. Preliminary Results
- LLM agent can generate application profiles
- Successfully chose and configured schedulers
- Performance improvements achieved:
  - **schbench** (benchmark recommended by sched_ext project):
    - 50% lower latency
    - 30% more throughput
  - **Linux kernel build**:
    - ~80% speedup (from ~11s to ~6s)
- Demonstrates potential for significant performance gains through AI-driven optimization

## III. System Design

### A. Design Philosophy and Constraints

1. **Separation of Concerns**
   - Let AI handle decision-making
   - System focuses on exposing right signals, tools, and abstractions
   - Future, more capable models can immediately perform better without system redesign

2. **AI Agents as Performance Engineers**
   - Treat AI agents as human performance engineers
   - Provide similar tools and interfaces

### B. Core System Components

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

#### 3. Implementation Framework
**Dual-Language Code Generation**:
- **eBPF C Code**: Kernel-space scheduler logic
  - Direct generation without DSL
  - Standard sched_ext framework patterns
  - BPF-compatible data structures
- **Rust Userspace Library**: Control plane and configuration
  - Parameter management
  - Runtime statistics collection
  - Dynamic configuration updates

**Safety and Validation Pipeline**:
- **Static Analysis**: Verify BPF code before compilation
  - Check for infinite loops, invalid memory access
  - Ensure BPF verifier compliance
- **Test Framework**: Isolated testing before kernel deployment
  - Userspace simulation of scheduler behavior
  - Performance micro-benchmarks
  - Stress testing with synthetic workloads
- **Early Feedback**: Get validation results before kernel installation
  - Reduces risk of system instability
  - Faster iteration cycles

#### 4. Retrieval and Matching Mechanism
**Multi-strategy Retrieval**:
- **Description Matching**: RAG-like but lightweight, matching workload descriptions
- **Tag-based Search**: Structured tags (e.g., "IO-heavy", "CPU-bound") for filtering
- **Historical Performance Matching**: Prioritize based on past effectiveness on similar workloads

#### 5. Tool Set Interface and Knowledge Database
- MCP server design (https://modelcontextprotocol.io/introduction)
- Helps AI understand current system state
- Provides workload analysis capabilities
- Enables dynamic configuration and programming
- Integration with system monitoring tools

#### 6. Performance Monitoring Daemon
- Continuous system monitoring
- Triggers AI agent when performance degrades
- Collects metrics for feedback loop
- Automatic rollback on performance regression

### C. Reinforcement Learning Integration

#### 1. Simple Memory-based RL (In-Context RL)
- Maintain feedback-driven memory file (e.g., claude.md)
- Performance feedback updates memory after each trial
- Influences future scheduler selection and tuning
- Similar to prompting-based feedback loops in language models

#### 2. Parameter-tuning with Small-scale RL
- Lightweight algorithms adjust retrieval parameters
- Iterative refinement based on workload outcomes
- Adjust similarity thresholds, performance weights, matching heuristics

#### 3. Feedback Mechanism
- Capture performance metrics after each scheduler deployment
- Store results in library for future reference
- Update agent's decision-making memory
- No complex training procedures required

### D. Self-Evolution Process
1. **Continuous Learning**: Each workload execution enriches the library
2. **Feedback Integration**: Performance data guides future decisions
3. **Adaptive Strategies**: Agent learns which schedulers work for which workloads
4. **Library Growth**: New schedulers added as needed for novel workloads

## IV. Key Technical Challenges and Solutions

### A. Code Generation Efficiency

**Challenge**: How can LLM estimate program execution time from code?
**Solution**: 
- Profile-guided optimization
- Historical data correlation
- Lightweight static analysis for early feedback

### B. Token Consumption Management

**Challenge**: Large code projects require excessive LLM tokens
**Solution**:
- Hierarchical code understanding
- Focus on critical paths
- Incremental refinement

### C. Safety Guarantees
**Challenge**: Prevent system crashes and performance degradation
**Solution**:
- Static analysis of generated eBPF code
- BPF verifier compliance checking
- Sandboxed testing environment
- Gradual rollout with monitoring
- Automatic rollback mechanisms

## V. Evaluation Plan

### A. Experimental Setup

#### Figure 1: Scheduler Configuration via LLM
**Objective**: Verify LLM can effectively select and configure existing schedulers

**Benchmarks**:
- schbench (scheduler benchmark)
- Linux kernel build (make linux)

**Configurations per benchmark**:
1. Baseline Linux default scheduler
2. First-time LLM-configured scheduler
3. RL-improved scheduler (with feedback loop)
4. (Optional) Traditional RL-tuned scheduler

#### Figure 2: Scheduler Generation for Specific Applications
**Objective**: Demonstrate LLM can generate new schedulers with 30-50% improvement

**Workloads** (typical batch processing scenarios with uneven process workloads):
- Data analytics (join table operations)
- Log analysis and processing
- Video editing and rendering
- Git operations (e.g., git add with large repositories)
- Unit testing suites with varying test durations

**Key Insights**:
- For minimizing average waiting time: Shortest Job First (SJF) optimal
- For minimizing total completion time (makespan): Longest Job First (LJF) optimal
- AI models show varying capabilities:
  - Claude Opus successfully identifies these optimization strategies
  - Sonnet and other models may not recognize these patterns
- AI-generated schedulers achieve 30-50% average speedup for batch processing workloads

### B. Expected Results
- Configuration improvements: 30-80% performance gains
- New scheduler generation: 30-50% speedup for specific workloads
- RL refinement: Additional 10-20% improvement over initial configuration

## VI. Related Work

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

## VII. Future Work and Impact

### A. Extended Scope
- Beyond schedulers: cache policies, DVFS, network, sysctl
- Unified framework for OS optimization
- Cross-component optimization

### C. Broader Impact
- Democratize OS optimization
- Enable application-specific kernel policies
- Bridge the gap between application needs and kernel capabilities

## VIII. Conclusion

We present the first framework for using fully automatic LLM agents to dynamically optimize Linux schedulers. By maintaining a self-evolving library of schedulers and leveraging reinforcement learning for continuous improvement, our system achieves significant performance gains while reducing the expertise required for OS optimization. This work opens new possibilities for adaptive, application-aware operating systems that can automatically optimize themselves for changing workloads.