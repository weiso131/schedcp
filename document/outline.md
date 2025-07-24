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
  - Example: In software building, prioritizing based on code dependencies could yield huge wins vs baseline
  - No one designs kernel schedulers for such specific cases, but AI agents can

### C. Research Significance
- Among the first frameworks for dynamically optimizing OS systems with LLM agents
- Covers schedulers, cache, DVFS, network, sysctl, and others
- Highly competitive and timely topic (companies like Huawei working on it)
- Only feasible now due to recent improvements in AI agent performance

## II. Motivation

### A. Key Motivations
1. **Domain Knowledge Gap**
   - Modern infrastructure (cloud, serverless, edge) separates system managers from deployment
   - System managers lack understanding of workload requirements and behavior
   - Deep domain knowledge required (e.g., DevOps cannot easily optimize ML workloads)
   - LLM Agents can bridge this gap by understanding workload patterns and requirements

2. **Technical Complexity**
   - Designing and implementing new Linux kernel schedulers requires deep domain knowledge
   - LLM domain knowledge can help bridge the expertise gap

3. **Dynamic Adaptation**
   - Workloads change over time
   - Impossible for humans to redesign scheduler algorithms frequently
   - AI can adapt dynamically

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
- Achieved for schbench (sched_ext benchmark):
  - 50% lower latency
  - 30% more throughput
- For Linux kernel build: ~80% speedup (from ~11s to ~6s)

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

#### 1. AI-Managed Scheduler Library
**Key Innovation**: AI agent maintains a self-evolving library of scheduler algorithms

**Capabilities**:
- **Selection**: Choose existing scheduler based on workload description and historical performance
- **Configuration**: Automatically tune parameters for new scenarios  
- **Modification**: Adapt existing schedulers for slightly different workloads
- **Creation**: Write new scheduler code when no suitable scheduler exists

**Library Entry Components**:
- **Description**: Textual summary of purpose, characteristics, use-cases
- **Configuration Parameters**: Clearly defined adjustable parameters
- **Source Code**: Executable eBPF C code
- **Historical Performance Data**: Metrics from previous uses (makespan, throughput, latency, fairness)

#### 2. Implementation Approach
**Direct eBPF C Code Generation**:
- AI generates eBPF C code directly (no DSL needed)
- Standard sched_ext framework patterns
- Clear parameter definitions
- Embedded documentation describing intent and applicability

#### 3. Retrieval and Matching Mechanism
**Multi-strategy Retrieval**:
- **Description Matching**: RAG-like but lightweight, matching workload descriptions
- **Tag-based Search**: Structured tags (e.g., "IO-heavy", "CPU-bound") for filtering
- **Historical Performance Matching**: Prioritize based on past effectiveness on similar workloads

#### 4. Tool Set Interface and Knowledge Database
- MCP server design (https://modelcontextprotocol.io/introduction)
- Helps AI understand current system state
- Provides workload analysis capabilities
- Enables dynamic configuration and programming

#### 5. Performance Monitoring Daemon
- Continuous system monitoring
- Triggers AI agent when performance degrades
- Collects metrics for feedback loop

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
- Lightweight static analysis

### B. Token Consumption Management
**Challenge**: Large code projects require excessive LLM tokens
**Solution**:
- Hierarchical code understanding
- Focus on critical paths
- Incremental refinement

### C. Safety Guarantees
**Challenge**: Prevent system crashes and performance degradation
**Solution**:
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

**Workloads** (typical batch processing scenarios):
- Data analytics (join operations)
- Log file processing
- Video processing/editing
- Git operations (e.g., git add)
- Unit testing suites

**Key Insights**:
- For minimizing average waiting time: Shortest Job First (SJF) optimal
- For minimizing total completion time: Longest Job First (LJF) optimal
- AI (Claude Opus) successfully identifies these heuristics automatically

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

### B. Publication Strategy
- Initial version targeting PLOS Workshop 2025
- ArXiv preprint for rapid dissemination
- Full system paper for OSDI with expanded scope

### C. Broader Impact
- Democratize OS optimization
- Enable application-specific kernel policies
- Bridge the gap between application needs and kernel capabilities

## VIII. Conclusion

We present the first framework for using fully automatic LLM agents to dynamically optimize Linux schedulers. By maintaining a self-evolving library of schedulers and leveraging reinforcement learning for continuous improvement, our system achieves significant performance gains while reducing the expertise required for OS optimization. This work opens new possibilities for adaptive, application-aware operating systems that can automatically optimize themselves for changing workloads.