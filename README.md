# Approximated Python Intelligence Quotient (APIQ)
APIQ is a measure of intelligence. It is an approximation of the [*Universal Intelligence Measure*](http://www.hutter1.net/ai/iorx.pdf) (UIM) by Legg and Hutter.

UIM is based on the following informal definition of intelligence.

> Intelligence measures an agent's ability to achieve goals in a wide range of environments.

Many environments have been implemented for agents to achieve goals in and a few agents have been implemented to have their intelligence measured and to test the measure.

Environments and agents communicate via bitstrings.

New environments and agents can be added easily. They need to implement their respective abstract classes Environment.py or Agent.py and be dropped into python/src/environments/ and python/src/agents/

Running python/src/Main.py will then trial new agents in all environments and all agents in new environments and calculate their APIQ from the results.
