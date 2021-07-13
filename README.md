<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Title

We present here our work investigating the effects of Reward Shaping on a Hierarchical Reinforcement Learning (HRL) environment while specifying increasingly complex temporal goals defined as LTL$_f$ formulas. Thanks to the flexibility of the LTL$_f$ language, it is trivial to define such goals and, thanks to existing Python libraries, convert them to DFA automatons that work best with RL tasks.
    
We also investigate the possibility to discard the LTL$_f$ goals and instead redefine the environment using STRIPS, which could later be extended to accept a multitude of temporal goals implicitly. 

## Reinforcement Learning
Reinforcement Learning (RL) is a branch of Machine Learning (ML), one of the most prominent and established field of study in the Artificial Intelligence (AI) sector.

In general, RL aims at learning an optimal behavior function, called policy, 
