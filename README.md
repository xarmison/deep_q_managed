# Master Thesis `Deep Q-Managed: A New Framework For Multi-Objective Deep Reinforcement Learning` Code Repository 

### Graduate Program in Electrical and Computer Engineering

#### Department of Computer Engineering and Automation 

<img width="800" alt="Building Photo" src="https://raw.githubusercontent.com/ivanovitchm/ppgeecmachinelearning/main/images/ct.jpeg">

## Abstract

The Deep Q-Managed algorithm, introduced in this paper, constitutes a notable improvement in the domain of multi-objective reinforcement learning (MORL). This new approach employs an enhanced methodology for multi-objective optimization, which strives to discover all policies within the Pareto Front. This demonstrates remarkable proficiency in attaining non-dominated multi-objective policies across environments characterized by deterministic transition functions. Its adaptability extends to scenarios featuring convex, concave, or mixed geometric complexities within the Pareto Front, thus rendering it a versatile solution suitable for addressing a diverse range of real-world challenges. To validate our proposal, we conducted extensive experiments using traditional MORL benchmarks and varied configurations of the Pareto front. The effectiveness of the policies generated by our algorithm was evaluated against prominent approaches in the literature using the hypervolume metric. The results obtained from these experiments unequivocally establish the Deep Q-Managed algorithm as a contender for addressing complex multi-objective problems. Its ability to consistently produce high-quality policies across a spectrum of environments underscores its potential for practical applications in numerous domains, ranging from robotics and finance to healthcare and logistics. Furthermore, the algorithm's robustness and scalability make it well-suited for tackling increasingly intricate multi-objective optimization tasks, thereby contributing to future advancements in the field of MORL.

## Installation

To run the provided code, the first step is to clone this repository, which can be accomplished by:

```console
user@computer:~$ git clone https://github.com/xarmison/deep_q_managed
```

### Python environment

This section is a guide to the installations of a python environment with the requirements of this repository.

First, install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), both of them give you the "conda" command line tool , but the latter requires less disk space.

Now, create a python virtual environment and install the required packages following the commands. Substitute **<environment_name>** with a name for your environment

Open your terminal and execute the following commands:

```console
user@computer:~$ conda create -n <enviroment_name> anaconda python=3
user@computer:~$ conda activate <enviroment_name> || source activate <enviroment_name>
(<enviroment_name>) user@computer:~$ conda install --yes --file requirements.txt
```

### Customized Environments Library

Next you'll need to install a customized [MO-Gymnasium](https://mo-gymnasium.farama.org/) module that provide the custom reinforcement leaning environments built for the benchmarks presented in the research. 

```console
user@computer: ~/deep_q_managed $ cd MO-Gymnasium
user@computer: ~/deep_q_managed/MO-Gymnasium $ pip install .
```

## Deep Sea Treasure


## Modified Resource Gathering 