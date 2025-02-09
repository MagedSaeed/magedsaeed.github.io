# Introduction

In January 2025, DeepSeek, a Chinese startup, unveiled its revolutionary AI model, transforming the artificial intelligence landscape with its exceptional performance-to-cost ratio compared to US and other West AI models. Alongside their model release, they launched a mobile app enabling real-time AI interactions similar to ChatGPT and Anthropic's Claude apps. The app cultivated over 10 million downloads on the Google Play Store within its first three weeks of release. Its rocketing growth reminds of the ChatGPT app initial growth on its first days.

What drove DeepSeek's R1 to such popularity? Well, the model delivers impressive performance comparable to leading Western AI counterparts like Llama and Mixtral with a surprisingly cost cutoffs. Interestingly, the model even rivals the closed-source models, OpenAI's ChatGPT o1 (that is very recently released) and Anthropic's Claude 3.5 Sonnet. In some settings, it even surpasses them. Most notably, they achieved this while reducing costs by more than 90%, causing significant market disruption that led to approximate collapse of $1 trillion in market value. Nvidia, the GPU industry leader, experienced an unprecedented $600 billion loss in market capital, marking the largest single-company loss in U.S. stock market history!

Perhaps most significantly, DeepSeek published their work as open-source under an MIT license. This makes their models commercially permissible given the compute. Additionally, their experimental work is thoroughly well-documented in published technical reports covering both DeepSeek R1 and DeepSeek V3, including all of its previous versions. Such initiative represents a major advancement for open-source progress in the AI race against closed-source AI enterprises.

**But Wait! _There is an Ahaa moment that I can flag here!_** How did DeepSeek achieve such impressive performance given China's export controls? Note that the company reported training this model on a cluster of 2,048 Nvidia H800 GPUs. These GPUs are less efficient than their powerful sibling, the Nvidia H100. While this is still a skeptical side of the story for many, especially conspiracy theorists, given the current US-China rivalry, one aspect of this notable performance boost is their through and numerous training innovations, advances and brilliant engineering efforts with various experimentation and ablation studies. In fact, they introduced wild conclusions. For instance, Reinforcement Learning can successfully replace the supervised fine-tuning phase of LLMs development pipeline. This step is usually a tedious task to collect high quality and typically involves cumbersome human annotation.

In this article, I will explain how this model was trained with some technical depth and what makes it unique among its counterparts. This will also cover the development of its predecessor, DeepSeek V3, as understanding both is instrumental for grasping the complete picture. For deeper insights, readers are encouraged to review the developers' technical reports.

This article begins with a preamble introducing key terminology to establish common ground. It then discusses DeepSeek-R1, followed by DeepSeek-V3.

# Preamble and Terminologies

Before diving deeper, it's useful to establish a common ground so that both of us can follow the technical discussion with a clear alignment to the context. This preamble will introduce three main topics that are relative to our discussion: The typical LLM development pipeline, and MoE architecture.

## An LLM Development Pipeline

The development of production-grade LLMs typically involves the following steps:

- **Dataset Creation:** Building a massive text dataset (usually containing trillions of tokens) comprising books, articles, high-quality web content, wiki pages, and other text sources. Nothing fancy here. The main focus in this stage is on collecting high-quality data with meticulous preprocessing to ensure the quality constrains.

- **Base Model Training:** Developing and training a transformer-based model for the next-token prediction objective on the collected dataset. The resulting trained model (usually called a base model) should be smart enough to predict the most probable next token given a sequence of previous tokens.

- **Supervised Fine-tuning (SFT):** The base model undergoes additional fine-tuning on datasets from downstream natural language processing tasks such as text classification, question answering, sentiment analysis, machine translation, language understanding, and language comprehension. These datasets follow a specific format with input-output pairs, and the model is fine-tuned to generate appropriate outputs for given inputs. In this stage, the model is trained with a variety of prompts to help generalizing to input quires variations.

- **Alignment:** While the model seems to be ready for the public use from the previous step, this final step ensures the model aligns its output with human preferences by avoiding harmful, offensive, or disrespectful content. This typically involves a process called Reinforcement Learning from Human Feedback (RLHF) where the model is tuned to follow human preferences. The resulting model after this step is a production-ready LLM, often called a Chat model (like ChatGPT). In some cases, this step is combined with the previous fine-tuning step, resulting in what's called an instruct model.

![Figure 1: LLM Development Pipeline](assets/llm-development-pipeline.png)

*Figure 1: LLM Development Pipeline*

## MoE architecture

DeepSeek V3 employs a Mixture of Experts (MoE) architecture. This design incorporates multiple specialized subnetworks (experts) trained together, with only a subset activated during inference. Each expert develops proficiency in processing different types of inputs or tasks. During user interaction, the architecture's gating mechanism routes to the most appropriate expert(s) based on the input.

A common misconception about MoE is that experts specialize in specific domains (like mathematics or creative writing). While theoretically possible (and already tried, see: [Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/html/2403.07816v1)), practical implementations typically have experts learn to process different data patterns rather than explicit domain specialization. These experts are integrated components within the larger network, not separate language models, though they could theoretically be implemented as individual LLMs.

The MoE concept dates back to 1991 (Introduced in this paper: [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)) but has gained renewed attention through its successful application in large-scale LLMs, particularly with the introduction of Mixtral LLMs. In transformer architecture, experts replace the feedforward network positioned after the attention and normalization layers.

![Figure 2: Mixture of Experts Architecture](assets/MoE.png)

*Figure 2: Mixture of Experts in a Transformers architecture*

The MoE architecture offers distinct advantages and challenges. Its primary benefit lies in cost-effective inference while enabling comprehensive learning during training. DeepSeek's architecture contains 671B total parameters, but only 37B are activated during inference. However, this efficiency comes with a trade-off: the entire model (all 671B parameters) must remain loaded in GPU VRAM for operation. Furthermore, training this architecture is associated with challenges. An example of such challenges is the following question: how to effecentinly load-balance the experts load during training so that the gating network does high-favor only few experts, low-favoring the rest?

# DeepSeek R1

DeepSeek-R1 marks a landmark in AI development timeline. It is the first model that shows Reinforcement Learning can work in a large-scale setting replacing the collection of high-quality SFT data as in the typical pipeline in Figure 1. This has been prominently proven with DeepSeek-R1-Zero model (explained next) where it proved that pure reinforcement learning could yield impressive result eliminating the need for SFT. This comes with the cost that its capabilities outside reasoning domains are hindered. While DeepSeek-R1 is meant to be a general-use model with polished, user-friendly outputs suitable for general public use, they performed some refinements combining the bests from both models DeepSeek-V2 and DeepSeek-R1-Zero. DeepSeek R1 development pipeline can be very briefly illustrated in the following sequence of steps:

1. SFT Cold Start fine-tuning of DeepSeek V3-Base -> DeepSeek-V3-Cold-Start
2. RL training on DeepSeek-V3-Cold-Start -> DeepSeek-V3-RL.
3. New SFT generated from DeepSeek-V3-Instruct-RL combined with another SFT data generated from DeepSeek-V3-Instruct. Let us call this data post-RL-SFT data.
4. DeepSeek-V3-RL fine-tuning on post-RL-SFT data -> DeepSeek-V3-RL-SFT
5. Finally DeepSeek-V3-RL-SFT undergoes another round of RL training where this step produces **DeepSeek-R1**.

![Figure 3: DeepSeek-R1 development pipeline](assets/DeepSeek-R1-pipeline.png)

*Figure 3: DeepSeek-R1 development pipeline*

Discussing in further details, they start by fine-tuning DeepSeek-V3-base with data they called cold start data. This data consists of thousands of high-quality SFT data. The reasoning SFT data was collected via few-shot prompting with long CoT DeepSeek-R1-Zero. The generated samples undergo verification steps extending to human annotators to maintain a high-quality constraints. This seems to play a major role in giving the model an overview of SFT data settings, beside, of course, its important role of maintaining stable fine-tuning for the upcoming phases alternating between SFT and reasoning setups. 

After this initial training, they moved on to the reinforcement learning phase, but with a twist. They added a new type of reward - a language consistency reward. Why? Because they noticed R1-Zero developed the habit of mixing languages while trying to reach the final answer. While this can be an emergent capability to learn from various lingual representation during the CoT (thinking) process, this kind of output is not user-friendly. This new reward encouraged the model to stick to one language and follow a specific format throughout its response. Sure, this slightly reduced the model's raw performance, but it made the outputs much more readable and user-friendly.

Once the first round of RL training converged (meaning the model got really good at reasoning), they used this improved model to generate new training data. But they were picky about it - they only kept the best responses, filtering out anything with mixed languages, overly long paragraphs, or messy code blocks through rejection sampling. They combined this with some regular language tasks (like writing and answering questions) to create a more well-rounded training set.

This iterative approach paid off. The final model, DeepSeek-R1, almost maintains all the impressive reasoning capabilities of R1-Zero but presents its solutions in a much more user-friendly way. It's like taking that brilliant but chaotic professor and teaching them how to explain things clearly to their students!

## DeepSeek-R1-Zero

DeepSeek R1-Zero represents a fascinating experiment in their development where the team took a bold approach: develop reasoning capabilities using only reinforcement learning, without any supervised fine-tuning! Yes, you read that right - they wanted to see if a model could learn to reason just through trial and error, like a child learning to solve puzzles without being shown examples first.
The training methodology was surprisingly straightforward. They started with DeepSeek V3-Base and designed a simple template where the model had to put its thinking process between <think> tags and its final answer between <answer> tags. That's it! No fancy prompting, no complex instructions - just "think first, answer second." The idea was to let the model figure out the best way to solve problems on its own.

But here's where it gets interesting, the reward system. Unlike many other approaches that use neural networks to evaluate responses (which can get messy with reward hacking), they kept it simple with two types of rewards:

- Accuracy rewards: Did you get the right answer? For math problems, programming challenges, and other tasks with clear right/wrong answers, this was straightforward to check.

- Format rewards: Did you use the thinking and answer tags correctly? This kept things organized.

Did this simple approach work> The model's performance on AIME 2024 (a notoriously difficult math competition) jumped from a modest 15.6% to an impressive 71.0%. When they let the model generate multiple solutions and take a majority vote (what they call consensus@64), it got even better reaching 86.7%! They also experimented on other math datasets reaching 95.9% on MATH-500. On code benchmarks, they achieved a respectable 1444 rating on Codeforces.

But perhaps the most fascinating part was what they call the "aha moment." During training, the model started developing human-like behaviors nobody programmed in. It would sometimes stop mid-solution and say things like "Wait, wait. Wait. That's an aha moment I can flag here" before correcting its approach. It learned to verify its work, try different solution methods, and even allocate more thinking time to harder problems.

![Figure 4: A DeepSeek-R1 Ahaa moment!](assets/Ahaa-moment.png)
*Figure 4: A DeepSeek-R1 Ahaa moment!*

What's particularly interesting is how the model naturally learned to use longer chains of thought as training progressed. Looking at the below figure,  you can see a clear upward trend in the average response length during the RL process. The model started with relatively short responses (around 500-1000 tokens) in the early stages but gradually increased its reasoning length to nearly 10,000 tokens by the end of training! This wasn't explicitly programmed - the model organically discovered that longer, more detailed reasoning chains led to better results. The graph shows some fluctuation (represented by the light blue shading), suggesting the model was actively experimenting with different reasoning lengths rather than following a preset pattern. It's almost like watching a student evolve from giving quick, instinctive answers to writing out detailed step-by-step solutions as they gain confidence and understanding. This evidence supports the idea that R1-Zero wasn't just learning to solve problems, but was developing a deeper understanding of how to approach complex reasoning tasks through extensive step-by-step analysis.

![Figure 5: CoT response length over training steps](assets/CoT-response-length.png)
*Figure 5: CoT response length over training steps*

However, it wasn't all perfect. R1-Zero had its own downsides too: its outputs could be hard to read, it would sometimes mix different languages in the same response (imagine getting a math solution that randomly switches between English and Chinese!), and its formatting wasn't always user-friendly. Think of it like a brilliant but slightly chaotic professor who solves problems brilliantly but writes their solutions in a way that only they can fully understand.

These limitations led the team to develop a more refined version, the R1 version with the above described pipeline (in Figure 3). But R1-Zero proved something important: pure reinforcement learning can teach a model to reason, and sometimes, just letting an AI figure things out on its own leads to surprisingly near-human behaviors.


## R1 distillation to small models

After developing R1, DeepSeek explored another practical angle to leverage such advances to the small, more economical models arena. Basically, they tried to approach the following two questions:

- What is the effect of reasoning data fine-tuning on the economical LLMs?
- Are economical LLMs better trained from scratch with reinforcement learning or distill knowledge from powerful large reasoners?

By distillation, the smaller model will be fine-tuned on the large LLM reasoner outputs. For fine-tuning, the collected 800,000 samples from R1 and use them to fine-tune various smaller models, ranging from tiny 1.5B parameter versions to larger 70B ones using both Qwen and Llama architectures.

The results were surprising. Their 7B model (DeepSeek-R1-Distill-Qwen-7B) outperformed GPT-4o-0513 across benchmarks, while their 14B version surpassed QwQ-32B-Preview despite being less than half its size. Most impressively, their 32B model achieved 72.6% on AIME 2024, 94.3% on MATH-500, and 57.2% on LiveCodeBench.

Another key finding they report from their experiments: distilling knowledge from R1 worked better than trying to train smaller models directly with reinforcement learning. When they applied R1-Zero's RL approach to a 32B model, it only matched QwQ-32B-Preview's performance. But the distilled version significantly outperformed both, suggesting that transferring knowledge from larger models is more effective than training smaller ones from scratch.


To give a concise summary, DeepSeek introduces the following **Ahaa moments in the LLMs research community:**

- Reinforcement Learning can completely replace supervised fine-tuning phase of LLMs development pipeline. However, cold-start data can be added for training and fine-tuning stability.
- Although current approaches for (RHLF) uses a reward model to score model responses then fine tune the LLM with this reward model using optimization methods like PPO, they showed that GRPO alone can perform both by scoring multiple LLM responses and score them relative to each other.
- They showed that LLMs can be good SFT data generators.
- They showed that distilling strong reasoning LLM to smaller models is more promising than training the small model with reinforcement learning.

# DeepSeek V3

As introduced earlier, DeepSeek-V3 is a Mixture-of-Experts (MoE) language model, featuring 671B total parameters while activating only 37B parameters for each token during inference time. The developers incorporate multiple architectural innovations to achieve both efficiency, impressive, and strong performance at this large scale.

The architecture consists of 61 transformer layers with a hidden dimension of 7168. For attention, it uses 128 heads with a per-head dimension of 128. While using standard FFNs in the first three layers, all subsequent layers utilize MoE with 1 shared expert and 256 routed experts. For each token, 8 experts are activated along with the shared expert.

## Training innovations and tricks:

This architecture allows DeepSeek-V3 to achieve impressive performance while maintaining efficient training and inference. The model was trained on 14.8 trillion diverse tokens with remarkable stability. Below is a list of innovations and tricks they employ to achieve such performance.


### Multi-head Latent Attention (MLA)

DeepSeek-V3 utilizes Multi-head Latent Attention (MLA), a technique carried over from DeepSeek-V2 for efficient inference.

For a glance, the attention mechanism is a mathematical operation that calculates scores (called attention) between sequence tokens. Each word is represented by 3 vectors called keys, values, and queries. The bottleneck here is that the number of keys, queries, and values grows linearly with the sequence length. That is, for longer sequences, the attention mechanism needs to perform more matrix operations adding more cost, especially during inference. In recent studies, caching the keys and values during inference has been shown to be a notable performance boost. However, DeepSeek advanced this one step further.

What MLA presenting (very briefly) is that they transform the keys and values into a lower dimension by down-projecting it with a lower-ranking matrix. This  transformation is cached during inference. During training where they want to attain the dimensionality of these two matrixes, they upper-project them with a higher-ranking matrix.

### Auxiliary Loss Free Load Balancing

DeepSeek implemented a MoE architecture. Following the hypothesis that parts of human brains are always active for all kinds of responses, they made a set of experts always active during inference. However, this introduces challenges on which expert got selected by the gating mechanism and thus has dominance during training and inference. This leads to inefficient training and waster compute during inference. For that sake, a loss-based load-balancing mechanism was introduced. Although it is effective, it confuses the training objective with its added gradients as the model will learn two objectives now. How to best select the best token and how to best balance its experts impairing the model performance as an additional regularization term. The innovation introduced by DeepSeek here is that they introduced a loss-free load balancing mechanism for the experts. This method does not rely on an auxiliary loss (as introduced in previous research works).

### Multi-token Prediction

Another key innovation in DeepSeek-V3 is its multi-token prediction capability. This trick was inspired by a previous research. However, they showed that this trick can also be scaled to this large number of parameter settings. The idea is simply rather than just predicting the next token, the model can predict multiple future tokens simultaneously through a causal chain of predictions. The future tokens after the next token are predicted with a shallower network compared to the main LLM. This is only activated during training. However, it is disabled during inference.

## Infrastructure and Training

Training large models like DeepSeek-V3 requires special attention to the training infrastructure. The model was trained on a cluster of 2048 NVIDIA H800 GPUs. The training framework, called HAI-LLM, is a lightweight framework built by DeepSeek engineers from scratch. The framework implements what they call DualPipe algorithm, allowing efficient pipeline parallelism by overlapping computation and communication. Reflecting my understanding here in this part, I can understand that experts can be trained on parallel. While they mentioned in their paper (section 3.2) that they even achieved such computation overlap during the forward and backward processes, it is not really clear, to me at least, how this overlap is happening as, in order to do the backward gradient update, the forward pass needs to be completed first.

### FP8 Training

One of the major advancements in DeepSeek-V3 is its pioneering use of 8-bit floating point (FP8) precision for training. Why this is important? Usually, training large language models requires high numerical precision (like FP32 or FP16) to maintain training stability. However, this comes with high memory and computational costs. DeepSeek showed that it is possible to train such large models with FP8 precision without compromising the training stability. They follow and improve on previous research (FP8-LM) introducing this idea at scale.

The framework they followed is a mixed-precision framework where GEMM operations (General Matrix Multiplication) are performed in FP8 and their outputs are upscaled back to FP32. However, other sensitive operations like attention, layer normalization, and embeddings are kept with FP32.

The team improved over this mixed-precision framework by introducing a fine-grained mixed-precision framework to overcome challenges due to this lower precision. Examples of these challenges are unstable training due to overflows and underflows. They group model parameters in small tiles (1x128 for activations and 128x128 for weights) and quantize each group independently. This helps manage outliers that typically cause training instability in low-precision training.

Among of the other tricks to introduce is the use of a smart accumulation strategy where intermediate computations are promoted to higher precision at regular intervals. This ensures that despite using FP8 for most operations, the critical accumulation of gradients remains accurate.

## Post-Training stage

DeepSeek-V3 undergoes an extensive post-training phase. Reading this section and the R1 paper, It seems that (my personal opinion) this section of the V3 paper is introduced almost twice with many overlaps in the R1 paper.My feeling is that the whole part of reinforcement learning got rewritten with further depth and evaluation in R1 paper. The R1 paper is just an extension to this section of the V3 paper with the detailed introduction of DeepSeek-R1-Zero. Here is an overview of the steps of V3 post training.

The first stage is Supervised Fine-Tuning (SFT), where the team curates a diverse dataset of 1.5M instances. They collected two kinds of data: reasoning-based and general language use data. For reasoning-related tasks (like mathematics and coding), they generated this data from DeepSeek-R1. While R1-generated data shows high accuracy, it tends to be overly verbose with excessive steps. To alleviate these challenges, they followed the following methodology. They generate two types of samples for each instance: one with the original response format and another incorporating R1's response with a system prompt designed to encourage reflection and verification patterns. The model, then, started an enforcement learning phase with this data. Finally, they applied rejection sampling to only select high quality SFT reasoning data.

In the reinforcement learning phase, rewards can be constructed from two sources with regard to the type of data. For data with closed-form answers, like math and code datasets where the output is usually in close format, the reward is based on whether the model reached to the correct answer or not. For the open-ended questions, they employed a model-reward method where the used the GRPO optimization introduced in their DeepSeekMath paper to update the model policy. This optimization will punish the model if it wend far from the expected output distribution or it went far from the expect output format they provided. The distribution distance is measured with the KL divergence score which measures the distance between two distributions.

The post-training phase concludes with various evaluations and optimizations. The results show impressive performance improvements, particularly in reasoning tasks. For instance, on LiveCodeBench-CoT, the model improves from 31.1% to 37.4% pass rate, and on MATH-500, accuracy jumps from 74.6% to 83.2%. However, these gains come with a trade-off - longer response lengths, which the team carefully balances through optimal distillation settings to maintain computational efficiency.


# Useful Links and resources:
- [The History of Mixture of Experts](https://www.linkedin.com/pulse/history-mixture-experts-upp-technology-ok9re/)
- [DeepSeek-R1: RL for LLMs Rethought](https://thegrigorian.medium.com/deepseek-r1-rl-for-llms-rethought-e148445d4381)
