# Introduction

In January 2025, a Chinese startup, called DeepSeek, unveiled its revolutionary AI model, shocking the landscape of artificial intelligence with its outstanding performance compared to its cost. Alongside their model release, they released a mobile app that allows users to chat with the AI model in real-time similar to ChatGPT and Anthropic. Their app has gathered over 10 million downloads on the Google Play Store within its first three weeks of release!

Why DeepSeek's R1 gains such popularity? Well, the model delivers an impressive, on-par performance compared to the leading U.S. and west AI counterparts like Llama and Mixtral covering closed source ones as well (OpenAI's ChatGPT, the most recent O1 version, and Anthropic Claude Sonnet 3.5), if not surpassing it. They deliver such performance with a more than 90% cost reduction! causing a massive disrupt in the markets, leading to erasing $1 trillion in value! Notably, Nvidia, the GPUs chips industry leader, experienced a record-breaking $600 billion loss in market capitalization, marking the largest single-company loss in U.S. stock market history. What is even more interesting is that they published their work as an open-source. The model is published under an MIT license, making it a commercially viable option with proper compute availability. Their experimental work is also well described on published research papers for both of their work: DeepSeek R1 and DeepSeek V3 with all of its previous versions. Making a giant leap for open source progress in the AI race with closed source enterprise.

"**But Wait**" how DeepSeek were able to achieve such impressive performance Given the Chaina's export controls? Note that they claimed that they trained this model on a cluster of 2048 Nvidia H800 GPUs which is less efficient to its powerful sibling: Nvidia H100. It turns out they achieved this by incorporating many training trick and advances. What is even more interesting is the introduction of Reinforcement Learning as an innovative approach to improve the performance. They were the first to show that Reinforcement Learning works at this large scale of LLMs development making a good end for all the previous stories and trials in this direction. In this article, I am going to illustrate how this model was trained with a bit of in-depth technicality. This will also cover how its elder brother DeepSeek V3 are developed as this is important to grasp the full picture. For deep dives, better to give an in-depth read to the developers' technical report they released.

# Preamble and Terminologies

Before star diving further, it is helpful to state the terminologies used so that both of us (me and you) are on landing on a common ground.

## An LLM development Pipeline

The current development of the production-grade LLMs undergoes the following steps:

- Building a massive dataset of text (usually with trillion tokens). This dataset is just a text, like books, articles, high quality web-content, wiki pages, etc.
- Build and train a transformer-based model for the next-token generation objective on the built dataset. The resulting trained model should be smart enough to predict the best next token given a previous sequence of tokens. This model is usually called a base model.
- The base model is taking another round of fine-tuning. This fine-tuning is called supervised fine-tuning or SFT in short. The model in this stage is trained on datasets from downstream natural language processing tasks like question answering, sentiment analysis, machine translation, text classification, language understanding and comprehension. These datasets are usually of special formats having two pairs apart: the input and the output. The model is fine-tuned to produce the output given the input.
- While the model seems to be ready for public use by the last step. However, it undergoes another step, an alignment step. This step is concerned with teaching the model to align its output to human preferences. That is: the model should not produce harmful, offensive, and disrespectful content. The model, in this step, is usually trained with a special process called Reinforcement Learning from Human Feedback (RHLF). The resulting model after this step is a production ready LLM, sometime called Chat model, like ChatGPT. In some cases, this step is merged with the previous step and the model is just called an instruct model where it covers both fine-tuning and alignment steps.

![Figure 1: LLM Development Pipeline](assets/llm-development-pipeline.png)

*Figure 1: LLM Development Pipeline*

## MoE architecture

DeepSeek V3 is a Mixture of Experts (MoE) architecture. What does this mean? The Mixture of Experts (MoE) architecture is a design where multiple specialized "expert" LLMs that are trained together. The hypothesis is that each one of these experts is expected to be better and proficient at handling different types of inputs or tasks. During inference where the user interact with the LLM, the MoE architecture will select the expert that is best suited to process the input.

MoE usually trains a gating mechanism that learns to route incoming inputs to the most appropriate expert(s), acting like a smart traffic controller that directs each input to the expert or a group of experts best suited to process it.

![Figure 2: Mixture of Experts Architecture](assets/MoE.png)

*Figure 2: Mixture of Experts Architecture*

Why this architecture is useful? It is useful because it is cost effective during inference time. That is, during training, many experts are trained. However, during inference, only few of them got activated to response to the user queries. DeepSeek architecture is an MoE expert with a total of 671B parameters with only 37B activated during inference.

## Reinforcement Learning

One of the innovations introduced by DeepSeek team is the incorporation of reinforcement learning as a major ingredient to improve the model reasoning abilities. But what is Reinforcement Learning in this context? How it is compared to the other learning paradigms: supervised and unsupervised learning?

Reinforcement learning is a machine learning paradigm where the model improve its abilities by getting punished if made mistakes or rewarded if made the right actions, similar to the "carrot and stick principle" learning approach. This is in contrast to the supervised learning approach where the model sees a lot of input and output pairs and learns to mimic this behavior.

# DeepSeek R1

DeepSeek R1 development pipeline can be very briefly illustrated in the following sequence of steps:

1. SFT Cold Start fine-tuning of DeepSeek V3-Base -> DeepSeek-V3-Cold-Start
2. RL training on DeepSeek-V3-Cold-Start -> DeepSeek-V3-RL.
3. New SFT generated from DeepSeek-V3-Instruct-RL combined with another SFT data generated from DeepSeek-V3-Instruct. Let us call this data post-RL-SFT data.
4. DeepSeek-V3-RL fine-tuning on post-RL-SFT data -> DeepSeek-V3-RL-SFT
5. Finally DeepSeek-V3-RL-SFT undergoes another round of RL training where this step produces **DeepSeek-R1**.

![Figure 3: DeepSeek-R1 development pipeline](assets/DeepSeek-R1-pipeline.png)

*Figure 3: DeepSeek-R1 development pipeline*

## DeepSeek V3

As introduced earlier, DeepSeek-V3 is a Mixture-of-Experts (MoE) language model, featuring 671B total parameters while activating only 37B parameters for each token during inference time. The developers incorporate multiple architectural innovations to achieve both efficiency, impressive, and strong performance at this large scale.

The architecture consists of 61 transformer layers with a hidden dimension of 7168. For attention, it uses 128 heads with a per-head dimension of 128. While using standard FFNs in the first three layers, all subsequent layers utilize MoE with 1 shared expert and 256 routed experts. For each token, 8 experts are activated along with the shared expert.

### Training innovations and tricks:

This architecture allows DeepSeek-V3 to achieve impressive performance while maintaining efficient training and inference. The model was trained on 14.8 trillion diverse tokens with remarkable stability. Below is a list of innovations and tricks they employ to achieve such performance.


#### Multi-head Latent Attention (MLA)

DeepSeek-V3 utilizes Multi-head Latent Attention (MLA), a technique carried over from DeepSeek-V2 for efficient inference.

For a glance, attention mechanism is a mathematical operation that calculates scores (called attention) between sequence tokens. Each word is represented by 3 vectors called keys, values, and queries. The bottleneck here is that the number of keys, queries, and values grows linearly with the sequence length. That is, for longer sequences, attention mechanism needs to perform more matrix operations adding more cost especially during inference. In recent studies, caching the keys and values during inference has been shown to be a notable performance boost. However, DeepSeek advanced this one step further.

What MLA presenting (very briefly) is that they transform the keys and values into a lower dimension by down-projecting it with a lower-ranking matrix. This  transformation is cached during inference. During training where they want to attain the dimensionality of these two matrixes, they upper-project them with a higher-ranking matrix.

#### Auxiliary Loss Free Load Balancing

DeepSeek implemented a MoE architecture. Following the hypothesis that parts of human brains are always active for all kind of responses, they made a set of experts always active during inference. However, this introduces challenges on which expert got selected by the gating mechanism and thus has dominance during training and inference. This lead to inefficient training and waster compute during inference. For that sake, a loss-based load balancing mechanism was introduced. Although it is effective, it confuses the training objective with its added gradients as the model will learn two objectives now. How to best select the best token and how to best balance its experts impairing the model performance as an additional regularization term. The innovation introduced by deep-seek here is that they introduced a loss-free load balancing mechanism for the experts. This method does not rely on an auxiliary losses (as introduced in previous research works).

#### Multi-token Prediction

Another key innovation in DeepSeek-V3 is its multi-token prediction capability. This trick was inspired by a previous research. However, they showed that this trick can also be scaled to this large number of parameters settings. The idea is simply rather than just predicting the next token, the model can predict multiple future tokens simultaneously through a causal chain of predictions. The future tokens after the next token are predicted with a shallower networks compared to the main LLM. This is only activated during training. However, it is disabled during inference.

### Infrastructure and Training

Training large models like DeepSeek-V3 requires special attention to the training infrastructure. The model was trained on a cluster of 2048 NVIDIA H800 GPUs. The training framework, called HAI-LLM, is a lightweight framework built by DeepSeek engineers from scratch. The framework implements what they call DualPipe algorithm, allowing efficient pipeline parallelism by overlapping computation and communication. Reflecting my understanding here in this part, I can understand that experts can be trained on parallel. While they mentioned in their paper (section 3.2) that they even achieved such computation overlap during the forward and backward processes, it is not really clear, to me at least, how this overlap is happening as, in order to do the backward gradient update, the forward pass needs to be completed first.

#### FP8 Training

One of the major advancements in DeepSeek-V3 is its pioneering use of 8-bit floating point (FP8) precision for training. Why this is important? Usually, training large language models requires high numerical precision (like FP32 or FP16) to maintain training stability. However, this comes with high memory and computational costs. DeepSeek showed that it is possible to train such large models with FP8 precision without compromising the training stability.

The team introduced a fine-grained mixed precision framework. Think of it as using different levels of numerical precision for different parts of the model - like using a precise ruler for critical measurements and a rough estimate for less critical ones. They group model parameters in small tiles (1x128 for activations and 128x128 for weights) and quantize each group independently. This helps manage outliers that typically cause training instability in low-precision training.

They also introduced a smart accumulation strategy where intermediate computations are promoted to higher precision at regular intervals. This ensures that despite using FP8 for most operations, the critical accumulation of gradients remains accurate. This advancement allowed them to achieve the same level of performance as higher precision training while significantly reducing memory usage and computational costs.
