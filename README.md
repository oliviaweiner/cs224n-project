# CS224N final project (2022 IID SQuAD track)

In this project, we aim to reproduce a coattention layer on the Stanford Question
Answering dataset (SQuAD) baseline model, and investigate its relationship with
other common SQuAD techniques. We start by testing how a coattention layer
improves the baseline model and find that when it is not paired with any other
techniques, it lowers performance. Next, we implement the Dynamic Decoder and
Highway Network, as well as Character Embeddings, and find that both increase
the performance of the coattention baseline, but still underperform the standard
baseline. In fact, the baseline only improves in the case where we use character
embeddings with the standard BiDirectional Attention Flow (BiDAF) layer and
single-pass decoder.
