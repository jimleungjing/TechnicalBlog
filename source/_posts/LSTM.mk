---
title: An Overview to LSTM
date: 2018-10-23 23:53:42
categories: 
    - [Paper Notes]
tags: 
    - [Machine Learning]
    - [Deep Learning]
    - [LSTM]
mathjax: true
comments: true
---
<!--
1.       Problem: the problem to be solved, proposed by the author

2.       Solution: how the author solves the proposed problem

3.       Novelty: the difference from previous related work, and pick out the most related paper

4.       Take-away: what you learn from this paper and want to remember
-->
## The Problem LSTM Addressed
&emsp;In the past decades, recurrent neural networks (RNNs) have been successufully applied to a variety problems: speech recognition, language modeling, prediction, etc. RNNs can use their internal state to process sequences of inputs and persist information as shown below in Fig. 1. 
&emsp;However, the RNNs don't seem to be able to learn of long term dependencies in the input/output sequences. It's believed the reason for that problem is that error signals flowing backward in time tend to blow up or vanish([Hochreiter 1991](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)).

## LSTM
&emsp;Fortunately, LSTM address this 
Learning to store information 

## RNN & IIR
　　The similarity between RNN(Recurrent Neural Network) and IIR(Infinite Inpulse Response) suddenly hit me when I learned the concept about LSTM at first.
  I realized that it is similar between RNN and IIR when I first came to the RNN.In the following study, the idea hits me again when learning Adam gredient descent. As a user of Reddit said, it is a two FIR.
  I guess that I'm not the first person to conceive the idea of similarity between RNN and IIR. So I Google the key words "IIR LSTM". Not surprisingly, there are several papers based on the similarity between the IIR and RNN appear in the search results list. In the most recent paper "Feedforward Sequential Memory Networks: A New Structure to Learn Long-term Dependency", the authors proposed a novel network structure to model sequential signals like speech or language. 

