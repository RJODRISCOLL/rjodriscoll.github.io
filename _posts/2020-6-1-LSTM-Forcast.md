---
layout: post
title: Forecasting energy intake
---

The prevalence of obesity has tripled in the last 40 years and it has been estimated that by 2050, 60% of males and 50% of females may be living with obesity. These increases to obesity are clearly multifaceted, as shown in the by the obesity foresight map below: 

Despite this complexity, a simple ineluctable law (first law of thermodynamics) underlies any change in human body weight; body weight can only change when the rate of energy intake exceeds the rate of energy expenditure. I.e. (change in energy stores = energy intake - energy expenditure). 

Evidence suggests that once this weight has been gained, it is hard to lose and even harder to prevent weight regain after weight loss. So this poises the question, can we forecast perturbations to energy intake with advanced neural network architectures? 

But first we need to be able to estimate energy intake...   

### An introduction to modelling energy intake

Above I introduced the first law of thermodynamics in the context of human bodyweight. The first which means that if we know just two of these values continuously, we can estimate the third, EI = EE + ΔES. 

Simple, right? Not so much.  

For this post energy expenditure is derived from a recently published accelerometery based machine learning algorithm  [O'Driscoll, et al. 2020](https://www.tandfonline.com/doi/full/10.1080/02640414.2020.1746088) and energy intake can be approximated from a validated mathematical model [Sanghvi, et al. 2015](https://academic.oup.com/ajcn/article/102/2/353/4564610)   

![](https://github.com/RJODRISCOLL/rjodriscoll.github.io/blob/master/images/Untitled.png)

The above model describes a linearised model to estimate change in energy intake relative to baseline requirements for the ith interval. 

The ρ parameter in this model describes the change energy density associated with the estimated change in body composition and *BW* refers to body weight. The parameter ε describes how the rate of energy expenditure depends on the body weight. 

The Δδ term describes changes to energy expenditure over time (derived from the algorithm above). The parameter β accounts for additional energy cost of digestive and adaptive processes. The parameter *f* refers to additional contributions to energy expenditure.
