# Optimizing Gradient Descent: Nesterov RMSprop with Momentum

## **📌 Project Overview**
This project implements **Nesterov RMSprop with Momentum** to optimize a quadratic function. The notebook explores the impact of different learning rate schedules and adaptive optimization techniques on convergence speed.

### **🚀 Key Features**
✅ **Mathematical Formulation** of an optimization function  
✅ **Gradient Computation** for parameter updates  
✅ **Implementation of Nesterov RMSprop with Momentum**  
✅ **Comparison of Constant vs. Power-based Learning Rate Schedules**  
✅ **Visualization of Optimization Path using Contour Plots**  

---

## **📌 Optimization Function**
The objective function used for testing optimization techniques:
\[
f(x_1, x_2) = x_1^2 + 2x_2^2
\]

Its gradient:
\[
\nabla f(x_1, x_2) = (2x_1, 4x_2)
\]

---

## **📌 Learning Rate Scheduling**
The optimizer supports two types of learning rate schedules:

1️⃣ **Constant Learning Rate**  
```python
def constant_lr(rate):
    return rate
```
2️⃣ **Power-based Decay Learning Rate
```python
def power_lr(rate, t, c=1, s=10):
    return rate / (1 + t/s)**c
```
