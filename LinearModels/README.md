# ✅Linear regression

**Model**: $f(x)=\sum_{i=1}^{N}w_i \cdot x_i+w_0$
**Loss**: $\mathcal{L}(w, X, y)=\frac{1}{N}\sum_{i}^{N}(f(x_i)-y_i)^2$
**Loss grad**: $\nabla_w​L=\frac{N}{2}​X^T(Xw−y)$

# ✅Logistic regression

**Model**: $\text{output}=\sum_{i=1}^{N}w_i \cdot x_i+w_0$; $f(x)=\text{sigmoid(output)}$
**Loss**: $\mathcal{L}(w, W, y)=-\frac{1}{N}\sum_{i}^{N}(y_i​log(σ(⟨w,x_i​⟩))+(1−y_i​)log(σ(−⟨w,xi​⟩)))$
**Loss grad**: $\nabla_w​L(y,X,w)=−\sum_i​x_i​(y_i​−σ(⟨w,x_i​⟩))$

# ✅ Optimizer

**Gradient descent**: $wj_​↦w_j​−α\frac{d}{w_jd}​L(f_w​,X,y)$
# ✅ Regularization techniques

Also uses with $\alpha$ - reqularization strength coeficient.
- **L1(lasso)**: $∥w∥_1​=\sum_{j=1}^D​∣w_j​∣$
- **L2(Ridge)**: $∥w∥_2^2​=\sum_{j=1}^D​w_j^2$
- **ElasticNet**: $\text{L1} + \text{L2}$

​