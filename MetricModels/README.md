# ✅ Idea

**Main idea**: Take N neighbors of an object and use them to make predictions about the object.

# ✅ Similiarity metrics

![](https://yastatic.net/s3/education-portal/media/2_2_8440e10f60_6a35a4f1bc.webp)

**Euclidean**: $\rho(x, y)=\sqrt{\sum_i(x_i-y_i)^2}$
**Cosine**: $\rho(x, y)=1-\text{cos}(\theta)=1-\frac{x \cdot y}{\left\| x \right\| \cdot \left\| y \right\|}$
**Manhattan**: $\rho(x, y)=\sum_i \left| x_i - y_i \right|$
**Jaccard**: $\rho(A, B)=1=\frac{\left| A \cap  B \right|}{\left| A \cup   B \right|}$
**Minkowski**: $\rho(x, y)=(\sum_i\left| x_i - y_i \right|^p)^\frac{1}{p}$
